package com.roadai.androidtest.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.roadai.androidtest.detection.Detection
import com.roadai.androidtest.detection.ModelResult
import com.roadai.androidtest.models.ModelSpec
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Runs ONE specialist and emits detections already normalized into the common
 * format. Model-specific knowledge lives in the [ModelSpec], so this class is
 * shared by every specialist and never needs editing to add one.
 *
 * The decode path handles the v8-style single-output head both current models
 * use: [1, 4+nc, anchors], boxes in input-space pixels, sigmoid independent
 * class scores, NMS not in the graph. It reads the real output shape at load
 * time and logs loudly if the asset disagrees with its spec, so a swapped or
 * re-exported model cannot silently mis-decode.
 *
 * Instances are independent and each owns its ORT session, so two can run
 * concurrently on different threads.
 */
class YoloOnnxDetector(
    context: Context,
    val spec: ModelSpec
) : AutoCloseable {

    companion object {
        private const val TAG = "YoloOnnxDetector"
        private const val PAD_VALUE = 114 // Ultralytics letterbox grey
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val inputSize = spec.inputSize

    /** Class names as reported by the model itself, falling back to the spec. */
    val classNames: Array<String>

    /** Actual anchor count read from the loaded graph, not assumed. */
    private val anchorCount: Int

    /** Actual class count read from the loaded graph, not assumed. */
    private val classCount: Int

    /** Reused across frames so nothing large is allocated per frame. */
    private val inputBuffer = FloatArray(3 * inputSize * inputSize)
    private val pixelBuffer = IntArray(inputSize * inputSize)

    init {
        val modelBytes = context.assets.open(spec.asset).use { it.readBytes() }

        val options = OrtSession.SessionOptions().apply {
            // Both specialists may run at once, so each is capped well below the
            // core count. Leaving headroom keeps the camera thread responsive.
            setIntraOpNumThreads(max(1, Runtime.getRuntime().availableProcessors() / 4))
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }

        session = env.createSession(modelBytes, options)
        inputName = session.inputNames.first()

        // Read the real output geometry instead of trusting the spec blindly.
        val outShape = session.outputInfo.values.first().info
            .let { it as ai.onnxruntime.TensorInfo }.shape
        val channels = if (outShape.size >= 3) outShape[1].toInt() else (4 + spec.numClasses)
        anchorCount = if (outShape.size >= 3) outShape[2].toInt() else spec.numAnchors
        classCount = channels - 4

        if (classCount != spec.numClasses || anchorCount != spec.numAnchors) {
            Log.w(
                TAG,
                "${spec.asset}: graph says nc=$classCount anchors=$anchorCount, " +
                    "spec says nc=${spec.numClasses} anchors=${spec.numAnchors}. " +
                    "Using the graph."
            )
        }

        classNames = readClassNamesFromMetadata() ?: spec.fallbackNames

        Log.i(
            TAG,
            "Loaded ${spec.asset} | in=$inputName ${inputSize}x$inputSize | " +
                "nc=$classCount anchors=$anchorCount | classes=${classNames.joinToString()}"
        )
    }

    /**
     * Ultralytics writes the class list into ONNX metadata_props as a Python
     * dict literal: {0: 'Pothole'}. Reading it from the model means labels
     * cannot drift if an asset is re-exported.
     */
    private fun readClassNamesFromMetadata(): Array<String>? {
        return try {
            val raw = session.metadata.customMetadata["names"] ?: return null
            val quote = '\''
            val pattern = Regex("(\\d+)\\s*:\\s*" + quote + "([^" + quote + "]*)" + quote)
            val pairs = pattern.findAll(raw)
                .map { it.groupValues[1].toInt() to it.groupValues[2] }
                .toList()
            if (pairs.isEmpty()) return null
            val maxId = pairs.maxOf { it.first }
            Array(maxId + 1) { idx -> pairs.firstOrNull { it.first == idx }?.second ?: "class_$idx" }
        } catch (t: Throwable) {
            Log.w(TAG, "Could not read class names from ${spec.asset}; using fallback", t)
            null
        }
    }

    /**
     * Run one frame. [bitmap] must already be upright (rotation applied).
     * Thread-confined: one detector instance must not be called concurrently
     * from two threads, because it reuses its input buffers. The scheduler
     * gives each specialist its own instance and its own thread.
     */
    fun detect(bitmap: Bitmap, confThreshold: Float, iouThreshold: Float): ModelResult {

        val t0 = SystemClock.elapsedRealtime()
        val lb = letterbox(bitmap)
        val t1 = SystemClock.elapsedRealtime()

        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val raw: Array<Array<FloatArray>>
        OnnxTensor.createTensor(env, FloatBuffer.wrap(inputBuffer), shape).use { input ->
            session.run(mapOf(inputName to input)).use { results ->
                @Suppress("UNCHECKED_CAST")
                raw = results[0].value as Array<Array<FloatArray>>
            }
        }
        val t2 = SystemClock.elapsedRealtime()

        val dets = decodeAndNms(raw[0], lb, confThreshold, iouThreshold)
        val t3 = SystemClock.elapsedRealtime()

        return ModelResult(
            source = spec.source,
            detections = dets,
            preprocessMs = t1 - t0,
            inferenceMs = t2 - t1,
            postprocessMs = t3 - t2
        )
    }

    // =====================================================================
    // Preprocessing
    // =====================================================================

    private class Letterboxed(
        val scale: Float,
        val padX: Int,
        val padY: Int,
        val srcWidth: Int,
        val srcHeight: Int
    )

    /**
     * Aspect-preserving resize into inputSize² with grey (114) padding, then
     * RGB / 255 / HWC->CHW. Writes into the reused buffer rather than allocating.
     * No ImageNet mean/std — neither model expects it.
     */
    private fun letterbox(src: Bitmap): Letterboxed {
        val sw = src.width
        val sh = src.height
        val scale = min(inputSize.toFloat() / sw, inputSize.toFloat() / sh)
        val nw = (sw * scale).roundToInt().coerceAtLeast(1)
        val nh = (sh * scale).roundToInt().coerceAtLeast(1)
        val padX = (inputSize - nw) / 2
        val padY = (inputSize - nh) / 2

        val resized = Bitmap.createScaledBitmap(src, nw, nh, true)
        resized.getPixels(pixelBuffer, 0, nw, 0, 0, nw, nh)
        if (resized !== src) resized.recycle()

        val area = inputSize * inputSize
        java.util.Arrays.fill(inputBuffer, PAD_VALUE / 255f)

        val gOff = area
        val bOff = 2 * area
        for (y in 0 until nh) {
            val dstRow = (y + padY) * inputSize + padX
            val srcRow = y * nw
            for (x in 0 until nw) {
                val p = pixelBuffer[srcRow + x]
                val i = dstRow + x
                inputBuffer[i] = ((p shr 16) and 0xFF) / 255f          // R
                inputBuffer[gOff + i] = ((p shr 8) and 0xFF) / 255f    // G
                inputBuffer[bOff + i] = (p and 0xFF) / 255f            // B
            }
        }

        return Letterboxed(scale, padX, padY, sw, sh)
    }

    // =====================================================================
    // Postprocessing
    // =====================================================================

    /**
     * [out] is [4+nc][anchors] — channels-first. Column-major access here is
     * the "transpose to [anchors, 4+nc]" step, done without allocating it.
     * Classes the spec's mapper rejects are dropped before NMS.
     */
    private fun decodeAndNms(
        out: Array<FloatArray>,
        lb: Letterboxed,
        confThreshold: Float,
        iouThreshold: Float
    ): List<Detection> {

        val cx = out[0]
        val cy = out[1]
        val w = out[2]
        val h = out[3]

        val candidates = ArrayList<Detection>()
        val anchors = min(anchorCount, cx.size)
        val now = System.currentTimeMillis()

        for (i in 0 until anchors) {
            var bestId = 0
            var bestScore = out[4][i]
            for (c in 1 until classCount) {
                val s = out[4 + c][i]
                if (s > bestScore) {
                    bestScore = s
                    bestId = c
                }
            }
            if (bestScore < confThreshold) continue

            val name = classNames.getOrElse(bestId) { "class_$bestId" }
            val mapping = spec.mapper(bestId, name) ?: continue

            val halfW = w[i] / 2f
            val halfH = h[i] / 2f
            var x1 = cx[i] - halfW
            var y1 = cy[i] - halfH
            var x2 = cx[i] + halfW
            var y2 = cy[i] + halfH

            // Undo the letterbox: remove padding, then remove the scale.
            x1 = (x1 - lb.padX) / lb.scale
            y1 = (y1 - lb.padY) / lb.scale
            x2 = (x2 - lb.padX) / lb.scale
            y2 = (y2 - lb.padY) / lb.scale

            x1 = x1.coerceIn(0f, lb.srcWidth.toFloat())
            y1 = y1.coerceIn(0f, lb.srcHeight.toFloat())
            x2 = x2.coerceIn(0f, lb.srcWidth.toFloat())
            y2 = y2.coerceIn(0f, lb.srcHeight.toFloat())
            if (x2 <= x1 || y2 <= y1) continue

            candidates.add(
                Detection(
                    source = spec.source,
                    className = name,
                    type = mapping.type,
                    subtype = mapping.subtype,
                    confidence = bestScore,
                    x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                    timestampMs = now
                )
            )
        }

        return nms(candidates, iouThreshold)
    }

    /** Greedy NMS within this model, applied per class name. */
    private fun nms(input: List<Detection>, iouThreshold: Float): List<Detection> {
        if (input.isEmpty()) return emptyList()
        val kept = ArrayList<Detection>()

        for (name in input.map { it.className }.distinct()) {
            val sorted = input.filter { it.className == name }
                .sortedByDescending { it.confidence }
                .toMutableList()

            while (sorted.isNotEmpty()) {
                val best = sorted.removeAt(0)
                kept.add(best)
                sorted.removeAll { iouOf(best, it) > iouThreshold }
            }
        }
        return kept.sortedByDescending { it.confidence }
    }

    override fun close() {
        try {
            session.close()
        } catch (_: Throwable) {
        }
    }
}

/** Shared IoU so the detector and the fusion engine cannot drift apart. */
fun iouOf(a: Detection, b: Detection): Float {
    val ix1 = max(a.x1, b.x1)
    val iy1 = max(a.y1, b.y1)
    val ix2 = min(a.x2, b.x2)
    val iy2 = min(a.y2, b.y2)
    val iw = (ix2 - ix1).coerceAtLeast(0f)
    val ih = (iy2 - iy1).coerceAtLeast(0f)
    val inter = iw * ih
    if (inter <= 0f) return 0f
    val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
    val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
    val union = areaA + areaB - inter
    return if (union <= 0f) 0f else inter / union
}
