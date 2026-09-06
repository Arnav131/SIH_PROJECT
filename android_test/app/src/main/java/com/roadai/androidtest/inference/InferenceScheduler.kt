package com.roadai.androidtest.inference

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.roadai.androidtest.config.DetectionConfig
import com.roadai.androidtest.config.ExecutionMode
import com.roadai.androidtest.detection.ModelResult
import com.roadai.androidtest.detection.PerceptionResult
import com.roadai.androidtest.fusion.DetectionFusionEngine
import com.roadai.androidtest.models.ModelRegistry
import com.roadai.androidtest.models.ModelSpec
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Owns every specialist, decides when a frame is worth running, runs them, and
 * hands the fused result back.
 *
 * Backpressure is deliberate and layered:
 *   - CameraX is configured KEEP_ONLY_LATEST, so stale frames never queue.
 *   - [busy] means a frame arriving mid-cycle is DROPPED, not queued, so work
 *     can never accumulate.
 *   - [DetectionConfig.targetInferenceFps] caps how often a cycle may start,
 *     independent of camera FPS, so the preview keeps its own frame rate even
 *     when inference is slow.
 *
 * The net effect is graceful degradation: on a slow phone the overlay updates
 * less often, but the camera never stutters and memory never grows.
 */
class InferenceScheduler(
    context: Context,
    initialConfig: DetectionConfig = DetectionConfig.DEFAULT,
    specs: List<ModelSpec> = ModelRegistry.ACTIVE
) : AutoCloseable {

    companion object {
        private const val TAG = "InferenceScheduler"
    }

    /** Read on the analyzer thread, written from the UI thread when the slider moves. */
    @Volatile
    var config: DetectionConfig = initialConfig

    private val detectors: List<YoloOnnxDetector> = specs.map { YoloOnnxDetector(context, it) }
    private val fusion = DetectionFusionEngine()

    /**
     * One worker per specialist, so PARALLEL mode genuinely runs them at the
     * same time. Each detector is confined to its own thread, which is what
     * lets the detectors reuse their input buffers safely.
     */
    private val workers: List<ExecutorService> =
        detectors.map { Executors.newSingleThreadExecutor() }

    private val busy = AtomicBoolean(false)
    private var lastStartMs = 0L

    /** What the UI shows about which specialists are live. */
    val loadedSummary: String
        get() = detectors.joinToString(" + ") { "${it.spec.source}(${it.classNames.size} cls)" }

    val modelCount: Int get() = detectors.size

    /**
     * True if a cycle may start now: nothing in flight, and the FPS cap allows it.
     * Callers that get false should close the frame and move on.
     */
    fun tryAcquire(): Boolean {
        val now = SystemClock.elapsedRealtime()
        if (now - lastStartMs < config.minFrameIntervalMs) return false
        if (!busy.compareAndSet(false, true)) return false
        lastStartMs = now
        return true
    }

    fun release() {
        busy.set(false)
    }

    /**
     * Run every specialist on the SAME frame and fuse the results.
     *
     * Must be called from a background thread, never the UI thread, and only
     * after [tryAcquire] returned true. The caller is responsible for calling
     * [release] afterwards.
     */
    fun runOnce(bitmap: Bitmap): PerceptionResult {
        // Snapshot once so the whole cycle uses a consistent config even if the
        // UI thread changes it mid-frame.
        val cfg = config
        val t0 = SystemClock.elapsedRealtime()

        val results: List<ModelResult> = when (cfg.executionMode) {
            ExecutionMode.PARALLEL -> runParallel(bitmap, cfg)
            ExecutionMode.SEQUENTIAL -> runSequential(bitmap, cfg)
        }

        val elapsed = SystemClock.elapsedRealtime() - t0
        if (elapsed > cfg.slowFrameWarningMs) {
            Log.w(TAG, "slow perception cycle: ${elapsed}ms (${cfg.executionMode})")
        }

        return fusion.fuse(results, bitmap.width, bitmap.height, elapsed, cfg)
    }

    /**
     * Both specialists see the same immutable bitmap concurrently. Nothing
     * writes to it, so no copy is needed.
     */
    private fun runParallel(bitmap: Bitmap, cfg: DetectionConfig): List<ModelResult> {
        val tasks = detectors.mapIndexed { i, det ->
            workers[i].submit(Callable { safeDetect(det, bitmap, cfg) })
        }
        return tasks.mapNotNull { future ->
            try {
                future.get()
            } catch (t: Throwable) {
                Log.e(TAG, "specialist failed", t)
                null
            }
        }
    }

    private fun runSequential(bitmap: Bitmap, cfg: DetectionConfig): List<ModelResult> =
        detectors.mapNotNull { safeDetect(it, bitmap, cfg) }

    /**
     * One failing specialist must not take the whole perception layer down —
     * the frame is still reported using whatever else succeeded.
     */
    private fun safeDetect(
        det: YoloOnnxDetector,
        bitmap: Bitmap,
        cfg: DetectionConfig
    ): ModelResult? = try {
        det.detect(bitmap, thresholdFor(det.spec, cfg), cfg.nmsIouThreshold)
    } catch (t: Throwable) {
        Log.e(TAG, "${det.spec.source} inference failed", t)
        null
    }

    /** Each specialist gets its own independently tunable threshold. */
    private fun thresholdFor(spec: ModelSpec, cfg: DetectionConfig): Float = when (spec.source) {
        ModelRegistry.POTHOLE.source -> cfg.potholeThreshold
        ModelRegistry.CRACK.source -> cfg.crackThreshold
        else -> cfg.potholeThreshold
    }

    override fun close() {
        workers.forEach { w ->
            w.shutdown()
            try {
                if (!w.awaitTermination(2, TimeUnit.SECONDS)) w.shutdownNow()
            } catch (_: InterruptedException) {
                w.shutdownNow()
            }
        }
        detectors.forEach { it.close() }
    }
}
