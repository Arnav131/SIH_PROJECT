package com.roadai.androidtest

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.SeekBar
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.roadai.androidtest.config.DetectionConfig
import com.roadai.androidtest.detection.DetectionType
import com.roadai.androidtest.detection.PerceptionResult
import com.roadai.androidtest.inference.InferenceScheduler
import com.roadai.androidtest.models.ModelRegistry
import com.roadai.androidtest.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * ROAD ANOMALY DETECTION — unified multi-specialist perception, one screen.
 *
 *     CameraX -> ImageAnalysis -> upright bitmap
 *             -> InferenceScheduler (pothole specialist || crack specialist)
 *             -> DetectionFusionEngine
 *             -> unified overlay
 *
 * This activity knows nothing about ONNX, class indices, or which model found
 * what. Adding a third specialist does not change this file.
 *
 * No IMU, GPS, networking or persistence — those come later.
 */
class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: ActivityMainBinding

    private var scheduler: InferenceScheduler? = null
    private lateinit var analysisExecutor: ExecutorService

    private var config = DetectionConfig.DEFAULT

    // Rolling FPS accounting, so the panel reports measured rates rather than
    // the configured target.
    //
    // NOTE: framesSeen counts frames DELIVERED TO THE ANALYZER, which is not the
    // preview's frame rate. With STRATEGY_KEEP_ONLY_LATEST, CameraX stops
    // delivering while the analyzer holds a frame, so this number tracks
    // inference throughput, not camera capture. The preview renders on its own
    // Surface and stays smooth regardless of what this reads.
    private var framesSeen = 0
    private var framesProcessed = 0
    private var windowStartMs = SystemClock.elapsedRealtime()

    // Written on the analyzer thread, read on the UI thread when building the
    // status text.
    @Volatile
    private var cameraFps = 0f

    @Volatile
    private var processedFps = 0f

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                setStatus("Status: CAMERA PERMISSION DENIED\nGrant it in Settings, then reopen the app.")
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // FIT_CENTER matches the geometry OverlayView assumes. See OverlayView.
        binding.previewView.scaleType = PreviewView.ScaleType.FIT_CENTER

        analysisExecutor = Executors.newSingleThreadExecutor()

        // One slider drives both specialists' thresholds together for quick
        // field tuning. They remain independently configurable in DetectionConfig.
        binding.thresholdSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                val t = progress / 100f
                config = config.copy(potholeThreshold = t, crackThreshold = t)
                scheduler?.config = config
                binding.thresholdLabel.text = "Confidence threshold: %.2f (both specialists)".format(t)
            }

            override fun onStartTrackingTouch(sb: SeekBar?) = Unit
            override fun onStopTrackingTouch(sb: SeekBar?) = Unit
        })

        loadModels()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    /**
     * Loading builds two ORT sessions over ~80 MB of assets, so it must not run
     * on the main thread.
     */
    private fun loadModels() {
        setStatus("Status: LOADING ${ModelRegistry.ACTIVE.size} SPECIALIST MODELS...")
        analysisExecutor.execute {
            try {
                val s = InferenceScheduler(applicationContext, config)
                scheduler = s
                runOnUiThread {
                    setStatus(
                        "Road anomaly detection: ${s.modelCount} specialists loaded\n" +
                            "Mode: ${config.executionMode}  target ${config.targetInferenceFps} FPS\n" +
                            "Status: RUNNING"
                    )
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Model load failed", t)
                runOnUiThread { setStatus("Status: LOAD FAILED\n${t.message}") }
            }
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = try {
                providerFuture.get()
            } catch (t: Throwable) {
                Log.e(TAG, "Camera provider failed", t)
                setStatus("Status: CAMERA INIT FAILED\n${t.message}")
                return@addListener
            }

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            // KEEP_ONLY_LATEST: stale frames are discarded by CameraX itself, so
            // the analyzer never falls behind the camera.
            // RGBA_8888 lets us use ImageProxy.toBitmap() and skips a manual YUV
            // conversion — fewer ways to get the colour channels wrong.
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(analysisExecutor) { proxy -> analyze(proxy) }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
            } catch (t: Throwable) {
                Log.e(TAG, "Camera bind failed", t)
                setStatus("Status: CAMERA BIND FAILED\n${t.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /** Runs on [analysisExecutor] — never on the main thread. */
    private fun analyze(proxy: ImageProxy) {
        framesSeen++

        val s = scheduler
        // tryAcquire enforces both "not already busy" and the FPS cap, so
        // inference work can never pile up behind a slow frame.
        if (s == null || !s.tryAcquire()) {
            updateRates()
            proxy.close()
            return
        }

        var upright: Bitmap? = null
        try {
            val raw = proxy.toBitmap()
            upright = rotate(raw, proxy.imageInfo.rotationDegrees)

            val result = s.runOnce(upright)
            framesProcessed++
            updateRates()

            runOnUiThread {
                binding.overlayView.setResult(result)
                setStatus(describe(result))
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Perception cycle failed", t)
            runOnUiThread { setStatus("Status: INFERENCE ERROR\n${t.message}") }
        } finally {
            upright?.recycle()
            s.release()
            proxy.close()
        }
    }

    /** Builds the status panel text. Model filenames stay out of the UI. */
    private fun describe(r: PerceptionResult): String {
        val potholes = r.detections.count { it.type == DetectionType.POTHOLE }
        val cracks = r.detections.count { it.type == DetectionType.CRACK }
        val other = r.detections.count { it.type == DetectionType.OTHER }

        val perModel = r.perModel.values.joinToString("  ") {
            "${it.source.removeSuffix("_model")}=${it.inferenceMs}ms"
        }

        val top = r.detections.take(3).joinToString(", ") {
            "${it.displayLabel} %.2f".format(it.confidence)
        }.ifEmpty { "none" }

        return buildString {
            append("ROAD ANOMALIES  pothole=$potholes  crack=$cracks")
            if (other > 0) append("  other=$other")
            append("\n")
            append("Latency ${r.totalLatencyMs}ms  [$perModel]  ${config.executionMode}\n")
            append("Analyzer %.1f FPS  processed %.1f FPS".format(cameraFps, processedFps))
            if (r.duplicatesRemoved > 0) append("  dedup=${r.duplicatesRemoved}")
            append("\n")
            append("Top: $top")
        }
    }

    /** Recomputes the rolling FPS figures once per second. */
    private fun updateRates() {
        val now = SystemClock.elapsedRealtime()
        val dt = now - windowStartMs
        if (dt >= 1000) {
            cameraFps = framesSeen * 1000f / dt
            processedFps = framesProcessed * 1000f / dt
            framesSeen = 0
            framesProcessed = 0
            windowStartMs = now
        }
    }

    private fun rotate(src: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return src
        val m = Matrix().apply { postRotate(degrees.toFloat()) }
        val out = Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
        if (out !== src) src.recycle()
        return out
    }

    private fun setStatus(text: String) {
        binding.statusText.text = text
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        scheduler?.close()
        scheduler = null
    }
}
