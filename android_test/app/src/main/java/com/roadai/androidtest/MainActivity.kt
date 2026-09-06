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
import android.content.Intent
import android.os.Build
import android.view.Gravity
import android.widget.Toast
import androidx.core.view.GravityCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.roadai.androidtest.config.DetectionConfig
import com.roadai.androidtest.fusion.ConfirmationStatus
import com.roadai.androidtest.fusion.RoadEvent
import com.roadai.androidtest.fusion.RoadEventConfirmation
import com.roadai.androidtest.sensors.AppState
import com.roadai.androidtest.sensors.CollectionService
import com.roadai.androidtest.sensors.Gps
import com.roadai.androidtest.sensors.UrbanEngine
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
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
 * The sensor engine migrated from UrbanSenseAI runs ALONGSIDE this, never inside it:
 * it owns the accelerometer, gyroscope, GPS, WebSocket and session, and publishes shocks.
 * This activity correlates those shocks with what the camera saw and emits road events.
 * There is exactly one camera in this app and it is the one bound below.
 */
class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: ActivityMainBinding

    private var scheduler: InferenceScheduler? = null
    private lateinit var analysisExecutor: ExecutorService

    private var config = DetectionConfig.DEFAULT

    /** Visual + physical corroboration. Separate from DetectionFusionEngine by design. */
    private val confirmation = RoadEventConfirmation()

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

    /**
     * Location and notifications are requested separately from CAMERA so a refusal
     * degrades the sensor side only — the detector keeps working without them.
     */
    private val requestSensorPermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
            UrbanEngine.startPreview()
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

        UrbanEngine.initialize(this)
        wireSensorDrawer()
        observeSensorState()
        requestSensorPermissions.launch(sensorPermissions())

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
            // Stamped BEFORE inference. On this device a cycle takes ~3.5 s, so a
            // detection's own timestamp trails the physical moment by seconds and is
            // useless for correlation. Everything downstream anchors on this instead.
            val frameCapturedAtMs = System.currentTimeMillis()
            val gpsAtCapture: Gps = UrbanEngine.state.value.gps

            val raw = proxy.toBitmap()
            upright = rotate(raw, proxy.imageInfo.rotationDegrees)

            val result = s.runOnce(upright)
            framesProcessed++
            updateRates()

            val events = confirmation.evaluate(
                result = result,
                frameCapturedAtMs = frameCapturedAtMs,
                shock = UrbanEngine.lastShock,
                gps = gpsAtCapture
            )
            events.forEach { transmit(it) }

            runOnUiThread {
                binding.overlayView.setResult(result)
                setStatus(describe(result))
                events.lastOrNull()?.let { showEventBanner(it) }
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

    /** Hands a decided road event to the sensor engine's existing transport. */
    private fun transmit(e: RoadEvent) {
        UrbanEngine.submitRoadEvent(
            eventId = e.eventId,
            eventType = e.type.name,
            visualConfidence = e.visualConfidence,
            confirmationStatus = e.status.name,
            sensorSupport = e.sensorSupport.name,
            detectionLabel = e.label,
            shock = e.shock,
            correlationDeltaMs = e.correlationDeltaMs,
            latitude = e.latitude,
            longitude = e.longitude,
            gpsAccuracy = e.gpsAccuracy,
            frameCapturedAtMs = e.frameCapturedAtMs
        )
    }

    private fun showEventBanner(e: RoadEvent) {
        val where = if (e.hasLocation) "%.6f, %.6f".format(e.latitude, e.longitude) else "no fix"
        val support = when (e.status) {
            ConfirmationStatus.CONFIRMED -> "CONFIRMED by IMU (%.1fx baseline)".format(e.shock?.score ?: 0f)
            ConfirmationStatus.SUPPORTED -> "SUPPORTED by IMU"
            ConfirmationStatus.VISUAL_ONLY -> "VISUAL ONLY"
        }
        binding.eventBanner.text =
            "${e.label.uppercase()}  %.2f\n$support\n$where".format(e.visualConfidence)
        binding.eventBanner.visibility = android.view.View.VISIBLE
        binding.eventBanner.removeCallbacks(hideBanner)
        binding.eventBanner.postDelayed(hideBanner, 6_000)
    }

    private val hideBanner = Runnable {
        binding.eventBanner.visibility = android.view.View.GONE
    }

    private fun sensorPermissions(): Array<String> {
        val wanted = mutableListOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            wanted += Manifest.permission.POST_NOTIFICATIONS
        }
        return wanted.toTypedArray()
    }

    private fun wireSensorDrawer() {
        val prefs = getSharedPreferences("roadai", MODE_PRIVATE)
        binding.receiverUrl.setText(prefs.getString("receiver", ""))

        binding.sensorsButton.setOnClickListener {
            if (binding.drawerLayout.isDrawerOpen(GravityCompat.END)) {
                binding.drawerLayout.closeDrawer(GravityCompat.END)
            } else {
                binding.drawerLayout.openDrawer(GravityCompat.END)
            }
        }

        binding.connectButton.setOnClickListener {
            val url = binding.receiverUrl.text.toString()
            val r = UrbanEngine.configure(url, "", "urbansenseai_demo_token")
            if (r.isFailure) {
                Toast.makeText(this, r.exceptionOrNull()?.message, Toast.LENGTH_LONG).show()
            } else {
                prefs.edit().putString("receiver", url).apply()
                UrbanEngine.connect()
            }
        }
        binding.disconnectButton.setOnClickListener { UrbanEngine.disconnect() }

        binding.startButton.setOnClickListener {
            val url = binding.receiverUrl.text.toString()
            val r = UrbanEngine.configure(url, "", "urbansenseai_demo_token")
            if (r.isFailure) {
                Toast.makeText(this, r.exceptionOrNull()?.message, Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }
            prefs.edit().putString("receiver", url).apply()
            // The service only keeps the process alive; a failure there must not take
            // collection down with it, so the engine starts either way.
            try {
                ContextCompat.startForegroundService(this, Intent(this, CollectionService::class.java))
            } catch (t: Throwable) {
                Log.w(TAG, "foreground service unavailable", t)
            }
            confirmation.reset()
            UrbanEngine.start()
        }
        binding.stopButton.setOnClickListener {
            UrbanEngine.stop()
            stopService(Intent(this, CollectionService::class.java))
        }
    }

    /**
     * StateFlow is already conflated, so a plain collect with a small pause renders the
     * newest sample at a readable rate. collectLatest would cancel each render before the
     * main thread could run it once the sensors emit at 20 Hz.
     */
    private fun observeSensorState() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                UrbanEngine.state.collect { st ->
                    renderSensors(st)
                    delay(200)
                }
            }
        }
    }

    private fun renderSensors(s: AppState) {
        binding.connectionValue.text = "● ${s.connection}\n${s.wsUrl.ifBlank { "no receiver set" }}"
        binding.accelValue.text = "X %+.3f\nY %+.3f\nZ %+.3f\n%s"
            .format(s.accel.x, s.accel.y, s.accel.z, if (s.accel.available) "available" else "unavailable")
        binding.gyroValue.text = "X %+.3f\nY %+.3f\nZ %+.3f\n%s"
            .format(s.gyro.x, s.gyro.y, s.gyro.z, if (s.gyro.available) "available" else "unavailable")
        binding.gpsValue.text = if (s.gps.active) {
            "Lat %.6f\nLon %.6f\nAlt %.1f m\n± %.1f m (%s)"
                .format(s.gps.latitude, s.gps.longitude, s.gps.altitude, s.gps.accuracy, s.gps.provider)
        } else if (s.gps.enabled) "waiting for first fix…" else "unavailable / disabled"
        binding.shockValue.text = "Score %.1f\nEvents %d\nLast %s"
            .format(s.shockScore, s.events, s.lastEvent.ifBlank { "none" })
        binding.roadEventValue.text = "Transmitted ${s.roadEventCount}"
        binding.transmissionValue.text =
            "Packets ${s.sent}\nAccel ${s.accelCount}  Gyro ${s.gyroCount}\nGPS ${s.gpsCount}\n" +
                "Buffered ${s.buffered}\nLast ${s.lastSent.ifBlank { "-" }}"
        binding.warningValue.text = s.warning
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

    @Deprecated("Back handling kept simple; the drawer must close before the activity finishes.")
    override fun onBackPressed() {
        if (binding.drawerLayout.isDrawerOpen(GravityCompat.END)) {
            binding.drawerLayout.closeDrawer(GravityCompat.END)
        } else {
            @Suppress("DEPRECATION")
            super.onBackPressed()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        scheduler?.close()
        scheduler = null
    }
}
