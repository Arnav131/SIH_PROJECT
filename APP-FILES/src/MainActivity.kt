package com.urbansenseai

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Size
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

 private lateinit var receiver: EditText
 private lateinit var device: EditText
 private lateinit var token: EditText
 private val labels = mutableMapOf<String, TextView>()
 private var capture: ImageCapture? = null
 private var previewView: PreviewView? = null
 private var autoButton: Button? = null
 private var sensitivityButton: Button? = null
 private val cameraExecutor = Executors.newSingleThreadExecutor()

 private val permissions = registerForActivityResult(
  ActivityResultContracts.RequestMultiplePermissions()
 ) {
  // Registration is retried only after the runtime permission result; previously
  // the listeners could remain absent for the entire Activity lifetime.
  UrbanEngine.startPreview()
  bindCamera()
 }

 override fun onCreate(savedInstanceState: Bundle?) {
  super.onCreate(savedInstanceState)
  UrbanEngine.initialize(this)
  val p = getSharedPreferences("urban", 0)
  receiver = EditText(this).apply {
   hint = "Receiver URL e.g. http://192.168.1.5:8000"
   setSingleLine()
   setText(p.getString("receiver", ""))
  }
  device = EditText(this).apply {
   hint = "Device ID"
   setSingleLine()
   setText(p.getString("device", "realme3pro_001"))
  }
  token = EditText(this).apply {
   hint = "Device token"
   setSingleLine()
   setText(p.getString("token", "urbansenseai_demo_token"))
  }
  buildUi()
  requestRuntimePermissions()
  observeState()
  // The engine spots the jolt; only the Activity owns a bound camera, so it takes the photo.
  UrbanEngine.onShock = { event -> runOnUiThread { takePhoto(event) } }
 }

 private fun requestRuntimePermissions() {
  val wanted = mutableListOf(
   Manifest.permission.ACCESS_FINE_LOCATION,
   Manifest.permission.ACCESS_COARSE_LOCATION,
   Manifest.permission.CAMERA
  )
  if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
   wanted += Manifest.permission.POST_NOTIFICATIONS
  }
  permissions.launch(wanted.toTypedArray())
 }

 /**
  * StateFlow already conflates, so a plain collect with a small pause renders the newest
  * sample at a readable rate. collectLatest cancelled each render before the main thread
  * could run it once the sensors emitted at 40 Hz, which froze the X/Y/Z readouts.
  */
 private fun observeState() {
  lifecycleScope.launch {
   repeatOnLifecycle(Lifecycle.State.STARTED) {
    UrbanEngine.state.collect { s ->
     render(s)
     delay(100)
    }
   }
  }
 }

 private fun text(t: String, size: Float = 16f) = TextView(this).apply {
  text = t
  textSize = size
  setPadding(0, 8, 0, 8)
 }

 private fun btn(t: String, act: () -> Unit) = Button(this).apply {
  text = t
  setOnClickListener { act() }
 }

 private fun value(key: String) = TextView(this).also {
  labels[key] = it
  it.textSize = 15f
  it.setPadding(0, 6, 0, 14)
 }

 private fun buildUi() {
  val scroll = ScrollView(this)
  val root = LinearLayout(this).apply {
   orientation = LinearLayout.VERTICAL
   setPadding(28, 24, 28, 24)
  }
  scroll.addView(root)

  root.addView(text("UrbanSenseAI", 28f))
  root.addView(text("Real-Time Sensor Data Collector"))

  root.addView(text("CONNECTION", 20f))
  root.addView(receiver)
  root.addView(device)
  root.addView(token)
  val line = LinearLayout(this)
  line.addView(btn("CONNECT") { configureAndConnect() }, LinearLayout.LayoutParams(0, -2, 1f))
  line.addView(btn("DISCONNECT") { UrbanEngine.disconnect() }, LinearLayout.LayoutParams(0, -2, 1f))
  root.addView(line)
  root.addView(value("connection"))
  root.addView(value("destination"))

  root.addView(text("ACCELEROMETER", 20f))
  root.addView(value("accel"))
  root.addView(text("GYROSCOPE", 20f))
  root.addView(value("gyro"))
  root.addView(text("GPS", 20f))
  root.addView(value("gps"))

  root.addView(text("CAMERA", 20f))
  val preview = PreviewView(this)
  previewView = preview
  root.addView(preview, LinearLayout.LayoutParams(-1, 420))
  root.addView(btn("CAPTURE") { takePhoto(null) })

  root.addView(text("AUTO-CAPTURE ON SHOCK", 20f))
  root.addView(text("A sharp, repeated jolt - a pothole, a kerb, a hard brake - photographs itself and tags the frame with the current fix. Keep this screen open: the camera is bound to it."))
  val autoRow = LinearLayout(this)
  autoButton = btn("AUTO-CAPTURE: ON") { toggleAuto() }
  sensitivityButton = btn("SENSITIVITY: MEDIUM") { cycleSensitivity() }
  autoRow.addView(autoButton, LinearLayout.LayoutParams(0, -2, 1f))
  autoRow.addView(sensitivityButton, LinearLayout.LayoutParams(0, -2, 1f))
  root.addView(autoRow)
  root.addView(value("shock"))

  val controls = LinearLayout(this)
  controls.addView(btn("START COLLECTION") { configureAndStart() }, LinearLayout.LayoutParams(0, -2, 1f))
  controls.addView(btn("STOP COLLECTION") { stopCollection() }, LinearLayout.LayoutParams(0, -2, 1f))
  root.addView(controls)

  root.addView(text("TRANSMISSION", 20f))
  root.addView(value("transmission"))
  root.addView(value("warning"))

  setContentView(scroll)
  preview.post { bindCamera() }
 }

 private fun configureAndConnect() {
  val r = UrbanEngine.configure(receiver.text.toString(), device.text.toString(), token.text.toString())
  if (r.isFailure) {
   Toast.makeText(this, r.exceptionOrNull()?.message, Toast.LENGTH_LONG).show()
   return
  }
  save()
  UrbanEngine.connect()
 }

 private fun configureAndStart() {
  val r = UrbanEngine.configure(receiver.text.toString(), device.text.toString(), token.text.toString())
  if (r.isFailure) {
   Toast.makeText(this, r.exceptionOrNull()?.message, Toast.LENGTH_LONG).show()
   return
  }
  save()
  // The foreground service only keeps the process alive; a failure there must not take
  // the collection down with it, so the engine starts either way.
  try {
   ContextCompat.startForegroundService(this, Intent(this, CollectionService::class.java))
  } catch (e: Exception) {
   Toast.makeText(this, "Background service unavailable: " + e.message, Toast.LENGTH_LONG).show()
  }
  UrbanEngine.start()
 }

 private fun toggleAuto() {
  val next = !UrbanEngine.state.value.autoCapture
  UrbanEngine.setAutoCapture(next)
  Toast.makeText(this, if (next) "Auto-capture on" else "Auto-capture off", Toast.LENGTH_SHORT).show()
 }

 private fun cycleSensitivity() {
  val next = UrbanEngine.cycleSensitivity()
  Toast.makeText(this, "Sensitivity: " + next.label, Toast.LENGTH_SHORT).show()
 }

 private fun stopCollection() {
  UrbanEngine.stop()
  stopService(Intent(this, CollectionService::class.java))
 }

 private fun save() {
  getSharedPreferences("urban", 0).edit()
   .putString("receiver", receiver.text.toString())
   .putString("device", device.text.toString())
   .putString("token", token.text.toString())
   .apply()
 }

 private fun render(s: AppState) {
  labels["connection"]?.text =
   "● " + s.connection + "   Session: " + (if (s.collection) s.session else "STOPPED")
  labels["destination"]?.text = "Receiver: " + s.receiver + "\nWebSocket: " + s.wsUrl
  labels["accel"]?.text = "X: %.3f m/s²\nY: %.3f m/s²\nZ: %.3f m/s²\nStatus: %s".format(
   s.accel.x, s.accel.y, s.accel.z, if (s.accel.available) "Available" else "Unavailable"
  )
  labels["gyro"]?.text = "X: %.3f rad/s\nY: %.3f rad/s\nZ: %.3f rad/s\nStatus: %s".format(
   s.gyro.x, s.gyro.y, s.gyro.z, if (s.gyro.available) "Available" else "Unavailable"
  )
  val gpsStatus = when {
   s.gps.active -> "Active (" + s.gps.provider + ")"
   s.gps.enabled -> "Waiting for first fix…"
   else -> "Unavailable / disabled"
  }
  labels["gps"]?.text =
   "Latitude: %.6f\nLongitude: %.6f\nAltitude: %.1f m\nSpeed: %.2f m/s\nAccuracy: %.1f m\nStatus: %s".format(
    s.gps.latitude, s.gps.longitude, s.gps.altitude, s.gps.speed, s.gps.accuracy, gpsStatus
   )
  labels["transmission"]?.text = "Packets Sent: " + s.sent +
   "\nAccelerometer: " + s.accelCount + "  Gyroscope: " + s.gyroCount +
   "\nGPS: " + s.gpsCount + "  Camera: " + s.cameraCount +
   "\nLast Sent: " + s.lastSent +
   "\nBuffered: " + s.buffered +
   "\nConnection: " + s.connection
  labels["shock"]?.text = "Shock score: %.1f  (fires above %.0f)\nEvents this session: %d\nLast event: %s".format(
   s.shockScore, s.sensitivity.scoreThreshold, s.events, if (s.lastEvent.isBlank()) "none" else s.lastEvent
  )
  autoButton?.text = if (s.autoCapture) "AUTO-CAPTURE: ON" else "AUTO-CAPTURE: OFF"
  sensitivityButton?.text = "SENSITIVITY: " + s.sensitivity.label.uppercase()
  labels["warning"]?.text = s.warning
 }

 private fun bindCamera() {
  val view = previewView ?: return
  if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
  if (capture != null) return
  val future = ProcessCameraProvider.getInstance(this)
  future.addListener({
   try {
    val provider = future.get()
    val preview = Preview.Builder().build().also { it.surfaceProvider = view.surfaceProvider }
    // The sensor shoots 12 MP by default: ~500 KB a frame over the phone's uplink and a
    // 12-megapixel decode per thumbnail in the browser. Road damage is legible at 1280x960.
    val resolution = ResolutionSelector.Builder()
     .setResolutionStrategy(
      ResolutionStrategy(Size(1280, 960), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)
     )
     .build()
    val imageCapture = ImageCapture.Builder()
     .setJpegQuality(75)
     .setResolutionSelector(resolution)
     .build()
    provider.unbindAll()
    provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
    capture = imageCapture
   } catch (e: Exception) {
    Toast.makeText(this, "Camera unavailable: " + e.message, Toast.LENGTH_SHORT).show()
   }
  }, ContextCompat.getMainExecutor(this))
 }

 private fun takePhoto(event: ShockEvent?) {
  val c = capture
  if (c == null) {
   // An automatic trigger must clear the in-flight flag itself, or the detector goes quiet.
   UrbanEngine.captureFinished()
   if (event == null) Toast.makeText(this, "Camera not ready", Toast.LENGTH_SHORT).show()
   return
  }
  val prefix = if (event != null) "shock_" else "frame_"
  c.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
   override fun onCaptureSuccess(image: ImageProxy) {
    try {
     val buffer = image.planes[0].buffer
     val bytes = ByteArray(buffer.remaining())
     buffer.get(bytes)
     UrbanEngine.submitCamera(bytes, prefix + System.currentTimeMillis() + ".jpg", image.width, image.height, event)
    } catch (e: Exception) {
     UrbanEngine.captureFinished()
    } finally {
     image.close()
    }
   }

   override fun onError(e: ImageCaptureException) {
    UrbanEngine.captureFinished()
    runOnUiThread { Toast.makeText(this@MainActivity, "Capture failed: " + e.message, Toast.LENGTH_SHORT).show() }
   }
  })
 }

 override fun onDestroy() {
  super.onDestroy()
  UrbanEngine.onShock = null
  cameraExecutor.shutdown()
 }
}
