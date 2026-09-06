package com.urbansenseai

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

data class Vec(
 val x: Float = 0f,
 val y: Float = 0f,
 val z: Float = 0f,
 val accuracy: Int = 0,
 val available: Boolean = false
)

data class Gps(
 val latitude: Double = 0.0,
 val longitude: Double = 0.0,
 val altitude: Double = 0.0,
 val speed: Float = 0f,
 val bearing: Float = 0f,
 val accuracy: Float = 0f,
 val active: Boolean = false,
 val enabled: Boolean = false,
 val provider: String = ""
)

data class AppState(
 val accel: Vec = Vec(),
 val gyro: Vec = Vec(),
 val gps: Gps = Gps(),
 val collection: Boolean = false,
 val session: String = "",
 val connection: String = "DISCONNECTED",
 val receiver: String = "",
 val wsUrl: String = "",
 val sent: Long = 0,
 val accelCount: Long = 0,
 val gyroCount: Long = 0,
 val gpsCount: Long = 0,
 val cameraCount: Long = 0,
 val buffered: Int = 0,
 val lastSent: String = "",
 val warning: String = "",
 val autoCapture: Boolean = true,
 val sensitivity: Sensitivity = Sensitivity.MEDIUM,
 val shockScore: Float = 0f,
 val events: Long = 0,
 val lastEvent: String = ""
)

/** Endpoint pair: [socket] is what OkHttp dials, [display] is what the operator reads. */
data class Endpoint(val socket: String, val display: String)

/**
 * OkHttp dials WebSockets over http(s) and performs the upgrade itself; HttpUrl rejects a
 * ws(s) scheme outright, so the dialled target stays http(s) and only the label shown in the
 * UI carries ws(s). Passing a ws:// URL to HttpUrl is what stopped the app connecting at all.
 */
fun deriveEndpoint(raw: String): Endpoint {
 var input = raw.trim().trimEnd('/')
 require(input.isNotBlank()) { "Receiver URL required" }
 // "192.168.1.5:8000" is the shape people actually type off a laptop banner, so the
 // scheme is assumed rather than thrown back at them.
 if (!input.contains("://")) input = "http://" + input
 val u = try {
  java.net.URI(input)
 } catch (e: java.net.URISyntaxException) {
  throw IllegalArgumentException("Invalid receiver URL, try http://192.168.1.5:8000")
 }
 val secure = when (u.scheme?.lowercase()) {
  "http", "ws" -> false
  "https", "wss" -> true
  else -> throw IllegalArgumentException("Use a full http(s) or ws(s) URL, e.g. http://192.168.1.5:8000")
 }
 require(!u.host.isNullOrBlank()) { "Invalid receiver URL: no host, try http://192.168.1.5:8000" }
 val path = u.rawPath?.takeIf { it.isNotBlank() && it != "/" } ?: "/ws/data"
 val http = java.net.URI(if (secure) "https" else "http", u.userInfo, u.host, u.port, path, null, null)
 val ws = java.net.URI(if (secure) "wss" else "ws", u.userInfo, u.host, u.port, path, null, null)
 return Endpoint(http.toString(), ws.toString())
}

object UrbanEngine {
 private const val TAG = "UrbanSense"
 private const val MAX_QUEUE = 2000

 private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
 private val _state = MutableStateFlow(AppState())
 val state: StateFlow<AppState> = _state

 private lateinit var context: Context
 private var sensorManager: SensorManager? = null
 private var locationManager: LocationManager? = null
 private val sensorThread = HandlerThread("UrbanSensorThread").apply { start() }
 private val sensorHandler = Handler(sensorThread.looper)

 private var webSocket: WebSocket? = null
 private val client = OkHttpClient.Builder()
  .pingInterval(20, TimeUnit.SECONDS)
  .connectTimeout(10, TimeUnit.SECONDS)
  .readTimeout(0, TimeUnit.MILLISECONDS)
  .retryOnConnectionFailure(true)
  .build()

 private data class Frame(val json: String? = null, val binary: ByteArray? = null, val kind: String = "")

 private val queue = ConcurrentLinkedDeque<Frame>()
 private val sequence = AtomicLong(0)
 private var retry = 1L
 private var connectWanted = false
 @Volatile private var socketUrl = ""

 private var deviceId = "realme3pro_001"
 private var token = "urbansenseai_demo_token"
 private var receiver = ""

 private val detector = ShockDetector()
 @Volatile private var captureInFlight = false

 /**
  * Raised when a jolt should be photographed. CameraX is bound to the Activity lifecycle, so
  * only the Activity can actually take the picture; the engine just says when.
  */
 @Volatile var onShock: ((ShockEvent) -> Unit)? = null

 fun initialize(c: Context) {
  if (::context.isInitialized) {
   startPreview()
   return
  }
  context = c.applicationContext
  sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
  locationManager = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager
  // Preview listeners are deliberately independent of a recording session. This
  // gives the UI real readings immediately; emitSensor still sends only while active.
  startPreview()
 }

 fun startPreview() {
  val sm = sensorManager ?: return
  val accelAvailable = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null
  val gyroAvailable = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE) != null
  val gpsEnabled = runCatching {
   locationManager?.isProviderEnabled(LocationManager.GPS_PROVIDER) == true
  }.getOrDefault(false)
  _state.update {
   it.copy(
    accel = it.accel.copy(available = accelAvailable),
    gyro = it.gyro.copy(available = gyroAvailable),
    gps = it.gps.copy(enabled = gpsEnabled)
   )
  }
  registerSensors()
 }

 fun configure(url: String, device: String, authToken: String): Result<String> = runCatching {
  val endpoint = deriveEndpoint(url)
  receiver = url.trim().trimEnd('/')
  socketUrl = endpoint.socket
  deviceId = device.trim().ifBlank { "android_" + Build.MODEL.replace(" ", "_") }
  token = authToken.trim()
  _state.update { it.copy(receiver = receiver, wsUrl = endpoint.display, warning = "") }
  endpoint.display
 }

 /** Delegates to the pure [deriveEndpoint] so the URL rules stay unit-testable off-device. */
 fun deriveEndpoint(raw: String): Endpoint = com.urbansenseai.deriveEndpoint(raw)

 /** Display form only; kept so callers that just want a label do not build the whole pair. */
 fun deriveWebSocketUrl(raw: String): String = deriveEndpoint(raw).display

 fun connect() {
  if (socketUrl.isBlank()) {
   _state.update { it.copy(connection = "ERROR", warning = "Enter a receiver URL first") }
   return
  }
  if (connectWanted && webSocket != null) return
  connectWanted = true
  retry = 1
  openSocket()
 }

 fun disconnect() {
  connectWanted = false
  retry = 1
  webSocket?.close(1000, "user disconnect")
  webSocket = null
  _state.update { it.copy(connection = "DISCONNECTED") }
 }

 private fun openSocket() {
  if (!connectWanted) return
  val target = try {
   socketUrl.toHttpUrl().newBuilder().addQueryParameter("token", token).build()
  } catch (e: Exception) {
   connectWanted = false
   _state.update { it.copy(connection = "ERROR", warning = "Invalid receiver URL: " + e.message) }
   return
  }
  _state.update { it.copy(connection = if (retry > 1) "RECONNECTING" else "CONNECTING") }
  try {
   webSocket = client.newWebSocket(Request.Builder().url(target).build(), object : WebSocketListener() {
    override fun onOpen(ws: WebSocket, response: Response) {
     Log.i(TAG, "WebSocket open; waiting for acknowledgement")
    }

    override fun onMessage(ws: WebSocket, text: String) {
     val j = runCatching { JSONObject(text) }.getOrNull() ?: return
     when (j.optString("type")) {
      "connection_ack" -> if (j.optString("status") == "connected") {
       retry = 1
       _state.update { it.copy(connection = "CONNECTED", warning = "") }
       Log.i(TAG, "Receiver acknowledged connection")
       flush()
      }
      "error" -> {
       val message = j.optString("message")
       Log.w(TAG, "Receiver rejected a frame: " + message)
       _state.update { it.copy(warning = "Receiver: " + message) }
      }
     }
    }

    override fun onFailure(ws: WebSocket, t: Throwable, r: Response?) {
     if (webSocket !== ws) return
     webSocket = null
     scheduleReconnect(t.message ?: "connection failure")
    }

    override fun onClosed(ws: WebSocket, code: Int, reason: String) {
     if (webSocket !== ws) return
     webSocket = null
     // 1008 is the receiver refusing our token; retrying that forever only hides the cause.
     if (code == 1008) {
      connectWanted = false
      _state.update { it.copy(connection = "ERROR", warning = "Receiver rejected the device token") }
      return
     }
     if (connectWanted) {
      scheduleReconnect(if (reason.isBlank()) "connection closed (" + code + ")" else reason)
     } else {
      _state.update { it.copy(connection = "DISCONNECTED") }
     }
    }
   })
  } catch (e: Exception) {
   connectWanted = false
   _state.update { it.copy(connection = "ERROR", warning = "Could not create WebSocket: " + e.message) }
  }
 }

 private fun scheduleReconnect(message: String) {
  if (!connectWanted) return
  _state.update { it.copy(connection = "RECONNECTING", warning = message) }
  val wait = retry.coerceAtMost(30)
  retry = (retry * 2).coerceAtMost(30)
  scope.launch {
   delay(wait * 1000)
   openSocket()
  }
 }

 fun start() {
  if (_state.value.collection) return
  startPreview()
  val id = "session_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
  sequence.set(0)
  detector.reset()
  captureInFlight = false
  _state.update { it.copy(collection = true, session = id, warning = "", events = 0, lastEvent = "") }
  // Starting a session with no live socket used to silently fill the buffer instead.
  if (!connectWanted) connect()
  sendSessionStart()
 }

 fun stop() {
  // The queue is deliberately kept so packets buffered offline still drain on reconnect.
  _state.update { it.copy(collection = false) }
 }

 private fun sendSessionStart() {
  if (!_state.value.collection) return
  val sensors = JSONObject()
   .put("accelerometer", sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null)
   .put("gyroscope", sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE) != null)
  val j = base("session_start")
   .put("device_model", Build.MANUFACTURER + " " + Build.MODEL)
   .put("android_version", Build.VERSION.RELEASE)
   .put("sensors", sensors)
  submit(j.toString(), "session_start")
 }

 private fun base(type: String) = JSONObject()
  .put("device_id", deviceId)
  .put("session_id", _state.value.session)
  .put("timestamp", System.currentTimeMillis())
  .put("type", type)

 private val sensorListener = object : SensorEventListener {
  override fun onAccuracyChanged(s: Sensor?, a: Int) {}

  override fun onSensorChanged(e: SensorEvent) {
   if (e.values.size < 3) return
   val v = Vec(e.values[0], e.values[1], e.values[2], e.accuracy, true)
   when (e.sensor.type) {
    Sensor.TYPE_ACCELEROMETER -> {
     _state.update { it.copy(accel = v) }
     emitSensor("accelerometer", v)
     inspectForShock(v)
    }
    Sensor.TYPE_GYROSCOPE -> {
     _state.update { it.copy(gyro = v) }
     emitSensor("gyroscope", v)
     detector.onGyroscope(v)
    }
   }
  }
 }

 private val locationListener = object : LocationListener {
  override fun onLocationChanged(l: Location) = acceptLocation(l)

  override fun onProviderEnabled(provider: String) {
   if (provider == LocationManager.GPS_PROVIDER) _state.update { it.copy(gps = it.gps.copy(enabled = true)) }
  }

  override fun onProviderDisabled(provider: String) {
   if (provider == LocationManager.GPS_PROVIDER) {
    _state.update { it.copy(gps = it.gps.copy(enabled = false, active = false)) }
   }
  }

  // LocationManager on API 30 and below still dispatches this; omitting it aborts the callback.
  override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
 }

 private fun acceptLocation(l: Location) {
  val g = Gps(
   latitude = l.latitude,
   longitude = l.longitude,
   altitude = l.altitude,
   speed = l.speed,
   bearing = l.bearing,
   accuracy = l.accuracy,
   active = true,
   enabled = true,
   provider = l.provider ?: ""
  )
  _state.update { it.copy(gps = g) }
  if (!_state.value.collection) return
  val data = JSONObject()
   .put("latitude", g.latitude)
   .put("longitude", g.longitude)
   .put("altitude", g.altitude)
   .put("speed", g.speed)
   .put("bearing", g.bearing)
   .put("accuracy", g.accuracy)
  val j = base("gps").put("sequence_number", sequence.incrementAndGet()).put("data", data)
  submit(j.toString(), "gps")
 }

 fun hasLocationPermission(): Boolean {
  if (!::context.isInitialized) return false
  val fine = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
  val coarse = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION)
  return fine == PackageManager.PERMISSION_GRANTED || coarse == PackageManager.PERMISSION_GRANTED
 }

 private fun registerSensors() {
  val sm = sensorManager ?: return
  sm.unregisterListener(sensorListener)
  sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let {
   sm.registerListener(sensorListener, it, 50_000, sensorHandler)
  }
  sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let {
   sm.registerListener(sensorListener, it, 50_000, sensorHandler)
  }
  registerLocation()
 }

 private fun registerLocation() {
  val lm = locationManager ?: return
  if (!hasLocationPermission()) {
   _state.update { it.copy(warning = "Location permission required: grant Precise location") }
   return
  }
  try {
   lm.removeUpdates(locationListener)
   // GPS alone yields nothing indoors, so every enabled provider is subscribed and the last
   // known fix seeds the UI immediately instead of leaving the readout at 0.000000.
   var subscribed = 0
   for (provider in listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)) {
    if (!lm.allProviders.contains(provider) || !lm.isProviderEnabled(provider)) continue
    lm.requestLocationUpdates(provider, 1000L, 0f, locationListener, sensorHandler.looper)
    subscribed++
    lm.getLastKnownLocation(provider)?.let { acceptLocation(it) }
   }
   _state.update {
    it.copy(
     gps = it.gps.copy(enabled = subscribed > 0),
     warning = if (subscribed == 0) "Turn on Location (GPS) in Android settings" else it.warning
    )
   }
  } catch (e: SecurityException) {
   _state.update { it.copy(warning = "Location permission required: grant Precise location") }
  }
 }

 private fun inspectForShock(v: Vec) {
  val event = detector.onAccelerometer(v, System.currentTimeMillis())
  val score = detector.lastScore
  // The score readout is throttled to whole tenths so the UI is not rewritten 20 times a second.
  if (_state.value.shockScore.toInt() != score.toInt() || event != null) {
   _state.update { it.copy(shockScore = score) }
  }
  if (event == null) return
  if (!_state.value.collection || !_state.value.autoCapture) return
  if (captureInFlight) {
   Log.i(TAG, "Shock ignored: a capture is still in flight")
   return
  }
  val handler = onShock
  if (handler == null) {
   Log.w(TAG, "Shock detected but no camera is bound")
   return
  }
  captureInFlight = true
  val stamp = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date(event.at))
  _state.update {
   it.copy(
    events = it.events + 1,
    lastEvent = stamp + "  " + String.format(Locale.US, "%.1f m/s² (x%.1f)", event.accelMagnitude, event.score)
   )
  }
  Log.i(TAG, "Shock: " + event.accelMagnitude + " m/s2 score " + event.score)
  handler(event)
 }

 /** The Activity calls this whether the capture succeeded or failed, or the detector jams. */
 fun captureFinished() {
  captureInFlight = false
 }

 fun setAutoCapture(enabled: Boolean) {
  _state.update { it.copy(autoCapture = enabled) }
 }

 fun cycleSensitivity(): Sensitivity {
  val next = _state.value.sensitivity.next()
  detector.sensitivity = next
  _state.update { it.copy(sensitivity = next) }
  return next
 }

 private fun emitSensor(kind: String, v: Vec) {
  if (!_state.value.collection) return
  val data = JSONObject().put("x", v.x).put("y", v.y).put("z", v.z).put("accuracy", v.accuracy)
  val j = base("sensor")
   .put("sensor", kind)
   .put("sequence_number", sequence.incrementAndGet())
   .put("data", data)
  submit(j.toString(), kind)
 }

 /**
  * [event] is null for a manual CAPTURE and set for an automatic one. Either way the current
  * fix rides along, so a photograph is never left without the place it was taken.
  */
 fun submitCamera(bytes: ByteArray, filename: String, width: Int, height: Int, event: ShockEvent? = null) {
  if (!_state.value.collection) {
   captureFinished()
   return
  }
  val gps = _state.value.gps
  val j = base("camera")
   .put("sequence_number", sequence.incrementAndGet())
   .put("filename", filename)
   .put("width", width)
   .put("height", height)
   .put("trigger", if (event != null) (if (event.sustained) "shock_sustained" else "shock") else "manual")
  if (event != null) {
   j.put("accel_magnitude", event.accelMagnitude.toDouble())
    .put("gyro_magnitude", event.gyroMagnitude.toDouble())
    .put("shock_score", event.score.toDouble())
  }
  if (gps.active) {
   j.put("latitude", gps.latitude)
    .put("longitude", gps.longitude)
    .put("gps_accuracy", gps.accuracy.toDouble())
    .put("gps_provider", gps.provider)
  }
  submit(j.toString(), "camera")
  sendFrame(Frame(binary = bytes, kind = "camera_blob"))
  captureFinished()
 }

 private fun submit(json: String, kind: String) = sendFrame(Frame(json = json, kind = kind))

 private fun sendFrame(frame: Frame) {
  if (_state.value.connection == "CONNECTED" && transmit(frame)) return
  queue.addLast(frame)
  var dropped = false
  while (queue.size > MAX_QUEUE) {
   queue.pollFirst()
   dropped = true
  }
  _state.update {
   it.copy(
    buffered = queue.size,
    warning = if (dropped) "Buffer full: oldest packets discarded" else it.warning
   )
  }
 }

 private fun transmit(frame: Frame): Boolean {
  val ws = webSocket ?: return false
  val ok = when {
   frame.json != null -> ws.send(frame.json)
   frame.binary != null -> ws.send(ByteString.of(*frame.binary))
   else -> false
  }
  if (ok) sent(frame.kind)
  return ok
 }

 private fun flush() {
  scope.launch {
   while (_state.value.connection == "CONNECTED") {
    val q = queue.pollFirst() ?: break
    if (!transmit(q)) {
     queue.addFirst(q)
     break
    }
    _state.update { it.copy(buffered = queue.size) }
   }
  }
 }

 private fun sent(kind: String) {
  val stamp = SimpleDateFormat("HH:mm:ss.SSS", Locale.US).format(Date())
  _state.update { s ->
   val counted = s.copy(sent = s.sent + 1, lastSent = stamp)
   when (kind) {
    "accelerometer" -> counted.copy(accelCount = s.accelCount + 1)
    "gyroscope" -> counted.copy(gyroCount = s.gyroCount + 1)
    "gps" -> counted.copy(gpsCount = s.gpsCount + 1)
    "camera" -> counted.copy(cameraCount = s.cameraCount + 1)
    else -> counted
   }
  }
 }
}
