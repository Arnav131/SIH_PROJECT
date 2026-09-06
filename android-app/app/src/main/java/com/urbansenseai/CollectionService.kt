package com.urbansenseai

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat

class CollectionService : Service() {

 private var wakeLock: PowerManager.WakeLock? = null

 override fun onBind(intent: Intent?): IBinder? = null

 override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
  createChannel()
  // startForeground must happen even on the STOP path: Android kills the process with
  // ForegroundServiceDidNotStartInTimeException if a started service never promotes itself.
  promote()
  if (intent?.action == STOP_ACTION) {
   UrbanEngine.stop()
   stopSelf()
   return START_NOT_STICKY
  }
  acquireWakeLock()
  return START_NOT_STICKY
 }

 /**
  * The declared type must match a granted permission or startForeground throws. Location is
  * only claimed when the runtime permission is actually held; otherwise the service runs as
  * a plain data-sync worker, which still keeps the process alive for the WebSocket.
  */
 private fun promote() {
  val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
   if (UrbanEngine.hasLocationPermission()) ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION
   else ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
  } else {
   0
  }
  try {
   ServiceCompat.startForeground(this, NOTIFICATION_ID, notification(), type)
  } catch (e: Exception) {
   Log.w("UrbanSense", "Typed foreground start refused, falling back", e)
   try {
    ServiceCompat.startForeground(this, NOTIFICATION_ID, notification(), 0)
   } catch (fallback: Exception) {
    Log.e("UrbanSense", "Foreground service could not start", fallback)
    stopSelf()
   }
  }
 }

 private fun acquireWakeLock() {
  if (wakeLock?.isHeld == true) return
  val pm = getSystemService(PowerManager::class.java) ?: return
  // Without this the CPU sleeps with the screen and the sensor stream stops mid-session.
  wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "UrbanSense::collection").apply {
   setReferenceCounted(false)
   acquire(4 * 60 * 60 * 1000L)
  }
 }

 private fun createChannel() {
  val channel = NotificationChannel(CHANNEL_ID, "Sensor collection", NotificationManager.IMPORTANCE_LOW)
  getSystemService(NotificationManager::class.java)?.createNotificationChannel(channel)
 }

 private fun notification(): Notification {
  val stop = PendingIntent.getService(
   this,
   1,
   Intent(this, CollectionService::class.java).setAction(STOP_ACTION),
   PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
  )
  val open = PendingIntent.getActivity(
   this,
   2,
   Intent(this, MainActivity::class.java),
   PendingIntent.FLAG_IMMUTABLE
  )
  return NotificationCompat.Builder(this, CHANNEL_ID)
   .setSmallIcon(android.R.drawable.ic_menu_compass)
   .setContentTitle("UrbanSenseAI")
   .setContentText("Sensor collection active")
   .setContentIntent(open)
   .setOngoing(true)
   .addAction(0, "STOP COLLECTION", stop)
   .build()
 }

 override fun onDestroy() {
  super.onDestroy()
  wakeLock?.let { if (it.isHeld) it.release() }
  wakeLock = null
 }

 private companion object {
  const val NOTIFICATION_ID = 9
  const val CHANNEL_ID = "collection"
  const val STOP_ACTION = "STOP"
 }
}
