package com.roadai.androidtest.sensors

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/** What set an automatic capture off, carried alongside the photo. */
data class ShockEvent(
 val at: Long,
 val accelMagnitude: Float,
 val gyroMagnitude: Float,
 val score: Float,
 val sustained: Boolean
)

enum class Sensitivity(val label: String, val scoreThreshold: Float, val accelFloor: Float, val gyroFloor: Float) {
 LOW("Low", 6f, 4.5f, 2.2f),
 MEDIUM("Medium", 4f, 2.5f, 1.3f),
 HIGH("High", 3f, 1.4f, 0.8f);

 fun next(): Sensitivity = entries[(ordinal + 1) % entries.size]
}

/**
 * Fires when the phone is shaken sharply — a pothole, a kerb, a hard brake.
 *
 * Gravity is a constant ~9.8 on whichever axis is down, so raw magnitude says nothing about
 * movement; it is tracked with a slow average and subtracted, leaving only the dynamic part.
 * That dynamic magnitude is then judged against its own recent baseline rather than a fixed
 * number, so a phone rattling in a car mount does not trip on every bump while a genuinely
 * sharp jolt still stands out.
 *
 * A single sample over the line is usually sensor noise, so a burst is required: several hits
 * inside a short window. After firing, a cooldown keeps one rough patch from producing a
 * hundred photographs.
 */
class ShockDetector(
 var sensitivity: Sensitivity = Sensitivity.MEDIUM,
 private val cooldownMs: Long = 4_000,
 private val burstWindowMs: Long = 1_200,
 private val burstHits: Int = 3
) {
 companion object {
  private const val GRAVITY_ALPHA = 0.08f      // ~0.6 s at 20 Hz
  private const val BASELINE_ALPHA = 0.02f     // ~2.5 s at 20 Hz
  private const val MIN_DEVIATION = 0.25f      // keeps the score finite when perfectly still
  private const val WARMUP_SAMPLES = 30
 }

 private var gx = 0f
 private var gy = 0f
 private var gz = 0f
 private var haveGravity = false
 private var baselineMean = 0f
 private var baselineDev = MIN_DEVIATION
 private var samples = 0

 private val hits = ArrayDeque<Long>()
 private var lastFired = 0L

 @Volatile var lastScore = 0f; private set
 @Volatile var lastDynamic = 0f; private set
 private var gyroMagnitude = 0f

 fun reset() {
  haveGravity = false
  samples = 0
  baselineMean = 0f
  baselineDev = MIN_DEVIATION
  hits.clear()
  lastFired = 0L
  lastScore = 0f
  lastDynamic = 0f
  gyroMagnitude = 0f
 }

 /** Gyroscope is advisory: it raises the reported magnitude and can trip the floor on its own. */
 fun onGyroscope(v: Vec) {
  gyroMagnitude = sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
 }

 /** Returns an event when this sample should trigger a capture, otherwise null. */
 fun onAccelerometer(v: Vec, now: Long): ShockEvent? {
  if (!haveGravity) {
   gx = v.x; gy = v.y; gz = v.z
   haveGravity = true
  } else {
   gx += (v.x - gx) * GRAVITY_ALPHA
   gy += (v.y - gy) * GRAVITY_ALPHA
   gz += (v.z - gz) * GRAVITY_ALPHA
  }

  val dx = v.x - gx
  val dy = v.y - gy
  val dz = v.z - gz
  val dynamic = sqrt(dx * dx + dy * dy + dz * dz)
  lastDynamic = dynamic

  samples++
  val excess = dynamic - baselineMean
  val score = excess / max(baselineDev, MIN_DEVIATION)
  lastScore = if (samples < WARMUP_SAMPLES) 0f else score

  val s = sensitivity
  val overFloor = dynamic >= s.accelFloor || gyroMagnitude >= s.gyroFloor
  val overBaseline = score >= s.scoreThreshold
  val armed = samples >= WARMUP_SAMPLES

  // The baseline only learns from calm samples; folding a jolt back into it would raise the
  // bar just as the rough stretch starts, and the detector would go deaf exactly when it matters.
  if (!(overFloor && overBaseline)) {
   baselineMean += (dynamic - baselineMean) * BASELINE_ALPHA
   baselineDev += (abs(dynamic - baselineMean) - baselineDev) * BASELINE_ALPHA
  }

  if (!armed || !overFloor || !overBaseline) return null

  hits.addLast(now)
  while (hits.isNotEmpty() && now - hits.first() > burstWindowMs) hits.removeFirst()
  if (hits.size < burstHits) return null
  if (now - lastFired < cooldownMs) return null

  lastFired = now
  val sustained = hits.size >= burstHits * 2
  hits.clear()
  return ShockEvent(
   at = now,
   accelMagnitude = dynamic,
   gyroMagnitude = gyroMagnitude,
   score = score,
   sustained = sustained
  )
 }
}
