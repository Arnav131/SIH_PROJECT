package com.urbansenseai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sin
import kotlin.random.Random

/**
 * The detector decides when the phone photographs itself, so a false positive wastes storage
 * and a miss loses the pothole. These drive it with sampled motion at the real 20 Hz rate.
 */
class ShockDetectorTest {

 private val stepMs = 50L      // 20 Hz, matching the 50_000 us registration
 private val g = 9.81f

 private fun rest(jitter: Float = 0.02f, seed: Int = 7): (Int) -> Vec {
  val rnd = Random(seed)
  return { Vec(rnd.nextFloat() * jitter, rnd.nextFloat() * jitter, g + rnd.nextFloat() * jitter) }
 }

 /** Feeds [count] samples and returns every event fired. */
 private fun run(
  d: ShockDetector,
  count: Int,
  startAt: Long = 0L,
  sample: (Int) -> Vec
 ): List<ShockEvent> {
  val fired = mutableListOf<ShockEvent>()
  for (i in 0 until count) {
   d.onAccelerometer(sample(i), startAt + i * stepMs)?.let { fired += it }
  }
  return fired
 }

 @Test fun aPhoneLyingStillNeverFires() {
  val d = ShockDetector()
  assertEquals(emptyList<ShockEvent>(), run(d, 400, sample = rest()))
 }

 @Test fun aPhoneCarriedWhileWalkingNeverFires() {
  val d = ShockDetector()
  // Walking is a steady ~2 Hz sway of roughly 1 m/s2 - rhythmic, not a jolt.
  val events = run(d, 600) { i ->
   val t = i * 0.05
   Vec((sin(t * 12.5) * 0.9).toFloat(), (sin(t * 12.5 + 1) * 0.6).toFloat(), g + (sin(t * 12.5) * 1.0).toFloat())
  }
  assertEquals("walking should not trigger", emptyList<ShockEvent>(), events)
 }

 @Test fun oneIsolatedSpikeIsIgnoredAsNoise() {
  val d = ShockDetector()
  val calm = rest()
  val events = run(d, 200) { i -> if (i == 120) Vec(14f, 3f, g) else calm(i) }
  assertEquals("a single sample is sensor noise, not an event", emptyList<ShockEvent>(), events)
 }

 @Test fun aBurstOfJoltsFires() {
  val d = ShockDetector()
  val calm = rest()
  // A pothole rattles the phone for a few samples in a row.
  val events = run(d, 200) { i ->
   if (i in 120..124) Vec(9f, -7f, g + 6f) else calm(i)
  }
  assertEquals(1, events.size)
  assertTrue("magnitude should be well above rest", events[0].accelMagnitude > 5f)
  assertTrue("score should clear the threshold", events[0].score >= Sensitivity.MEDIUM.scoreThreshold)
 }

 @Test fun theCooldownStopsOneRoughPatchFloodingTheGallery() {
  val d = ShockDetector(cooldownMs = 4_000)
  val calm = rest()
  // Six seconds of continuous hammering at 20 Hz: 120 samples that all clear the threshold.
  val events = run(d, 200) { i -> if (i in 40..159) Vec(11f, -8f, g + 7f) else calm(i) }
  assertTrue("continuous shaking must not fire on every sample, got ${events.size}", events.size <= 3)
  assertTrue("but it must fire at least once", events.isNotEmpty())
  for (k in 1 until events.size) {
   assertTrue("captures must be at least the cooldown apart",
    events[k].at - events[k - 1].at >= 4_000)
  }
 }

 @Test fun sustainedShakingIsLabelledSustained() {
  val d = ShockDetector()
  val calm = rest()
  val events = run(d, 200) { i -> if (i in 60..120) Vec(11f, -8f, g + 7f) else calm(i) }
  assertTrue(events.isNotEmpty())
  assertTrue("a long rough stretch should be marked sustained", events[0].sustained)
 }

 @Test fun aShortBurstIsNotLabelledSustained() {
  val d = ShockDetector()
  val calm = rest()
  val events = run(d, 200) { i -> if (i in 100..102) Vec(9f, -7f, g + 6f) else calm(i) }
  assertEquals(1, events.size)
  assertTrue("three hits is a bump, not a stretch", !events[0].sustained)
 }

 @Test fun theWarmupSuppressesTheVeryFirstSamples() {
  val d = ShockDetector()
  // Jolting from sample zero: the baseline knows nothing yet, so nothing should fire.
  val events = run(d, 20) { Vec(12f, -9f, g + 8f) }
  assertEquals(emptyList<ShockEvent>(), events)
 }

 @Test fun aRattlingMountRaisesTheBarInsteadOfFiringConstantly() {
  val d = ShockDetector()
  val rnd = Random(3)
  // A loose car mount buzzes at +/- 2 m/s2 the whole time. That is the new normal, not an event.
  val events = run(d, 800) { i ->
   val t = i * 0.05
   Vec(
    (sin(t * 40) * 2.0 + rnd.nextFloat() * 0.4).toFloat(),
    (sin(t * 37) * 1.8).toFloat(),
    g + (sin(t * 43) * 2.0).toFloat()
   )
  }
  assertTrue("a constant rattle should settle into the baseline, got ${events.size}", events.size <= 1)
 }

 @Test fun aRealJoltStillFiresThroughARattlingMount() {
  val d = ShockDetector()
  val rnd = Random(3)
  val events = run(d, 800) { i ->
   val t = i * 0.05
   if (i in 500..506) Vec(18f, -14f, g + 12f)
   else Vec(
    (sin(t * 40) * 2.0 + rnd.nextFloat() * 0.4).toFloat(),
    (sin(t * 37) * 1.8).toFloat(),
    g + (sin(t * 43) * 2.0).toFloat()
   )
  }
  assertTrue("a hard pothole must still be seen over a rattle", events.any { it.at in 25_000L..25_600L })
 }

 @Test fun gravityOnAnyAxisIsSubtracted() {
  val d = ShockDetector()
  // Phone held upright: gravity sits on Y, not Z. Magnitude is still ~9.8 but nothing moves.
  val events = run(d, 400) { Vec(0.01f, g, 0.01f) }
  assertEquals("orientation must not look like motion", emptyList<ShockEvent>(), events)
 }

 @Test fun higherSensitivityFiresOnSofterBumps() {
  val soft = { i: Int -> if (i in 150..154) Vec(1.6f, -1.0f, 9.81f + 1.2f) else rest()(i) }
  val low = run(ShockDetector(sensitivity = Sensitivity.LOW), 250, sample = soft)
  val high = run(ShockDetector(sensitivity = Sensitivity.HIGH), 250, sample = soft)
  assertEquals("a soft bump is below the LOW floor", emptyList<ShockEvent>(), low)
  assertTrue("HIGH should catch it", high.isNotEmpty())
 }

 @Test fun gyroscopeAloneCanTripTheFloor() {
  val d = ShockDetector(sensitivity = Sensitivity.HIGH)
  val calm = rest()
  // A sharp twist barely moves the accelerometer but spins the gyroscope hard.
  val fired = mutableListOf<ShockEvent>()
  for (i in 0 until 250) {
   if (i in 150..156) {
    d.onGyroscope(Vec(2.4f, 1.1f, 0.6f))
    d.onAccelerometer(Vec(1.0f, 0.8f, 9.81f + 0.9f), i * stepMs)?.let { fired += it }
   } else {
    d.onGyroscope(Vec(0.002f, 0.001f, 0.001f))
    d.onAccelerometer(calm(i), i * stepMs)?.let { fired += it }
   }
  }
  assertTrue("a hard twist should be catchable", fired.isNotEmpty())
  assertTrue("the gyro magnitude should be recorded", fired[0].gyroMagnitude > 2f)
 }

 @Test fun resetClearsTheCooldownAndBaseline() {
  val d = ShockDetector()
  val calm = rest()
  val first = run(d, 200) { i -> if (i in 120..126) Vec(11f, -8f, g + 7f) else calm(i) }
  assertTrue(first.isNotEmpty())
  d.reset()
  assertEquals(0f, d.lastScore, 0.001f)
  // A new session starts from scratch, so the warmup applies again.
  val immediate = run(d, 10, startAt = 20_000L) { Vec(12f, -9f, g + 8f) }
  assertEquals(emptyList<ShockEvent>(), immediate)
 }

 @Test fun sensitivityCyclesThroughAllThree() {
  assertEquals(Sensitivity.MEDIUM, Sensitivity.LOW.next())
  assertEquals(Sensitivity.HIGH, Sensitivity.MEDIUM.next())
  assertEquals(Sensitivity.LOW, Sensitivity.HIGH.next())
 }

 @Test fun theScoreIsReportedForTheUi() {
  val d = ShockDetector()
  run(d, 200, sample = rest())
  assertTrue("a still phone should score near zero, was ${d.lastScore}", d.lastScore < 3f)
  assertNull(d.onAccelerometer(Vec(0.01f, 0.01f, g), 10_000L))
  assertNotNull("dynamic magnitude should be exposed", d.lastDynamic)
 }
}
