package com.roadai.androidtest.fusion

import com.roadai.androidtest.detection.CrackSubtype
import com.roadai.androidtest.detection.Detection
import com.roadai.androidtest.detection.DetectionType
import com.roadai.androidtest.detection.PerceptionResult
import com.roadai.androidtest.sensors.Gps
import com.roadai.androidtest.sensors.ShockEvent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The confirmation rules are the whole point of merging the two systems, so they are
 * pinned here rather than left to on-road observation.
 */
class RoadEventConfirmationTest {

    private val T = 1_000_000L      // an arbitrary frame-capture instant

    private fun det(
        type: DetectionType,
        confidence: Float,
        label: String = type.name
    ) = Detection(
        source = "test",
        className = label,
        type = type,
        subtype = if (type == DetectionType.CRACK) CrackSubtype.LONGITUDINAL else null,
        confidence = confidence,
        x1 = 10f, y1 = 10f, x2 = 100f, y2 = 100f,
        timestampMs = T + 3_500          // deliberately late: inference took ~3.5 s
    )

    private fun frame(vararg d: Detection) = PerceptionResult(
        detections = d.toList(),
        sourceWidth = 640, sourceHeight = 480,
        perModel = emptyMap(), totalLatencyMs = 3_500, duplicatesRemoved = 0
    )

    private fun shock(atOffsetMs: Long, score: Float = 12f, sustained: Boolean = false) =
        ShockEvent(at = T + atOffsetMs, accelMagnitude = 9f, gyroMagnitude = 1f,
            score = score, sustained = sustained)

    private val fix = Gps(
        latitude = 30.763940, longitude = 76.573267, altitude = 310.0,
        speed = 8.3f, bearing = 90f, accuracy = 6f,
        active = true, enabled = true, provider = "gps"
    )

    // ---------------- POTHOLE: visual + physical ----------------

    @Test fun potholeWithStrongShockInWindowIsConfirmed() {
        val e = RoadEventConfirmation().evaluate(
            frame(det(DetectionType.POTHOLE, 0.91f)), T, shock(1_200, score = 12f), fix
        )
        assertEquals(1, e.size)
        assertEquals(ConfirmationStatus.CONFIRMED, e[0].status)
        assertEquals(SensorSupport.STRONG, e[0].sensorSupport)
        assertEquals(1_200L, e[0].correlationDeltaMs)
    }

    @Test fun potholeWithWeakShockIsOnlySupported() {
        val e = RoadEventConfirmation().evaluate(
            frame(det(DetectionType.POTHOLE, 0.91f)), T, shock(1_200, score = 5f), fix
        )
        assertEquals(ConfirmationStatus.SUPPORTED, e[0].status)
        assertEquals(SensorSupport.WEAK, e[0].sensorSupport)
    }

    /** A pothole with no jolt is still reported — just not dressed up as confirmed. */
    @Test fun potholeWithNoShockIsVisualOnlyAndStillEmitted() {
        val e = RoadEventConfirmation().evaluate(
            frame(det(DetectionType.POTHOLE, 0.91f)), T, null, fix
        )
        assertEquals(1, e.size)
        assertEquals(ConfirmationStatus.VISUAL_ONLY, e[0].status)
        assertEquals(SensorSupport.NONE, e[0].sensorSupport)
        assertNull(e[0].shock)
    }

    @Test fun aSustainedShockCountsAsStrongEvenAtModestScore() {
        val e = RoadEventConfirmation().evaluate(
            frame(det(DetectionType.POTHOLE, 0.8f)), T, shock(900, score = 5f, sustained = true), fix
        )
        assertEquals(ConfirmationStatus.CONFIRMED, e[0].status)
    }

    // ---------------- CRACKS ARE NEVER PENALISED FOR SILENCE ----------------

    /** The rule that matters most: a hairline crack produces no jolt and must still stand. */
    @Test fun crackWithNoShockIsNeverRejected() {
        val e = RoadEventConfirmation().evaluate(
            frame(det(DetectionType.CRACK, 0.87f, "Longitudinal Crack")), T, null, fix
        )
        assertEquals(1, e.size)
        assertEquals(ConfirmationStatus.VISUAL_ONLY, e[0].status)
        assertEquals(0.87f, e[0].visualConfidence, 1e-6f)
    }

    /** A crack may be SUPPORTED by a coincident jolt but must never be CONFIRMED by it. */
    @Test fun crackNeverReachesConfirmedEvenWithAStrongShock() {
        val e = RoadEventConfirmation().evaluate(
            frame(det(DetectionType.CRACK, 0.87f)), T, shock(1_000, score = 40f, sustained = true), fix
        )
        assertEquals(ConfirmationStatus.SUPPORTED, e[0].status)
    }

    // ---------------- TEMPORAL WINDOW ----------------

    @Test fun aJoltLongAfterTheSightingDoesNotCorrelate() {
        val e = RoadEventConfirmation().evaluate(
            frame(det(DetectionType.POTHOLE, 0.9f)), T, shock(9_000), fix
        )
        assertEquals(ConfirmationStatus.VISUAL_ONLY, e[0].status)
    }

    @Test fun aJoltLongBeforeTheSightingDoesNotCorrelate() {
        val e = RoadEventConfirmation().evaluate(
            frame(det(DetectionType.POTHOLE, 0.9f)), T, shock(-9_000), fix
        )
        assertEquals(ConfirmationStatus.VISUAL_ONLY, e[0].status)
    }

    /**
     * The window is asymmetric on purpose: a forward-facing camera sees the defect
     * before the wheel reaches it, so "after" is given far more room than "before".
     */
    @Test fun theWindowIsAsymmetricForwardInTime() {
        val cfg = RoadEventConfig()
        val after = RoadEventConfirmation(cfg).evaluate(
            frame(det(DetectionType.POTHOLE, 0.9f)), T, shock(2_500), fix
        )
        val before = RoadEventConfirmation(cfg).evaluate(
            frame(det(DetectionType.POTHOLE, 0.9f)), T, shock(-2_500), fix
        )
        assertEquals("2.5 s after the sighting should correlate",
            ConfirmationStatus.CONFIRMED, after[0].status)
        assertEquals("2.5 s before the sighting should not",
            ConfirmationStatus.VISUAL_ONLY, before[0].status)
    }

    // ---------------- NO ARITHMETIC BLENDING ----------------

    /** The model's own confidence must survive untouched, whatever the sensors said. */
    @Test fun visualConfidenceIsNeverCombinedWithTheShockScore() {
        val e = RoadEventConfirmation().evaluate(
            frame(det(DetectionType.POTHOLE, 0.91f)), T, shock(1_000, score = 30f), fix
        )
        assertEquals(0.91f, e[0].visualConfidence, 1e-6f)
        // and the shock is carried alongside, not folded in
        assertEquals(30f, e[0].shock!!.score, 1e-6f)
    }

    // ---------------- SPAM CONTROL ----------------

    @Test fun theSameDefectDoesNotEmitOnEveryFrame() {
        val c = RoadEventConfirmation()
        val first = c.evaluate(frame(det(DetectionType.POTHOLE, 0.9f)), T, null, fix)
        val second = c.evaluate(frame(det(DetectionType.POTHOLE, 0.9f)), T + 500, null, fix)
        val third = c.evaluate(frame(det(DetectionType.POTHOLE, 0.9f)), T + 2_000, null, fix)
        assertEquals(1, first.size)
        assertEquals("still inside the cooldown", 0, second.size)
        assertEquals("still inside the cooldown", 0, third.size)
    }

    @Test fun aNewDefectIsEmittedOnceTheCooldownExpires() {
        val c = RoadEventConfirmation()
        c.evaluate(frame(det(DetectionType.POTHOLE, 0.9f)), T, null, fix)
        val later = c.evaluate(frame(det(DetectionType.POTHOLE, 0.9f)), T + 9_000, null, fix)
        assertEquals(1, later.size)
    }

    /** A pothole and a crack are different defects and must not share a cooldown. */
    @Test fun cooldownIsPerTypeNotGlobal() {
        val c = RoadEventConfirmation()
        val e = c.evaluate(
            frame(det(DetectionType.POTHOLE, 0.9f), det(DetectionType.CRACK, 0.8f)), T, null, fix
        )
        assertEquals(2, e.size)
        assertEquals(setOf(DetectionType.POTHOLE, DetectionType.CRACK), e.map { it.type }.toSet())
    }

    @Test fun oneJoltCannotConfirmTwoSeparateSightings() {
        val c = RoadEventConfirmation()
        val s = shock(1_000, score = 20f)
        val first = c.evaluate(frame(det(DetectionType.POTHOLE, 0.9f)), T, s, fix)
        assertEquals(ConfirmationStatus.CONFIRMED, first[0].status)
        // same shock object, a later frame past the cooldown: must not confirm again
        val second = c.evaluate(frame(det(DetectionType.POTHOLE, 0.9f)), T + 9_000, s, fix)
        assertEquals(ConfirmationStatus.VISUAL_ONLY, second[0].status)
    }

    @Test fun lowConfidenceDetectionsDoNotBecomeEvents() {
        val e = RoadEventConfirmation().evaluate(
            frame(det(DetectionType.POTHOLE, 0.30f)), T, shock(1_000), fix
        )
        assertTrue("0.30 is below the event threshold", e.isEmpty())
    }

    // ---------------- LOCATION ----------------

    @Test fun theEventCarriesTheFixAsItWasAtFrameCapture() {
        val e = RoadEventConfirmation().evaluate(
            frame(det(DetectionType.POTHOLE, 0.9f)), T, shock(1_000), fix
        )
        assertTrue(e[0].hasLocation)
        assertEquals(30.763940, e[0].latitude!!, 1e-9)
        assertEquals(76.573267, e[0].longitude!!, 1e-9)
        assertEquals(T, e[0].frameCapturedAtMs)
    }

    @Test fun anEventWithoutAFixIsStillEmitted() {
        val e = RoadEventConfirmation().evaluate(
            frame(det(DetectionType.POTHOLE, 0.9f)), T, shock(1_000), Gps()
        )
        assertEquals(1, e.size)
        assertTrue("no fix yet", !e[0].hasLocation)
        assertNotNull(e[0].eventId)
    }

    @Test fun resetClearsCooldownsBetweenSessions() {
        val c = RoadEventConfirmation()
        c.evaluate(frame(det(DetectionType.POTHOLE, 0.9f)), T, null, fix)
        c.reset()
        val after = c.evaluate(frame(det(DetectionType.POTHOLE, 0.9f)), T + 100, null, fix)
        assertEquals(1, after.size)
    }
}
