package com.roadai.androidtest.fusion

import com.roadai.androidtest.detection.Detection
import com.roadai.androidtest.detection.DetectionType
import com.roadai.androidtest.detection.PerceptionResult
import com.roadai.androidtest.sensors.Gps
import com.roadai.androidtest.sensors.ShockEvent
import java.util.UUID
import kotlin.math.abs

/**
 * How strongly the physical sensors back up what the camera saw.
 *
 * This is NOT a confidence and never mixes with one. It is a separate axis of evidence.
 */
enum class SensorSupport { NONE, WEAK, STRONG }

/**
 * The event's standing, derived from two independent evidence sources.
 *
 * VISUAL_ONLY  a model saw it; no correlated physical disturbance.
 *              For a CRACK this is a perfectly normal, valid outcome.
 * SUPPORTED    a model saw it AND a jolt landed in the correlation window,
 *              but not hard enough (or not the right kind) to call it confirmed.
 * CONFIRMED    a model saw it AND a strong jolt landed in the window.
 *              Only ever reachable for defects that are supposed to be felt.
 */
enum class ConfirmationStatus { VISUAL_ONLY, SUPPORTED, CONFIRMED }

/** One road defect, as decided by visual + physical + location evidence. */
data class RoadEvent(
    val eventId: String,
    val type: DetectionType,
    val label: String,
    /** The producing model's OWN confidence, carried through untouched. */
    val visualConfidence: Float,
    val status: ConfirmationStatus,
    val sensorSupport: SensorSupport,
    val shock: ShockEvent?,
    /** shock time minus frame-capture time. Positive = jolt came after the sighting. */
    val correlationDeltaMs: Long?,
    val latitude: Double?,
    val longitude: Double?,
    val gpsAccuracy: Float?,
    val frameCapturedAtMs: Long,
    val emittedAtMs: Long = System.currentTimeMillis()
) {
    val hasLocation: Boolean get() = latitude != null && longitude != null
}

/**
 * Correlation and emission policy.
 *
 * ==========================================================================
 * WHY THE WINDOW IS ASYMMETRIC, AND WHY IT IS SECONDS RATHER THAN MILLISECONDS
 *
 * Two measured facts drive these numbers:
 *
 * 1. INFERENCE LATENCY IS ~3.5 s ON THE TEST DEVICE.
 *    android_test/test_results.md records ~3100 ms (pothole) and ~3550 ms
 *    (crack) per cycle on device, roughly 0.25 FPS. A [Detection]'s own
 *    timestampMs is stamped when the object is built — AFTER inference — so it
 *    trails the physical moment by seconds. Correlating against it would be
 *    meaningless. We therefore correlate against FRAME-CAPTURE time, which the
 *    caller records the instant CameraX hands the frame over, before any
 *    inference runs. Pipeline latency then cancels out entirely.
 *
 * 2. A FORWARD-FACING CAMERA SEES A POTHOLE BEFORE THE WHEEL REACHES IT.
 *    At ~30 km/h (8.3 m/s) a defect spotted 10 m ahead is struck ~1.2 s later;
 *    at 20 m ahead, ~2.4 s later. So the jolt is expected to ARRIVE AFTER the
 *    sighting, which is why [windowAfterMs] is much larger than
 *    [windowBeforeMs]. The smaller backward window still allows for a rough
 *    stretch already underway when the frame was taken, plus the detector's own
 *    burst window (1.2 s in ShockDetector) and 50 ms sensor sampling.
 *
 * These are starting values, not optima. They should be re-measured on the
 * mounting and speeds actually used. See docs/SENSOR_YOLO_FUSION.md.
 * ==========================================================================
 */
data class RoadEventConfig(
    /** A jolt this far BEFORE the frame still counts. */
    val windowBeforeMs: Long = 1_500,

    /** A jolt this far AFTER the frame still counts (vehicle reaching the defect). */
    val windowAfterMs: Long = 3_000,

    /**
     * Shock score (multiples of the sensor's own rolling baseline, straight out of
     * ShockDetector) at or above which physical support counts as STRONG.
     * 4x is ShockDetector's own MEDIUM firing threshold; 8x is comfortably above
     * the noise it already rejected.
     */
    val strongShockScore: Float = 8f,

    /**
     * Only detections at or above this fire an event. Deliberately higher than the
     * overlay's display threshold (0.25): the overlay may show a maybe, but an event
     * gets transmitted and reverse-geocoded, so it should be worth the trip.
     */
    val minVisualConfidence: Float = 0.45f,

    /**
     * Per-type quiet period. One pothole stays in frame for many cycles; without this
     * it would emit an event on each one and trigger a geocoding call every time.
     */
    val perTypeCooldownMs: Long = 8_000
)

/**
 * Decides which visual detections become road events, and how well the physical
 * sensors back them up.
 *
 * ==========================================================================
 * THE RULE THIS CLASS EXISTS TO ENFORCE
 *
 * VISUAL AND PHYSICAL CONFIDENCE ARE NEVER COMBINED ARITHMETICALLY.
 *
 * `visualConfidence` stays exactly what the model produced. The shock score stays
 * exactly what ShockDetector produced. There is no 0.91 + 0.8, no (0.91 + 0.8)/2,
 * no weighted blend. They are different quantities on different scales measuring
 * different things — a blended number would be arithmetic with no meaning behind it.
 * What the two together produce is a STATUS, not a score.
 *
 * This mirrors the rule DetectionFusionEngine already enforces between the two
 * models, and is kept in a separate class precisely because it is a different
 * question: that one asks "are these the same object?", this one asks "did the
 * road agree with the camera?".
 * ==========================================================================
 *
 * CRACKS ARE NEVER PENALISED FOR SILENCE. A hairline crack produces no jolt; a
 * crack detection therefore stands on its own visual evidence and can never be
 * rejected or downgraded for lack of a shock. Only POTHOLE (and OTHER, which in
 * RDD2022 is usually surface damage) can reach CONFIRMED, because those are the
 * defects a vehicle is actually supposed to feel.
 */
class RoadEventConfirmation(
    private val config: RoadEventConfig = RoadEventConfig()
) {

    /** Last emission per type, for the cooldown. */
    private val lastEmittedByType = HashMap<DetectionType, Long>()

    /** Shocks already spent confirming an event, so one jolt cannot confirm many frames. */
    private var lastConsumedShockAt: Long = 0

    /**
     * @param result           the fused visual result for this frame.
     * @param frameCapturedAtMs wall-clock time the frame arrived from CameraX, taken
     *                          BEFORE inference. See the note on [RoadEventConfig].
     * @param shock            the most recent shock the sensor engine reported, or null.
     * @param gps              the fix as it was AT FRAME CAPTURE — not at emission, which
     *                          on this device would be ~3.5 s and ~30 m stale.
     */
    fun evaluate(
        result: PerceptionResult,
        frameCapturedAtMs: Long,
        shock: ShockEvent?,
        gps: Gps?
    ): List<RoadEvent> {

        val correlated = correlate(shock, frameCapturedAtMs)
        val events = ArrayList<RoadEvent>()

        // Strongest first, so if two of a type are in frame the cooldown keeps the better one.
        for (d in result.detections.sortedByDescending { it.confidence }) {
            if (d.confidence < config.minVisualConfidence) continue
            if (!cooldownExpired(d.type, frameCapturedAtMs)) continue

            val support = supportFor(d, correlated)
            val status = statusFor(d, support)

            // A pothole whose only claim is a weak visual with no physical backing is
            // still reported, but plainly labelled VISUAL_ONLY rather than dressed up.
            events += RoadEvent(
                eventId = UUID.randomUUID().toString(),
                type = d.type,
                label = d.displayLabel,
                visualConfidence = d.confidence,
                status = status,
                sensorSupport = support,
                shock = if (support == SensorSupport.NONE) null else correlated?.first,
                correlationDeltaMs = if (support == SensorSupport.NONE) null else correlated?.second,
                latitude = gps?.takeIf { it.active }?.latitude,
                longitude = gps?.takeIf { it.active }?.longitude,
                gpsAccuracy = gps?.takeIf { it.active }?.accuracy,
                frameCapturedAtMs = frameCapturedAtMs
            )

            lastEmittedByType[d.type] = frameCapturedAtMs
            if (status == ConfirmationStatus.CONFIRMED) {
                lastConsumedShockAt = correlated?.first?.at ?: lastConsumedShockAt
            }
        }
        return events
    }

    /** Returns the shock and its signed offset if it falls inside the window, else null. */
    private fun correlate(shock: ShockEvent?, frameAtMs: Long): Pair<ShockEvent, Long>? {
        if (shock == null) return null
        val delta = shock.at - frameAtMs          // positive: jolt came after the sighting
        val inWindow = delta >= -config.windowBeforeMs && delta <= config.windowAfterMs
        if (!inWindow) return null
        // One jolt confirms one event; a second frame must not reuse a spent shock.
        if (shock.at == lastConsumedShockAt) return null
        return shock to delta
    }

    private fun supportFor(d: Detection, correlated: Pair<ShockEvent, Long>?): SensorSupport {
        if (correlated == null) return SensorSupport.NONE
        val (shock, _) = correlated
        val strong = shock.score >= config.strongShockScore || shock.sustained
        return if (strong) SensorSupport.STRONG else SensorSupport.WEAK
    }

    private fun statusFor(d: Detection, support: SensorSupport): ConfirmationStatus = when {
        // A crack is a visual defect. Silence from the accelerometer is expected and
        // must never count against it — at most, a coincident jolt adds context.
        d.type == DetectionType.CRACK ->
            if (support == SensorSupport.NONE) ConfirmationStatus.VISUAL_ONLY
            else ConfirmationStatus.SUPPORTED

        support == SensorSupport.STRONG -> ConfirmationStatus.CONFIRMED
        support == SensorSupport.WEAK -> ConfirmationStatus.SUPPORTED
        else -> ConfirmationStatus.VISUAL_ONLY
    }

    private fun cooldownExpired(type: DetectionType, nowMs: Long): Boolean {
        val last = lastEmittedByType[type] ?: return true
        return abs(nowMs - last) >= config.perTypeCooldownMs
    }

    /** Call when a new session starts so cooldowns do not leak across sessions. */
    fun reset() {
        lastEmittedByType.clear()
        lastConsumedShockAt = 0
    }
}
