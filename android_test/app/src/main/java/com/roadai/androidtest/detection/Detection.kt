package com.roadai.androidtest.detection

/**
 * The common detection format every specialist model is normalized into.
 *
 * Nothing downstream of the adapters — fusion, overlay, status panel — is
 * allowed to know which model produced a detection or how that model numbers
 * its classes. That is what makes adding a third specialist cheap.
 */
data class Detection(
    /** Which specialist produced this. Diagnostic only; never shown to the user. */
    val source: String,

    /** The model's OWN label, preserved verbatim (e.g. "Longitudinal Crack"). */
    val className: String,

    /** Normalized category shared across models. */
    val type: DetectionType,

    /** Finer detail where the model provides it (e.g. LONGITUDINAL). Null if not applicable. */
    val subtype: CrackSubtype? = null,

    /** The producing model's own confidence. Never combined with another model's. */
    val confidence: Float,

    /**
     * xyxy in the pixel space of the UPRIGHT SOURCE BITMAP handed to the
     * detector — letterboxing already undone. Mapping to view space is the
     * overlay's job.
     */
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,

    /** Wall-clock time the frame was processed. */
    val timestampMs: Long = System.currentTimeMillis()
) {
    /** What the overlay shows. Model-agnostic, human-readable. */
    val displayLabel: String get() = className

    val width: Float get() = x2 - x1
    val height: Float get() = y2 - y1
}

/**
 * The unified taxonomy. Specialists map their own classes onto this; adding a
 * new specialist usually means adding one entry here plus an adapter.
 */
enum class DetectionType {
    POTHOLE,
    CRACK,

    /**
     * Damage a model reports but cannot categorize further (RDD2022 "Other").
     * Kept rather than dropped — throwing it away would lose real information —
     * but rendered neutrally.
     */
    OTHER;
}

/** Crack subtypes, preserved so the crack model's detail is not flattened away. */
enum class CrackSubtype {
    LONGITUDINAL,
    TRANSVERSE,
    ALLIGATOR
}

/**
 * One specialist's output for one frame, plus its own timings.
 * Timings are per-model so a slow specialist can be identified.
 */
data class ModelResult(
    val source: String,
    val detections: List<Detection>,
    val preprocessMs: Long,
    val inferenceMs: Long,
    val postprocessMs: Long
) {
    val totalMs: Long get() = preprocessMs + inferenceMs + postprocessMs
}

/**
 * The fused, unified view of one frame — what the UI actually renders.
 */
data class PerceptionResult(
    val detections: List<Detection>,
    val sourceWidth: Int,
    val sourceHeight: Int,
    /** Per-model timings, keyed by source, for the performance panel. */
    val perModel: Map<String, ModelResult>,
    /** Wall-clock time from frame arrival to fused result. */
    val totalLatencyMs: Long,
    /** How many detections fusion removed as duplicates. */
    val duplicatesRemoved: Int
) {
    val potholes: List<Detection> get() = detections.filter { it.type == DetectionType.POTHOLE }
    val cracks: List<Detection> get() = detections.filter { it.type == DetectionType.CRACK }

    companion object {
        fun empty() = PerceptionResult(emptyList(), 0, 0, emptyMap(), 0, 0)
    }
}
