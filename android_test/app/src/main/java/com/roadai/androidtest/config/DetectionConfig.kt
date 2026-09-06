package com.roadai.androidtest.config

/**
 * All tunable knobs in one place.
 *
 * Every threshold here is an INITIAL value, not an optimum. The numbers in the
 * comments come from a run of `android_test/tools/evaluate_fusion.py` against
 * the labelled test split; they describe those images, not the road outside.
 */
data class DetectionConfig(

    /**
     * Pothole specialist threshold. Measured on 40 ground-truth pothole images
     * (close-range photos) and 40 non-pothole images:
     *   0.25 -> 85% of pothole frames fired,  ~5% crack-model cross-fire
     *   0.40 -> markedly fewer, near-zero false alarms
     * Recall is favoured over precision for a prototype.
     */
    val potholeThreshold: Float = 0.25f,

    /** Crack specialist threshold. 0.25 fired on 57% of ground-truth crack frames. */
    val crackThreshold: Float = 0.25f,

    /** Within-model NMS. Standard Ultralytics default. */
    val nmsIouThreshold: Float = 0.7f,

    /**
     * Cross-model duplicate suppression, applied ONLY between detections that
     * normalized to the same [com.roadai.androidtest.detection.DetectionType].
     *
     * Deliberately high (0.80). Two boxes must almost coincide before one is
     * discarded. A pothole overlapping a crack is never treated as a duplicate —
     * they are different physical events and both are kept.
     */
    val duplicateIouThreshold: Float = 0.80f,

    /**
     * Whether to surface RDD2022's catch-all "Other" damage class.
     *
     * It is kept (not discarded) because it is real information, but it fires
     * often and generically — on close-range pothole photos it frequently
     * co-fires with the pothole specialist on the same defect. Set false for a
     * cleaner overlay that shows only POTHOLE and CRACK.
     */
    val includeOtherType: Boolean = true,

    /**
     * Upper bound on how often inference runs, independent of camera FPS.
     * The camera keeps rendering at its own rate; frames arriving while the
     * pipeline is busy or inside this interval are dropped, never queued.
     * Lower this if the preview stutters on a slow device.
     */
    val targetInferenceFps: Int = 8,

    /** How the two specialists share the frame. */
    val executionMode: ExecutionMode = ExecutionMode.PARALLEL,

    /**
     * If a full cycle takes longer than this, the scheduler logs a warning and
     * (in PARALLEL mode) the effective rate degrades naturally, because frames
     * are dropped rather than queued. The preview is never blocked.
     */
    val slowFrameWarningMs: Long = 1500L
) {
    val minFrameIntervalMs: Long
        get() = if (targetInferenceFps <= 0) 0L else 1000L / targetInferenceFps

    companion object {
        val DEFAULT = DetectionConfig()
    }
}

/**
 * PARALLEL runs both specialists concurrently on the same frame — lower
 * latency when the device has spare cores.
 * SEQUENTIAL runs them one after another — lower peak memory and CPU
 * contention, better on weak devices.
 *
 * Switching modes must not require touching the camera pipeline.
 */
enum class ExecutionMode {
    PARALLEL,
    SEQUENTIAL
}
