package com.roadai.androidtest.fusion

import com.roadai.androidtest.config.DetectionConfig
import com.roadai.androidtest.detection.Detection
import com.roadai.androidtest.detection.DetectionType
import com.roadai.androidtest.detection.ModelResult
import com.roadai.androidtest.detection.PerceptionResult
import com.roadai.androidtest.inference.iouOf

/**
 * Combines the specialists' outputs into one unified view of the frame.
 *
 * ==========================================================================
 * TWO RULES THIS CLASS EXISTS TO ENFORCE
 *
 * 1. CONFIDENCES ARE NEVER COMBINED ARITHMETICALLY.
 *    Each detection keeps the score its own model produced. Two models'
 *    confidences are not on a common scale — they were trained separately, on
 *    different data, with different class counts. Averaging or summing them
 *    would invent a number that means nothing. A calibrated fusion score is
 *    possible later, but only with validation data to calibrate against.
 *
 * 2. A POTHOLE AND A CRACK ARE NEVER DUPLICATES OF EACH OTHER.
 *    Overlap is evidence about geometry, not about semantics. A cracked rim
 *    around a pothole is two real, co-located defects. Suppression therefore
 *    only ever applies WITHIN one DetectionType.
 * ==========================================================================
 */
class DetectionFusionEngine {

    /**
     * @param results one entry per specialist that ran on this frame.
     *                Specialists that failed or were skipped are simply absent —
     *                fusion degrades to whatever did run.
     * @param config  passed per call rather than held, so live threshold changes
     *                from the UI take effect immediately instead of fusing
     *                against a stale copy.
     */
    fun fuse(
        results: List<ModelResult>,
        sourceWidth: Int,
        sourceHeight: Int,
        totalLatencyMs: Long,
        config: DetectionConfig
    ): PerceptionResult {

        // Highest confidence first, so the survivor of any duplicate pair is
        // the more confident one regardless of which model produced it.
        val all = results.flatMap { it.detections }
            .filter { config.includeOtherType || it.type != DetectionType.OTHER }
            .sortedByDescending { it.confidence }

        val kept = ArrayList<Detection>(all.size)
        var duplicatesRemoved = 0

        for (candidate in all) {
            if (isDuplicateOfKept(candidate, kept, config.duplicateIouThreshold)) {
                duplicatesRemoved++
            } else {
                kept.add(candidate)
            }
        }

        return PerceptionResult(
            detections = kept,
            sourceWidth = sourceWidth,
            sourceHeight = sourceHeight,
            perModel = results.associateBy { it.source },
            totalLatencyMs = totalLatencyMs,
            duplicatesRemoved = duplicatesRemoved
        )
    }

    /**
     * A candidate is a duplicate only when an already-kept detection has the
     * SAME normalized type and overlaps above the configured IoU.
     *
     * Note this also collapses same-type near-duplicates coming from a single
     * model that survived its own NMS (possible when one model reports two
     * different class names for the same object, e.g. a longitudinal and a
     * transverse crack on the same box — per-class NMS cannot merge those).
     */
    private fun isDuplicateOfKept(
        candidate: Detection,
        kept: List<Detection>,
        duplicateIou: Float
    ): Boolean {
        for (k in kept) {
            if (k.type != candidate.type) continue           // rule 2
            if (iouOf(k, candidate) > duplicateIou) return true
        }
        return false
    }
}
