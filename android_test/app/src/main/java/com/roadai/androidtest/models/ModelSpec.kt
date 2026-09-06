package com.roadai.androidtest.models

import com.roadai.androidtest.detection.CrackSubtype
import com.roadai.androidtest.detection.DetectionType

/**
 * A specialist's inference contract plus its class mapping.
 *
 * ==========================================================================
 * Every numeric value in the two specs below was READ OUT OF THE ACTUAL ASSET
 * with onnxruntime (see android_test/MULTI_MODEL_ARCHITECTURE.md for the dump).
 * Nothing here is assumed, and the two models are NOT assumed to agree — they
 * happen to share an input size, but each spec states its own.
 * ==========================================================================
 */
data class ModelSpec(
    /** Diagnostic id, also the [com.roadai.androidtest.detection.Detection.source]. */
    val source: String,
    val asset: String,
    val inputSize: Int,
    val numClasses: Int,
    val numAnchors: Int,
    /** Used only if the ONNX metadata has no "names" entry. */
    val fallbackNames: Array<String>,
    /**
     * Maps a class id + the model's own label onto the unified taxonomy.
     * Returning null DROPS the class — used for deliberate routing decisions.
     */
    val mapper: (Int, String) -> ClassMapping?
) {
    override fun equals(other: Any?) = this === other
    override fun hashCode() = System.identityHashCode(this)
}

/** Result of normalizing one model-specific class onto the shared taxonomy. */
data class ClassMapping(
    val type: DetectionType,
    val subtype: CrackSubtype? = null
)

object ModelRegistry {

    /**
     * POTHOLE SPECIALIST — pothole_yolov12s.onnx
     * (md5 daa0a3ee9f3f3711cf4479ddb6870141)
     *
     *   architecture : YOLOv12s, 9,074,595 params
     *   input        : "images"  [1, 3, 640, 640] float32 NCHW, STATIC
     *   output       : "output0" [1, 5, 8400] float32, channels-first
     *                  5 = cx,cy,w,h + 1 class score. No objectness column.
     *   boxes        : 640-space PIXELS (observed -6.8 .. 637.9), not normalized
     *   scores       : already sigmoid, independent (observed 0.0 .. 1.0)
     *   NMS in graph : NO (op scan found no NonMaxSuppression / TopK)
     *   preprocessing: letterbox 640 pad=114, RGB, /255, HWC->CHW. No mean/std.
     *   classes      : {0: 'Pothole'}
     */
    val POTHOLE = ModelSpec(
        source = "pothole_model",
        asset = "pothole_yolov12s.onnx",
        inputSize = 640,
        numClasses = 1,
        numAnchors = 8400,
        fallbackNames = arrayOf("Pothole"),
        mapper = { _, _ -> ClassMapping(DetectionType.POTHOLE) }
    )

    /**
     * CRACK SPECIALIST — crack_rdd2022.onnx
     * (md5 6fc2f37ec4e8d10e9aba1f33a1b6a74b — byte-identical to
     *  model_training/models/best.onnx, which is left untouched)
     *
     *   architecture : YOLOv8s, 11,137,535 params, trained on RDD2022
     *   input        : "images"  [1, 3, 640, 640] float32 NCHW, STATIC
     *   output       : "output0" [1, 9, 8400] float32, channels-first
     *                  9 = cx,cy,w,h + 5 class scores. No objectness column.
     *   boxes        : 640-space PIXELS (observed 3.2 .. 645.9)
     *   scores       : already sigmoid, independent (per-anchor sums << 1,
     *                  so not a softmax — do not re-activate)
     *   NMS in graph : NO
     *   opset 19 / IR 9 -> requires onnxruntime >= 1.16
     *   preprocessing: identical to the pothole model, verified separately
     *   classes      : {0: Longitudinal Crack, 1: Transverse Crack,
     *                   2: Alligator Crack, 3: Pothole, 4: Other}
     */
    val CRACK = ModelSpec(
        source = "crack_model",
        asset = "crack_rdd2022.onnx",
        inputSize = 640,
        numClasses = 5,
        numAnchors = 8400,
        fallbackNames = arrayOf(
            "Longitudinal Crack",
            "Transverse Crack",
            "Alligator Crack",
            "Pothole",
            "Other"
        ),
        mapper = { _, name ->
            when (name) {
                "Longitudinal Crack" -> ClassMapping(DetectionType.CRACK, CrackSubtype.LONGITUDINAL)
                "Transverse Crack" -> ClassMapping(DetectionType.CRACK, CrackSubtype.TRANSVERSE)
                "Alligator Crack" -> ClassMapping(DetectionType.CRACK, CrackSubtype.ALLIGATOR)

                // ROUTING DECISION, not a bug: this model has its own Pothole
                // class, but the pothole specialist owns potholes. Dropping it
                // here stops two models from arguing over the same object with
                // two incomparable confidences. To let both vote instead,
                // return ClassMapping(DetectionType.POTHOLE) — fusion already
                // de-duplicates same-type overlaps.
                "Pothole" -> null

                // RDD2022 'Other' is real damage the model could not categorize.
                // Kept as OTHER rather than discarded.
                else -> ClassMapping(DetectionType.OTHER)
            }
        }
    )

    /**
     * The specialists the scheduler runs. Adding a future specialist
     * (MANHOLE, ROAD_SIGN, ...) means: add the asset, declare a ModelSpec with
     * its verified contract, add its mapper, and list it here. The camera
     * pipeline, fusion engine and overlay need no changes.
     */
    val ACTIVE: List<ModelSpec> = listOf(POTHOLE, CRACK)
}
