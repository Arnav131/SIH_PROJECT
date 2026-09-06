# Multi-Model Road Anomaly Detection — Architecture

Two independent specialist models run on the same camera frame and their
results are merged into one unified detection list.

**Their weights are never merged and neither model was retrained or modified.**
This is multi-model *inference* plus result *fusion*.

```
                        CameraX frame
                              |
                    upright RGBA bitmap
                              |
              +---------------+---------------+
              |                               |
     POTHOLE SPECIALIST              CRACK SPECIALIST
     pothole_yolov12s.onnx           crack_rdd2022.onnx
              |                               |
      PotholeAdapter mapper          CrackAdapter mapper
      (all -> POTHOLE)               (-> CRACK + subtype,
              |                       Pothole dropped, -> OTHER)
              |                               |
              +---------------+---------------+
                              |
                    DetectionFusionEngine
                (per-type duplicate suppression,
                 confidences preserved as-is)
                              |
                    PerceptionResult
                              |
                        OverlayView
```

---

## 1. Pothole specialist

| Property | Value |
|---|---|
| Asset | `app/src/main/assets/pothole_yolov12s.onnx` |
| md5 | `daa0a3ee9f3f3711cf4479ddb6870141` |
| Origin | `Pothole-Computer-Vision-Project-main.zip` you supplied |
| Architecture | **YOLOv12s**, 9,074,595 params |
| Trained with | Ultralytics 8.3.63, Roboflow `Pothole-1` dataset, 30 epochs, imgsz 640 |
| Format | ONNX, opset 12 |
| Input | `images` `[1, 3, 640, 640]` float32, **NCHW**, static batch |
| Output | `output0` `[1, 5, 8400]` float32, channels-first |
| Output meaning | `5 = cx, cy, w, h + 1 class score`. No objectness column |
| Box space | 640-space **pixels** (observed −6.8 … 637.9), not normalized |
| Activation | Class score already **sigmoid**. Do not re-activate |
| NMS in graph | **No** — verified by op scan. External NMS required |
| Classes | `{0: 'Pothole'}` |
| Size | 35 MB |

### A note on how this ONNX was produced

The supplied `best.pt` could not be exported directly. It was trained with
Ultralytics 8.3.63, whose YOLOv12 `AAttn` attention block stores **separate
`qk` and `v` projections**. Ultralytics 8.4.33 (installed here) expects a
**single fused `qkv`** projection, so loading crashed with
`'AAttn' object has no attribute 'qkv'`.

Rather than retrain or download a different model, the 8 attention blocks were
converted to the newer layout by concatenating the `qk` and `v` convolution and
BatchNorm parameters **interleaved per attention head**
(`[h0_q, h0_k, h0_v, h1_q, …]`, which is the layout the newer forward pass
indexes). Verified numerically: the converted block reproduces the original
block's output with **max absolute difference 0.0**.

The patched checkpoint is saved beside the original as
`model_training/models/experimental_pothole_cv/Pothole-Computer-Vision-Project-main/pothole_yolov12s.pt`.
**The original `best.pt` is untouched.**

---

## 2. Crack specialist

| Property | Value |
|---|---|
| Asset | `app/src/main/assets/crack_rdd2022.onnx` |
| md5 | `6fc2f37ec4e8d10e9aba1f33a1b6a74b` — byte-identical to `model_training/models/best.onnx` |
| Architecture | **YOLOv8s**, 11,137,535 params |
| Trained on | RDD2022 (not by us; this is your pre-existing model) |
| Format | ONNX, **opset 19 / IR 9** → requires onnxruntime ≥ 1.16 |
| Input | `images` `[1, 3, 640, 640]` float32, **NCHW**, static batch |
| Output | `output0` `[1, 9, 8400]` float32, channels-first |
| Output meaning | `9 = cx, cy, w, h + 5 class scores`. No objectness column |
| Box space | 640-space **pixels** (observed 3.2 … 645.9) |
| Activation | Sigmoid, **independent** (per-anchor sums ≪ 1, so not softmax) |
| NMS in graph | **No** — external NMS required |
| Classes | `0 Longitudinal Crack, 1 Transverse Crack, 2 Alligator Crack, 3 Pothole, 4 Other` |
| Size | 43 MB |

The master copy in `model_training/models/` was **copied, not moved**, and is
unmodified.

---

## 3. Preprocessing — verified separately for each model

Both models happen to require the **same** preprocessing. This was confirmed
independently for each, not assumed from the other:

1. Letterbox to 640×640 preserving aspect ratio, padding value **114** (grey)
2. RGB channel order
3. Divide by 255 — **no ImageNet mean/std normalization**
4. HWC → CHW (NCHW tensor layout), float32

Because both use 640 input, one letterboxed bitmap could in principle be
shared. The code deliberately does **not** share it: each detector letterboxes
independently from its own `ModelSpec.inputSize`, so a future specialist with a
different input size needs no special-casing.

## 4. Postprocessing — same shape family, different class handling

Both decode identically (`[1, 4+nc, anchors]` → transpose → threshold →
per-class greedy NMS at IoU 0.7). The difference is entirely in the **adapter
mapping**, see below.

The detector reads the real `nc` and anchor count out of the loaded graph and
logs a warning if the asset disagrees with its declared spec, so a swapped or
re-exported model cannot silently mis-decode.

---

## 5. Class normalization

| Model | Model's class | → `type` | → `subtype` |
|---|---|---|---|
| pothole | `Pothole` | `POTHOLE` | — |
| crack | `Longitudinal Crack` | `CRACK` | `LONGITUDINAL` |
| crack | `Transverse Crack` | `CRACK` | `TRANSVERSE` |
| crack | `Alligator Crack` | `CRACK` | `ALLIGATOR` |
| crack | `Other` | `OTHER` | — |
| crack | `Pothole` | **dropped** | — |

The model's own label is preserved verbatim in `Detection.className` and is what
the overlay displays, so crack subtype detail is never flattened away.

### Why the crack model's `Pothole` class is dropped

The crack model has its own pothole class (id 3), but the pothole specialist
owns potholes. Dropping it stops two models arguing over the same object with
two incomparable confidences.

**This is a routing decision, not a bug**, and it is one line to reverse in
`models/ModelSpec.kt`:

```kotlin
"Pothole" -> null   // change to ClassMapping(DetectionType.POTHOLE) to let both vote
```

Fusion already de-duplicates same-type overlaps, so enabling it is safe.

---

## 6. Confidence thresholds

Each specialist has its **own** independently tunable threshold in
`config/DetectionConfig.kt`:

```kotlin
potholeThreshold = 0.25f
crackThreshold   = 0.25f
```

These are **initial values, not optimal ones.** They were chosen from the
measurements in `test_results.md` — 0.25 favours recall, which suits a
prototype. The in-app slider moves both at once for quick field tuning.

---

## 7. Fusion logic

`fusion/DetectionFusionEngine.kt` enforces exactly two rules.

### Rule 1 — confidences are never combined arithmetically

Each detection keeps the score its own model produced. The two models were
trained separately, on different datasets, with different class counts; their
confidences are not on a common scale. Averaging or summing them would invent a
number that means nothing.

```
Pothole model:  Pothole 0.57            ->  Pothole 0.57
Crack model:    Alligator Crack 0.56    ->  Alligator Crack 0.56
                                            (NOT 1.13, NOT 0.565)
```

A calibrated fusion score is possible later, but only with validation data to
calibrate against.

### Rule 2 — a pothole and a crack are never duplicates of each other

Overlap is evidence about geometry, not semantics. Cracking around a pothole rim
is two real, co-located defects. Suppression therefore applies **only within one
`DetectionType`**.

### Duplicate handling

| Parameter | Value | Rationale |
|---|---|---|
| Within-model NMS IoU | `0.7` | Ultralytics default |
| Cross-model duplicate IoU | **`0.80`** | Deliberately high — two boxes must almost coincide before one is discarded |

Candidates are sorted by confidence descending, so the survivor of a duplicate
pair is always the more confident detection, regardless of which model produced
it. Same-type near-duplicates from a *single* model that survived per-class NMS
(e.g. the same box labelled both longitudinal and transverse) are also collapsed
here.

---

## 8. Inference scheduling

`inference/InferenceScheduler.kt`.

Backpressure is layered so work can never accumulate:

1. **CameraX `STRATEGY_KEEP_ONLY_LATEST`** — stale frames are dropped by CameraX.
2. **Busy flag** — a frame arriving mid-cycle is dropped, never queued.
3. **FPS cap** (`targetInferenceFps`, default **8**) — bounds how often a cycle
   may *start*, independent of camera frame rate.

The result is graceful degradation: on a slow phone the overlay updates less
often, but the camera preview keeps its own frame rate and memory stays flat.

### Execution modes

| Mode | Behaviour | When to use |
|---|---|---|
| `MODE_PARALLEL` (default) | Both specialists run concurrently, one dedicated thread each | Device has spare cores |
| `MODE_SEQUENTIAL` | One after another | Weak device; lower peak CPU/memory |

Switching is a one-line config change and requires no pipeline rewrite.
Each detector is confined to its own thread, which is what makes reusing its
input buffers safe. Each ORT session is capped at `cores/4` intra-op threads so
the two specialists plus the camera thread do not starve each other.

A specialist that throws is logged and skipped; the frame is still reported
using whatever else succeeded.

---

## 9. Performance

See `test_results.md` for measured numbers. **Laptop CPU figures are not phone
figures** — no on-device measurement has been taken yet, because the build could
not be run on a phone from this environment.

---

## 10. Known limitations

1. **The pothole specialist is a close-range model.** It was trained on
   close-up pothole photographs. On distant dashcam-style frames — which is
   exactly the vehicle-mounted use case — it detected **0%** of ground-truth
   potholes. This is the single most important limitation. See `test_results.md`.
2. **The crack specialist is a dashcam model** (RDD2022) and is the stronger of
   the two on vehicle-view imagery. The two models' competencies are close to
   complementary, but the pothole model's domain is the wrong one for driving.
3. `OTHER` (RDD2022 catch-all) fires often and generically, frequently
   co-firing with the pothole specialist on the same defect. Set
   `includeOtherType = false` for a cleaner overlay.
4. Assets total ~78 MB, so the APK is large.
5. No on-device performance measurement yet.
6. Confidence values across the two models are not calibrated against each
   other; a higher number from one model does not mean "more certain" than a
   lower number from the other.

---

## 11. Adding another specialist

The camera pipeline, fusion engine and overlay do **not** change. Four steps:

1. **Add the asset** to `app/src/main/assets/`.
2. **Declare a `ModelSpec`** in `models/ModelSpec.kt` with the contract read out
   of the actual file (input size, `nc`, anchors, box space, activation, whether
   NMS is in the graph). Do not copy another model's values.
3. **Write its mapper** — return a `ClassMapping` per class, or `null` to drop
   one. Add a new `DetectionType` (e.g. `MANHOLE`) if the taxonomy needs it, and
   a colour for it in `OverlayView.colorFor`.
4. **Register it** in `ModelRegistry.ACTIVE`, and give it a threshold in
   `DetectionConfig` if it needs its own.

The scheduler will allocate it a worker thread and run it automatically;
fusion will merge its output by type.
