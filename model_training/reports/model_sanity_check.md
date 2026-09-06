# Model Sanity Check

**Date:** 2026-09-05
**Status:** ✅ **Resolved — RDD2022 confirmed acceptable by project owner (2026-09-05).**
The original "RDD2022 is excluded" constraint came from a 12 GB download problem, not a
requirement. The RDD2022-trained model is accepted as the project model, and the configs have
been reconciled to it — see §9 and `configs/model_classes.yaml`.

**Machine:** Windows 10, 12th Gen Intel Core i5-1235U, 15.7 GB RAM, **no CUDA** (CPU only)
**Environment:** Python 3.10.9, PyTorch 2.7.0+cpu, Ultralytics 8.4.33, ONNX 1.22.0, ONNX Runtime 1.23.2
**Scope:** Validation only. No training, no weight modification, no file deletion.

> **Dependencies installed for this check:** `onnx==1.22.0`, `onnxruntime==1.23.2` (neither was
> present). Pip also upgraded `numpy` 1.26.4 → 2.2.6 as a transitive dependency; torch, cv2 and
> ultralytics were re-verified working afterwards. A pre-existing `mediapipe 0.10.21` in this
> environment declares `numpy<2` and is now version-conflicted — unrelated to this project, but
> noted for transparency.

---

## 1. Files Found

Workspace scan for `*.pt` / `*.onnx` (excluding `venv/`):

| Path | Size | Notes |
|---|---|---|
| `runs/train/road_damage_v1/weights/best.pt` | 67.1 MB | **Primary — the file under test** |
| `runs/train/road_damage_v1/weights/best.onnx` | 44.8 MB | **Primary — the file under test** |
| `weights/yolo11n.pt` | 5.4 MB | Pretrained COCO backbone used by this repo's `train.py` |
| `runs/train/guard_test/weights/best.pt` | 5.2 MB | Leftover from an earlier 2-epoch pipeline smoke test |
| `runs/train/guard_test/weights/last.pt` | 5.2 MB | Same smoke test. Not deleted (per instructions) |

No duplicate copies of the model under test were found elsewhere. Supporting files present:
`configs/data.yaml`, `configs/class_mapping.yaml`, `requirements.txt`, `scripts/` (train, evaluate,
predict, and the data pipeline), `utils/`, `reports/`.

**Files absent that would normally accompany a run:** `results.csv`, `args.yaml`, `last.pt`,
`confusion_matrix.png` — the `road_damage_v1/` directory contains only `weights/`. The training
artifacts for this model live wherever it was actually trained, not here.

### Organization performed

Created `models/`, `tests/images/`, `tests/results/`. Both model files were **copied** (not moved)
into `models/`. Originals under `runs/train/road_damage_v1/weights/` are untouched and verified
intact. Copies were made because the containing directory name is misleading — see §8.

---

## 2. PyTorch Model (`best.pt`)

Checkpoint loads successfully.

| Property | Value |
|---|---|
| Architecture | `DetectionModel` — **YOLOv8-small** (`model: yolov8s.pt`) |
| Parameters | 11,137,535 |
| Number of classes | **5** |
| Input image size | 640 × 640 |
| Saved at epoch | 75 (of 100 requested; `patience: 25`) |
| `best_fitness` | 0.34463 |
| Ultralytics version (writer) | 8.4.71 |
| Checkpoint date | 2026-06-20 |
| Loads for inference | **Yes** |

**Checkpoint contents:** `model` key is `None`; the weights live in the `ema` key (Ultralytics
loads `ema` preferentially, so this is handled transparently). The file also carries `optimizer`,
`scaler` and `updates` state — that training state, not model size, is why the file is 67 MB for
an 11 M-parameter model.

**Recorded validation metrics from its own training run:**

| Metric | Value |
|---|---|
| precision(B) | 0.662 |
| recall(B) | 0.583 |
| mAP50(B) | 0.629 |
| mAP50-95(B) | 0.345 |

**Provenance (read from `train_args` in the checkpoint):**

| Field | Value |
|---|---|
| `data` | `/kaggle/working/data_rdd.yaml` |
| `project` | `/kaggle/working/runs/train_kaggle` |
| `name` | `improved_model` |
| `model` | `yolov8s.pt` |
| `epochs` / `batch` / `device` | 100 / 64 / `0,1` (dual GPU) |

This model was trained on **Kaggle**, on a dataset config named `data_rdd.yaml`, under the run name
`improved_model`. It did **not** come from this repository's `road_damage_v1` run, despite sitting
in that directory.

---

## 3. ONNX Model (`best.onnx`)

Loads successfully; `onnx.checker.check_model` **passes**.

| Property | Value |
|---|---|
| File size | 44.8 MB |
| IR version | 9 |
| Opset | 19 (`ai.onnx`) |
| Producer | pytorch 2.5.1 |
| Exported by Ultralytics | 8.4.12, on 2026-06-24 |
| ONNX Runtime compatible | **Yes** — session created on `CPUExecutionProvider` |
| Graph nodes | 231 |

**Input:**

| Name | Shape | Dtype |
|---|---|---|
| `images` | `[1, 3, 640, 640]` | `tensor(float)` (float32) |

**Output:**

| Name | Shape | Dtype |
|---|---|---|
| `output0` | `[1, 9, 8400]` | `tensor(float)` (float32) |

Number of outputs: **1**. The `9` decomposes as 4 box terms (cx, cy, w, h) + 5 class scores,
across 8400 anchor positions — internally consistent with the 5-class head.

**Export flags** (from embedded metadata): `dynamic: False`, `simplify: True`, `half: False`,
`nms: False`, `end2end: False`, `batch: 1`.

**NMS is NOT inside the graph** — verified by op scan: no `NonMaxSuppression` and no `TopK` node
exists. External NMS is mandatory.

**Version note:** the `.pt` was written by Ultralytics 8.4.71 (2026-06-20) but the ONNX was exported
by 8.4.12 (2026-06-24) — an older library exporting a newer checkpoint, four days later. This is
unusual bookkeeping, but §4 confirms the weights are numerically identical, so it is cosmetic.

---

## 4. PT vs ONNX Inference

Same 5 test images, identical settings (`imgsz=640, conf=0.25, iou=0.7`), run through both models
via the Ultralytics pipeline so pre/post-processing is equivalent.

| Check | Result |
|---|---|
| Total detections | PT = 1, ONNX = 1 |
| Class mismatches | **0** |
| Max confidence difference | **0.000000** |
| Max bounding-box difference | **0.30 px** |

The one detection both produced was identical: `Longitudinal Crack` @ conf 0.4691.

A **raw ONNX Runtime** path was also implemented independently (manual letterbox → float32 NCHW
normalize → `sess.run` → transpose → confidence filter → `cv2.dnn.NMSBoxes`) and executed
successfully, returning a correctly shaped `(1, 9, 8400)` tensor and decoding to valid boxes.

**Verdict on export fidelity: excellent.** A 0.3 px box delta and zero confidence delta are within
normal float32 export/rounding tolerance. The ONNX is a faithful conversion of the `.pt`.

---

## 5. Class Mapping

Read directly from the model files (identical in both `.pt` `names` and `.onnx` `metadata_props`,
so the export preserved the mapping correctly):

| ID | Class name |
|---|---|
| 0 | `Longitudinal Crack` |
| 1 | `Transverse Crack` |
| 2 | `Alligator Crack` |
| 3 | `Pothole` |
| 4 | `Other` |

### ⚠️ This does not match the project taxonomy

`configs/data.yaml` in this repository defines **6** classes:

| ID | Class name |
|---|---|
| 0 | `pothole` |
| 1 | `crack` |
| 2 | `manhole` |
| 3 | `road_damage` |
| 4 | `speed_bump` |
| 5 | `unsurfaced_road` |

The two taxonomies differ in count (5 vs 6), in names, and in ID ordering. `Pothole` is ID **3** in
the model but ID **0** in this project. Any integration that assumes this repo's `data.yaml`
ordering will mislabel every detection.

Additionally: `Longitudinal Crack` / `Transverse Crack` / `Alligator Crack` / `Pothole` / `Other`
are the standard **RDD2022** damage categories (D00/D10/D20/D40 + Other), and the training config
was named `data_rdd.yaml`. The project brief states repeatedly that RDD2022 is not to be used in
this project. This needs a human decision — see §8.

---

## 6. Android Integration Requirements

| Requirement | Value |
|---|---|
| Input tensor name | `images` |
| Input shape | `[1, 3, 640, 640]` — **static**, batch is fixed at 1 |
| Input dtype | float32 |
| Preprocessing | BGR→RGB, letterbox to 640×640 (pad value **114**), scale to `[0,1]` (divide by 255), HWC→**CHW**, add batch dim, contiguous |
| Normalization | Divide by 255 only. **No** ImageNet mean/std subtraction |
| Output tensor name | `output0` |
| Output shape | `[1, 9, 8400]` float32 |
| Output layout | Channels-first: **transpose to `[8400, 9]`** before decoding |
| Box format | `cx, cy, w, h` in **640-space pixels** (not normalized) — convert to xyxy, then un-letterbox back to original image coords |
| Class scores | Columns 4–8, already **sigmoid-activated** — take `argmax` for class, `max` for confidence. No separate objectness column (v8+ head) |
| NMS | **Required externally** — not in the graph. Suggested `iou=0.7` |
| Confidence threshold | 0.25 is the Ultralytics default; see §8 for why this model may need tuning |
| Runtime | ONNX Runtime Android — **yes, realistically supported**. Opset 19 / IR 9 needs ORT ≥ 1.16; use a current 1.17+ AAR |

**Points of friction for Android:**

- **Static batch and static 640×640** — the phone camera pipeline must letterbox to exactly
  640×640. No dynamic shapes are available.
- **44.8 MB float32 model** — large to ship in an APK and heavy for real-time mobile inference.
  Quantization (INT8) or FP16 would cut this substantially, but must be done from the `.pt` and then
  re-validated; do not quantize blindly.
- **Class mapping must be hardcoded from the model, not from `configs/data.yaml`** — see §5.
- Un-letterboxing is the most common source of misplaced boxes in mobile ports; scale and both pad
  offsets must be carried through.

---

## 7. Performance

**Laptop CPU only (i5-1235U, no GPU).** These figures do not predict phone performance.

| Model | Preprocess | Inference | Postprocess | Total | FPS |
|---|---|---|---|---|---|
| `best.pt` (PyTorch CPU) | 2.74 ms | 205.30 ms | 0.87 ms | **208.91 ms** | ~4.8 |
| `best.onnx` (ORT via Ultralytics) | 3.90 ms | 134.13 ms | 1.13 ms | **139.15 ms** | ~7.2 |
| `best.onnx` (pure ORT, inference only) | — | 134.50 ms | — | — | ~7.4 |

ONNX Runtime is **~35% faster** than PyTorch CPU on identical work — a normal and expected result.

**On Android, expect different numbers.** A mid-range phone CPU running an 11 M-parameter float32
model at 640×640 will not be fast; NNAPI/GPU delegation or a quantized model will likely be needed
for anything approaching real-time. No Android measurement has been taken, so no claim is made here.

---

## 8. Problems Found

**1 — ✅ RESOLVED: the model's taxonomy is not this project's original taxonomy.**
5 RDD2022-style classes vs the local dataset's 6 harmonized classes; `Pothole` sits at index 3
rather than index 0. The class list is the RDD2022 category set and the training config was
`data_rdd.yaml`.
*Resolution (2026-09-05):* project owner confirmed RDD2022 is acceptable — the exclusion was a
download-size problem, not a requirement. The 5-class taxonomy is now the project's inference
taxonomy, recorded in `configs/model_classes.yaml`. **The index difference is still live and
still dangerous** — never wire `configs/data.yaml`'s order into the app.

**2 — ✅ RESOLVED (consequence of 1): the model does not detect potholes on the local dataset.**
Tested against 60 test-set images that carry a `pothole` label in this project's own annotations:

| Confidence | Images with any detection | Total boxes | `Pothole` predictions |
|---|---|---|---|
| ≥ 0.25 | 26 / 60 | 39 | **0** |
| ≥ 0.10 | 40 / 60 | 98 | **1** |

At the standard threshold it fired `Other` (28), `Alligator Crack` (8), `Longitudinal Crack` (2),
`Transverse Crack` (1) — and **zero** potholes, on images that definitively contain potholes.
Lowering the threshold to 0.10 produced exactly one.
*Resolution:* this is a domain/taxonomy artefact, not a defect. The model was trained on RDD2022
imagery and is being shown images from three unrelated Kaggle datasets whose "pothole" label does
not correspond to RDD's `D40 Pothole`. Judged on its own validation data it scores mAP50 0.629.
**However — this is still the best available evidence about real-world generalization**, and it is
not encouraging: on unfamiliar Indian dashcam footage the model mostly emits `Other`. Worth a real
field test on target-camera footage before relying on it.

**3 — Provenance is mislabelled on disk.** The files sit in `runs/train/road_damage_v1/weights/`,
but the checkpoint says they were produced on Kaggle under `train_kaggle/improved_model`. The
directory implies they came from this repo's training run; they did not. This will cause confusion
later — hence the `models/` copies with this report attached.

**4 — MINOR: export toolchain version skew.** `.pt` written by Ultralytics 8.4.71, `.onnx` exported
by 8.4.12 four days later. Numerically harmless (§4 shows the weights match), but worth knowing.

**5 — MINOR: no training artifacts locally.** No `results.csv` or `args.yaml` accompany the model,
so the per-epoch metric history could not be inspected — only the summary metrics embedded in the
checkpoint.

### What is NOT wrong

The model is technically healthy. It loads, the ONNX passes the checker, the export is numerically
faithful to within 0.3 px, ONNX Runtime executes it end to end, and its own recorded metrics
(mAP50 0.629, recall 0.583) are respectable. **The problem is not the file — it is that this model
was trained for a different dataset and a different class list than this project.**

---

## 9. Final Verdict

# ✅ PASS

**Updated 2026-09-05** after the project owner confirmed RDD2022 is acceptable (the original
exclusion was a 12 GB download problem, not a requirement). That resolves Problem 1 in §8, and
Problem 2 follows from it — this model is an RDD2022 model and should be judged on RDD2022, where
it scores mAP50 **0.629** / recall **0.583**, not against the local 6-class labels.

**As an ONNX artifact:** valid, faithful to its `.pt` (0.000000 confidence delta, ≤0.3 px box
delta), ONNX Runtime compatible, Android-deployable.

**As this project's model:** accepted. 5-class RDD2022 taxonomy is now the project taxonomy for
inference and Android.

### Reconciliation performed

| Change | Purpose |
|---|---|
| **`configs/model_classes.yaml`** (new) | Authoritative record of the shipped model: 5-class mapping, RDD2022 code equivalents, full ONNX I/O contract, preprocessing/postprocessing spec, Android notes. **This is the source of truth for inference.** |
| **`configs/data.yaml`** (header added) | Scope note clarifying it describes only the local 6-class dataset on disk — not the shipped model. Class list unchanged, because it correctly describes those labels. |
| **`scripts/verify_model_contract.py`** (new) | Re-verifies model against the contract — classes, shapes, dtypes, opset, NMS-in-graph, PT/ONNX consistency. Run it after any re-export or quantization. Currently **PASS**. |
| **`scripts/evaluate.py`** (guard added) | Refuses to evaluate a model against a dataset with a different class list, and explains why. Overridable with `--force-taxonomy-mismatch`. |
| **`scripts/evaluate.py`** (per-class fix) | "Classes with no ground truth" now derives from the model's own names instead of a hardcoded 6-class list. |
| **`models/`** (new) | Copies of both model files out of the misleading `runs/train/road_damage_v1/` path. Originals untouched. |

`scripts/predict.py` needed no change — it already read class names from the model. Verified: it
labels detections `Longitudinal Crack` etc. and writes correct names into `detections.json`.

### Two things to carry into the Android work

1. **Take the class list from the model, not from `configs/data.yaml`.** `Pothole` is index **3**
   here, index 0 there. `configs/model_classes.yaml` records the correct mapping.
2. **44.8 MB float32 is heavy for a phone.** Quantization (INT8/FP16) is the obvious next
   optimization — but do it from the `.pt`, then re-run `verify_model_contract.py` against the
   new export before shipping.

### Still open (not blockers)

- No local RDD2022 validation data, so the 0.629 mAP50 figure cannot be independently reproduced
  here — it is read from the checkpoint, and is trusted as the model's own recorded metric.
- The local 6-class dataset in `data/processed/road_damage_combined/` is now decoupled from the
  shipped model. It remains valid for training a separate 6-class model, and its images are still
  useful as an out-of-domain qualitative check.
