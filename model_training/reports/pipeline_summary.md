# Pipeline Summary — Pre-Training Report

**Project:** AI-Powered Mobile Road Intelligence & Road Damage Detection System
**Phase:** Computer vision model only (no GPS / IMU / LiDAR / ESP32 / Android / dashboard)
**Status:** Data pipeline complete and verified. Ready for training on a GPU machine.

> RDD2022 is not used anywhere in this project. Only the three team-provided datasets below.

---

## 1. Dataset Summary

| Dataset | Images | Videos | Annotations | Native format | Image size | Origin |
|---|---|---|---|---|---|---|
| RAD — Road Anomaly Detection | 8,394 | 364 | 29,941 | YOLO txt, `train/valid/test` | 1920×1080 | India (dashcam) |
| Road Damage: Potholes, Cracks, Manholes | 2,009 | 0 | 4,737 | YOLO txt (`labels-YOLO`) + originals in `labels` | 640×360 | Europe (dashcam) |
| Pothole Detection 3900+ (YOLOv11 optimized) | 3,940 | 0 | 10,111 | YOLO txt, `train/valid/test` | 640×640 | Mixed / Roboflow-augmented |
| **Total** | **14,343** | **364** | **44,789** | | | |

### Original class taxonomies (read from the actual label files, not the dataset titles)

| Dataset | Class IDs found | Names |
|---|---|---|
| RAD | 0–5 | HMV, LMV, Pedestrian, RoadDamages, SpeedBump, UnsurfacedRoad |
| Road Damage | 0–2 | pothole, crack, manhole |
| Pothole | 0 | pothole |

### RAD raw videos

364 videos, ~5.5 GB total, 1920×1080 @ 30 fps, ~10–13 s each.

**Decision: no frames extracted.** The videos carry no annotations, and RAD's labelled
images are already video-derived frames. Extracting frames would require fabricating
labels, which is explicitly out of scope. Videos are kept only as inference test material.

---

## 2. Class Mapping

Final taxonomy (6 classes) — full rationale in `configs/class_mapping.yaml`.

| Source dataset | Original ID / name | → Final ID | Final name | Reason |
|---|---|---|---|---|
| Road Damage | 0 pothole | 0 | pothole | Direct semantic match |
| Pothole | 0 pothole | 0 | pothole | Direct semantic match |
| Road Damage | 1 crack | 1 | crack | Direct semantic match |
| Road Damage | 2 manhole | 2 | manhole | Direct semantic match |
| RAD | 3 RoadDamages | 3 | road_damage | Generic, undifferentiated damage — cannot be split into pothole/crack without re-annotation |
| RAD | 4 SpeedBump | 4 | speed_bump | Useful for road intelligence |
| RAD | 5 UnsurfacedRoad | 5 | unsurfaced_road | Useful for road condition assessment |
| RAD | 0 HMV | — | **excluded** | Vehicle class, not road damage |
| RAD | 1 LMV | — | **excluded** | Vehicle class, not road damage |
| RAD | 2 Pedestrian | — | **excluded** | Not road damage |

**RAD `RoadDamages` was deliberately NOT merged into `pothole`.** Its annotations mark
generic surface damage from a moving dashcam, without distinguishing pothole from crack.
Merging it would have injected label noise into both classes.

Excluding vehicles/pedestrians removed 4,460 RAD images that contained no
road-damage-relevant objects (8,394 → 3,934 images kept).

---

## 3. Data Quality

### Annotation validation (`reports/annotation_validation_report.md`)

| Dataset | Label files | Annotations | Empty labels | Missing images | Missing labels | Malformed |
|---|---|---|---|---|---|---|
| RAD (raw) | 8,394 | 29,941 | 91 | 0 | 0 | 0 |
| Road Damage (raw) | 2,009 | 4,737 | 0 | 0 | 0 | 1 |
| Pothole (raw) | 3,940 | 10,111 | 0 | 0 | 0 | 0 |
| FINAL/train | 7,413 | 17,045 | 0 | 0 | 0 | 0 |
| FINAL/val | 1,482 | 3,225 | 0 | 0 | 0 | 0 |
| FINAL/test | 988 | 2,132 | 0 | 0 | 0 | 0 |

The single malformed annotation in Dataset 2 is a zero-height box
(`w=0.003125, h=0.0`). `split_dataset.py` drops degenerate boxes when writing the
processed labels, so the final splits are clean. **The raw dataset is untouched.**

Corrupted images: 0 in every sampled check. No missing image/label counterparts anywhere.

### Duplicates (`reports/duplicate_detection_report.md`)

| Type | Count | Notes |
|---|---|---|
| Exact duplicates (MD5) | 5 pairs | All within Dataset 2 (near-identical video grabs) |
| Near duplicates (pHash ≤ 8) | 5,416 pairs | RAD↔RAD 4,303 · Pothole↔Pothole 626 · RoadDamage↔RoadDamage 487 |
| **Cross-dataset duplicates** | **0** | The three datasets do not overlap |

Nothing was deleted. Duplicate clusters are instead fed into the splitter (below) so
that twins cannot land on opposite sides of a split. Removal remains available via
`detect_duplicates.py --remove`.

That zero cross-dataset overlap is a meaningful result: the combined dataset genuinely
covers three distinct visual domains (Indian dashcam, European dashcam, close-up
pothole photos) rather than re-counting the same roads.

---

## 4. Split Strategy (leakage-aware)

Splitting operates on **groups**, never on individual images. A group is formed by:

1. **Filename provenance** — RAD frames from the same source video (`XX_DDMMYYYY_mp4-N`),
   Road Damage grabs from the same capture session, Pothole images sharing a base name
   (Roboflow augmentation copies).
2. **Duplicate detection** — exact and near-duplicate pairs are union-found into the
   groups from step 1. This merged 113 additional groups.

Groups are then stratified per source dataset and assigned largest-first to whichever
split is furthest below its target. Result: 1,092 atomic groups → exact target ratios
with every source and every class present in every split.

| Split | Images | % | RAD | Road Damage | Pothole | Annotations |
|---|---|---|---|---|---|---|
| train | 7,413 | 75.0% | 2,951 | 1,507 | 2,955 | 17,045 |
| val | 1,482 | 15.0% | 590 | 301 | 591 | 3,225 |
| test | 988 | 10.0% | 393 | 201 | 394 | 2,132 |

Official upstream splits were **not** reused: RAD and Pothole each ship their own
train/valid/test, but merging three independent split schemes gives no leakage guarantee
across datasets and none of them account for the near-duplicate structure found above.
A single leakage-aware split over the combined pool is the defensible choice, and it is
fully reproducible (`--seed 42`).

### Class distribution per split

| ID | Class | train | val | test |
|---|---|---|---|---|
| 0 | pothole | 8,708 | 1,626 | 1,038 |
| 1 | crack | 1,886 | 309 | 324 |
| 2 | manhole | 757 | 156 | 44 |
| 3 | road_damage | 4,864 | 989 | 610 |
| 4 | speed_bump | 303 | 104 | 92 |
| 5 | unsurfaced_road | 527 | 42 | 24 |

---

## 5. Visual Verification

Previews in `runs/dataset_preview/` were generated and inspected before training was
allowed to proceed:

- `rad_preview.jpg` — vehicle/pedestrian boxes correct; `RoadDamages` boxes sit on real
  surface damage but are small and far from the camera.
- `road_damage_preview.jpg` — pothole/crack/manhole boxes are tight and correct.
- `pothole_preview.jpg` — correct.
- `combined_{train,val,test}_preview.jpg` — harmonized class IDs render with the right
  names and colours, confirming the remap did not shuffle classes.

**Conclusion: annotations and the class remap are verified. Training is unblocked.**

---

## 6. Pipeline Verification (smoke test)

A 1-epoch run at `--imgsz 320 --batch 4 --fraction 0.01` completed end to end on CPU:

- `configs/data.yaml` resolved to an absolute dataset root on this machine
- Ultralytics scanned all splits: **0 corrupt, 0 background** images
- all 6 classes present in the validation scan
- `best.pt` produced (5.2 MB)
- `evaluate.py --split test` produced overall + per-class metrics
- `predict.py` verified on an image folder and on a 153-frame RAD video, producing
  `annotated_video.mp4` (with frame number + timestamp overlay) and `detections.json`

Smoke artifacts were deleted afterwards; metrics from that run are meaningless by design.

---

## 7. Model Configuration

| Property | Value |
|---|---|
| Architecture | YOLO11-nano, transfer learning from `weights/yolo11n.pt` (bundled) |
| Trained from scratch | No |
| Classes | 6 |
| Image size | 640 |
| Default epochs / batch / patience | 100 / 16 / 20 |
| Augmentation | hsv, ±5° rotation, translate 0.1, scale 0.3, shear 2°, perspective 0.0005, horizontal flip 0.5, mosaic 1.0, mixup 0.1, erasing 0.1 |
| Vertical flip | Disabled — physically unrealistic for road scenes |
| Target deployment | Android edge (TFLite / NCNN export, future phase) |
| Image caching | Disabled (ultralytics default) — images stream from disk |
| Measured peak system RAM | **~4.2 GB** at batch 16 / imgsz 640 / workers 8 |

---

## 8. Known Limitations

1. **`road_damage` overlaps semantically with `pothole` and `crack`.** RAD's generic
   damage class covers the same physical phenomena the other two datasets label
   specifically. Expect confusion between class 3 and classes 0/1 in the confusion
   matrix. This is a property of the source annotations, not a pipeline bug.
2. **Class imbalance.** pothole 11,372 boxes vs speed_bump 499 and unsurfaced_road 593.
   The rare classes will have weak metrics; per-class evaluation will show this honestly.
3. **RAD damage objects are small and distant** at 1920×1080 downscaled to 640. If
   `road_damage` recall is poor, `--imgsz 960` is the first thing to try.
4. **Domain gap between datasets** is real (Indian dashcam vs European dashcam vs
   close-up pothole photos). This makes cross-dataset generalization tests meaningful —
   and makes strong combined-test numbers less impressive than they look.
5. **`manhole` and `unsurfaced_road` have few test instances** (44 and 24). Their test
   metrics will be high-variance.
6. **RAD videos are unused for training** (no labels). Only as inference material.

---

## 9. What Is Still Pending

These require the GPU machine — see `GUIDE.md` for step-by-step (Hinglish) instructions.

| Step | Command |
|---|---|
| Train | `python scripts/train.py --model yolo11n.pt --epochs 100 --batch 16 --imgsz 640 --name road_damage_v1` |
| Evaluate on test | `python scripts/evaluate.py --model runs/train/road_damage_v1/weights/best.pt --split test` |
| Cross-dataset analysis | add `--cross-dataset` |
| Image / folder inference | `python scripts/predict.py --source <path> --model .../best.pt` |
| Video inference | `python scripts/predict.py --source <video.mp4> --model .../best.pt` |

### Cross-dataset generalization experiments (optional, each needs its own training run)

Configs are generated by `python scripts/make_cross_dataset_configs.py`, which writes
image-list based data configs — no images are copied or moved.

| Experiment | Train on | Test on | Train / Val / Test images | Config |
|---|---|---|---|---|
| A | Combined | Combined hold-out | 7,413 / 1,482 / 988 | `configs/data.yaml` |
| B | RAD + Pothole | Road Damage | 5,906 / 1,181 / 2,009 | `configs/cross/exp_b.yaml` |
| C | RAD + Road Damage | Pothole | 4,458 / 891 / 3,940 | `configs/cross/exp_c.yaml` |
| D | Road Damage + Pothole | RAD | 4,462 / 892 / 3,934 | `configs/cross/exp_d.yaml` |

```
python scripts/train.py --data configs/cross/exp_b.yaml --epochs 50 --name cross_b
python scripts/evaluate.py --model runs/train/cross_b/weights/best.pt \
    --data configs/cross/exp_b.yaml --split test --name cross_b_eval
```

B/C/D are only partly comparable: `crack`/`manhole` exist solely in Dataset 2, and
`road_damage`/`speed_bump`/`unsurfaced_road` solely in RAD, so the held-out source always
contains classes the model never trained on. Read those runs on the shared `pothole`
class; the remaining classes measure zero-shot behaviour and are expected to score ~0.
The generated configs use absolute paths — regenerate them after copying the project.

---

## 10. Next Step After Training

Depending on results: `best.pt` → TFLite / NCNN export → Android (Phone 1),
then ESP32 + TF-Luna, GPS gateway (Phone 2), and the fusion engine.
None of that is in scope for the current phase.
