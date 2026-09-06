# Road Damage AI — YOLO Training Pipeline

AI-Powered Road Damage Detection using YOLOv11 with transfer learning.
Part of the **AI-Powered Mobile Road Intelligence & Road Damage Detection System**.

## Overview

This pipeline builds a robust YOLO object detection model trained on three combined datasets,
targeting 6 road damage classes for eventual Android edge deployment.

## Current Status

> ### ⚠️ Read this first — the shipped model is NOT trained on this pipeline's dataset
>
> As of **2026-09-05** the project model is `models/best.pt` / `models/best.onnx` — a **YOLOv8s
> trained on RDD2022** (on Kaggle, separately from this repo), with **5 classes**:
>
> `0 Longitudinal Crack · 1 Transverse Crack · 2 Alligator Crack · 3 Pothole · 4 Other`
>
> That is a **different taxonomy** from the 6-class local dataset this pipeline builds
> (where `pothole` is 0, not 3). Both are valid; they are not interchangeable.
>
> | Use this | For |
> |---|---|
> | **`configs/model_classes.yaml`** | Inference, Android, anything reading the shipped model |
> | `configs/data.yaml` | Training/evaluating a model on the **local 6-class dataset** |
>
> Validation record: [`reports/model_sanity_check.md`](reports/model_sanity_check.md) — verdict
> **PASS** (ONNX export faithful to 0.000000 conf / 0.3 px).
> Re-check any new export with `python scripts/verify_model_contract.py`.

**Data pipeline: complete and verified.**

| Stage | Status |
|---|---|
| Dataset inspection | ✅ `reports/dataset_inspection_report.md` |
| Annotation validation (raw + final splits) | ✅ `reports/annotation_validation_report.md` |
| Class harmonization | ✅ `configs/class_mapping.yaml`, `reports/harmonization_report.md` |
| Duplicate / near-duplicate detection | ✅ `reports/duplicate_detection_report.md` |
| Leakage-aware train/val/test split | ✅ 7,413 / 1,482 / 988 — `reports/split_report.md` |
| Annotation previews (visually verified) | ✅ `runs/dataset_preview/` |
| Pipeline smoke test (train → eval → predict) | ✅ passed end to end |
| Training on the local 6-class dataset | ⬜ not pursued — superseded by the RDD2022 model |
| **Shipped model (RDD2022, 5-class)** | ✅ `models/best.pt`, `models/best.onnx` |
| **Model sanity check + ONNX validation** | ✅ **PASS** — `reports/model_sanity_check.md` |
| Android integration | ⬜ next — contract in `configs/model_classes.yaml` |

📄 **Full pre-training report:** [`reports/pipeline_summary.md`](reports/pipeline_summary.md)
📄 **Step-by-step training guide (Hinglish):** [`GUIDE.md`](GUIDE.md) — read this if you are
running the training on another laptop.

### Class Taxonomies — there are two, keep them straight

**A. Shipped model** (`models/best.onnx`, RDD2022-trained) — **use this for inference/Android.**
Authoritative copy: `configs/model_classes.yaml`

| ID | Class | RDD2022 code |
|---|---|---|
| 0 | Longitudinal Crack | D00 |
| 1 | Transverse Crack | D10 |
| 2 | Alligator Crack | D20 |
| 3 | **Pothole** | D40 |
| 4 | Other | — |

Recorded metrics (its own RDD2022 val split): P 0.662 · R 0.583 · **mAP50 0.629** · mAP50-95 0.345

**B. Local dataset** (`data/processed/road_damage_combined/`, built by this pipeline from the three
Kaggle datasets) — use this only for training/evaluating on that data. Config: `configs/data.yaml`

| ID | Class | Description |
|---|---|---|
| 0 | pothole | Road potholes |
| 1 | crack | Road surface cracks |
| 2 | manhole | Manhole covers |
| 3 | road_damage | General road damage (undifferentiated) |
| 4 | speed_bump | Speed bumps |
| 5 | unsurfaced_road | Unpaved/unsurfaced road segments |

> `scripts/evaluate.py` will refuse to evaluate a model from taxonomy A against a dataset from
> taxonomy B (it produces meaningless near-zero metrics). Override only if you mean it:
> `--force-taxonomy-mismatch`

### Datasets

| Dataset | Images | Key Classes |
|---|---|---|
| RAD (Road Anomaly Detection) | 8,394 | road_damage, speed_bump, unsurfaced_road |
| Road Damage Dataset | 2,009 | pothole, crack, manhole |
| Pothole Detection Dataset | 3,940 | pothole |

## Setup

```bash
cd road_damage_ai
pip install -r requirements.txt
```

## Pipeline Execution

Run scripts **in order**. Do NOT skip to training.

```bash
# Step 1: Inspect raw datasets
python scripts/inspect_datasets.py

# Step 2: Validate annotations
python scripts/validate_annotations.py

# Step 3: Verify annotation formats
python scripts/convert_annotations.py

# Step 4: Harmonize classes across datasets
python scripts/harmonize_classes.py

# Step 5: Detect duplicate images
python scripts/detect_duplicates.py

# Step 6: Create train/val/test splits (leakage-aware)
python scripts/split_dataset.py

# Step 7: Generate visual previews (INSPECT BEFORE TRAINING!)
python scripts/visualize_annotations.py

# Step 8: Train (run on GPU machine)
python scripts/train.py --model yolo11n.pt --epochs 100 --batch 16 --imgsz 640 --name road_damage_v1

# Step 9: Evaluate on the held-out test split
python scripts/evaluate.py --model runs/train/road_damage_v1/weights/best.pt --split test

# Step 10: Inference (image, folder, video, or webcam)
python scripts/predict.py --source <image_or_video> --model runs/train/road_damage_v1/weights/best.pt
```

Steps 1–7 have already been run; their outputs are in `reports/` and `runs/dataset_preview/`.
Re-run them only if you re-copy `data/raw/` and want to rebuild the processed dataset.

## Project Structure

```
road_damage_ai/
├── data/
│   ├── raw/                          # Original datasets (NEVER modified)
│   │   ├── rad/
│   │   ├── road_damage/
│   │   └── pothole_dataset/
│   └── processed/
│       └── road_damage_combined/     # Harmonized + split dataset
│           ├── images/{train,val,test}/
│           └── labels/{train,val,test}/
├── models/                           # ⭐ SHIPPED MODEL (RDD2022, 5-class)
│   ├── best.pt                       #    YOLOv8s checkpoint
│   └── best.onnx                     #    ONNX export — Android target
├── configs/
│   ├── model_classes.yaml            # ⭐ SHIPPED MODEL contract — classes,
│   │                                 #    ONNX I/O, pre/postprocessing (Android)
│   ├── data.yaml                     # LOCAL 6-class dataset config (training only)
│   └── class_mapping.yaml            # Local dataset class mapping documentation
├── scripts/
│   ├── inspect_datasets.py           # Dataset structure inspection
│   ├── validate_annotations.py       # Annotation validation
│   ├── convert_annotations.py        # Format verification
│   ├── harmonize_classes.py          # Class ID remapping
│   ├── detect_duplicates.py          # Duplicate image detection
│   ├── split_dataset.py              # Leakage-aware splitting
│   ├── visualize_annotations.py      # Annotation preview generation
│   ├── make_cross_dataset_configs.py # Cross-dataset experiment configs
│   ├── train.py                      # YOLO training
│   ├── evaluate.py                   # Model evaluation (guards taxonomy mismatch)
│   ├── verify_model_contract.py      # ⭐ Re-verify model vs model_classes.yaml
│   └── predict.py                    # Inference (image/video/webcam)
├── utils/
│   ├── config.py                     # Central configuration
│   ├── dataset_utils.py              # File discovery, parsing
│   ├── annotation_utils.py           # Validation, remapping
│   └── visualization.py             # Bounding box drawing
├── weights/
│   └── yolo11n.pt                    # Bundled pretrained checkpoint (offline-safe)
├── reports/                          # Generated reports
├── runs/                             # Training runs & previews
├── requirements.txt
├── GUIDE.md                          # Step-by-step training guide (Hinglish)
└── README.md
```

## Hardware

- **Data preparation**: Runs on CPU (this machine)
- **Training**: Requires CUDA GPU (transfer to GPU machine)
- **Inference**: Auto-detects CUDA/MPS/CPU

## Key Design Decisions

1. **RAD "RoadDamages"** kept as separate `road_damage` class (not mapped to pothole)
2. **SpeedBump + UnsurfacedRoad** included for complete road intelligence
3. **RAD vehicles/pedestrians** (HMV, LMV, Pedestrian) excluded from training
4. **Leakage-aware splitting**: Video frames from same source grouped in same split
5. **Transfer learning** from YOLOv11n for mobile deployment compatibility

## Future Architecture

```
Camera → YOLO → Visual Detection
IMU → Motion Features              → Future Event
TF-Luna → Surface Distance            Fusion Engine → Confirmed Road Event
GPS → Location
```

**Current phase**: Computer vision model only.
Sensor fusion, GPS, Android app, and dashboard are future phases.
