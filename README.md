# SIH Road Damage Detection

AI-powered road damage detection project with an Android camera app and a YOLO
training/evaluation pipeline. The current mobile demo runs two specialist ONNX
models on-device, then fuses their detections into one road-anomaly overlay.

## What This Project Contains

- `android_test/` - Android CameraX test app using ONNX Runtime.
- `model_training/` - YOLO training, evaluation, conversion, and reporting tools.
- `model_training/models/` - shipped model checkpoints and ONNX exports.
- `model_training/reports/` - validation, dataset, duplicate, evaluation, and model sanity reports.
- `.github/modernize/` - GitHub modernization hook scripts.

The dataset is intentionally not committed. Put local training data under
`model_training/data/` when reproducing the pipeline.

## Android App

The Android app performs live camera road-anomaly detection using two independent
models:

| Role | Asset | Detects |
| --- | --- | --- |
| Pothole specialist | `android_test/app/src/main/assets/pothole_yolov12s.onnx` | Potholes |
| Crack specialist | `android_test/app/src/main/assets/crack_rdd2022.onnx` | Longitudinal, transverse, alligator cracks, potholes, other damage |

The app uses ONNX Runtime for Android and fuses detections in app code. Pothole
and crack detections are kept as separate road defects; confidence scores from
different models are not combined.

Build from Android Studio, or from a compatible Gradle/JDK setup:

```bash
cd android_test
./gradlew assembleDebug
```

Then install the debug APK:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

For deeper architecture notes, see
[`android_test/MULTI_MODEL_ARCHITECTURE.md`](android_test/MULTI_MODEL_ARCHITECTURE.md).

## Training Pipeline

The training pipeline prepares road-damage datasets, validates annotations,
harmonizes classes, detects duplicates, creates leakage-aware train/val/test
splits, and supports YOLO training/evaluation.

Typical setup:

```bash
cd model_training
pip install -r requirements.txt
```

Pipeline scripts are in `model_training/scripts/`:

```bash
python scripts/inspect_datasets.py
python scripts/validate_annotations.py
python scripts/convert_annotations.py
python scripts/harmonize_classes.py
python scripts/detect_duplicates.py
python scripts/split_dataset.py
python scripts/visualize_annotations.py
python scripts/train.py --model yolo11n.pt --epochs 100 --batch 16 --imgsz 640 --name road_damage_v1
python scripts/evaluate.py --model runs/train/road_damage_v1/weights/best.pt --split test
python scripts/predict.py --source <image_or_video> --model runs/train/road_damage_v1/weights/best.pt
```

The shipped Android crack model uses the class contract in
`model_training/configs/model_classes.yaml`. The local training dataset config is
`model_training/configs/data.yaml`; keep these taxonomies separate.

## Testing

Fusion rule tests:

```bash
python android_test/tools/test_fusion_rules.py
```

Offline fusion evaluation:

```bash
python android_test/tools/evaluate_fusion.py --limit 40
```

Model contract verification:

```bash
cd model_training
python scripts/verify_model_contract.py
```

## Important Notes

- `model_training/data/` is ignored and not pushed to GitHub.
- Android build outputs, APKs, Gradle caches, and ML run outputs are ignored.
- Large model files are committed because the app needs the ONNX/checkpoint
  artifacts for reproducibility.
- The pothole specialist is close-range focused, so vehicle-mounted dashcam
  pothole performance still needs more field validation.

