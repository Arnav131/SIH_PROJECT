# Model Evaluation Report

Generated: 2026-09-05 12:44:34

## Model
- Path: `runs/train/road_damage_v1/weights/best.pt`

## Overall Metrics

| Metric | Value |
|---|---|
| metrics/precision(B) | 0.9272 |
| metrics/recall(B) | 0.0633 |
| metrics/mAP50(B) | 0.0668 |
| metrics/mAP50-95(B) | 0.0261 |
| fitness | 0.0261 |

## Per-Class Metrics

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| pothole | 0.5629 | 0.3796 | 0.4003 | 0.1567 |
| crack | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| manhole | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| road_damage | 1.0000 | 0.0000 | 0.0001 | 0.0000 |
| speed_bump | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| unsurfaced_road | 1.0000 | 0.0000 | 0.0001 | 0.0000 |

## Recommended Cross-Dataset Experiments

| Experiment | Training Data | Test Data |
|---|---|---|
| A (standard) | Combined | Combined hold-out |
| B | RAD + Pothole | Road Damage |
| C | RAD + Road Damage | Pothole |
| D | Road Damage + Pothole | RAD |
