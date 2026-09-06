# Experimental model — NOT deployed

**Status: parked, not wired into the Android app or model_training pipeline.**

## What this is

`iris_v3_best.pt` was dropped into the project root on 2026-09-06 by the user
("another model best.pt"). It is **not** the production model.

- Checkpoint metadata: `train_args.name = "iris-v3"`, base `yolov8n.pt`,
  trained on `data=/tmp/iris-train/merged/data.yaml`, `imgsz=320`, 80 epochs.
- Architecture: YOLOv8n scale (depth 0.33 / width 0.25), 16 classes, 3.0M params.
- Classes (index: name):
  ```
  0 POTHOLE          8  PEDESTRIAN_CROSSING
  1 CRACK            9  ROAD_MARKING
  2 PATCH           10  MANHOLE
  3 UNPAVED_ROAD    11  DRAINAGE
  4 SPEED_BUMP      12  VEHICLE
  5 ROAD_SIGN       13  MOTORCYCLE
  6 TRAFFIC_LIGHT   14  CONSTRUCTION
  7 GUARDRAIL       15  NUMBER_PLATE
  ```

`iris_v3_best.onnx` was exported from it here with:
```
YOLO('best.pt').export(format='onnx', imgsz=320, opset=12, simplify=True, dynamic=False, half=False)
```
Result: input `images` [1,3,320,320] float32 NCHW, output `output0` [1,20,2100]
(4 box terms + 16 sigmoid, independent class scores — same v8-style head/NMS-free
contract as the production model, just different imgsz/nc/anchor count).

## Why it's parked, not deployed

Ran this ONNX export against `android_test/test_images/*.jpg` (the images the
production model is validated against). Result:

- **POTHOLE confidence was exactly 0.0 on every test image**, including the
  five labeled pothole shots. Same for CRACK, MANHOLE, SPEED_BUMP, and most
  other road-damage classes.
- Instead it fired confidently on **MOTORCYCLE / VEHICLE / NUMBER_PLATE** on
  almost every image — including `06_no_damage_negative_control.jpg`
  (MOTORCYCLE 0.79).
- Ruled out a preprocessing/export bug: raw tensor output matches Ultralytics'
  own `.predict()`, and synthetic gray/black/noise inputs behave sanely
  (near-zero confidence everywhere), so the network itself is intact — it's
  just not detecting road damage in these images.
- The `data=/tmp/iris-train/merged/data.yaml` training path and the strong
  vehicle/motorcycle/plate bias suggest this checkpoint was trained on a
  traffic-scene / vehicle-detection-heavy dataset, not close-up road-damage
  photos like the ones the Android app is tested against.

**Conclusion: this checkpoint is not currently usable as a road-damage
detector for this app.** It may be useful later for vehicle/plate detection,
or if retrained/fine-tuned further on road-damage-specific imagery, but it
should not replace `model_training/models/best.pt` / `best.onnx` without a
lot more validation.

## What was NOT touched

- `model_training/models/best.pt`, `model_training/models/best.onnx` — original production files, unmodified.
- `android_test/app/src/main/assets/best.onnx` — deployed Android model, unmodified.
- `android_test/app/src/main/java/.../YoloOnnxDetector.kt` — detector contract, unmodified.
