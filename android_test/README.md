# Road Anomaly Detection — Android Test App

Live camera road-damage detection using **two independent specialist models**
fused into a single perception layer.

```
CameraX → [ pothole specialist ‖ crack specialist ] → fusion → overlay
```

Neither model's weights were merged, retrained, or modified.

---

## Models in use

| Role | Asset | Architecture | Input | Classes |
|---|---|---|---|---|
| Pothole specialist | `pothole_yolov12s.onnx` (35 MB) | YOLOv12s | `[1,3,640,640]` NCHW f32 | `Pothole` |
| Crack specialist | `crack_rdd2022.onnx` (43 MB) | YOLOv8s (RDD2022) | `[1,3,640,640]` NCHW f32 | `Longitudinal Crack`, `Transverse Crack`, `Alligator Crack`, `Pothole`*, `Other` |

\* The crack model's own `Pothole` class is deliberately dropped — the pothole
specialist owns potholes. One line to re-enable; see
[MULTI_MODEL_ARCHITECTURE.md](MULTI_MODEL_ARCHITECTURE.md) §5.

Both are ONNX running on **ONNX Runtime for Android 1.19.2**. Full verified
contracts — output shapes, box space, activation, NMS — are in
[MULTI_MODEL_ARCHITECTURE.md](MULTI_MODEL_ARCHITECTURE.md).

The master copies in `model_training/models/` are untouched; the assets are
copies.

---

## Preprocessing / postprocessing

**Preprocessing** (verified independently for each model, identical in both):
letterbox to 640×640 with pad value **114** → RGB → `/255` → HWC→CHW float32.
No ImageNet mean/std.

**Postprocessing:** transpose `[1, 4+nc, 8400]` → threshold → per-class greedy
NMS at IoU 0.7 → undo letterbox → clamp to bitmap. **NMS is not in either
graph**, so it is done in app code.

---

## Detection thresholds

In `config/DetectionConfig.kt`, each specialist tuned independently:

```kotlin
potholeThreshold      = 0.25f   // initial value, NOT optimal
crackThreshold        = 0.25f   // initial value, NOT optimal
nmsIouThreshold       = 0.70f   // within a model
duplicateIouThreshold = 0.80f   // cross-model, same type only
targetInferenceFps    = 8
executionMode         = PARALLEL
includeOtherType      = true    // set false for a cleaner overlay
```

The in-app slider moves both thresholds together for quick field tuning.

---

## Fusion rules

1. **Confidences are never combined.** `Pothole 0.57` and `Alligator Crack 0.56`
   stay separate — never summed, never averaged. The two models' scores are not
   on a comparable scale.
2. **A pothole and a crack are never duplicates.** Duplicate suppression applies
   only *within* one detection type, at IoU 0.80. Cracking around a pothole rim
   is two real defects and both are kept.

---

## Architecture

```
config/    DetectionConfig.kt        thresholds, FPS cap, execution mode
detection/ Detection.kt              unified format + DetectionType + results
models/    ModelSpec.kt              per-model contract + class mapping + registry
inference/ YoloOnnxDetector.kt       generic ONNX runner (one per specialist)
           InferenceScheduler.kt     parallel/sequential, backpressure, fusion call
fusion/    DetectionFusionEngine.kt  merge + duplicate handling
           OverlayView.kt            unified overlay
           MainActivity.kt           CameraX wiring
```

Adding a third specialist (manhole, road sign, …) requires **no change** to the
camera pipeline — see [MULTI_MODEL_ARCHITECTURE.md](MULTI_MODEL_ARCHITECTURE.md) §11.

---

## Build

```bash
cd android_test
./gradlew assembleDebug
```

> **Note:** this project has `gradle/wrapper/gradle-wrapper.properties` but **no
> `gradlew` script or `gradle-wrapper.jar`**. Easiest path is to open
> `android_test/` in Android Studio, which supplies its own Gradle and will
> generate the wrapper. Requires a JDK Gradle 8.9 supports (17–21); the JDKs
> installed here are 24/25, which Gradle 8.9 rejects — Android Studio's bundled
> JDK handles this for you.

Install and run:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Grant the camera permission when prompted.

---

## Running it

The status panel shows, live:

```
ROAD ANOMALIES  pothole=1  crack=2
Latency 812ms  [pothole=780ms  crack=735ms]  PARALLEL
Camera 29.7 FPS  processed 1.2 FPS
Top: Pothole 0.57, Alligator Crack 0.56
```

Red boxes = potholes, cyan = cracks, grey = uncategorized damage. Model
filenames are never shown in the UI — it presents as one system.

---

## Testing

### Fusion rules (fast, no models loaded)

```bash
python android_test/tools/test_fusion_rules.py
```

Encodes the two fusion rules as assertions — including "an overlapping pothole
and crack are both kept" and "confidences are never combined". Currently 7/7
pass. Update these if you change fusion behaviour in Kotlin.

### Offline (laptop, against labelled ground truth)

```bash
python android_test/tools/evaluate_fusion.py --limit 40
```

This is a reference implementation of the exact on-phone pipeline — same
letterbox, decode, adapters and fusion rules. **If you change fusion logic in
Kotlin, change it here too.** Results: [test_results.md](test_results.md).

### Laptop-screen camera test

Display a road-damage image full-screen on the laptop, point the phone camera at
it, and check the overlay. Use a **close-up** pothole photo — see the limitation
below. This is a pipeline sanity check only, **not** an accuracy measurement:
screen glare, moiré and rescaling all distort the input.

---

## Known limitations

1. **The pothole specialist is a close-range model.** Trained on close-up
   pothole photos, it detected **0%** of ground-truth potholes in distant
   dashcam-style frames, versus **85%** on close-ups. Since a windscreen-mounted
   phone produces exactly the dashcam view, **this pairing is not yet fit for
   vehicle-mounted pothole detection.** The crack specialist (RDD2022) handles
   that view far better.
2. **Not yet run on a phone.** Build, launch, camera preview, overlay alignment
   and on-device performance are **unverified** from this environment.
3. Laptop-CPU inference is ~730–790 ms per model. Phone figures are unknown.
4. `OTHER` (RDD2022 catch-all) fires often and generically; set
   `includeOtherType = false` to hide it.
5. ~78 MB of model assets makes the APK large.
6. The two models' confidences are not calibrated against each other.
7. No shadow / night / rain testing.

---

## Scope

Camera + two models + fusion + overlay. Deliberately **no** GPS, IMU, LiDAR,
networking, backend, database or sensor fusion — those come later.
