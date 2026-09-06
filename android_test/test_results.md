# Test Results — Two-Specialist Road Anomaly Detection

**Date:** 2026-09-06
**Harness:** `android_test/tools/evaluate_fusion.py` (laptop reference
implementation of the on-phone pipeline — same letterbox, same decode, same
adapters, same fusion rules)
**Machine:** Windows 11, 12th Gen Intel Core i5-1235U, CPU only (no CUDA)
**Runtime:** onnxruntime 1.23.2, CPUExecutionProvider

> **Read this first.** Sections 1–4 measure the two ONNX assets on **still
> images from your own labelled test split**; section 5a reports what happened
> on a **real phone**. Together they show the *integration* works. Neither is a
> real-world accuracy benchmark — no model was evaluated against matched labels
> in its own taxonomy, and nothing was tested on an actual road.

---

## 1. What ground truth was used

`model_training/data/processed/road_damage_combined/` test split, 988 labelled
images. Its local 6-class taxonomy (`pothole=0, crack=1, …`) is used **only** to
choose and describe images — neither model was trained on that class order, so
these are not per-class accuracy scores against matched labels.

| Bucket | Images available | Character |
|---|---|---|
| Ground-truth **pothole + crack** in the same frame | 51 | `rd_vlcsnap-*`, 640×360 dashcam video frames |
| Ground-truth **pothole** only | 398 | `pot_*`, 640×640 close-up pothole photos |
| Ground-truth **crack** only | 139 | mixed |

Thresholds for every run below: **pothole 0.25, crack 0.25**, NMS IoU 0.7,
cross-model duplicate IoU 0.80.

---

## 2. Headline result: the two models have *different domains*

This is the most important finding of the whole exercise.

| Bucket (n=40 each) | Fusion reported POTHOLE | Fusion reported CRACK | Reported BOTH |
|---|---|---|---|
| pothole only (close-up photos) | **85%** | 5% | 5% |
| crack only | 2% | **57%** | 2% |
| pothole + crack (dashcam frames) | **0%** | 40% | **0%** |

The pothole specialist detects 85% of close-range pothole photos and **0%** of
distant dashcam potholes. Confidence on the dashcam frames collapses to
0.001–0.10, well under any usable threshold:

```
BOTH bucket (dashcam)        POTHOLE-ONLY bucket (close-up)
rd_vlcsnap-00033  0.005      pot_02_...   0.421
rd_vlcsnap-00040  0.013      pot_08_...   0.498
rd_vlcsnap-00049  0.102      pot_110_...  0.676
rd_vlcsnap-00085  0.001      pot_113_...  0.616
```

**Cause:** the pothole model was trained on the Roboflow `Pothole-1` dataset —
close-up photographs where the pothole fills much of the frame. In a dashcam
frame the pothole is a small patch of texture in the mid-distance. The crack
model, trained on RDD2022 (genuinely dashcam imagery), handles those frames far
better (40%).

**Consequence for the product:** a phone mounted on a windscreen sees the
dashcam view, which is the pothole specialist's *weak* domain. Integration is
complete and correct, but this model pairing is not yet fit for vehicle-mounted
pothole detection.

---

## 3. Fusion behaviour — the critical test case

**Test: one frame containing both a pothole and a crack.**

Searching 180 frames, **18 frames had both specialists fire simultaneously**.
Fusion returned both, with confidences untouched. Examples:

| Frame | Fused output |
|---|---|
| `pot_121_….ed09533a` | `Pothole 0.57` + `Alligator Crack 0.56` + `Alligator Crack 0.27` |
| `pot_121_….9cd31094` | `Alligator Crack 0.60` + `Pothole 0.47` + `Alligator Crack 0.27` |
| `pot_148_….f07a27a6` | `Pothole 0.41` + `Alligator Crack 0.26` |

Annotated proof images: `test_results/fusion_case1.jpg`, `fusion_case2.jpg`,
`fusion_case3.jpg` (red = POTHOLE, cyan = CRACK).

`fusion_case1.jpg` is the clearest: the pothole specialist boxes the pothole
itself while the crack specialist boxes the alligator cracking spreading around
it. The boxes overlap substantially and were **correctly not merged**, because
they are different `DetectionType`s.

Verified in that run:
- ✅ Both detections survive fusion
- ✅ `0.57` and `0.56` are reported separately — never summed, never averaged
- ✅ Overlapping pothole/crack boxes are not treated as duplicates
- ✅ Crack subtype (`Alligator`) preserved, not flattened to a generic "crack"

---

## 4. Test cases from the plan

| # | Case | Result |
|---|---|---|
| 1 | Normal road | ✅ Non-pothole images: 2% pothole false-alarm rate at 0.25 |
| 2 | Clear pothole | ✅ 85% of close-up pothole frames fire; median max conf 0.471 |
| 3 | Multiple potholes | ✅ Mean 2.50 fused detections/frame in the pothole bucket; up to 12 boxes on `pot_17_*` |
| 4 | Clear crack | ✅ 57% of ground-truth crack frames fire |
| 5 | Multiple cracks | ✅ Multiple `Alligator Crack` boxes returned per frame |
| 6 | **Pothole + crack in one frame** | ✅ **See §3 — both returned, both preserved** |
| 7 | Different crack types | ⚠️ Longitudinal / Transverse / Alligator all mapped and renderable, but `Alligator Crack` dominated real output. Longitudinal/Transverse were rarely produced on these images |
| 8 | Noisy / low-quality | ⚠️ Partially covered — the 640×360 dashcam frames are the low-quality end, and the pothole model failed on them (§2) |
| 9 | Shadows | ❌ **Not tested** — no shadow-labelled subset exists |
| 10 | Road markings as false positives | ⚠️ Anecdotal only: on `fusion_case1.jpg` the zebra crossing did **not** trigger a false pothole |
| — | Laptop-screen camera test (Step 16) | ❌ **Not performed** — needs someone to physically aim the phone; see §5a |

---

## 5a. ON-DEVICE results (real phone)

**Device:** Xiaomi `24048RN6CI`, SoC `T603` (budget octa-core), Android 15
**Build:** `assembleDebug` → `app-debug.apk`, 159 MB
**Installed and launched via adb; camera permission granted.**

### Integration — all verified on hardware

| Check | Result |
|---|---|
| App builds | ✅ `BUILD SUCCESSFUL`, 39 tasks |
| Both assets packaged | ✅ `assets/pothole_yolov12s.onnx` (36.6 MB), `assets/crack_rdd2022.onnx` (44.8 MB) |
| App installs & launches | ✅ |
| Camera permission | ✅ granted, preview live |
| **Both models load** | ✅ see logcat below |
| Contracts read from graph | ✅ both reported `640x640`, `anchors=8400`, correct `nc` |
| No load crash | ✅ |
| No runtime crash | ✅ no `FATAL EXCEPTION`, no `OutOfMemory` |
| Inference runs each frame | ✅ continuous cycles logged |
| UI responsive | ✅ preview and slider remain interactive |

```
I YoloOnnxDetector: Loaded pothole_yolov12s.onnx | in=images 640x640 | nc=1 anchors=8400 | classes=Pothole
I YoloOnnxDetector: Loaded crack_rdd2022.onnx    | in=images 640x640 | nc=5 anchors=8400 |
                    classes=Longitudinal Crack, Transverse Crack, Alligator Crack, Pothole, Other
```

Note both models self-reported `nc` and anchor counts matching their declared
specs — the mismatch warning never fired, confirming the assets are what the
code expects.

### Measured latency — **this is the main problem**

Per-cycle wall clock in `MODE_PARALLEL`, steady state after warm-up:

```
5808ms  4672ms  4404ms  3914ms  4208ms
```

From the on-screen panel: `Latency 3579ms [pothole=3114ms crack=3558ms] PARALLEL`

| Metric | Value |
|---|---|
| Pothole model, on device | ~3100 ms |
| Crack model, on device | ~3550 ms |
| **Parallel cycle total** | **~3.6–4.7 s** |
| **Effective processed rate** | **~0.25 FPS** |
| TOTAL PSS | ~350 MB (348 MB → 354 MB across runs, stable) |

Parallel mode is working as designed — the cycle (~3.58 s) tracks the *slower*
specialist (3.56 s) rather than the sum (6.7 s), so it is saving ~45% versus
sequential. The problem is that both models are simply too heavy for this SoC:
two 640×640 YOLO-small networks on CPU with no NNAPI/GPU delegate.

**~0.25 FPS is far below the 5–15 FPS target** and is not usable for
vehicle-mounted detection at road speed. The camera preview itself stays smooth
because it renders on its own Surface — the backpressure design works — but the
overlay updates only every ~4 seconds.

### Negative-control behaviour on device

Pointed at an indoor floor scene (no road, no damage), the fused output was
`pothole=0 crack=0`, `Top: none` — no false positives. See
`test_results/device_running.png`.

### Not verified on device

- ❌ **Positive pothole detection on device** — requires physically aiming the
  camera at a pothole or a pothole image; cannot be done from this environment.
  Use the laptop-screen procedure in README.
- ❌ SEQUENTIAL mode not measured on device (requires a config change + rebuild)
- ❌ Bounding-box alignment not visually confirmed against a real detection

---

## 5b. Performance (laptop CPU — **not** phone figures)

Measured over 120 frames:

| Model | Mean inference | Median |
|---|---|---|
| Pothole (YOLOv12s @ 640) | 784.6 ms | 731.8 ms |
| Crack (YOLOv8s @ 640) | 727.7 ms | 665.6 ms |
| **Sequential total** | **~1512 ms** | → ~0.7 FPS |

These come from a laptop CPU with no NNAPI/GPU delegate and a Python harness.
A phone with ORT's Android build will differ substantially — possibly better on
a modern SoC, possibly worse on a budget one. **No on-device measurement has
been taken.**

In `MODE_PARALLEL` the wall-clock cycle is bounded by the *slower* specialist
rather than their sum, so parallel mode should land nearer ~800 ms/cycle on
similar hardware. The app reports measured camera FPS, processed FPS and
per-model latency live in its status panel, which is where real device numbers
should be read from.

Memory was not profiled.

---

## 6. Not yet verified

Being explicit about what has **not** been demonstrated:

- ❌ **On-device positive detection** — the app has never been shown a real
  pothole. Only negative behaviour (clean indoor scene → no detections) was
  observed on hardware.
- ❌ **Overlay box alignment** against a real detection — the mapping is
  implemented and reasoned about, but with no on-device detection to look at it
  has not been visually confirmed. Rotation handling likewise.
- ❌ Laptop-screen camera sanity test not performed
- ❌ SEQUENTIAL mode never measured on device
- ❌ No shadow / night / rain testing
- ❌ No testing on an actual road

---

## 6a. How to make it faster (untested suggestions)

The ~0.25 FPS on-device result is the biggest practical blocker. Options, roughly
by expected payoff per unit of effort — **none of these have been tried yet**:

1. **Drop the input size.** Both models run at 640×640. The pothole model's
   sibling export at 320×320 is ~4× less compute. Accuracy will drop; the
   trade needs measuring.
2. **Enable NNAPI or XNNPACK** in the ORT session options. Currently plain CPU
   with no delegate. On this SoC NNAPI support is uncertain but worth testing.
3. **Quantize to INT8.** Typically 2–4× faster on ARM CPU, some accuracy loss.
4. **Run the specialists on alternating frames** rather than both every cycle —
   halves per-cycle latency at the cost of temporal alignment.
5. **Drop to one specialist** for the vehicle-mounted case. Given §2, the crack
   model is the one that works on dashcam views.

---

## 7. Conclusions

**Integration: verified working, on real hardware.** Both specialists load on
the phone with the contracts the code expects, run on the same frame, normalize
into one format, and fuse correctly with confidences preserved and semantics
respected. No crashes, no memory growth, camera stays responsive.

**Accuracy and performance: not yet fit for the intended use.** Two separate
problems, both independent of the integration being correct:

1. **Domain mismatch (§2).** The pothole specialist is a close-range model being
   asked to do a dashcam job; on dashcam frames it detects nothing.
2. **Speed (§5a).** ~4 s per cycle / ~0.25 FPS on the test device, against a
   5–15 FPS target.

**Suggested next steps, in order:**
1. Do the laptop-screen test with a **close-up** pothole image to confirm the
   on-device positive-detection path end to end (the one integration check still
   unverified).
2. Address speed — see §6a. Smaller input size is the cheapest experiment.
3. Address the domain gap: either accept close-range mounting, or replace the
   pothole specialist with one trained on dashcam-perspective data. The
   architecture supports swapping it without touching the pipeline.
