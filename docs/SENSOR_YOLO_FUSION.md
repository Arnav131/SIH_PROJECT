# YOLO + IMU/Gyro corroboration

## The rule

**Visual confidence and shock score are never combined arithmetically.**

`visual_confidence` is exactly what the YOLO model produced. `shock_score` is exactly what
`ShockDetector` produced. There is no `0.91 + 0.8`, no `(0.91 + 0.8) / 2`, no weighted blend.
They are different quantities, on different scales, measuring different things — one is a
trained classifier's output, the other is a multiple of a rolling accelerometer baseline. A
blended number would be arithmetic with nothing behind it.

What the two evidence sources produce together is a **status**, not a score.

This mirrors the rule `DetectionFusionEngine` already enforces between the two YOLO models.
The sensor corroboration deliberately lives in a **separate class**
(`fusion/RoadEventConfirmation.kt`) because it answers a different question:

| Class | Question |
|---|---|
| `DetectionFusionEngine` | Are these two boxes the same physical object? |
| `RoadEventConfirmation` | Did the road agree with what the camera saw? |

## Event states

| Status | Meaning |
|---|---|
| `VISUAL_ONLY` | A model saw it; no correlated physical disturbance. **A normal, valid outcome for a crack.** |
| `SUPPORTED` | A model saw it AND a jolt landed in the window, but not hard enough to confirm. |
| `CONFIRMED` | A model saw it AND a strong jolt landed in the window. |

`CONFIRMED` is never reachable from YOLO confidence alone, however high. It requires
independent physical evidence.

Sensor support is reported on its own axis: `NONE` / `WEAK` / `STRONG`.
`STRONG` means `shock_score >= 8.0` **or** the shock was flagged `sustained`.

## Cracks are never penalised for silence

A hairline crack produces no measurable jolt. A crack detection therefore stands on its own
visual evidence and **can never be rejected or downgraded** for lack of a shock.

- `CRACK` + no shock → `VISUAL_ONLY` (emitted normally)
- `CRACK` + coincident shock → `SUPPORTED` (context only)
- `CRACK` → **never** `CONFIRMED`

Only `POTHOLE` and `OTHER` can reach `CONFIRMED`, because those are the defects a vehicle is
actually supposed to feel. Pinned by `crackWithNoShockIsNeverRejected` and
`crackNeverReachesConfirmedEvenWithAStrongShock`.

## Temporal correlation

### Why we do NOT use the detection's own timestamp

`android_test/test_results.md` records measured on-device inference of **~3100 ms (pothole)
and ~3550 ms (crack)**, roughly **0.25 FPS**. A `Detection.timestampMs` is stamped when the
object is constructed — *after* inference — so it trails the physical moment by seconds.
Correlating against it would be meaningless.

Instead `MainActivity.analyze()` stamps `frameCapturedAtMs` the instant CameraX hands the
frame over, **before any inference runs**, and everything downstream anchors on that.
Pipeline latency then cancels out entirely and stays correct if the models get faster.

### Why the window is asymmetric

A forward-facing camera sees a pothole **before** the wheel reaches it. At ~30 km/h (8.3 m/s)
a defect spotted 10 m ahead is struck ~1.2 s later; at 20 m ahead, ~2.4 s later. The jolt is
therefore expected to arrive *after* the sighting.

| Parameter | Value | Reasoning |
|---|---|---|
| `windowBeforeMs` | **1500 ms** | Rough stretch already underway when the frame was taken. Covers `ShockDetector`'s own 1.2 s burst window plus 50 ms sensor sampling. |
| `windowAfterMs` | **3000 ms** | Vehicle travelling to a defect seen up to ~25 m ahead at urban speed. |
| `strongShockScore` | **8.0** | `ShockDetector` MEDIUM already fires at 4x baseline; 8x is comfortably above noise it rejected. |
| `minVisualConfidence` | **0.45** | Higher than the overlay's 0.25 display threshold: an event is transmitted and geocoded, so it should be worth the trip. |
| `perTypeCooldownMs` | **8000 ms** | One pothole stays in frame for many cycles; without this it would emit per cycle and geocode every time. |

`correlation_delta_ms` is transmitted with every correlated event (`shock.at - frameCapturedAtMs`,
positive = jolt after sighting), so these values can be re-tuned from real road data rather
than argued about. **They are starting values, not optima** — re-measure for your mounting
and speeds.

### Spam control

- One jolt cannot confirm two sightings (`lastConsumedShockAt`).
- Cooldown is **per detection type**, so a pothole and a crack in the same frame both emit.
- Below `minVisualConfidence`, nothing is emitted at all.

## Location accuracy caveat

Because a cycle takes ~3.5 s, "where the phone is now" is ~29 m from where the frame was taken
at 8.3 m/s. The GPS fix is therefore **snapshotted at frame-capture time** in
`MainActivity.analyze()` and passed through to the event, not read at emission time.

## Tests

`app/src/test/java/com/roadai/androidtest/fusion/RoadEventConfirmationTest.kt` — 18 tests
covering every state transition, both window edges, the asymmetry, the no-blending rule,
cooldown behaviour, and the crack exemption.
