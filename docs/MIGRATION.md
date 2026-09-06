# Migration: three branches → one system

## Branch audit (performed before any edit)

The three branches sit on **two unrelated root commits**:

```
0e56f63 (root)  Initial UrbanSense project upload
  └─ 2addd18    Exclude local editor settings          <- akshveer TIP

031b067 (root)  Add README
  └─ 184ea28    Add project files                      <- arnav TIP
       └─ 3cddfc4  Merge PR #1                         <- main TIP
```

`main` and `akshveer` share **no history**. `git diff origin/main origin/arnav` is empty —
**main and arnav are content-identical** (89 files); arnav is fully contained in main.

| Branch | Content | Verdict |
|---|---|---|
| `main` | YOLO Android app + model training | Base for everything |
| `arnav` | identical to main | No unique content to salvage |
| `akshveer` | UrbanSenseAI sensor app + FastAPI + dashboard | Source of the sensor subsystem |

### Which branch supplied which subsystem

| Subsystem | Taken from |
|---|---|
| YOLO inference, ONNX Runtime, model contracts | `main` / `arnav` |
| CameraX preview + ImageAnalysis | `main` / `arnav` |
| Bounding-box coordinate mapping (`OverlayView`) | `main` / `arnav` |
| Model-vs-model fusion | `main` / `arnav` |
| Accelerometer, gyroscope, GPS | `akshveer` |
| Shock/event detection | `akshveer` |
| WebSocket client, session, bounded queue | `akshveer` |
| FastAPI receiver, storage | `akshveer` |
| Dashboard | `akshveer` |
| Visual↔physical corroboration | **new** |
| Reverse geocoding | **new** |

## What was preserved byte-for-byte

`ShockDetector.kt` is **identical below its package line** — verified by diff. Its 16 tests
came across unchanged and still pass.

`Core.kt` moved package and received four surgical edits (below). Its sensor sampling, GPS
provider strategy, WebSocket reconnect/backoff, bounded queue and session logic are untouched.

The YOLO side — `YoloOnnxDetector`, `InferenceScheduler`, `DetectionFusionEngine`,
`OverlayView`, `ModelRegistry`, `DetectionConfig`, `Detection` — was **not modified at all**.
Both `.onnx` assets are byte-identical and still ship in the APK (verified: 36.6 MB + 44.8 MB
present in `app-debug.apk`).

## Changes to migrated code

### `sensors/Core.kt`
1. `package com.urbansenseai` → `com.roadai.androidtest.sensors`
2. **Removed `submitCamera()`** — Requirement #6, no second camera pipeline
3. Removed `captureInFlight` gating and the "no camera bound" path from `inspectForShock`
4. Added `lastShock` (published for correlation) and `submitRoadEvent(...)`
5. `AppState.cameraCount` → `roadEventCount`

The generic binary-frame branch in `transmit()` is retained. Nothing produces binary frames
now, but it is transport code rather than camera code, and removing it would be surgery for
no benefit.

### `sensors/CollectionService.kt`
1. Package line
2. `import com.roadai.androidtest.MainActivity` (the notification's content intent)
3. **`createChannel()` now returns early below API 26** — the base app supports minSdk 24 and
   UrbanSenseAI assumed 28. Keeping minSdk 24 preserves the base app's device reach.

### Not migrated
`com.urbansenseai.MainActivity` — the standalone sensor UI. Requirement #1/#5: there is one
app and the camera is its main screen. Its readouts were re-expressed in the drawer; its
CameraX capture code was dropped.

## Dashboard

Only the **camera display** was disabled, and it was commented out rather than deleted:
the shock-photo gallery and the latest-frame card. Everything else is untouched —
accelerometer/gyroscope charts, GPS card, packet counters, live rate, session handling,
packet log, JSON download, storage, and every API endpoint.

The server retains full image capability (`/api/latest-frame.jpg`,
`/api/sessions/{id}/images/…`, `/api/events`, thumbnailing). Re-enabling the view is
uncommenting three lines.

## Verification

| Check | Result |
|---|---|
| `:app:assembleDebug` | **SUCCESS** — 153 MB APK, both ONNX assets present |
| Android unit tests | **46 / 46** (16 shock + 12 endpoint + 18 fusion) |
| Server road-event suite | **25 / 25** |
| Pre-existing server suites re-run against the consolidated server | e2e 24/24, dashboard 22/22, rejoin 7/7, events 27/27, monitor pass |
| Geocoding cache | 4 lookups → 2 Google calls |

The one pre-existing assertion that changed was `events_test`'s "dashboard page renders the
gallery" — the gallery is now deliberately disabled, so the assertion was updated to pin the
intended new behaviour rather than the old one.
