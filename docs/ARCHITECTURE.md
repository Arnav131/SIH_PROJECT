# Consolidated architecture

## One app, one camera, one receiver

```
                        ONE ANDROID APP  (com.roadai.androidtest)
                                    |
              +---------------------+---------------------+
              |                                           |
              v                                           v
      PRIMARY CAMERA UI                          SENSOR SIDE PANEL
      (always on screen)                         (right drawer, toggled)
              |                                           |
              v                                           +-- Accelerometer
      CameraX -> ImageAnalysis                            +-- Gyroscope
              |                                           +-- GPS
              v                                           +-- ShockDetector
      InferenceScheduler                                  +-- Connection / session
      (pothole || crack, ONNX Runtime)                    +-- Transmission stats
              |                                           |
              v                                           |
      DetectionFusionEngine                               |
      (model vs model: same object?)                      |
              |                                           |
              +---------------+---------------------------+
                              |
                              v
                    RoadEventConfirmation
                    (visual vs physical: did the road agree?)
                              |
                              v
                         ROAD EVENT
                    + GPS snapshotted at frame capture
                              |
                              v
                   WebSocket  /ws/data
                              |
                              v
                    FastAPI receiver (server/)
                    + reverse geocoding -> address
                              |
                              v
                    Dashboard  /ws/dashboard
                    (sensor + GPS + system monitoring)
```

There is exactly **one** Android application and exactly **one** camera pipeline.

## Repository layout

```
SIH_PROJECT/
├── android_test/              THE single Android app (see naming note below)
│   └── app/src/main/
│       ├── assets/            pothole_yolov12s.onnx, crack_rdd2022.onnx (unchanged)
│       ├── java/com/roadai/androidtest/
│       │   ├── MainActivity.kt        camera + drawer + event emission
│       │   ├── OverlayView.kt         bounding boxes            (unchanged)
│       │   ├── config/                DetectionConfig           (unchanged)
│       │   ├── detection/             Detection, PerceptionResult (unchanged)
│       │   ├── inference/             YoloOnnxDetector, InferenceScheduler (unchanged)
│       │   ├── models/                ModelRegistry, verified contracts (unchanged)
│       │   ├── fusion/
│       │   │   ├── DetectionFusionEngine.kt   model-vs-model    (unchanged)
│       │   │   └── RoadEventConfirmation.kt   visual-vs-physical  [NEW]
│       │   └── sensors/               migrated from UrbanSenseAI
│       │       ├── Core.kt            UrbanEngine: IMU, GPS, WebSocket, session, queue
│       │       ├── ShockDetector.kt   jolt detection (byte-identical logic)
│       │       └── CollectionService.kt foreground service
│       └── res/layout/activity_main.xml  DrawerLayout wrapper   [MODIFIED]
├── server/                    FastAPI receiver + dashboard + reverse geocoding
│   ├── server.py              WebSocket, storage, road events, geocoding
│   ├── dashboard.html         live dashboard (camera display disabled)
│   ├── monitor.py             terminal packet view
│   ├── export_json.py         session -> single JSON
│   └── tests/test_road_events.py
├── model_training/            untouched
└── docs/
```

### Two naming decisions, made deliberately

**`android_test/` keeps its name.** It is now the single production app, not a test harness.
Renaming it to `android/` would touch every path in Arnav's `README.md`,
`MULTI_MODEL_ARCHITECTURE.md` and `test_results.md` for zero functional gain, and would bury
the real change in a rename diff. The name is documented here instead; renaming is a safe
follow-up commit whenever the team wants it.

**`dashboard.html` stays inside `server/`.** `server.py` serves it with
`(ROOT / "dashboard.html").read_text()`. Moving it to a top-level `dashboard/` folder would
break the working receiver to satisfy a folder diagram.

## Threading

| Work | Thread |
|---|---|
| Camera preview | CameraX render surface |
| YOLO inference | `analysisExecutor` + one worker per specialist |
| Sensor sampling | `UrbanSensorThread` (HandlerThread), 20 Hz |
| WebSocket I/O | OkHttp dispatcher |
| UI rendering | Main, throttled to 200 ms per sensor redraw |

Backpressure is unchanged from the base app: `KEEP_ONLY_LATEST` + `busy` flag +
`targetInferenceFps` cap mean frames are **dropped, never queued**. Sensor sampling is
independent of UI redraws; the drawer collects a conflated `StateFlow` with a 200 ms pause,
so 20 Hz sensor updates cannot flood the main thread.

## Wire protocol

Existing packet types are **unchanged**: `session_start`, `sensor`, `gps`, `camera`.
One additive type, `road_event`, carries the fused decision. Existing consumers
(`monitor.py`, `export_json.py`, the session JSON API, stored CSVs) are unaffected.
