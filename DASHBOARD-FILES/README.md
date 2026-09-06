# DASHBOARD-FILES — UrbanSenseAI receiver and dashboard

The laptop side. Drop all of these into one folder and run `server.py` from it.

## Contents

| File | What it does |
|---|---|
| `server.py` | FastAPI receiver. Accepts the phone's WebSocket, writes CSV + JSONL + JPEG, serves the dashboard and the JSON API. Prints the LAN URL to type into the app at startup |
| `dashboard.html` | The live dashboard. Served at `/`, fed by `/ws/dashboard`. Must sit next to `server.py` |
| `monitor.py` | Terminal view of the live stream, one line per packet |
| `export_json.py` | Writes a captured session out as a single JSON file |
| `requirements.txt` | Python dependencies |
| `.env.example` | Copy to `.env` and change `DEVICE_TOKEN` before exposing the receiver |

## Run

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python server.py
```

The startup banner prints the address to enter in the app. Dashboard: `http://localhost:8000`.

Two things that bite here:

- **Restart the receiver after replacing `server.py`.** A running process keeps the old code,
  so new endpoints answer 404 and new CSV columns never appear.
- **A virtualenv is a different `python.exe`**, so a Windows Firewall allowance granted to your
  system Python does not carry over. If the phone cannot reach the port, open it once from an
  Administrator prompt:

```
netsh advfirewall firewall add rule name="UrbanSenseAI" dir=in action=allow protocol=TCP localport=8000
```

`DATA_DIR` moves the capture folder somewhere other than `./data` — useful for a second
receiver or a test run. `export_json.py` follows it.

## The dashboard

Pushed over a WebSocket, not polled. Accelerometer and gyroscope axes with rolling 300-sample
charts, GPS with a map link, packet counters and live rate in Hz, the newest camera frame, a
packet log, and the shock-event gallery.

## Shock events

A sharp, repeated jolt on the phone photographs itself. The frame arrives tagged with what set
it off and where, and lands in the **Shock events** gallery: newest first, with a `SHOCK` or
`SUSTAINED` badge, the jolt in m/s², how many times over its own baseline it was, the
coordinates with their accuracy, and a map link. Clicking a card opens the full-size capture.

The gallery survives both a browser reload and a receiver restart: a joining dashboard is
replayed the recent events in its greeting, and on startup the server rebuilds that list by
reading `packets.jsonl` off disk. Without that second part, restarting the receiver hid every
capture taken before it — the photographs were still there but nothing pointed at them.

Cards load a 480px thumbnail (`?thumb=1`), built once and cached in `images/.thumbs/`. Without
it a gallery of full-size captures asks the browser to decode dozens of large JPEGs at once and
the cards stay blank for seconds. Pillow does the resizing; if it is missing the endpoint
serves the original instead of failing.

Manual captures are stored and shown in the Camera card but stay out of the gallery — that
list is for what the phone decided on its own.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | The dashboard page |
| `GET /health` | `{"status":"ok"}` — the quickest reachability check from the phone's browser |
| `GET /api/status` | Connected devices, active sessions, packet counters, last packet |
| `GET /api/events` | Recent shock captures, newest last (last 50, restored from disk at startup) |
| `GET /api/sessions` | Every session on disk |
| `GET /api/sessions/{id}/json` | One session consolidated into a single JSON document |
| `GET /api/sessions/{id}/images/{file}` | One stored frame; `?thumb=1` for the 480px copy |
| `GET /api/sessions/{id}/download` | The session folder as a zip |
| `GET /api/latest-frame.jpg` | Newest camera frame |
| `WS /ws/data?token=…` | Where the phone sends. Token required |
| `WS /ws/dashboard` | Read-only live feed. No token from localhost, token required from anywhere else, so tunnelling the receiver does not expose the stream |

## Stored data

```
data/<session_id>/
  metadata.json        device model, Android version, sensors present
  packets.jsonl        every accepted frame verbatim, one JSON object per line
  accelerometer.csv    timestamp, sequence_number, x, y, z, accuracy
  gyroscope.csv        same shape
  gps.csv              timestamp, sequence_number, latitude, longitude, altitude, speed, bearing, accuracy
  camera.csv           timestamp, sequence_number, filename, width, height, trigger,
                       accel_magnitude, gyro_magnitude, shock_score,
                       latitude, longitude, gps_accuracy, gps_provider
  images/              shock_*.jpg for automatic, frame_*.jpg for manual
  images/.thumbs/      generated 480px copies
```

JSONL rather than one big array: a running session can be appended to and tailed, which an
array cannot.

```
python export_json.py --list
python export_json.py                      # newest session -> <session_id>.json
python export_json.py --session session_...
```

## Wire protocol

Every frame carries `device_id`, `session_id`, epoch-ms `timestamp`, `type`, and — except for
`session_start` — a rising `sequence_number`.

```json
{"device_id":"realme3pro_001","session_id":"session_20260906_105747",
 "timestamp":1788672475481,"type":"sensor","sensor":"accelerometer",
 "sequence_number":2,"data":{"x":-0.152,"y":-0.334,"z":9.957,"accuracy":3}}
```

A camera capture is a metadata frame followed immediately by the JPEG as a binary frame. An
automatic one carries the trigger and the fix:

```json
{"type":"camera","filename":"shock_1788672475481.jpg","width":1280,"height":960,
 "trigger":"shock","accel_magnitude":4.7,"gyro_magnitude":1.82,"shock_score":17.5,
 "latitude":30.7655565,"longitude":76.5748829,"gps_accuracy":8.5,"gps_provider":"gps"}
```

`trigger` is `shock`, `shock_sustained` or `manual`. All of it lands in `camera.csv`, in
`packets.jsonl` and in the consolidated session JSON.

## Fixes in this version

- The receive loop treated `websocket.disconnect` as neither text nor bytes and spun without
  awaiting real I/O, blocking the event loop. One phone dropping Wi-Fi wedged the receiver for
  every later request until restart.
- Packets were echoed back to the producing device, doubling its uplink.
- The dashboard polled `/api/status` once a second. It receives pushed frames now.
- A dashboard opening or reloading mid-session is sent the running session, counters, newest
  frame and event history in its greeting, so a refresh no longer blanks the page.
- Rate and freshness moved off `requestAnimationFrame`: a background browser tab stops
  animating, and the rate read 0.0 Hz while packets were still arriving.
- Axis readouts could split mid-number (`-4.92` / `7`) on a narrow window.
- `/api/sessions/{id}/images/{file}` strips any path traversal from the filename.
- `export_json.py` had its own hardcoded data path and ignored `DATA_DIR`, so it looked in the
  wrong folder whenever the receiver was pointed elsewhere.
- The charts scaled purely to the values in view. A resting gyroscope idles around
  0.0005 rad/s, so normalising to that stretched pure noise across the full height; shaking the
  phone raised the peak and the same noise went flat — exactly backwards. Measured: a still
  phone and a shaken one both drew across 79% of the chart, indistinguishable. Each chart now
  has a floor it will not zoom past (2.0 m/s², 0.5 rad/s) and the axis eases between values
  instead of snapping. A resting phone now draws across 1%, gentle handling 42%, a real twist
  79%. The label reads `(at rest)` while the chart is sitting on its floor.
