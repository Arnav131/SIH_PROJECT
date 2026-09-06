# UrbanSenseAI

UrbanSenseAI is a clean prototype for using an Android phone as a mobile accelerometer, gyroscope, GPS, and on-demand JPEG camera collector. It sends real hardware measurements to a configurable FastAPI receiver over WebSocket, with bounded offline buffering and reconnection.

## Layout

`android-app/` is the Android Studio Kotlin project. `server/` is the laptop receiver. Every receiver is chosen at runtime in the app; there is no hard-coded LAN address, localhost address, or Cloudflare URL.

```
Phone sensors -> foreground collection service -> bounded queue -> WebSocket -> FastAPI
                                                                                  |
                                            packets.jsonl + CSV + JPEG on disk <--+
                                                                                  |
                                                     browser dashboard (live) <---+
```

## Run the laptop receiver

From `server`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python server.py
```

Create the environment inside `server/`. An unrelated project's venv on the PATH will be
missing `python-dotenv` (`ModuleNotFoundError: No module named 'dotenv'`) and, more quietly,
`websockets` — without which uvicorn accepts no WebSocket at all and the phone can never
deliver a packet.

On startup the receiver prints the exact address to type into the phone, for example:

```
  Type ONE of these into the app's Receiver URL field:
      http://192.168.1.5:8000
```

Visit `http://localhost:8000/health` and expect `{"status":"ok"}`. The dashboard is at `http://localhost:8000/`.

If the phone cannot reach the laptop, open the port once from an Administrator PowerShell:

```powershell
netsh advfirewall firewall add rule name="UrbanSenseAI" dir=in action=allow protocol=TCP localport=8000
```

The app sends its configured token as the WebSocket query parameter. Keep `DEVICE_TOKEN` in `server/.env` and the app's Device Token field identical. Change the example token before exposing the receiver.

## Android setup

Open `android-app` in Android Studio (JDK 17+; Android SDK 36 recommended), let Gradle sync, then select **Build > Build APK(s)**. Install the resulting debug APK on the Realme 3 Pro and approve Location, Camera, and notification permissions. Android 9/10 is supported by `minSdk 28`.

Enter a receiver into the visible **Receiver URL** field, for example `http://YOUR_LAPTOP_LAN_IP:8000`; it derives and shows `ws://YOUR_LAPTOP_LAN_IP:8000/ws/data`. `https` becomes `wss`; explicit `ws(s)` URLs are used directly, and a bare `192.168.1.5:8000` is assumed to be `http`. The socket is always dialled over http(s) because OkHttp performs the upgrade itself and rejects a `ws(s)` scheme outright. The field persists between launches; the app does not auto-connect.

Press **CONNECT** and wait for the server acknowledgement before the UI says CONNECTED. Press **START COLLECTION** to create a session and show the foreground notification. Sensor listeners are owned by the collection engine, not by UI redraws; live state is a latest-value `StateFlow`, so sudden sensor changes cannot recreate the activity or enqueue unlimited UI work.

## LAN then Internet testing

1. Put phone and laptop on the same Wi-Fi network and discover the laptop IPv4 address with `ipconfig`.
2. Run server on port 8000 and allow it through the Windows firewall if prompted.
3. Enter `http://<laptop-ip>:8000` on the phone; confirm Receiver and WebSocket values match.
4. After LAN works, run `cloudflared tunnel --url http://localhost:8000`, then enter the issued `https://...trycloudflare.com` URL. The app derives `wss://.../ws/data`.

## Live dashboard

`http://localhost:8000/` is a WebSocket dashboard, not a polling page. It attaches to
`/ws/dashboard` and renders each frame as it arrives: accelerometer and gyroscope axes with
rolling 300-sample charts, GPS with a map link, packet counters, live rate in Hz, and the
newest camera frame. A dashboard that opens or reloads mid-session is sent the running
session, counters and latest frame in its greeting, so a refresh never blanks the page.

`/ws/dashboard` needs no token from localhost and requires one from anywhere else, so
tunnelling the receiver does not expose the live stream. Terminal-only viewing is available
too:

```powershell
python monitor.py                       # one line per packet
python monitor.py --raw                 # full JSON per packet
python monitor.py --url http://host:8000
```

## Data protocol and storage

All JSON frames contain device ID, session ID, epoch-ms timestamp, type, and (except `session_start`) a monotonically increasing sequence number. Sensor values are raw physical Android readings. Camera capture sends a camera metadata JSON frame followed by a binary JPEG WebSocket frame.

Received sessions are stored under `server/data/<session_id>/`: metadata JSON, an append-only
`packets.jsonl` holding every accepted frame verbatim, accelerometer/gyroscope/GPS/camera CSV
files, and `images/`. One JSON object per line means a running session can be tailed or
resumed, which a single JSON array cannot.

`GET /api/status`, `/api/sessions`, `/api/sessions/{id}`, `/api/sessions/{id}/json`,
`/api/sessions/{id}/download`, and `/api/latest-frame.jpg` provide inspection and download.
The `/json` endpoint consolidates a session into one document: metadata, per-stream arrays,
counts, and the image list. The same document is written to disk by:

```powershell
python export_json.py --list                  # sessions on disk
python export_json.py                         # newest session -> server/<session_id>.json
python export_json.py --session session_...
```

## Operational notes

The current camera control provides real preview and manual 75% JPEG capture. The prompt’s periodic camera modes are intentionally not yet exposed in the UI; this needs final physical-device validation before enabling recurring background CameraX capture, because Android camera lifecycle behavior differs by vendor/device. Accelerometer, gyroscope, GPS, transmission, and bounded queue collection stay independent of the UI and network connection.

When the connection fails, packets buffer up to 2,000 JSON frames; the oldest is discarded with a visible warning. Reconnect delay is 1, 2, 4, 8, 16, then 30 seconds. A reconnect never changes the receiver, session, state, or sequence number.

## Troubleshooting

* **Connection failed:** confirm receiver URL, matching token, firewall, and server `/health` endpoint.
* **No GPS:** enable location services and grant precise location permission.
* **Camera unavailable:** grant camera permission and close any competing camera app.
* **No packets:** tap Connect first, confirm it becomes CONNECTED only after acknowledgement, then start collection.
* **Readings frozen at zero:** accelerometer and gyroscope need no permission and update immediately; GPS shows `Waiting for first fix…` until a real fix arrives, which needs an outdoor view of the sky. The network provider is subscribed alongside GPS and the last known fix seeds the display.
* **"Receiver rejected the device token":** the app stops retrying on a 1008 close. Match the app's Device Token to `DEVICE_TOKEN` in `server/.env`.
* **Cloudflare unavailable:** verify the tunnel process is still running and re-enter its newly issued URL.
* **Dashboard shows Device: none:** the receiver is up but no phone is attached. Counters shown are the running totals from earlier packets.
* **Dashboard rate reads 0.0 Hz:** no packets have arrived in the last two seconds. Counters and rate keep updating in a background browser tab.
