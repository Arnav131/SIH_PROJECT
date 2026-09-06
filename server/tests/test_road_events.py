"""Road events must reach the dashboard with status, coordinates and (when keyed) an address,
without disturbing the existing sensor packet path."""
import asyncio, csv, json, os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

import websockets

SCRATCH = Path(__file__).parent / "testdata_roadevent"
SERVER = Path(__file__).resolve().parent.parent      # server/ , next to server.py
PORT = 8087
BASE = f"http://127.0.0.1:{PORT}"
TOKEN = "urbansenseai_demo_token"
SESSION = "session_road_20260906_230000"

results = []


def check(name, ok, detail=""):
    results.append((ok, name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=8) as r:
        return json.loads(r.read())


def frame(kind, seq=None, **extra):
    m = {"device_id": "realme3pro_001", "session_id": SESSION,
         "timestamp": int(time.time() * 1000), "type": kind}
    if seq is not None:
        m["sequence_number"] = seq
    m.update(extra)
    return json.dumps(m)


async def wait_health(proc):
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError("server exited:\n" + proc.stdout.read())
        try:
            if get("/health")["status"] == "ok":
                return
        except Exception:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError("server never healthy")


async def main():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    env = {**os.environ, "PORT": str(PORT), "HOST": "127.0.0.1", "DEVICE_TOKEN": TOKEN,
           "DATA_DIR": str(SCRATCH), "PYTHONUNBUFFERED": "1"}
    env.pop("GOOGLE_MAPS_API_KEY", None)   # unkeyed run: must degrade, not crash
    proc = subprocess.Popen([sys.executable, "server.py"], cwd=SERVER, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        await wait_health(proc)

        dash = await websockets.connect(f"ws://127.0.0.1:{PORT}/ws/dashboard")
        hello = json.loads(await asyncio.wait_for(dash.recv(), timeout=5))
        check("hello advertises geocoding state", "geocoding" in hello, str(hello.get("geocoding")))
        check("hello carries a road_events list", isinstance(hello.get("road_events"), list))

        received = []
        stop = asyncio.Event()

        async def pump():
            try:
                while not stop.is_set():
                    received.append(json.loads(await asyncio.wait_for(dash.recv(), timeout=5)))
            except (asyncio.TimeoutError, websockets.WebSocketException):
                pass
        pumping = asyncio.create_task(pump())

        async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws/data?token={TOKEN}") as phone:
            await phone.recv()
            await phone.send(frame("session_start", device_model="Realme RMX1851",
                                   android_version="11",
                                   sensors={"accelerometer": True, "gyroscope": True}))

            # --- existing packet types must still work unchanged ---
            await phone.send(frame("sensor", 1, sensor="accelerometer",
                                   data={"x": 1.5, "y": 2.5, "z": 9.8, "accuracy": 3}))
            await phone.send(frame("sensor", 2, sensor="gyroscope",
                                   data={"x": 0.1, "y": 0.2, "z": 0.3, "accuracy": 3}))
            await phone.send(frame("gps", 3, data={"latitude": 30.763940, "longitude": 76.573267,
                                                   "altitude": 310.0, "speed": 8.3,
                                                   "bearing": 90.0, "accuracy": 6.0}))

            # --- the new type ---
            await phone.send(frame("road_event", 4,
                                   event_id="evt-confirmed-1", event_type="POTHOLE",
                                   detection_label="Pothole", visual_confidence=0.94,
                                   confirmation_status="CONFIRMED", sensor_support="STRONG",
                                   accel_magnitude=13.4, gyro_magnitude=1.8, shock_score=12.2,
                                   shock_sustained=False, correlation_delta_ms=1200,
                                   frame_captured_at=int(time.time() * 1000) - 3500,
                                   latitude=30.763940, longitude=76.573267,
                                   gps_accuracy=6.0, gps_provider="gps"))
            await phone.send(frame("road_event", 5,
                                   event_id="evt-crack-1", event_type="CRACK",
                                   detection_label="Longitudinal Crack", visual_confidence=0.87,
                                   confirmation_status="VISUAL_ONLY", sensor_support="NONE",
                                   frame_captured_at=int(time.time() * 1000) - 3500,
                                   latitude=30.764100, longitude=76.573400,
                                   gps_accuracy=7.0, gps_provider="gps"))
            await asyncio.sleep(1.2)

        stop.set(); await asyncio.sleep(0.3); pumping.cancel(); await dash.close()

        # --- dashboard received them live ---
        live = [m["packet"] for m in received if m.get("type") == "live_packet"]
        road = [p for p in live if p.get("type") == "road_event"]
        check("dashboard received both road events live", len(road) == 2, str(len(road)))
        check("confirmed event keeps its status", road[0]["confirmation_status"] == "CONFIRMED")
        check("crack stays VISUAL_ONLY with no shock", road[1]["confirmation_status"] == "VISUAL_ONLY")
        check("visual confidence passes through untouched",
              road[0]["visual_confidence"] == 0.94 and road[1]["visual_confidence"] == 0.87)
        check("shock score travels separately, never blended",
              road[0]["shock_score"] == 12.2 and "shock_score" not in road[1])
        check("address key present even unkeyed", "address" in road[0], str(road[0].get("address")))
        check("unkeyed server leaves address null", road[0]["address"] is None)

        # --- existing sensor path untouched ---
        sensors = [p for p in live if p.get("type") == "sensor"]
        gps = [p for p in live if p.get("type") == "gps"]
        check("accelerometer + gyroscope packets still flow", len(sensors) == 2, str(len(sensors)))
        check("gps packets still flow", len(gps) == 1, str(len(gps)))

        st = get("/api/status")
        check("existing counters still populated",
              st["sensor_counts"].get("accelerometer") == 1
              and st["sensor_counts"].get("gyroscope") == 1
              and st["sensor_counts"].get("gps") == 1, str(st["sensor_counts"]))
        check("road events counted separately",
              st["sensor_counts"].get("road_events") == 2, str(st["sensor_counts"].get("road_events")))

        # --- API ---
        api = get("/api/road-events")
        check("/api/road-events lists them", len(api) == 2, str(len(api)))

        # --- storage ---
        folder = SCRATCH / SESSION
        with (folder / "road_event.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        check("road_event.csv written", len(rows) == 2, f"{len(rows)} rows")
        check("csv keeps status + coordinates + label",
              rows[0]["confirmation_status"] == "CONFIRMED"
              and rows[0]["latitude"] == "30.76394"
              and rows[1]["detection_label"] == "Longitudinal Crack", str(rows[0])[:90])
        check("existing csvs still written",
              (folder / "accelerometer.csv").exists() and (folder / "gyroscope.csv").exists()
              and (folder / "gps.csv").exists())

        lines = (folder / "packets.jsonl").read_text(encoding="utf-8").strip().splitlines()
        check("packets.jsonl holds every type", len(lines) == 6, f"{len(lines)} lines")

        # --- session JSON stays backward compatible ---
        doc = get(f"/api/sessions/{SESSION}/json")
        check("session JSON keeps its original stream keys",
              {"accelerometer", "gyroscope", "gps", "camera"} <= set(doc["data"]), str(list(doc["data"])))
        check("session JSON counts unchanged for existing streams",
              doc["counts"]["accelerometer"] == 1 and doc["counts"]["gps"] == 1, str(doc["counts"]))

        # --- restart: road events restored from disk ---
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill()
        proc = subprocess.Popen([sys.executable, "server.py"], cwd=SERVER, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        await wait_health(proc)
        check("road events restored after restart", len(get("/api/road-events")) == 2)

        # --- dashboard markup ---
        with urllib.request.urlopen(BASE + "/", timeout=8) as r:
            html = r.read().decode()
        check("camera card is commented out, not deleted",
              "CAMERA DISPLAY DISABLED" in html and "showFrame(msg.frame);" in html)
        check("road events section rendered", 'id="roadEvents"' in html and "addRoadEvent(" in html)
        check("sensor charts preserved",
              'id="chartA"' in html and 'id="chartG"' in html and "MIN_SCALE" in html)
        check("gps + counters + packet log preserved",
              'id="lat"' in html and 'id="packets"' in html and 'id="log"' in html)

    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill()

    print()
    bad = [r for r in results if not r[0]]
    print(f"{len(results) - len(bad)}/{len(results)} checks passed")
    if bad:
        for _, n, d in bad: print("  FAILED:", n, d)
        sys.exit(1)


asyncio.run(main())
