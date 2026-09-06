"""UrbanSenseAI receiver: JSON sensor frames + metadata followed by JPEG binary frames."""
import asyncio, csv, json, logging, os, time, zipfile
from collections import Counter, deque
from pathlib import Path
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("urbansense")
ROOT = Path(__file__).parent
# DATA_DIR lets a second receiver, or a test run, keep its captures away from the real ones.
DATA = Path(os.getenv("DATA_DIR") or (ROOT / "data")); DATA.mkdir(parents=True, exist_ok=True)
TOKEN = os.getenv("DEVICE_TOKEN", "urbansenseai_demo_token")
app = FastAPI(title="UrbanSenseAI Receiver")
connections: dict[str, dict[str, Any]] = {}
viewers: dict[str, WebSocket] = {}        # dashboards; receive, never produce
sessions: dict[str, dict[str, Any]] = {}
stats = Counter(); last_packet: dict[str, Any] | None = None
pending_image: dict[str, Any] = {}
latest_frame: dict[str, Any] = {}         # newest JPEG, served to the dashboard
events: deque[dict[str, Any]] = deque(maxlen=50)   # recent auto-captures, newest last
lock = asyncio.Lock()

CSV_FIELDS = {
 "accelerometer": ["timestamp","sequence_number","x","y","z","accuracy"],
 "gyroscope": ["timestamp","sequence_number","x","y","z","accuracy"],
 "gps": ["timestamp","sequence_number","latitude","longitude","altitude","speed","bearing","accuracy"],
 "camera": ["timestamp","sequence_number","filename","width","height","trigger","accel_magnitude","gyro_magnitude","shock_score","latitude","longitude","gps_accuracy","gps_provider"],
}
def clean_id(value: str) -> str:
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("IDs may only contain letters, numbers, underscore, and hyphen")
    return value
def session_dir(sid: str) -> Path:
    return DATA / clean_id(sid)
def append_csv(sid: str, kind: str, row: dict[str, Any]):
    fields = CSV_FIELDS[kind]; path = session_dir(sid) / f"{kind}.csv"; new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new: writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})
def append_jsonl(sid: str, msg: dict[str, Any]):
    """The append-only wire log. One JSON object per line, so a running session can be
    tailed or resumed; a single JSON array could not be appended to safely."""
    with (session_dir(sid) / "packets.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg, separators=(",", ":")) + "\n")

def read_jsonl(sid: str) -> list[dict[str, Any]]:
    path = session_dir(sid) / "packets.jsonl"
    if not path.exists(): return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except json.JSONDecodeError: LOG.warning("skipping malformed line in %s", path.name)
    return out

def build_session_json(sid: str) -> dict[str, Any]:
    """Consolidates a session into one JSON document: metadata plus per-stream arrays."""
    folder = session_dir(sid)
    if not folder.exists(): raise HTTPException(404, "session not found")
    meta_path = folder / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    streams: dict[str, list[dict[str, Any]]] = {"accelerometer": [], "gyroscope": [], "gps": [], "camera": []}
    for msg in read_jsonl(sid):
        kind = msg.get("type")
        if kind == "sensor" and msg.get("sensor") in streams:
            streams[msg["sensor"]].append({"timestamp": msg["timestamp"], "sequence_number": msg["sequence_number"], **msg.get("data", {})})
        elif kind == "gps":
            streams["gps"].append({"timestamp": msg["timestamp"], "sequence_number": msg["sequence_number"], **msg.get("data", {})})
        elif kind == "camera":
            streams["camera"].append({k: msg.get(k) for k in CSV_FIELDS["camera"] if k in msg})
    stamps = [m["timestamp"] for m in read_jsonl(sid) if isinstance(m.get("timestamp"), int)]
    return {
        "session_id": sid,
        "device_id": meta.get("device_id"),
        "device_model": meta.get("device_model"),
        "android_version": meta.get("android_version"),
        "started": min(stamps) if stamps else None,
        "ended": max(stamps) if stamps else None,
        "counts": {k: len(v) for k, v in streams.items()},
        "images": sorted(p.name for p in (folder / "images").glob("*")) if (folder / "images").exists() else [],
        "data": streams,
    }

def validate(msg: dict[str, Any]) -> tuple[str, str]:
    kind = msg.get("type")
    for key in ("device_id", "session_id", "timestamp"):
        if key not in msg: raise ValueError(f"missing {key}")
    if kind not in ("session_start", "sensor", "gps", "camera"): raise ValueError("unsupported type")
    if kind != "session_start" and not isinstance(msg.get("sequence_number"), int): raise ValueError("missing sequence_number")
    sid = clean_id(str(msg["session_id"])); device = clean_id(str(msg["device_id"]))
    if kind == "sensor":
        sensor = msg.get("sensor")
        if sensor not in ("accelerometer", "gyroscope"): raise ValueError("invalid sensor")
        for k in ("x","y","z","accuracy"):
            if k not in msg.get("data", {}): raise ValueError(f"missing data.{k}")
    elif kind == "gps":
        for k in ("latitude","longitude","altitude","speed","bearing","accuracy"):
            if k not in msg.get("data", {}): raise ValueError(f"missing data.{k}")
    return sid, device
async def broadcast(payload: dict[str, Any], exclude: str | None = None):
    """Fan out to observers. The producing device is skipped: echoing its own packets
    back over a phone uplink doubled the traffic and starved the sensor stream."""
    dead=[]
    for ws in list(connections.values()):
        if ws["id"] == exclude: continue
        try: await ws["socket"].send_json(payload)
        except Exception: dead.append(ws["id"])
    for ident in dead: connections.pop(ident, None)
    stale=[]
    for ident, sock in list(viewers.items()):
        try: await sock.send_json(payload)
        except Exception: stale.append(ident)
    for ident in stale: viewers.pop(ident, None)

@app.get("/health")
async def health(): return {"status":"ok"}
@app.get("/api/status")
async def status():
    return {"status":"ok","connected_devices":len(connections),"active_sessions":list(sessions),"packets_received":stats["packets"],"sensor_counts":dict(stats),"last_packet":last_packet}
@app.get("/api/sessions")
async def list_sessions(): return [{"session_id":p.name, **sessions.get(p.name,{})} for p in DATA.iterdir() if p.is_dir()]
@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    folder=session_dir(session_id)
    if not folder.exists(): raise HTTPException(404,"session not found")
    meta=folder/"metadata.json"
    return json.loads(meta.read_text()) if meta.exists() else {"session_id":session_id}
@app.get("/api/sessions/{session_id}/download")
async def download(session_id: str):
    folder=session_dir(session_id)
    if not folder.exists(): raise HTTPException(404,"session not found")
    archive=DATA/f"{session_id}.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file(): z.write(p,p.relative_to(DATA))
    return FileResponse(archive, filename=archive.name)

@app.websocket("/ws/data")
async def receiver(ws: WebSocket):
    global last_packet
    token=ws.query_params.get("token","")
    if token != TOKEN:
        LOG.warning("Rejected device: invalid token %r", token[:8])
        await ws.accept(); await ws.close(code=1008, reason="invalid token"); return
    await ws.accept(); ident=f"conn_{id(ws)}"; connections[ident]={"id":ident,"socket":ws,"connected_at":time.time()}
    await ws.send_json({"type":"connection_ack","status":"connected","server_time":int(time.time()*1000)})
    try:
      while True:
        incoming=await ws.receive()
        # A dropped phone delivers websocket.disconnect, which matches neither branch below.
        # Falling through span the loop without awaiting real I/O and wedged the event loop,
        # so every later HTTP request timed out until the process was restarted.
        if incoming.get("type") == "websocket.disconnect": break
        if incoming.get("text") is not None:
            try:
                msg=json.loads(incoming["text"]); sid,device=validate(msg); kind=msg["type"]
                folder=session_dir(sid); (folder/"images").mkdir(parents=True, exist_ok=True)
                if kind=="session_start":
                    (folder/"metadata.json").write_text(json.dumps(msg,indent=2)); sessions[sid]={"device_id":device,"started":msg["timestamp"]}
                elif kind=="sensor":
                    row={"timestamp":msg["timestamp"],"sequence_number":msg["sequence_number"],**msg["data"]}; append_csv(sid,msg["sensor"],row); stats[msg["sensor"]]+=1
                elif kind=="gps":
                    row={"timestamp":msg["timestamp"],"sequence_number":msg["sequence_number"],**msg["data"]}; append_csv(sid,"gps",row); stats["gps"]+=1
                elif kind=="camera": pending_image[ident]=msg
                append_jsonl(sid,msg)
                stats["packets"]+=1; last_packet={"timestamp":msg["timestamp"],"type":kind,"device_id":device,"session_id":sid}
                await broadcast({"type":"live_packet","packet":msg,"stats":{"packets":stats["packets"],"accelerometer":stats["accelerometer"],"gyroscope":stats["gyroscope"],"gps":stats["gps"],"camera":stats["camera"]}}, exclude=ident)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                LOG.warning("Rejected message: %s",e); await ws.send_json({"type":"error","message":str(e)})
        elif incoming.get("bytes") is not None:
            meta=pending_image.pop(ident,None)
            if not meta: await ws.send_json({"type":"error","message":"image metadata required"}); continue
            sid,_=validate(meta); image=incoming["bytes"]
            if len(image)>8_000_000: await ws.send_json({"type":"error","message":"image too large"}); continue
            filename=Path(meta["filename"]).name; (session_dir(sid)/"images"/filename).write_bytes(image)
            append_csv(sid,"camera",meta); stats["camera"]+=1
            latest_frame.update({"bytes":image,"filename":filename,"session_id":sid,"received":int(time.time()*1000)})
            frame_event={"type":"camera_frame","session_id":sid,"filename":filename,
                         "width":meta.get("width"),"height":meta.get("height"),
                         "received":latest_frame["received"],
                         "trigger":meta.get("trigger","manual"),
                         "accel_magnitude":meta.get("accel_magnitude"),
                         "gyro_magnitude":meta.get("gyro_magnitude"),
                         "shock_score":meta.get("shock_score"),
                         "latitude":meta.get("latitude"),"longitude":meta.get("longitude"),
                         "gps_accuracy":meta.get("gps_accuracy"),
                         "url":f"/api/sessions/{sid}/images/{filename}"}
            if str(meta.get("trigger","")).startswith("shock"):
                events.append(frame_event); stats["events"]+=1
            await broadcast(frame_event, exclude=ident)
    except WebSocketDisconnect: LOG.info("client disconnected")
    except RuntimeError as e: LOG.info("socket closed: %s", e)
    finally:
        connections.pop(ident,None); pending_image.pop(ident,None)
        LOG.info("connection %s released (%d still open)", ident, len(connections))

@app.get("/api/sessions/{session_id}/json")
async def session_json(session_id: str):
    """The whole session as one JSON document, ready to save or feed to a notebook."""
    return build_session_json(clean_id(session_id))

def load_recent_events(limit: int = 50):
    """Rebuild the gallery from disk at startup.

    The event list lives in memory, so without this a restart hid every capture taken before
    it - the photographs were still on disk but nothing pointed at them any more."""
    found: list[dict[str, Any]] = []
    folders = sorted((d for d in DATA.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    for folder in folders:
        log = folder / "packets.jsonl"
        if not log.exists(): continue
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or '"camera"' not in line: continue
            try: msg = json.loads(line)
            except json.JSONDecodeError: continue
            if msg.get("type") != "camera": continue
            if not str(msg.get("trigger", "")).startswith("shock"): continue
            name = Path(str(msg.get("filename", ""))).name
            if not name or not (folder / "images" / name).is_file(): continue
            found.append({"type": "camera_frame", "session_id": folder.name, "filename": name,
                          "width": msg.get("width"), "height": msg.get("height"),
                          "received": msg.get("timestamp"), "trigger": msg.get("trigger"),
                          "accel_magnitude": msg.get("accel_magnitude"),
                          "gyro_magnitude": msg.get("gyro_magnitude"),
                          "shock_score": msg.get("shock_score"),
                          "latitude": msg.get("latitude"), "longitude": msg.get("longitude"),
                          "gps_accuracy": msg.get("gps_accuracy"),
                          "url": f"/api/sessions/{folder.name}/images/{name}"})
        if len(found) >= limit: break
    found.sort(key=lambda e: e.get("received") or 0)
    for e in found[-limit:]: events.append(e)
    if found: LOG.info("restored %d shock events from disk", min(len(found), limit))

load_recent_events()

@app.get("/api/events")
async def list_events(): return list(events)

THUMB_WIDTH = 480

def thumbnail(path: Path) -> Path:
    """A 480px copy beside the original, built once and reused.

    A gallery of full-size captures asks the browser to decode a dozen multi-megapixel
    JPEGs at once, which leaves the cards blank for seconds. Falls back to the original
    if Pillow is not installed."""
    cache = path.parent / ".thumbs"
    small = cache / path.name
    if small.is_file() and small.stat().st_mtime >= path.stat().st_mtime: return small
    try:
        from PIL import Image
    except ImportError:
        return path
    try:
        cache.mkdir(exist_ok=True)
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > THUMB_WIDTH:
                im = im.resize((THUMB_WIDTH, max(1, round(im.height * THUMB_WIDTH / im.width))), Image.LANCZOS)
            im.save(small, "JPEG", quality=72)
        return small
    except Exception as e:
        LOG.warning("thumbnail failed for %s: %s", path.name, e)
        return path

@app.get("/api/sessions/{session_id}/images/{filename}")
async def session_image(session_id: str, filename: str, thumb: int = 0):
    # Path().name strips any traversal a caller tried to smuggle into the filename.
    path = session_dir(clean_id(session_id)) / "images" / Path(filename).name
    if not path.is_file(): raise HTTPException(404, "image not found")
    served = thumbnail(path) if thumb else path
    return FileResponse(served, media_type="image/jpeg")

@app.get("/api/latest-frame.jpg")
async def latest_frame_jpeg():
    if not latest_frame: raise HTTPException(404, "no frame captured yet")
    return Response(content=latest_frame["bytes"], media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})

@app.websocket("/ws/dashboard")
async def dashboard_socket(ws: WebSocket):
    """Read-only live feed. A dashboard on this laptop needs no token; anything arriving
    through a public tunnel does, so exposing the receiver does not expose the stream."""
    host = ws.client.host if ws.client else ""
    if host not in ("127.0.0.1", "::1", "localhost") and ws.query_params.get("token","") != TOKEN:
        await ws.accept(); await ws.close(code=1008, reason="invalid token"); return
    await ws.accept(); ident=f"view_{id(ws)}"; viewers[ident]=ws
    LOG.info("dashboard attached from %s (%d viewing)", host or "?", len(viewers))
    # A dashboard that reloads mid-session has missed every event so far, so the greeting
    # carries the current session and newest frame instead of leaving the page blank.
    await ws.send_json({"type":"hello","status":"connected","server_time":int(time.time()*1000),
                        "devices":len(connections),"stats":dict(stats),"last_packet":last_packet,
                        "sessions":list(sessions),
                        "frame":{k:latest_frame[k] for k in ("filename","session_id","received")
                                 if k in latest_frame} or None,
                        "events":list(events)})
    try:
        while True:
            incoming = await ws.receive()
            if incoming.get("type") == "websocket.disconnect": break
    except (WebSocketDisconnect, RuntimeError): pass
    finally:
        viewers.pop(ident, None)
        LOG.info("dashboard detached (%d viewing)", len(viewers))

@app.get("/", response_class=HTMLResponse)
async def dashboard(): return (ROOT/"dashboard.html").read_text(encoding="utf-8")
def lan_addresses() -> list[str]:
    """Every IPv4 the laptop answers on, so the phone can be told exactly what to type."""
    import socket
    found: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            # 169.254.x is a link-local address from an adapter with no DHCP lease; the
            # phone can never reach it, so listing it only sends people down a rabbit hole.
            if ip.startswith(("127.", "169.254.")) or ip in found: continue
            found.append(ip)
    except OSError: pass
    if not found:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80))
            found.append(s.getsockname()[0]); s.close()
        except OSError: pass
    return found

def banner(port: int):
    rule = ('      netsh advfirewall firewall add rule name="UrbanSenseAI" '
            f"dir=in action=allow protocol=TCP localport={port}")
    lines = ["", "=" * 62, "  UrbanSenseAI receiver", "=" * 62,
             "  Type ONE of these into the app's Receiver URL field:"]
    for ip in lan_addresses() or ["<no LAN address found - check Wi-Fi>"]:
        lines.append(f"      http://{ip}:{port}")
    lines += ["",
              f"  Dashboard on this laptop: http://localhost:{port}",
              f"  Device token:             {TOKEN}",
              "",
              "  Phone and laptop must sit on the SAME Wi-Fi network.",
              "  If the phone cannot reach it, open the port once as Administrator:",
              rule, "=" * 62, ""]
    print("\n".join(lines), flush=True)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    banner(port)
    uvicorn.run("server:app", host=os.getenv("HOST", "0.0.0.0"), port=port, reload=False)
