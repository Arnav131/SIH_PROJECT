"""Live packet monitor: attaches to the receiver as an observer and prints what the phone sends.

The receiver broadcasts every accepted frame to all connected clients except the one that
produced it, so this connects to the same /ws/data endpoint and simply listens.

    python monitor.py                      # follow a receiver on localhost:8000
    python monitor.py --url http://host:8000
    python monitor.py --raw                # dump full JSON instead of one line per packet
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

try:
    import websockets
except ImportError:
    sys.exit("websockets is missing. Run: pip install websockets")

from dotenv import load_dotenv

load_dotenv()


def socket_url(base: str, token: str) -> str:
    base = base.strip().rstrip("/")
    if "://" not in base:
        base = "http://" + base
    scheme, _, rest = base.partition("://")
    ws = {"http": "ws", "ws": "ws", "https": "wss", "wss": "wss"}.get(scheme.lower())
    if ws is None:
        sys.exit(f"Unsupported scheme in {base!r}; use http(s) or ws(s)")
    host, _, path = rest.partition("/")
    return f"{ws}://{host}/{path or 'ws/data'}?token={token}"


def line(packet: dict) -> str:
    kind = packet.get("type")
    ts = packet.get("timestamp", 0)
    clock = datetime.fromtimestamp(ts / 1000).strftime("%H:%M:%S.%f")[:-3] if ts else "--:--:--"
    seq = packet.get("sequence_number", "-")
    d = packet.get("data", {})
    if kind == "sensor":
        unit = "m/s2" if packet.get("sensor") == "accelerometer" else "rad/s"
        body = (f"{packet.get('sensor', ''):<13} "
                f"x={d.get('x', 0):>9.4f} y={d.get('y', 0):>9.4f} z={d.get('z', 0):>9.4f} {unit}")
    elif kind == "gps":
        body = (f"{'gps':<13} lat={d.get('latitude', 0):>11.6f} lon={d.get('longitude', 0):>11.6f} "
                f"alt={d.get('altitude', 0):>7.1f}m spd={d.get('speed', 0):>5.2f} "
                f"acc={d.get('accuracy', 0):>5.1f}m")
    elif kind == "camera":
        body = f"{'camera':<13} {packet.get('filename', '')} {packet.get('width')}x{packet.get('height')}"
    elif kind == "session_start":
        body = (f"{'SESSION':<13} {packet.get('session_id')} on {packet.get('device_model')} "
                f"(Android {packet.get('android_version')})")
    else:
        body = f"{str(kind):<13} {json.dumps(d)[:80]}"
    return f"{clock}  #{str(seq):<6} {body}"


async def follow(url: str, raw: bool):
    attempt = 0
    while True:
        try:
            async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
                attempt = 0
                print("connected - waiting for packets (Ctrl+C to stop)\n", flush=True)
                totals: dict[str, int] = {}
                async for message in ws:
                    if isinstance(message, bytes):
                        print(f"{'':>12}  <binary frame {len(message)} bytes>", flush=True)
                        continue
                    msg = json.loads(message)
                    if msg.get("type") == "connection_ack":
                        continue
                    if msg.get("type") != "live_packet":
                        print(msg, flush=True)
                        continue
                    packet = msg["packet"]
                    if raw:
                        print(json.dumps(packet, indent=2), flush=True)
                    else:
                        print(line(packet), flush=True)
                    stats = msg.get("stats", {})
                    if stats != totals:
                        totals = stats
                        if stats.get("packets", 0) % 50 == 0:
                            print(f"{'':>12}  -- totals: {stats}", flush=True)
        except (OSError, websockets.WebSocketException) as e:
            attempt += 1
            if getattr(e, "code", None) == 1008:
                sys.exit("Receiver rejected the token. Match DEVICE_TOKEN in server/.env.")
            wait = min(2 ** attempt, 30)
            print(f"disconnected ({e}); retrying in {wait}s", file=sys.stderr, flush=True)
            await asyncio.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description="Live view of packets arriving at the receiver")
    ap.add_argument("--url", default=os.getenv("MONITOR_URL", "http://localhost:8000"),
                    help="receiver base URL (default: http://localhost:8000)")
    ap.add_argument("--token", default=os.getenv("DEVICE_TOKEN", "urbansenseai_demo_token"))
    ap.add_argument("--raw", action="store_true", help="print full JSON per packet")
    args = ap.parse_args()
    url = socket_url(args.url, args.token)
    print(f"monitoring {url.split('?')[0]}", flush=True)
    try:
        asyncio.run(follow(url, args.raw))
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
