"""Write a captured session out as a single JSON file.

    python export_json.py                       # newest session -> <session_id>.json
    python export_json.py --session session_...  # a specific one
    python export_json.py --list                 # what is on disk
    python export_json.py --out C:\\path\\run.json

Runs straight off the files in data/, so the receiver does not have to be running.
"""
import argparse
import json
import sys
from pathlib import Path

import server  # reuses build_session_json so the CLI and the API cannot drift apart

# Taken from the server rather than rebuilt, so DATA_DIR moves both of them together.
DATA = server.DATA


def sessions() -> list[Path]:
    if not DATA.exists():
        return []
    return sorted((p for p in DATA.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser(description="Export one session as a single JSON file")
    ap.add_argument("--session", help="session id (default: the most recent)")
    ap.add_argument("--out", help="output path (default: <session_id>.json beside data/)")
    ap.add_argument("--list", action="store_true", help="list sessions and exit")
    ap.add_argument("--indent", type=int, default=2, help="0 for a compact file")
    args = ap.parse_args()

    found = sessions()
    if args.list:
        if not found:
            print("no sessions in", DATA)
            return
        for p in found:
            packets = p / "packets.jsonl"
            n = sum(1 for _ in packets.open(encoding="utf-8")) if packets.exists() else 0
            print(f"{p.name:<40} {n:>7} packets")
        return

    if not found:
        sys.exit(f"No sessions in {DATA}. Run a collection first.")

    sid = args.session or found[-1].name
    if not (DATA / sid).is_dir():
        sys.exit(f"Session {sid!r} not found. Try --list.")

    doc = server.build_session_json(sid)
    out = Path(args.out) if args.out else DATA.parent / f"{sid}.json"
    out.write_text(json.dumps(doc, indent=args.indent or None), encoding="utf-8")

    counts = doc["counts"]
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    print(f"  device   {doc.get('device_model') or doc.get('device_id') or 'unknown'}")
    print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    if doc["images"]:
        print(f"  images   {len(doc['images'])} in data/{sid}/images/")


if __name__ == "__main__":
    main()
