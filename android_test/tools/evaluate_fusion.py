"""
Laptop-side reference implementation of the Android multi-model perception stack.

This mirrors, step for step, what the Kotlin code does on the phone:
letterbox -> two independent ONNX sessions -> per-model adapters (class
normalization + per-model thresholds) -> DetectionFusionEngine.

Keeping the two implementations in sync matters: this is the only place the
fusion logic can be checked against ground-truth labels, because the phone has
no labels. If you change the fusion rules in Kotlin, change them here too.

Usage:
    python android_test/tools/evaluate_fusion.py
    python android_test/tools/evaluate_fusion.py --limit 40

Nothing here writes to the datasets. Read-only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np
import onnxruntime as ort
from PIL import Image

# --------------------------------------------------------------------------
# Paths. Repo-relative so this runs from anywhere.
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

POTHOLE_MODEL = os.path.join(ROOT, "android_test", "app", "src", "main", "assets",
                             "pothole_yolov12s.onnx")
CRACK_MODEL = os.path.join(ROOT, "android_test", "app", "src", "main", "assets",
                           "crack_rdd2022.onnx")

DATASET = os.path.join(ROOT, "model_training", "data", "processed", "road_damage_combined")
IMAGES = os.path.join(DATASET, "images", "test")
LABELS = os.path.join(DATASET, "labels", "test")

# Local dataset taxonomy (configs/class_mapping.yaml), used ONLY to pick which
# images to test and to state what is actually in them. It is NOT the models'
# taxonomy — neither model was trained on this class order.
GT_POTHOLE, GT_CRACK = 0, 1

# --------------------------------------------------------------------------
# Config — mirrors config/DetectionConfig.kt
# --------------------------------------------------------------------------
POTHOLE_THRESHOLD = 0.25   # tunable
CRACK_THRESHOLD = 0.25     # tunable
NMS_IOU = 0.7              # within a single model
CROSS_MODEL_DUPLICATE_IOU = 0.80  # see DetectionFusionEngine notes

INPUT_SIZE = 640
PAD = 114


# --------------------------------------------------------------------------
# Common detection format — mirrors detection/Detection.kt
# --------------------------------------------------------------------------
@dataclass
class Detection:
    source: str          # "pothole_model" | "crack_model"
    class_name: str      # model's own label, preserved
    class_type: str      # POTHOLE | CRACK | OTHER
    subtype: str | None  # LONGITUDINAL | TRANSVERSE | ALLIGATOR | None
    confidence: float
    box: tuple           # x1, y1, x2, y2 in source-image pixels
    timestamp_ms: int = 0


def letterbox(im: Image.Image, size: int = INPUT_SIZE):
    w, h = im.size
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = Image.new("RGB", (size, size), (PAD, PAD, PAD))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (px, py))
    x = (np.asarray(canvas).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    return x, s, px, py, w, h


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets: list[Detection], thr: float) -> list[Detection]:
    """Greedy NMS, per class name — mirrors YoloOnnxDetector.nms."""
    kept: list[Detection] = []
    for cname in {d.class_name for d in dets}:
        pool = sorted([d for d in dets if d.class_name == cname],
                      key=lambda d: d.confidence, reverse=True)
        while pool:
            best = pool.pop(0)
            kept.append(best)
            pool = [d for d in pool if iou(best.box, d.box) <= thr]
    return sorted(kept, key=lambda d: d.confidence, reverse=True)


# --------------------------------------------------------------------------
# Adapters — mirror adapters/PotholeModelAdapter.kt and CrackModelAdapter.kt
# --------------------------------------------------------------------------
class YoloAdapter:
    """Shared decode for both models: [1, 4+nc, anchors], sigmoid scores, no in-graph NMS."""

    def __init__(self, path: str, source: str, threshold: float,
                 mapper: Callable[[int, str], tuple | None]):
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.source = source
        self.threshold = threshold
        self.mapper = mapper
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = eval(meta["names"]) if "names" in meta else {}
        out_shape = self.session.get_outputs()[0].shape
        self.num_classes = int(out_shape[1]) - 4

    def infer(self, im: Image.Image):
        x, s, px, py, w, h = letterbox(im)
        t0 = time.perf_counter()
        y = self.session.run(None, {self.input_name: x})[0][0]
        infer_ms = (time.perf_counter() - t0) * 1000

        dets: list[Detection] = []
        cx, cy, bw, bh = y[0], y[1], y[2], y[3]
        scores = y[4:]
        best_id = scores.argmax(axis=0)
        best_sc = scores.max(axis=0)
        keep = np.where(best_sc >= self.threshold)[0]

        for i in keep:
            cid = int(best_id[i])
            mapped = self.mapper(cid, self.names.get(cid, f"class_{cid}"))
            if mapped is None:      # adapter chose to drop this class
                continue
            class_type, subtype = mapped
            x1 = (cx[i] - bw[i] / 2 - px) / s
            y1 = (cy[i] - bh[i] / 2 - py) / s
            x2 = (cx[i] + bw[i] / 2 - px) / s
            y2 = (cy[i] + bh[i] / 2 - py) / s
            x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
            y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append(Detection(self.source, self.names.get(cid, f"class_{cid}"),
                                  class_type, subtype, float(best_sc[i]), (x1, y1, x2, y2)))
        return nms(dets, NMS_IOU), infer_ms


def pothole_mapper(cid: int, name: str):
    # Single-class model. Everything it emits is a pothole.
    return ("POTHOLE", None)


CRACK_SUBTYPES = {
    "Longitudinal Crack": "LONGITUDINAL",
    "Transverse Crack": "TRANSVERSE",
    "Alligator Crack": "ALLIGATOR",
}


def crack_mapper(cid: int, name: str):
    """
    The crack model is RDD2022 5-class and has its OWN 'Pothole' class (id 3).
    The pothole specialist owns potholes, so this adapter drops class 3 rather
    than letting two models argue about the same object. This is a deliberate
    routing decision, documented in MULTI_MODEL_ARCHITECTURE.md, not a bug.
    """
    if name in CRACK_SUBTYPES:
        return ("CRACK", CRACK_SUBTYPES[name])
    if name == "Pothole":
        return None          # routed to the pothole specialist instead
    return ("OTHER", None)   # RDD2022 'Other' — kept, not discarded


# --------------------------------------------------------------------------
# Fusion — mirrors fusion/DetectionFusionEngine.kt
# --------------------------------------------------------------------------
def fuse(pothole_dets: list[Detection], crack_dets: list[Detection]) -> list[Detection]:
    """
    Confidences are NEVER combined arithmetically — each detection keeps the
    score its own model produced.

    Duplicate handling only removes a box when two detections of the SAME
    class_type overlap above CROSS_MODEL_DUPLICATE_IOU. A pothole and a crack
    that overlap are left alone: they are semantically different events and one
    does not invalidate the other.
    """
    merged = list(pothole_dets) + list(crack_dets)
    merged.sort(key=lambda d: d.confidence, reverse=True)

    out: list[Detection] = []
    for d in merged:
        dup = any(
            k.class_type == d.class_type and iou(k.box, d.box) > CROSS_MODEL_DUPLICATE_IOU
            for k in out
        )
        if not dup:
            out.append(d)
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def gt_classes(stem: str) -> set:
    p = os.path.join(LABELS, stem + ".txt")
    if not os.path.exists(p):
        return set()
    return {int(l.split()[0]) for l in open(p) if l.strip()}


def find_image(stem: str):
    for ext in (".jpg", ".png", ".jpeg"):
        p = os.path.join(IMAGES, stem + ext)
        if os.path.exists(p):
            return p
    return None


def bucket_images():
    both, pot, crk = [], [], []
    for lf in sorted(glob.glob(os.path.join(LABELS, "*.txt"))):
        stem = os.path.splitext(os.path.basename(lf))[0]
        cs = gt_classes(stem)
        if GT_POTHOLE in cs and GT_CRACK in cs:
            both.append(stem)
        elif GT_POTHOLE in cs:
            pot.append(stem)
        elif GT_CRACK in cs:
            crk.append(stem)
    return both, pot, crk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40,
                    help="images per bucket (default 40)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    for p, label in ((POTHOLE_MODEL, "pothole"), (CRACK_MODEL, "crack")):
        if not os.path.exists(p):
            raise SystemExit(f"missing {label} model asset: {p}")

    pothole = YoloAdapter(POTHOLE_MODEL, "pothole_model", POTHOLE_THRESHOLD, pothole_mapper)
    crack = YoloAdapter(CRACK_MODEL, "crack_model", CRACK_THRESHOLD, crack_mapper)

    print(f"pothole model: {os.path.basename(POTHOLE_MODEL)}  "
          f"nc={pothole.num_classes} names={pothole.names}")
    print(f"crack   model: {os.path.basename(CRACK_MODEL)}  "
          f"nc={crack.num_classes} names={crack.names}")
    print()

    both, pot_only, crk_only = bucket_images()
    buckets = {
        "BOTH pothole+crack": both[:args.limit],
        "pothole only": pot_only[:args.limit],
        "crack only": crk_only[:args.limit],
    }

    results = {}
    p_times, c_times = [], []

    for bname, stems in buckets.items():
        stats = {"n": 0, "pothole_fired": 0, "crack_fired": 0, "both_fired": 0,
                 "fused_counts": [], "pothole_conf": [], "crack_conf": []}
        for stem in stems:
            path = find_image(stem)
            if not path:
                continue
            im = Image.open(path).convert("RGB")
            pd, pt = pothole.infer(im)
            cd, ct = crack.infer(im)
            p_times.append(pt)
            c_times.append(ct)
            fused = fuse(pd, cd)

            stats["n"] += 1
            has_p = any(d.class_type == "POTHOLE" for d in fused)
            has_c = any(d.class_type == "CRACK" for d in fused)
            stats["pothole_fired"] += has_p
            stats["crack_fired"] += has_c
            stats["both_fired"] += (has_p and has_c)
            stats["fused_counts"].append(len(fused))
            if pd:
                stats["pothole_conf"].append(max(d.confidence for d in pd))
            if cd:
                stats["crack_conf"].append(max(d.confidence for d in cd))
        results[bname] = stats

        n = max(stats["n"], 1)
        print(f"--- {bname}  (n={stats['n']}) ---")
        print(f"    frames where fusion reported a POTHOLE : {stats['pothole_fired']:3d}  "
              f"({stats['pothole_fired']/n*100:.0f}%)")
        print(f"    frames where fusion reported a CRACK   : {stats['crack_fired']:3d}  "
              f"({stats['crack_fired']/n*100:.0f}%)")
        print(f"    frames reporting BOTH                  : {stats['both_fired']:3d}  "
              f"({stats['both_fired']/n*100:.0f}%)")
        if stats["fused_counts"]:
            print(f"    mean fused detections/frame            : "
                  f"{statistics.mean(stats['fused_counts']):.2f}")
        print()

    print("--- laptop inference time (CPU, not phone) ---")
    if p_times:
        print(f"    pothole model : mean {statistics.mean(p_times):6.1f} ms  "
              f"median {statistics.median(p_times):6.1f} ms")
    if c_times:
        print(f"    crack   model : mean {statistics.mean(c_times):6.1f} ms  "
              f"median {statistics.median(c_times):6.1f} ms")
    if p_times and c_times:
        seq = statistics.mean(p_times) + statistics.mean(c_times)
        print(f"    sequential total ~{seq:.0f} ms  -> ~{1000/seq:.1f} FPS (laptop CPU)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
                      f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
