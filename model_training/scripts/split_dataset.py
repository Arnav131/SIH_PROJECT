"""
Dataset Splitting Script
=========================

Creates leakage-aware train/val/test splits from the harmonized staging area.

Key principles:
1. RAD images from the same video source stay in the same split
   (filename pattern: XX_DDMMYYYY_mp4-N → group by XX_DDMMYYYY)
2. Road Damage images from the same capture session stay together
   (filename pattern: YYYYMMDD_HHMMSS → group by date)
3. Pothole dataset images are grouped by base name prefix
4. Target split: 75% train / 15% val / 10% test

Usage:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""

import sys
import json
import shutil
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import COMBINED_DIR, REPORTS_DIR, FINAL_CLASS_NAMES
from utils.dataset_utils import find_images, find_labels, get_class_distribution


def extract_group_key(filename: str) -> str:
    """
    Extract the group key for leakage-aware splitting.
    
    Files are prefixed by source dataset:
    - rad_XX_DDMMYYYY_mp4-N_... → group by 'rad_XX_DDMMYYYY'
    - rd_YYYYMMDD_HHMMSS.jpg → group by 'rd_YYYYMMDD'
    - pot_NNN_jpg.rf... → group by 'pot_NNN'
    """
    stem = Path(filename).stem

    if stem.startswith('rad_'):
        # RAD: rad_XX_DDMMYYYY_mp4-N or rad_XX_DD-MM-YYYY_mp4-N
        rest = stem[4:]  # Remove 'rad_' prefix
        if '_mp4-' in rest:
            group = rest.split('_mp4-')[0]
            return f"rad_{group}"
        return f"rad_{rest}"

    elif stem.startswith('rd_'):
        # Road Damage: rd_YYYYMMDD_HHMMSS
        rest = stem[3:]
        # Group by date (first 8 chars if they look like a date)
        parts = rest.split('_')
        if len(parts) >= 1 and len(parts[0]) == 8 and parts[0].isdigit():
            return f"rd_{parts[0]}"
        return f"rd_{rest[:20]}"  # Fallback

    elif stem.startswith('pot_'):
        # Pothole: pot_NNN_jpg.rf.HASH or pot_potholesNNN_png.rf.HASH
        rest = stem[4:]
        # Extract base number/name before _jpg or _png
        for sep in ['_jpg', '_png']:
            if sep in rest:
                base = rest.split(sep)[0]
                return f"pot_{base}"
        return f"pot_{rest[:15]}"

    return stem  # Fallback


def load_duplicate_pairs():
    """
    Load duplicate/near-duplicate pairs produced by detect_duplicates.py.

    Returns a list of (name_a, name_b) filename pairs, or [] if the report
    has not been generated yet.
    """
    dupe_json = REPORTS_DIR / "duplicate_groups.json"
    if not dupe_json.exists():
        return []
    with open(dupe_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for group in data.get("exact_groups", []):
        for other in group[1:]:
            pairs.append((group[0], other))
    for entry in data.get("near_pairs", []):
        pairs.append((entry[0], entry[1]))
    return pairs


class _Union:
    """Minimal union-find over group keys."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(image_paths: list, use_duplicates: bool = True):
    """
    Build the atomic units of the split.

    Two images end up in the same unit when either
      (a) their filename group key matches (same video / same capture session /
          same base image), or
      (b) duplicate detection flagged them as identical or near identical.

    Merging (b) into (a) is what stops an augmented copy or a neighbouring
    video frame from sitting in train while its twin sits in val.
    """
    key_of = {img.name: extract_group_key(img.name) for img in image_paths}

    uf = _Union()
    for key in set(key_of.values()):
        uf.find(key)

    linked = 0
    if use_duplicates:
        for name_a, name_b in load_duplicate_pairs():
            ka, kb = key_of.get(name_a), key_of.get(name_b)
            if ka and kb and uf.find(ka) != uf.find(kb):
                uf.union(ka, kb)
                linked += 1

    groups = defaultdict(list)
    for img in image_paths:
        groups[uf.find(key_of[img.name])].append(img)
    return groups, linked


def _source_of(name: str) -> str:
    for prefix, label in (("rad_", "rad"), ("rd_", "road_damage"), ("pot_", "pothole")):
        if name.startswith(prefix):
            return label
    return "other"


def split_by_groups(
    image_paths: list,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
    use_duplicates: bool = True,
):
    """
    Split images into train/val/test with group-level separation.

    Groups are stratified per source dataset and assigned largest-first to
    whichever split is furthest below its target. A plain sequential fill
    (fill train, then val, then test) overshoots badly here because RAD video
    groups are hundreds of images while pothole groups are 1-2, which starves
    the test split.
    """
    random.seed(seed)

    groups, linked = build_groups(image_paths, use_duplicates=use_duplicates)

    # Stratify per source so every split sees all three datasets
    by_source = defaultdict(list)
    for key, imgs in groups.items():
        by_source[_source_of(imgs[0].name)].append((key, imgs))

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    splits = {"train": [], "val": [], "test": []}

    for source, source_groups in by_source.items():
        random.shuffle(source_groups)
        # Largest groups first: they constrain the split the most
        source_groups.sort(key=lambda kv: len(kv[1]), reverse=True)

        total = sum(len(imgs) for _, imgs in source_groups)
        targets = {k: total * r for k, r in ratios.items()}
        counts = {k: 0 for k in ratios}

        for _key, imgs in source_groups:
            # Whichever split is furthest below its target takes the group
            pick = max(ratios, key=lambda k: targets[k] - counts[k])
            splits[pick].extend(imgs)
            counts[pick] += len(imgs)

    return splits, len(groups), linked


def copy_split(split_name: str, image_paths: list):
    """Copy images and labels to their final split directories."""
    img_dir = COMBINED_DIR / "images" / split_name
    lbl_dir = COMBINED_DIR / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    staging_lbl_dir = COMBINED_DIR / "staging" / "labels"
    copied = 0
    dropped = 0

    for img_path in image_paths:
        # Copy image
        dst_img = img_dir / img_path.name
        shutil.copy2(str(img_path), str(dst_img))

        # Write label, dropping degenerate boxes.
        # Dataset 2 contains a zero-height box; YOLO would silently ignore it,
        # but keeping it out of the processed set keeps validation clean.
        lbl_path = staging_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            keep = []
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    dropped += 1
                    continue
                try:
                    w, h = float(parts[3]), float(parts[4])
                except ValueError:
                    dropped += 1
                    continue
                if w <= 0 or h <= 0:
                    dropped += 1
                    continue
                keep.append(line)
            dst_lbl = lbl_dir / lbl_path.name
            dst_lbl.write_text(("\n".join(keep) + "\n") if keep else "",
                               encoding="utf-8")

        copied += 1

    return copied, dropped


def main():
    parser = argparse.ArgumentParser(description="Split dataset with leakage prevention")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-duplicates", action="store_true",
                        help="Do not merge groups linked by duplicate detection")
    args = parser.parse_args()

    print("=" * 60)
    print("DATASET SPLITTING — Road Damage AI Pipeline")
    print("=" * 60)
    print(f"\n  Target split: {args.train_ratio:.0%} train / "
          f"{args.val_ratio:.0%} val / {args.test_ratio:.0%} test")

    staging_img = COMBINED_DIR / "staging" / "images"
    if not staging_img.exists():
        print(f"\n  ERROR: Staging directory not found: {staging_img}")
        print("  Run harmonize_classes.py first!")
        return

    images = find_images(staging_img, recursive=False)
    print(f"  Total images in staging: {len(images)}")

    # Clean existing splits
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            split_dir = COMBINED_DIR / subdir / split
            if split_dir.exists():
                shutil.rmtree(str(split_dir))

    # Perform leakage-aware split
    splits, num_groups, linked = split_by_groups(
        images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        use_duplicates=not args.ignore_duplicates,
    )

    print(f"\n  Total groups (for leakage prevention): {num_groups}")
    print(f"  Groups merged by duplicate detection: {linked}")

    # Copy to final directories
    for split_name, split_images in splits.items():
        copied, dropped = copy_split(split_name, split_images)
        note = f" — {dropped} degenerate box(es) dropped" if dropped else ""
        print(f"  {split_name}: {copied} images ({copied / len(images) * 100:.1f}%){note}")

    # Verify class distribution per split
    print(f"\n  Per-split class distribution:")
    split_distributions = {}
    for split_name in ['train', 'val', 'test']:
        lbl_dir = COMBINED_DIR / "labels" / split_name
        labels = find_labels(lbl_dir, recursive=False)
        dist = get_class_distribution(labels)
        split_distributions[split_name] = dist

        print(f"\n    {split_name}:")
        for cls_id in sorted(dist.keys()):
            name = FINAL_CLASS_NAMES[cls_id] if cls_id < len(FINAL_CLASS_NAMES) else f"class_{cls_id}"
            print(f"      {cls_id} ({name}): {dist[cls_id]}")

    # Check source dataset representation per split
    print(f"\n  Source dataset representation per split:")
    for split_name, split_images in splits.items():
        sources = Counter()
        for img in split_images:
            if img.name.startswith('rad_'):
                sources['RAD'] += 1
            elif img.name.startswith('rd_'):
                sources['Road Damage'] += 1
            elif img.name.startswith('pot_'):
                sources['Pothole'] += 1
        print(f"    {split_name}: {dict(sources)}")

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "split_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Dataset Split Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Split Configuration\n")
        f.write(f"- Ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}\n")
        f.write(f"- Seed: {args.seed}\n")
        f.write(f"- Leakage prevention groups: {num_groups}\n")
        f.write(f"- Groups merged via duplicate detection: {linked}\n\n")
        f.write(f"## Split Sizes\n\n")
        f.write("| Split | Images | % |\n|---|---|---|\n")
        for split_name, split_images in splits.items():
            pct = len(split_images) / len(images) * 100
            f.write(f"| {split_name} | {len(split_images)} | {pct:.1f}% |\n")
        f.write(f"\n## Class Distribution Per Split\n\n")
        for split_name, dist in split_distributions.items():
            f.write(f"\n### {split_name}\n")
            f.write("| ID | Class | Count |\n|---|---|---|\n")
            for cls_id in sorted(dist.keys()):
                name = FINAL_CLASS_NAMES[cls_id] if cls_id < len(FINAL_CLASS_NAMES) else f"class_{cls_id}"
                f.write(f"| {cls_id} | {name} | {dist[cls_id]} |\n")

    print(f"\n  Report saved: {report_path}")
    print(f"\n{'=' * 60}")
    print(f"Dataset splitting complete!")
    print(f"Next step: python scripts/visualize_annotations.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
