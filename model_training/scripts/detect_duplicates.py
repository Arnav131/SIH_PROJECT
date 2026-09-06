"""
Duplicate Detection Script
============================

Detects exact and near-duplicate images across the harmonized staging area.

Methods:
1. Exact duplicates: MD5 file hash
2. Near-duplicates: Perceptual hash (pHash) via imagehash library

Reports duplicates but does NOT automatically delete them.
Removal is configurable via --remove flag.

Usage:
    python scripts/detect_duplicates.py
    python scripts/detect_duplicates.py --remove    # Remove exact duplicates
    python scripts/detect_duplicates.py --threshold 8  # pHash threshold
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import COMBINED_DIR, REPORTS_DIR
from utils.dataset_utils import find_images, file_md5


def detect_exact_duplicates(image_paths: list) -> dict:
    """Find exact duplicate images using MD5 hash."""
    print("\n  Computing MD5 hashes...")
    hash_to_files = defaultdict(list)

    for i, img_path in enumerate(image_paths):
        if (i + 1) % 1000 == 0:
            print(f"    Hashed {i + 1}/{len(image_paths)} images...")
        h = file_md5(img_path)
        hash_to_files[h].append(img_path)

    # Find groups with >1 file
    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
    total_dupes = sum(len(f) - 1 for f in duplicates.values())

    print(f"  Exact duplicate groups: {len(duplicates)}")
    print(f"  Total duplicate files: {total_dupes}")

    # Show sample
    for h, files in list(duplicates.items())[:5]:
        print(f"    Hash {h[:12]}...: {len(files)} copies")
        for f in files[:3]:
            print(f"      {f.name}")

    return duplicates


def detect_near_duplicates(image_paths: list, threshold: int = 8) -> list:
    """
    Find near-duplicate images using perceptual hashing.
    
    A lower threshold means stricter matching.
    Threshold of 8 is a reasonable default for near-duplicates.
    """
    try:
        from PIL import Image
        import imagehash
    except ImportError:
        print("  ⚠ imagehash not installed. Run: pip install imagehash Pillow")
        print("    Skipping near-duplicate detection.")
        return []

    print(f"\n  Computing perceptual hashes (threshold={threshold})...")

    hashes = []
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1}/{len(image_paths)} images...")
        try:
            img = Image.open(str(img_path))
            phash = imagehash.phash(img)
            hashes.append((img_path, phash))
        except Exception:
            continue

    # Find near-duplicate pairs
    print(f"  Comparing {len(hashes)} hashes...")
    near_dupes = []

    # For efficiency, only compare within source-dataset groups
    # and use a threshold-based approach
    groups = defaultdict(list)
    for path, h in hashes:
        # Group by prefix (rad_, rd_, pot_)
        prefix = path.name.split('_')[0]
        groups[prefix].append((path, h))

    # Cross-group comparison (between datasets — most important)
    prefixes = list(groups.keys())
    for i in range(len(prefixes)):
        for j in range(i + 1, len(prefixes)):
            group_a = groups[prefixes[i]]
            group_b = groups[prefixes[j]]
            print(f"    Comparing {prefixes[i]} ({len(group_a)}) vs {prefixes[j]} ({len(group_b)})...")

            for path_a, hash_a in group_a:
                for path_b, hash_b in group_b:
                    dist = hash_a - hash_b
                    if dist <= threshold:
                        near_dupes.append({
                            'file_a': str(path_a),
                            'file_b': str(path_b),
                            'distance': dist,
                        })

    # Within-group near-duplicates.
    # Sorting by the hash value itself puts visually similar images next to each
    # other, so a sliding window finds real clusters — comparing images in
    # filesystem order would only catch dupes that happen to be named adjacently.
    for prefix, group in groups.items():
        print(f"    Within-group {prefix} ({len(group)})...")
        ordered = sorted(group, key=lambda t: str(t[1]))
        window = 100
        for a in range(len(ordered)):
            for b in range(a + 1, min(a + window, len(ordered))):
                dist = ordered[a][1] - ordered[b][1]
                if dist <= threshold:
                    near_dupes.append({
                        'file_a': str(ordered[a][0]),
                        'file_b': str(ordered[b][0]),
                        'distance': dist,
                    })

    print(f"  Near-duplicate pairs found: {len(near_dupes)}")
    for pair in near_dupes[:5]:
        print(f"    {Path(pair['file_a']).name} <-> {Path(pair['file_b']).name} "
              f"(dist={pair['distance']})")

    return near_dupes


def main():
    parser = argparse.ArgumentParser(description="Detect duplicate images")
    parser.add_argument("--remove", action="store_true",
                        help="Remove exact duplicate files (keeps first copy)")
    parser.add_argument("--threshold", type=int, default=8,
                        help="Perceptual hash distance threshold (default: 8)")
    parser.add_argument("--skip-near", action="store_true",
                        help="Skip near-duplicate detection (faster)")
    args = parser.parse_args()

    print("=" * 60)
    print("DUPLICATE DETECTION — Road Damage AI Pipeline")
    print("=" * 60)

    staging_img = COMBINED_DIR / "staging" / "images"
    if not staging_img.exists():
        print(f"\n  ERROR: Staging directory not found: {staging_img}")
        print("  Run harmonize_classes.py first!")
        return

    images = find_images(staging_img, recursive=False)
    print(f"\n  Total images in staging: {len(images)}")

    # Exact duplicates
    exact_dupes = detect_exact_duplicates(images)

    # Near duplicates
    near_dupes = []
    if not args.skip_near:
        near_dupes = detect_near_duplicates(images, threshold=args.threshold)

    # Remove exact duplicates if requested
    removed = 0
    if args.remove and exact_dupes:
        print(f"\n  Removing exact duplicates...")
        for h, files in exact_dupes.items():
            # Keep the first file, remove the rest
            for f in files[1:]:
                # Also remove corresponding label
                lbl = COMBINED_DIR / "staging" / "labels" / f"{f.stem}.txt"
                f.unlink()
                if lbl.exists():
                    lbl.unlink()
                removed += 1
        print(f"  Removed {removed} duplicate files")

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "duplicate_detection_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Duplicate Detection Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n")
        f.write(f"- Total images scanned: {len(images)}\n")
        f.write(f"- Exact duplicate groups: {len(exact_dupes)}\n")
        f.write(f"- Exact duplicate files: {sum(len(f) - 1 for f in exact_dupes.values())}\n")
        f.write(f"- Near-duplicate pairs: {len(near_dupes)}\n")
        if args.remove:
            f.write(f"- Files removed: {removed}\n")
        f.write("\n")

        if exact_dupes:
            f.write("## Exact Duplicates\n\n")
            for h, files in list(exact_dupes.items())[:20]:
                f.write(f"### Hash: `{h[:16]}...`\n")
                for fp in files:
                    f.write(f"- `{fp.name}`\n")
                f.write("\n")

        if near_dupes:
            f.write("## Near Duplicates (top 50)\n\n")
            f.write("| File A | File B | Distance |\n|---|---|---|\n")
            for pair in near_dupes[:50]:
                f.write(f"| `{Path(pair['file_a']).name}` | "
                        f"`{Path(pair['file_b']).name}` | {pair['distance']} |\n")

    # Machine-readable output consumed by split_dataset.py so that duplicate
    # clusters are never spread across train/val/test.
    dupe_json = REPORTS_DIR / "duplicate_groups.json"
    with open(dupe_json, "w", encoding="utf-8") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "threshold": args.threshold,
            "exact_groups": [[Path(fp).name for fp in files]
                             for files in exact_dupes.values()],
            "near_pairs": [[Path(pr["file_a"]).name, Path(pr["file_b"]).name,
                            int(pr["distance"])] for pr in near_dupes],
        }, f, indent=2)
    print(f"\n  Duplicate groups (for splitter): {dupe_json}")

    print(f"\n  Report saved: {report_path}")
    print(f"\n{'=' * 60}")
    print(f"Duplicate detection complete!")
    print(f"Next step: python scripts/split_dataset.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
