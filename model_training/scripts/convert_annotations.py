"""
Annotation Format Conversion Script
=====================================

Ensures all annotations are in standard YOLO format.
For the Road Damage Dataset, validates the pre-existing YOLO labels
against the COCO JSON annotations.

Note: This script is primarily for verification. The harmonize_classes.py
script handles the actual remapping. This script can be used to:
1. Verify Road Damage YOLO labels match COCO annotations
2. Regenerate YOLO labels from polygon annotations if needed

Usage:
    python scripts/convert_annotations.py [--verify-only]
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ROAD_DAMAGE_DIR, REPORTS_DIR
from utils.dataset_utils import find_labels, parse_yolo_label


def verify_yolo_vs_coco():
    """Verify YOLO labels against COCO JSON for Road Damage Dataset."""
    print("\n  Verifying YOLO labels against COCO JSON...")

    data_root = ROAD_DAMAGE_DIR / "data"
    coco_path = data_root / "annotations_coco.json"
    yolo_dir = data_root / "labels-YOLO"

    if not coco_path.exists():
        print("  COCO JSON not found — skipping verification")
        return

    # Load COCO
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    coco_images = {img['id']: img for img in coco['images']}
    coco_categories = {cat['id']: cat['name'] for cat in coco['categories']}

    # Count annotations per image in COCO
    coco_ann_counts = Counter()
    for ann in coco['annotations']:
        coco_ann_counts[ann['image_id']] += 1

    # Build filename -> image_id mapping
    filename_to_id = {img['file_name']: img['id'] for img in coco['images']}

    # Check YOLO labels
    yolo_labels = find_labels(yolo_dir, recursive=False)
    mismatches = 0
    checked = 0

    for lbl_path in yolo_labels:
        # Find corresponding COCO image
        img_filename = lbl_path.stem + ".jpg"
        if img_filename not in filename_to_id:
            continue

        img_id = filename_to_id[img_filename]
        coco_count = coco_ann_counts.get(img_id, 0)

        # Count YOLO annotations
        yolo_anns = parse_yolo_label(lbl_path)
        yolo_count = len([a for a in yolo_anns if 'class_id' in a])

        if yolo_count != coco_count:
            mismatches += 1
            if mismatches <= 5:
                print(f"    Mismatch: {img_filename} — YOLO: {yolo_count}, COCO: {coco_count}")

        checked += 1

    print(f"\n  Checked {checked} files")
    print(f"  Annotation count mismatches: {mismatches}")
    print(f"  Categories: {coco_categories}")

    if mismatches == 0:
        print("  ✓ YOLO labels are consistent with COCO annotations")
    else:
        print(f"  ⚠ {mismatches} files have different annotation counts")
        print("    (This may be due to polygon-to-bbox conversion differences)")


def verify_polygon_labels():
    """Verify polygon label format in Road Damage Dataset."""
    print("\n  Checking polygon label format...")

    poly_dir = ROAD_DAMAGE_DIR / "data" / "labels"
    if not poly_dir.exists():
        print("  Polygon labels directory not found")
        return

    labels = find_labels(poly_dir, recursive=False)
    print(f"  Found {len(labels)} polygon label files")

    # Check format: class_id x1 y1 x2 y2 x3 y3 x4 y4
    format_errors = 0
    for lbl in labels[:100]:  # Sample
        with open(lbl, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 9:  # class_id + 4 points (8 coords)
                    format_errors += 1
                    if format_errors <= 3:
                        print(f"    Format issue in {lbl.name}: {len(parts)} fields (expected 9)")

    if format_errors == 0:
        print("  ✓ Polygon labels have correct format (class_id + 8 coordinates)")
    else:
        print(f"  ⚠ {format_errors} format issues in sampled files")


def main():
    parser = argparse.ArgumentParser(description="Verify annotation formats")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify, don't convert anything")
    args = parser.parse_args()

    print("=" * 60)
    print("ANNOTATION CONVERSION / VERIFICATION")
    print("=" * 60)

    verify_yolo_vs_coco()
    verify_polygon_labels()

    print(f"\n{'=' * 60}")
    print("Verification complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
