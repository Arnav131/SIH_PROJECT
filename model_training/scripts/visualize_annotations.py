"""
Annotation Visualization Script
=================================

Generates visual previews of annotations for:
1. Each raw source dataset (with original class names)
2. The final merged/harmonized dataset (with unified class names)

Saves preview grids to runs/dataset_preview/

Usage:
    python scripts/visualize_annotations.py
    python scripts/visualize_annotations.py --num-samples 24
"""

import sys
import random
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    RAD_DIR, ROAD_DAMAGE_DIR, POTHOLE_DIR,
    COMBINED_DIR, RUNS_DIR, FINAL_CLASS_NAMES,
)
from utils.dataset_utils import find_images, find_labels
from utils.visualization import (
    save_dataset_preview, DEFAULT_COLORS,
)
from utils.annotation_utils import find_image_label_pairs


def preview_rad(output_dir: Path, num_samples: int):
    """Preview RAD dataset annotations with original class names."""
    print("\n  Previewing RAD dataset...")
    rad_classes = ['HMV', 'LMV', 'Pedestrian', 'RoadDamages', 'SpeedBump', 'UnsurfacedRoad']
    rad_colors = {
        0: (80, 80, 200),    # HMV
        1: (200, 80, 80),    # LMV
        2: (80, 200, 200),   # Pedestrian
        3: (50, 180, 255),   # RoadDamages
        4: (200, 80, 255),   # SpeedBump
        5: (255, 255, 80),   # UnsurfacedRoad
    }

    image_root = RAD_DIR / "images"
    pairs = []
    for split in ['train', 'valid', 'test']:
        img_dir = image_root / split / "images"
        lbl_dir = image_root / split / "labels"
        if img_dir.exists() and lbl_dir.exists():
            result = find_image_label_pairs(img_dir, lbl_dir)
            pairs.extend(result['paired'])

    save_dataset_preview(
        pairs, output_dir / "rad_preview.jpg",
        class_names=rad_classes, colors=rad_colors,
        num_samples=num_samples, title="RAD - Original Annotations",
    )


def preview_road_damage(output_dir: Path, num_samples: int):
    """Preview Road Damage dataset annotations."""
    print("\n  Previewing Road Damage dataset...")
    rd_classes = ['pothole', 'crack', 'manhole']
    rd_colors = {
        0: (80, 80, 255),    # pothole
        1: (255, 180, 80),   # crack
        2: (80, 255, 80),    # manhole
    }

    img_dir = ROAD_DAMAGE_DIR / "data" / "images"
    lbl_dir = ROAD_DAMAGE_DIR / "data" / "labels-YOLO"

    if img_dir.exists() and lbl_dir.exists():
        result = find_image_label_pairs(img_dir, lbl_dir)
        save_dataset_preview(
            result['paired'], output_dir / "road_damage_preview.jpg",
            class_names=rd_classes, colors=rd_colors,
            num_samples=num_samples, title="Road Damage - Original Annotations",
        )


def preview_pothole(output_dir: Path, num_samples: int):
    """Preview Pothole dataset annotations."""
    print("\n  Previewing Pothole dataset...")
    pot_classes = ['pothole']
    pot_colors = {0: (80, 80, 255)}

    pairs = []
    for split in ['train', 'valid', 'test']:
        img_dir = POTHOLE_DIR / split / "images"
        lbl_dir = POTHOLE_DIR / split / "labels"
        if img_dir.exists() and lbl_dir.exists():
            result = find_image_label_pairs(img_dir, lbl_dir)
            pairs.extend(result['paired'])

    save_dataset_preview(
        pairs, output_dir / "pothole_preview.jpg",
        class_names=pot_classes, colors=pot_colors,
        num_samples=num_samples, title="Pothole - Original Annotations",
    )


def preview_combined(output_dir: Path, num_samples: int):
    """Preview the final harmonized dataset."""
    print("\n  Previewing combined/harmonized dataset...")

    for split in ['train', 'val', 'test']:
        img_dir = COMBINED_DIR / "images" / split
        lbl_dir = COMBINED_DIR / "labels" / split

        if not img_dir.exists():
            print(f"    SKIP: {split} — not found")
            continue

        result = find_image_label_pairs(img_dir, lbl_dir)
        save_dataset_preview(
            result['paired'],
            output_dir / f"combined_{split}_preview.jpg",
            class_names=FINAL_CLASS_NAMES,
            colors=DEFAULT_COLORS,
            num_samples=num_samples,
            title=f"Combined Dataset - {split} (Harmonized Classes)",
        )


def main():
    parser = argparse.ArgumentParser(description="Generate annotation previews")
    parser.add_argument("--num-samples", type=int, default=16,
                        help="Number of sample images per preview")
    args = parser.parse_args()

    print("=" * 60)
    print("ANNOTATION VISUALIZATION — Road Damage AI Pipeline")
    print("=" * 60)

    random.seed(42)  # Reproducible samples

    output_dir = RUNS_DIR / "dataset_preview"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Source dataset previews
    preview_rad(output_dir, args.num_samples)
    preview_road_damage(output_dir, args.num_samples)
    preview_pothole(output_dir, args.num_samples)

    # Combined dataset preview
    preview_combined(output_dir, args.num_samples)

    print(f"\n{'=' * 60}")
    print(f"All previews saved to: {output_dir}")
    print(f"{'=' * 60}")
    print(f"\n  ⚠ STOP: Visually inspect the previews before training!")
    print(f"  Verify that bounding boxes align with actual damage features.")
    print(f"\n  If annotations look correct:")
    print(f"    python scripts/train.py")


if __name__ == "__main__":
    main()
