"""
Class Harmonization Script
===========================

Reads class_mapping.yaml and creates the unified processed dataset by:
1. Remapping class IDs from each source dataset to the final taxonomy
2. Filtering out excluded classes (vehicles, pedestrians from RAD)
3. Copying images + remapped labels to data/processed/road_damage_combined/
4. Skipping images that have NO road-damage annotations after filtering

Raw datasets are NEVER modified.

Usage:
    python scripts/harmonize_classes.py
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    RAD_DIR, ROAD_DAMAGE_DIR, POTHOLE_DIR,
    COMBINED_DIR, CONFIGS_DIR, REPORTS_DIR,
    load_yaml, FINAL_CLASS_NAMES,
)
from utils.dataset_utils import find_images, find_labels
from utils.annotation_utils import remap_label_file


def process_rad(class_config: dict, stats: dict):
    """Process RAD dataset: remap road-damage classes, exclude vehicles/pedestrians."""
    print("\n  Processing RAD dataset...")

    config = class_config['rad']
    class_map = {int(k): v for k, v in config['class_map'].items()}
    exclude_ids = set(config['exclude_class_ids'])

    image_root = RAD_DIR / "images"
    processed = 0
    kept = 0
    excluded_images = 0
    annotation_stats = Counter()

    for split in ['train', 'valid', 'test']:
        img_dir = image_root / split / "images"
        lbl_dir = image_root / split / "labels"

        if not img_dir.exists():
            print(f"    SKIP: {img_dir} not found")
            continue

        images = find_images(img_dir, recursive=False)
        print(f"    {split}: {len(images)} images")

        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            processed += 1

            if not lbl_path.exists():
                excluded_images += 1
                continue

            # We don't assign to final splits yet — that happens in split_dataset.py
            # For now, put everything in a staging area
            dst_img_dir = COMBINED_DIR / "staging" / "images"
            dst_lbl_dir = COMBINED_DIR / "staging" / "labels"
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            # Prefix to avoid filename collisions across datasets
            prefix = "rad_"
            dst_img = dst_img_dir / f"{prefix}{img_path.name}"
            dst_lbl = dst_lbl_dir / f"{prefix}{img_path.stem}.txt"

            # Remap labels
            label_stats = remap_label_file(lbl_path, dst_lbl, class_map, exclude_ids)

            if label_stats['kept'] > 0:
                shutil.copy2(str(img_path), str(dst_img))
                kept += 1
                for cls_id in class_map.values():
                    # Count the actual annotations we kept
                    pass
            else:
                excluded_images += 1
                # Remove empty label if created
                if dst_lbl.exists():
                    dst_lbl.unlink()

            annotation_stats['kept'] += label_stats['kept']
            annotation_stats['excluded'] += label_stats['excluded']
            annotation_stats['errors'] += label_stats['errors']

    stats['rad'] = {
        'processed': processed,
        'kept': kept,
        'excluded_images': excluded_images,
        'annotations': dict(annotation_stats),
    }
    print(f"    Result: {kept}/{processed} images kept, "
          f"{annotation_stats['kept']} annotations kept, "
          f"{annotation_stats['excluded']} excluded")


def process_road_damage(class_config: dict, stats: dict):
    """Process Road Damage Dataset: direct mapping, no exclusions."""
    print("\n  Processing Road Damage dataset...")

    config = class_config['road_damage']
    class_map = {int(k): v for k, v in config['class_map'].items()}

    data_root = ROAD_DAMAGE_DIR / "data"
    img_dir = data_root / "images"
    lbl_dir = data_root / "labels-YOLO"

    images = find_images(img_dir, recursive=False)
    print(f"    Images: {len(images)}")

    dst_img_dir = COMBINED_DIR / "staging" / "images"
    dst_lbl_dir = COMBINED_DIR / "staging" / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    annotation_stats = Counter()

    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        prefix = "rd_"
        dst_img = dst_img_dir / f"{prefix}{img_path.name}"
        dst_lbl = dst_lbl_dir / f"{prefix}{img_path.stem}.txt"

        label_stats = remap_label_file(lbl_path, dst_lbl, class_map)

        if label_stats['kept'] > 0:
            shutil.copy2(str(img_path), str(dst_img))
            kept += 1

        annotation_stats['kept'] += label_stats['kept']
        annotation_stats['errors'] += label_stats['errors']

    stats['road_damage'] = {
        'processed': len(images),
        'kept': kept,
        'annotations': dict(annotation_stats),
    }
    print(f"    Result: {kept}/{len(images)} images kept, "
          f"{annotation_stats['kept']} annotations kept")


def process_pothole(class_config: dict, stats: dict):
    """Process Pothole Dataset: direct mapping (0 -> 0)."""
    print("\n  Processing Pothole dataset...")

    config = class_config['pothole_dataset']
    class_map = {int(k): v for k, v in config['class_map'].items()}

    kept = 0
    total = 0
    annotation_stats = Counter()

    dst_img_dir = COMBINED_DIR / "staging" / "images"
    dst_lbl_dir = COMBINED_DIR / "staging" / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'valid', 'test']:
        img_dir = POTHOLE_DIR / split / "images"
        lbl_dir = POTHOLE_DIR / split / "labels"

        if not img_dir.exists():
            continue

        images = find_images(img_dir, recursive=False)
        total += len(images)
        print(f"    {split}: {len(images)} images")

        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            prefix = "pot_"
            dst_img = dst_img_dir / f"{prefix}{img_path.name}"
            dst_lbl = dst_lbl_dir / f"{prefix}{img_path.stem}.txt"

            label_stats = remap_label_file(lbl_path, dst_lbl, class_map)

            if label_stats['kept'] > 0:
                shutil.copy2(str(img_path), str(dst_img))
                kept += 1

            annotation_stats['kept'] += label_stats['kept']
            annotation_stats['errors'] += label_stats['errors']

    stats['pothole'] = {
        'processed': total,
        'kept': kept,
        'annotations': dict(annotation_stats),
    }
    print(f"    Result: {kept}/{total} images kept, "
          f"{annotation_stats['kept']} annotations kept")


def verify_staging():
    """Verify the staging area contents."""
    staging_img = COMBINED_DIR / "staging" / "images"
    staging_lbl = COMBINED_DIR / "staging" / "labels"

    images = find_images(staging_img, recursive=False)
    labels = find_labels(staging_lbl, recursive=False)

    print(f"\n  Staging area verification:")
    print(f"    Images: {len(images)}")
    print(f"    Labels: {len(labels)}")

    # Class distribution in harmonized labels
    from utils.dataset_utils import get_class_distribution
    class_dist = get_class_distribution(labels)
    print(f"\n  Harmonized class distribution:")
    for cls_id in sorted(class_dist.keys()):
        name = FINAL_CLASS_NAMES[cls_id] if cls_id < len(FINAL_CLASS_NAMES) else f"class_{cls_id}"
        print(f"    {cls_id} ({name}): {class_dist[cls_id]}")
    print(f"    TOTAL: {sum(class_dist.values())}")

    return len(images), len(labels)


def main():
    print("=" * 60)
    print("CLASS HARMONIZATION — Road Damage AI Pipeline")
    print("=" * 60)

    # Load class mapping config
    mapping_path = CONFIGS_DIR / "class_mapping.yaml"
    class_config = load_yaml(mapping_path)
    print(f"\n  Loaded class mapping from: {mapping_path}")
    print(f"  Final taxonomy: {class_config['final_taxonomy']['classes']}")

    # Clean staging area if it exists
    staging = COMBINED_DIR / "staging"
    if staging.exists():
        print(f"\n  Cleaning existing staging area...")
        shutil.rmtree(str(staging))

    # Process each dataset
    stats = {}
    process_rad(class_config, stats)
    process_road_damage(class_config, stats)
    process_pothole(class_config, stats)

    # Verify
    num_images, num_labels = verify_staging()

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "harmonization_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Class Harmonization Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Final Staging Area\n")
        f.write(f"- Images: {num_images}\n")
        f.write(f"- Labels: {num_labels}\n\n")
        for ds_name, ds_stats in stats.items():
            f.write(f"## {ds_name}\n")
            f.write(f"- Processed: {ds_stats['processed']}\n")
            f.write(f"- Kept: {ds_stats['kept']}\n")
            f.write(f"- Annotations kept: {ds_stats['annotations'].get('kept', 0)}\n")
            if 'excluded_images' in ds_stats:
                f.write(f"- Excluded images: {ds_stats['excluded_images']}\n")
            f.write("\n")

    print(f"\n  Report saved: {report_path}")
    print(f"\n{'=' * 60}")
    print(f"Harmonization complete!")
    print(f"Next step: python scripts/detect_duplicates.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
