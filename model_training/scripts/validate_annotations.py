"""
Annotation Validation Script
=============================

Validates all YOLO annotations across the three datasets.

Checks:
- Missing images (labels without corresponding images)
- Missing labels (images without corresponding labels)
- Malformed labels (wrong field count, non-numeric values)
- Invalid class IDs
- Normalized coordinates outside [0, 1]
- Zero-area boxes
- Negative dimensions
- Boxes outside image boundaries
- Empty labels
- Duplicate annotations within a file

Usage:
    python scripts/validate_annotations.py
"""

import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    RAD_DIR, ROAD_DAMAGE_DIR, POTHOLE_DIR, REPORTS_DIR,
    COMBINED_DIR, NUM_CLASSES,
)
from utils.dataset_utils import find_images, find_labels
from utils.annotation_utils import (
    validate_label_file, find_duplicate_annotations, find_image_label_pairs,
)


def validate_dataset(
    name: str,
    image_dirs: list,
    label_dirs: list,
    valid_class_ids: set,
):
    """Validate annotations for a dataset."""
    print(f"\n{'=' * 60}")
    print(f"Validating: {name}")
    print(f"{'=' * 60}")

    all_errors = Counter()
    total_files = 0
    total_annotations = 0
    files_with_errors = 0
    empty_files = 0
    duplicate_annotation_files = 0
    missing_images = 0
    missing_labels = 0
    error_details = []

    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"  SKIP: {img_dir} or {lbl_dir} not found")
            continue

        # Find pairs and mismatches
        pairing = find_image_label_pairs(img_dir, lbl_dir)
        missing_labels += len(pairing['images_without_labels'])
        missing_images += len(pairing['labels_without_images'])

        if pairing['images_without_labels']:
            print(f"  Images without labels in {img_dir.name}: {len(pairing['images_without_labels'])}")

        if pairing['labels_without_images']:
            print(f"  Labels without images in {lbl_dir.name}: {len(pairing['labels_without_images'])}")

        # Validate each label file
        for img_path, lbl_path in pairing['paired']:
            total_files += 1
            result = validate_label_file(lbl_path, valid_class_ids)

            if result['empty']:
                empty_files += 1
                continue

            total_annotations += result['total_lines']

            if result['error_lines'] > 0:
                files_with_errors += 1
                for ann in result['annotations']:
                    if not ann.get('valid', True):
                        for err in ann.get('errors', []):
                            all_errors[err] += 1
                            if len(error_details) < 50:  # Cap details
                                error_details.append({
                                    "file": lbl_path.name,
                                    "error": err,
                                    "raw": ann.get('raw', ''),
                                })

            # Check for duplicate annotations
            dupes = find_duplicate_annotations(lbl_path)
            if dupes:
                duplicate_annotation_files += 1

    # Report
    print(f"\n  --- Validation Summary ---")
    print(f"  Total label files: {total_files}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Empty label files: {empty_files}")
    print(f"  Missing images (labels without images): {missing_images}")
    print(f"  Missing labels (images without labels): {missing_labels}")
    print(f"  Files with errors: {files_with_errors}")
    print(f"  Files with duplicate annotations: {duplicate_annotation_files}")

    if all_errors:
        print(f"\n  Error types:")
        for err, count in all_errors.most_common():
            print(f"    {err}: {count}")

    return {
        "name": name,
        "total_files": total_files,
        "total_annotations": total_annotations,
        "empty_files": empty_files,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "files_with_errors": files_with_errors,
        "duplicate_annotation_files": duplicate_annotation_files,
        "errors": dict(all_errors),
        "error_details": error_details,
    }


def validate_final_splits(results):
    """
    Validate the processed dataset that training actually consumes.

    Raw-dataset validation cannot catch mistakes introduced by harmonization,
    remapping or splitting, so the final splits get checked on their own.
    """
    for split in ("train", "val", "test"):
        img_dir = COMBINED_DIR / "images" / split
        lbl_dir = COMBINED_DIR / "labels" / split
        if not img_dir.exists():
            continue
        results.append(validate_dataset(
            f"FINAL/{split}", [img_dir], [lbl_dir],
            valid_class_ids=set(range(NUM_CLASSES)),
        ))
    return results


def main():
    print("=" * 60)
    print("ANNOTATION VALIDATION — Road Damage AI Pipeline")
    print("=" * 60)

    results = []

    # RAD: has train/valid/test splits
    rad_image_root = RAD_DIR / "images"
    rad_splits = ['train', 'valid', 'test']
    rad_img_dirs = [rad_image_root / s / "images" for s in rad_splits]
    rad_lbl_dirs = [rad_image_root / s / "labels" for s in rad_splits]
    results.append(validate_dataset(
        "RAD", rad_img_dirs, rad_lbl_dirs,
        valid_class_ids={0, 1, 2, 3, 4, 5},
    ))

    # Road Damage: flat structure
    rd_root = ROAD_DAMAGE_DIR / "data"
    results.append(validate_dataset(
        "Road Damage",
        [rd_root / "images"],
        [rd_root / "labels-YOLO"],
        valid_class_ids={0, 1, 2},
    ))

    # Pothole: has train/valid/test splits
    pot_splits = ['train', 'valid', 'test']
    pot_img_dirs = [POTHOLE_DIR / s / "images" for s in pot_splits]
    pot_lbl_dirs = [POTHOLE_DIR / s / "labels" for s in pot_splits]
    results.append(validate_dataset(
        "Pothole", pot_img_dirs, pot_lbl_dirs,
        valid_class_ids={0},
    ))

    # Final processed splits (what training actually reads)
    validate_final_splits(results)

    # Generate report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "annotation_validation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Annotation Validation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for r in results:
            f.write(f"## {r['name']}\n\n")
            f.write(f"| Metric | Count |\n|---|---|\n")
            for key in ['total_files', 'total_annotations', 'empty_files',
                        'missing_images', 'missing_labels', 'files_with_errors',
                        'duplicate_annotation_files']:
                f.write(f"| {key} | {r[key]} |\n")

            if r['errors']:
                f.write(f"\n### Error Types\n\n")
                for err, count in sorted(r['errors'].items(), key=lambda x: -x[1]):
                    f.write(f"- `{err}`: {count}\n")
            f.write("\n")

    print(f"\n{'=' * 60}")
    print(f"Validation report saved: {report_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
