"""
Annotation utilities for the Road Damage AI pipeline.

Provides functions for:
- YOLO annotation validation
- Annotation format conversion
- Coordinate bounds checking
- Annotation remapping
"""

import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter


# ──────────────────────────────────────────────────────────────
# YOLO annotation validation
# ──────────────────────────────────────────────────────────────

def validate_yolo_annotation(line: str, valid_class_ids: set = None) -> Dict:
    """
    Validate a single YOLO annotation line.
    Returns a dict with 'valid' bool and any 'errors' found.
    """
    result = {"valid": True, "errors": [], "raw": line.strip()}
    line = line.strip()

    if not line:
        result["valid"] = False
        result["errors"].append("empty_line")
        return result

    parts = line.split()

    # Check field count
    if len(parts) < 5:
        result["valid"] = False
        result["errors"].append(f"too_few_fields:{len(parts)}")
        return result

    # Parse values
    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
    except ValueError as e:
        result["valid"] = False
        result["errors"].append(f"parse_error:{e}")
        return result

    # Validate class ID
    if class_id < 0:
        result["valid"] = False
        result["errors"].append(f"negative_class_id:{class_id}")

    if valid_class_ids is not None and class_id not in valid_class_ids:
        result["valid"] = False
        result["errors"].append(f"invalid_class_id:{class_id}")

    # Validate coordinates (normalized 0-1)
    for name, val in [("x_center", x_center), ("y_center", y_center),
                      ("width", width), ("height", height)]:
        if val < 0:
            result["valid"] = False
            result["errors"].append(f"negative_{name}:{val}")
        if val > 1.0:
            # Allow small tolerance for floating point
            if val > 1.01:
                result["valid"] = False
                result["errors"].append(f"{name}_exceeds_1:{val}")

    # Zero-area box
    if width <= 0 or height <= 0:
        result["valid"] = False
        result["errors"].append(f"zero_area_box:w={width},h={height}")

    # Box extends outside image
    if x_center - width / 2 < -0.01 or x_center + width / 2 > 1.01:
        result["errors"].append(f"x_outside_bounds")
    if y_center - height / 2 < -0.01 or y_center + height / 2 > 1.01:
        result["errors"].append(f"y_outside_bounds")

    result["class_id"] = class_id
    result["bbox"] = (x_center, y_center, width, height)
    return result


def validate_label_file(label_path: Path, valid_class_ids: set = None) -> Dict:
    """
    Validate an entire YOLO label file.
    Returns detailed validation results.
    """
    result = {
        "file": str(label_path),
        "exists": label_path.exists(),
        "annotations": [],
        "total_lines": 0,
        "valid_lines": 0,
        "error_lines": 0,
        "empty": False,
        "errors": [],
    }

    if not label_path.exists():
        result["errors"].append("file_not_found")
        return result

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        result["errors"].append(f"read_error:{e}")
        return result

    # Filter out empty lines for "empty file" check
    non_empty_lines = [l for l in lines if l.strip()]
    result["total_lines"] = len(non_empty_lines)

    if not non_empty_lines:
        result["empty"] = True
        return result

    for line in non_empty_lines:
        validation = validate_yolo_annotation(line, valid_class_ids)
        result["annotations"].append(validation)
        if validation["valid"]:
            result["valid_lines"] += 1
        else:
            result["error_lines"] += 1

    return result


# ──────────────────────────────────────────────────────────────
# Annotation remapping
# ──────────────────────────────────────────────────────────────

def remap_label_file(
    src_label: Path,
    dst_label: Path,
    class_map: Dict[int, int],
    exclude_classes: set = None,
) -> Dict:
    """
    Read a YOLO label file, remap class IDs, and write to destination.

    Args:
        src_label: Source label file path
        dst_label: Destination label file path
        class_map: Mapping from original class ID to new class ID
        exclude_classes: Set of original class IDs to exclude

    Returns:
        Dict with stats: kept, excluded, errors
    """
    stats = {"kept": 0, "excluded": 0, "errors": 0}
    output_lines = []

    if not src_label.exists():
        return stats

    with open(src_label, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                stats["errors"] += 1
                continue

            try:
                orig_class = int(parts[0])
            except ValueError:
                stats["errors"] += 1
                continue

            # Skip excluded classes
            if exclude_classes and orig_class in exclude_classes:
                stats["excluded"] += 1
                continue

            # Remap class ID
            if orig_class not in class_map:
                stats["errors"] += 1
                continue

            new_class = class_map[orig_class]
            new_line = f"{new_class} {' '.join(parts[1:5])}"
            output_lines.append(new_line)
            stats["kept"] += 1

    # Only write if there are annotations to keep
    if output_lines:
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_label, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines) + '\n')

    return stats


def copy_image_with_label(
    src_image: Path,
    src_label: Path,
    dst_image_dir: Path,
    dst_label_dir: Path,
    class_map: Dict[int, int],
    exclude_classes: set = None,
    prefix: str = "",
) -> Dict:
    """
    Copy an image and its remapped label to the destination.

    Args:
        src_image: Source image path
        src_label: Source label path
        dst_image_dir: Destination image directory
        dst_label_dir: Destination label directory
        class_map: Class ID mapping
        exclude_classes: Classes to exclude
        prefix: Optional prefix for filenames to avoid collisions

    Returns:
        Dict with operation stats
    """
    stats = {"copied": False, "label_stats": {}}

    # Determine destination filenames
    img_name = f"{prefix}{src_image.name}" if prefix else src_image.name
    lbl_name = f"{prefix}{src_label.stem}.txt" if prefix else f"{src_label.stem}.txt"

    dst_image = dst_image_dir / img_name
    dst_label = dst_label_dir / lbl_name

    # Remap label
    label_stats = remap_label_file(src_label, dst_label, class_map, exclude_classes)
    stats["label_stats"] = label_stats

    # Only copy image if label has annotations
    if label_stats["kept"] > 0:
        dst_image_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_image), str(dst_image))
        stats["copied"] = True

    return stats


# ──────────────────────────────────────────────────────────────
# Duplicate annotation detection
# ──────────────────────────────────────────────────────────────

def find_duplicate_annotations(label_path: Path) -> List[Tuple[int, int]]:
    """
    Find duplicate annotation lines within a single label file.
    Returns list of (line_a, line_b) pairs that are duplicates.
    """
    duplicates = []
    lines = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)

    seen = {}
    for i, line in enumerate(lines):
        if line in seen:
            duplicates.append((seen[line], i))
        else:
            seen[line] = i

    return duplicates


# ──────────────────────────────────────────────────────────────
# Image-label pairing
# ──────────────────────────────────────────────────────────────

def find_image_label_pairs(
    image_dir: Path,
    label_dir: Path,
) -> Dict:
    """
    Find paired and unpaired images/labels.

    Returns dict with:
        paired: list of (image_path, label_path)
        images_without_labels: list of image paths
        labels_without_images: list of label paths
    """
    from .dataset_utils import find_images, find_labels

    images = {p.stem: p for p in find_images(image_dir, recursive=False)}
    labels = {p.stem: p for p in find_labels(label_dir, recursive=False)}

    image_stems = set(images.keys())
    label_stems = set(labels.keys())

    paired_stems = image_stems & label_stems
    images_only = image_stems - label_stems
    labels_only = label_stems - image_stems

    return {
        "paired": [(images[s], labels[s]) for s in sorted(paired_stems)],
        "images_without_labels": [images[s] for s in sorted(images_only)],
        "labels_without_images": [labels[s] for s in sorted(labels_only)],
    }
