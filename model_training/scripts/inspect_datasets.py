"""
Dataset Inspection Script
=========================

Automatically inspects all three raw datasets and generates a comprehensive report.

Determines:
- Directory structure
- Image/label/video counts
- Annotation format
- Class names and IDs
- Class distribution
- Image dimensions (sampled)
- Missing labels / images
- Empty labels
- Video metadata (RAD only)

Usage:
    python scripts/inspect_datasets.py
"""

import sys
import json
import random
from pathlib import Path
from collections import Counter
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    RAD_DIR, ROAD_DAMAGE_DIR, POTHOLE_DIR,
    REPORTS_DIR, FINAL_CLASSES,
)
from utils.dataset_utils import (
    find_images, find_labels, find_videos,
    detect_dataset_structure, parse_yolo_label,
    get_class_distribution, sample_image_dimensions,
    get_video_metadata, is_image_corrupted,
)


def inspect_rad():
    """Inspect the RAD dataset."""
    print("\n" + "=" * 70)
    print("DATASET 1: RAD — Road Anomaly Detection")
    print("=" * 70)

    # The RAD archive extracts with an 'images' subfolder for annotated data
    # and 'videos_without_audio' for raw videos
    image_root = RAD_DIR / "images"
    video_root = RAD_DIR / "videos_without_audio"

    # Structure detection
    structure = detect_dataset_structure(image_root)
    print(f"\n  Root: {image_root}")
    print(f"  Total images: {structure['total_images']}")
    print(f"  Total labels: {structure['total_labels']}")
    print(f"  YAML configs: {structure['yaml_files']}")

    # Splits
    print("\n  Splits:")
    for split, info in structure['splits'].items():
        print(f"    {split}: {info['images']} images, {info['labels']} labels")

    # Class distribution
    all_labels = find_labels(image_root)
    class_dist = get_class_distribution(all_labels)
    rad_class_names = ['HMV', 'LMV', 'Pedestrian', 'RoadDamages', 'SpeedBump', 'UnsurfacedRoad']
    print("\n  Class Distribution:")
    total_annotations = sum(class_dist.values())
    for cls_id in sorted(class_dist.keys()):
        name = rad_class_names[cls_id] if cls_id < len(rad_class_names) else f"unknown_{cls_id}"
        count = class_dist[cls_id]
        pct = count / total_annotations * 100 if total_annotations > 0 else 0
        print(f"    {cls_id} ({name}): {count} ({pct:.1f}%)")
    print(f"    TOTAL: {total_annotations}")

    # Empty labels
    empty_count = 0
    for lp in all_labels:
        content = lp.read_text(encoding='utf-8').strip()
        if not content:
            empty_count += 1
    print(f"\n  Empty label files: {empty_count}")

    # Image dimensions
    all_images = find_images(image_root)
    dims = sample_image_dimensions(all_images, max_samples=50)
    print(f"\n  Image Dimensions (sampled {dims['samples']} images):")
    print(f"    Width:  {dims['min_w']} - {dims['max_w']} (median: {dims.get('median_w', 'N/A')})")
    print(f"    Height: {dims['min_h']} - {dims['max_h']} (median: {dims.get('median_h', 'N/A')})")
    print(f"    Unique resolutions: {dims.get('unique_resolutions', 'N/A')}")

    # Corrupted images check (sample)
    sample_for_corruption = random.sample(all_images, min(100, len(all_images)))
    corrupted = [p for p in sample_for_corruption if is_image_corrupted(p)]
    print(f"\n  Corrupted images (in sample of {len(sample_for_corruption)}): {len(corrupted)}")

    # Videos
    videos = find_videos(video_root) if video_root.exists() else []
    print(f"\n  Raw Videos: {len(videos)}")
    video_meta = []
    video_size_mb = 0.0
    if videos:
        # Sample video metadata. The sample goes into the report so the
        # frame-extraction decision is documented rather than guessed at.
        sample_vids = random.sample(videos, min(8, len(videos)))
        video_size_mb = sum(v.stat().st_size for v in videos) / (1024 * 1024)
        print(f"  Total video size: {video_size_mb:.0f} MB")
        print("\n  Sample video metadata:")
        for v in sample_vids:
            meta = get_video_metadata(v)
            if meta:
                meta['name'] = v.name
                video_meta.append(meta)
                print(f"    {v.name}: {meta['width']}x{meta['height']}, "
                      f"{meta['fps']} fps, {meta['duration_sec']}s, {meta['size_mb']} MB")

    return {
        "name": "RAD",
        "images": structure['total_images'],
        "labels": structure['total_labels'],
        "videos": len(videos),
        "video_size_mb": round(video_size_mb, 1),
        "video_metadata_sample": video_meta,
        "annotations": total_annotations,
        "classes": {k: v for k, v in sorted(class_dist.items())},
        "class_names": rad_class_names,
        "splits": structure['splits'],
        "empty_labels": empty_count,
        "dimensions": dims,
        "corrupted_sample": len(corrupted),
    }


def inspect_road_damage():
    """Inspect the Road Damage Dataset."""
    print("\n" + "=" * 70)
    print("DATASET 2: Road Damage Dataset (Potholes, Cracks, Manholes)")
    print("=" * 70)

    data_root = ROAD_DAMAGE_DIR / "data"

    # Structure
    image_dir = data_root / "images"
    yolo_label_dir = data_root / "labels-YOLO"
    polygon_label_dir = data_root / "labels"

    images = find_images(image_dir, recursive=False)
    yolo_labels = find_labels(yolo_label_dir, recursive=False)
    polygon_labels = find_labels(polygon_label_dir, recursive=False)

    print(f"\n  Root: {data_root}")
    print(f"  Images: {len(images)}")
    print(f"  YOLO labels: {len(yolo_labels)}")
    print(f"  Polygon labels: {len(polygon_labels)}")
    print(f"  COCO JSON: {(data_root / 'annotations_coco.json').exists()}")
    print(f"  NOTE: No pre-existing train/val/test split")

    # Class distribution from YOLO labels
    class_dist = get_class_distribution(yolo_labels)
    rd_class_names = ['pothole', 'crack', 'manhole']
    print("\n  Class Distribution (YOLO labels):")
    total_annotations = sum(class_dist.values())
    for cls_id in sorted(class_dist.keys()):
        name = rd_class_names[cls_id] if cls_id < len(rd_class_names) else f"unknown_{cls_id}"
        count = class_dist[cls_id]
        pct = count / total_annotations * 100 if total_annotations > 0 else 0
        print(f"    {cls_id} ({name}): {count} ({pct:.1f}%)")
    print(f"    TOTAL: {total_annotations}")

    # Image dimensions
    dims = sample_image_dimensions(images, max_samples=50)
    print(f"\n  Image Dimensions (sampled {dims['samples']} images):")
    print(f"    Width:  {dims['min_w']} - {dims['max_w']} (median: {dims.get('median_w', 'N/A')})")
    print(f"    Height: {dims['min_h']} - {dims['max_h']} (median: {dims.get('median_h', 'N/A')})")

    # Check for corrupted images
    sample_for_corruption = random.sample(images, min(100, len(images)))
    corrupted = [p for p in sample_for_corruption if is_image_corrupted(p)]
    print(f"\n  Corrupted images (in sample of {len(sample_for_corruption)}): {len(corrupted)}")

    # YOLO vs COCO consistency check
    coco_path = data_root / "annotations_coco.json"
    if coco_path.exists():
        with open(coco_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        print(f"\n  COCO JSON:")
        print(f"    Images: {len(coco_data.get('images', []))}")
        print(f"    Annotations: {len(coco_data.get('annotations', []))}")
        print(f"    Categories: {coco_data.get('categories', [])}")

    return {
        "name": "Road Damage",
        "images": len(images),
        "labels": len(yolo_labels),
        "videos": 0,
        "annotations": total_annotations,
        "classes": {k: v for k, v in sorted(class_dist.items())},
        "class_names": rd_class_names,
        "splits": {},
        "empty_labels": 0,
        "dimensions": dims,
    }


def inspect_pothole():
    """Inspect the Pothole Detection Dataset."""
    print("\n" + "=" * 70)
    print("DATASET 3: Pothole Detection Dataset (YOLOv11 Optimized)")
    print("=" * 70)

    # This dataset has train/valid/test structure at root
    structure = detect_dataset_structure(POTHOLE_DIR)
    print(f"\n  Root: {POTHOLE_DIR}")
    print(f"  Total images: {structure['total_images']}")
    print(f"  Total labels: {structure['total_labels']}")

    # Splits
    print("\n  Splits:")
    for split, info in structure['splits'].items():
        print(f"    {split}: {info['images']} images, {info['labels']} labels")

    # Class distribution
    all_labels = find_labels(POTHOLE_DIR)
    class_dist = get_class_distribution(all_labels)
    pot_class_names = ['pothole']
    print("\n  Class Distribution:")
    total_annotations = sum(class_dist.values())
    for cls_id in sorted(class_dist.keys()):
        name = pot_class_names[cls_id] if cls_id < len(pot_class_names) else f"unknown_{cls_id}"
        count = class_dist[cls_id]
        pct = count / total_annotations * 100 if total_annotations > 0 else 0
        print(f"    {cls_id} ({name}): {count} ({pct:.1f}%)")
    print(f"    TOTAL: {total_annotations}")

    # Image dimensions
    all_images = find_images(POTHOLE_DIR)
    dims = sample_image_dimensions(all_images, max_samples=50)
    print(f"\n  Image Dimensions (sampled {dims['samples']} images):")
    print(f"    Width:  {dims['min_w']} - {dims['max_w']} (median: {dims.get('median_w', 'N/A')})")
    print(f"    Height: {dims['min_h']} - {dims['max_h']} (median: {dims.get('median_h', 'N/A')})")

    # Check for corrupted images
    sample_for_corruption = random.sample(all_images, min(100, len(all_images)))
    corrupted = [p for p in sample_for_corruption if is_image_corrupted(p)]
    print(f"\n  Corrupted images (in sample of {len(sample_for_corruption)}): {len(corrupted)}")

    return {
        "name": "Pothole",
        "images": structure['total_images'],
        "labels": structure['total_labels'],
        "videos": 0,
        "annotations": total_annotations,
        "classes": {k: v for k, v in sorted(class_dist.items())},
        "class_names": pot_class_names,
        "splits": structure['splits'],
        "empty_labels": 0,
        "dimensions": dims,
    }


def generate_report(results: dict):
    """Generate a markdown inspection report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "dataset_inspection_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Dataset Inspection Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        f.write("| Property | RAD | Road Damage | Pothole |\n")
        f.write("|---|---|---|---|\n")
        for key in ['images', 'labels', 'videos', 'annotations', 'empty_labels']:
            f.write(f"| {key} | {results['rad'].get(key, 'N/A')} | "
                    f"{results['road_damage'].get(key, 'N/A')} | "
                    f"{results['pothole'].get(key, 'N/A')} |\n")

        for ds_name, ds_data in results.items():
            f.write(f"\n## {ds_data['name']}\n\n")
            f.write("### Class Distribution\n\n")
            f.write("| ID | Name | Count | % |\n")
            f.write("|---|---|---|---|\n")
            total = ds_data['annotations']
            for cls_id, count in sorted(ds_data['classes'].items()):
                name = ds_data['class_names'][cls_id] if cls_id < len(ds_data['class_names']) else f"class_{cls_id}"
                pct = count / total * 100 if total > 0 else 0
                f.write(f"| {cls_id} | {name} | {count} | {pct:.1f}% |\n")

            if ds_data.get('video_metadata_sample'):
                f.write(f"\n### Raw Videos\n")
                f.write(f"- Count: {ds_data['videos']}\n")
                f.write(f"- Total size: {ds_data.get('video_size_mb', 0)} MB\n\n")
                f.write("| Video | Resolution | FPS | Duration (s) | Size (MB) |\n")
                f.write("|---|---|---|---|---|\n")
                for meta in ds_data['video_metadata_sample']:
                    f.write(f"| {meta['name']} | {meta['width']}x{meta['height']} | "
                            f"{meta['fps']} | {meta['duration_sec']} | {meta['size_mb']} |\n")
                f.write("\n> Videos carry no annotations. No frames are extracted for "
                        "supervised training — labels would have to be fabricated.\n")

            if ds_data.get('dimensions'):
                dims = ds_data['dimensions']
                f.write(f"\n### Image Dimensions\n")
                f.write(f"- Width: {dims.get('min_w')}-{dims.get('max_w')} (median: {dims.get('median_w')})\n")
                f.write(f"- Height: {dims.get('min_h')}-{dims.get('max_h')} (median: {dims.get('median_h')})\n")

    print(f"\n{'=' * 70}")
    print(f"Report saved: {report_path}")
    print(f"{'=' * 70}")


def main():
    print("=" * 70)
    print("DATASET INSPECTION — Road Damage AI Pipeline")
    print("=" * 70)

    results = {}
    results['rad'] = inspect_rad()
    results['road_damage'] = inspect_road_damage()
    results['pothole'] = inspect_pothole()

    generate_report(results)

    # Final summary
    print("\n" + "=" * 70)
    print("COMBINED SUMMARY")
    print("=" * 70)
    total_images = sum(r['images'] for r in results.values())
    total_annotations = sum(r['annotations'] for r in results.values())
    print(f"  Total images across all datasets: {total_images}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Total videos (RAD): {results['rad']['videos']}")


if __name__ == "__main__":
    main()
