"""
Dataset utilities for the Road Damage AI pipeline.

Provides functions for:
- Path discovery and file listing
- Image metadata reading
- YOLO label parsing
- Video metadata extraction
- Dataset structure detection
"""

import cv2
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter


# ──────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
LABEL_EXTENSIONS = {'.txt'}


def find_files(directory: Path, extensions: Set[str], recursive: bool = True) -> List[Path]:
    """Find all files with given extensions in a directory."""
    files = []
    if not directory.exists():
        return files
    pattern = '**/*' if recursive else '*'
    for ext in extensions:
        files.extend(directory.glob(f'{pattern}{ext}'))
    return sorted(files)


def find_images(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all image files in a directory."""
    return find_files(directory, IMAGE_EXTENSIONS, recursive)


def find_labels(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all label (.txt) files in a directory."""
    return find_files(directory, LABEL_EXTENSIONS, recursive)


def find_videos(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all video files in a directory."""
    return find_files(directory, VIDEO_EXTENSIONS, recursive)


# ──────────────────────────────────────────────────────────────
# Image metadata
# ──────────────────────────────────────────────────────────────

def get_image_dimensions(image_path: Path) -> Optional[Tuple[int, int]]:
    """Get (width, height) of an image without loading full data."""
    try:
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            return (w, h)
    except Exception:
        pass
    return None


def sample_image_dimensions(image_paths: List[Path], max_samples: int = 100) -> Dict[str, any]:
    """Sample image dimensions from a list of image paths."""
    import random
    samples = random.sample(image_paths, min(max_samples, len(image_paths)))
    widths, heights = [], []
    for p in samples:
        dims = get_image_dimensions(p)
        if dims:
            widths.append(dims[0])
            heights.append(dims[1])
    if not widths:
        return {"min_w": 0, "max_w": 0, "min_h": 0, "max_h": 0, "samples": 0}
    return {
        "min_w": min(widths), "max_w": max(widths),
        "min_h": min(heights), "max_h": max(heights),
        "median_w": int(np.median(widths)), "median_h": int(np.median(heights)),
        "unique_resolutions": len(set(zip(widths, heights))),
        "samples": len(widths),
    }


def is_image_corrupted(image_path: Path) -> bool:
    """Check if an image file is corrupted (cannot be read by OpenCV)."""
    try:
        img = cv2.imread(str(image_path))
        return img is None
    except Exception:
        return True


# ──────────────────────────────────────────────────────────────
# YOLO label parsing
# ──────────────────────────────────────────────────────────────

def parse_yolo_label(label_path: Path) -> List[Dict]:
    """
    Parse a YOLO format label file.
    Returns list of dicts with keys: class_id, x_center, y_center, width, height
    """
    annotations = []
    if not label_path.exists():
        return annotations
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    annotations.append({
                        "line_num": line_num,
                        "error": f"Too few fields ({len(parts)})",
                        "raw": line,
                    })
                    continue
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append({
                        "line_num": line_num,
                        "class_id": class_id,
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                    })
                except (ValueError, IndexError) as e:
                    annotations.append({
                        "line_num": line_num,
                        "error": str(e),
                        "raw": line,
                    })
    except Exception as e:
        return [{"error": f"Cannot read file: {e}"}]
    return annotations


def get_class_distribution(label_paths: List[Path]) -> Counter:
    """Count class occurrences across all label files."""
    counter = Counter()
    for lp in label_paths:
        anns = parse_yolo_label(lp)
        for a in anns:
            if "class_id" in a:
                counter[a["class_id"]] += 1
    return counter


# ──────────────────────────────────────────────────────────────
# Video metadata
# ──────────────────────────────────────────────────────────────

def get_video_metadata(video_path: Path) -> Optional[Dict]:
    """Extract metadata from a video file."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return {
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": round(duration, 2),
            "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# File hashing
# ──────────────────────────────────────────────────────────────

def file_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────
# Dataset structure detection
# ──────────────────────────────────────────────────────────────

def detect_dataset_structure(root: Path) -> Dict:
    """Detect the structure of a dataset directory."""
    images = find_images(root)
    labels = find_labels(root)
    videos = find_videos(root)

    # Detect splits
    splits = {}
    for split_name in ['train', 'valid', 'val', 'test']:
        split_images = [p for p in images if split_name in p.parts]
        split_labels = [p for p in labels if split_name in p.parts]
        if split_images or split_labels:
            splits[split_name] = {
                "images": len(split_images),
                "labels": len(split_labels),
            }

    # Detect yaml configs
    yaml_files = list(root.rglob('*.yaml')) + list(root.rglob('*.yml'))
    json_files = list(root.rglob('*.json'))

    return {
        "root": str(root),
        "total_images": len(images),
        "total_labels": len(labels),
        "total_videos": len(videos),
        "splits": splits,
        "yaml_files": [str(y) for y in yaml_files],
        "json_files": [str(j) for j in json_files],
    }


# ──────────────────────────────────────────────────────────────
# Video frame grouping for RAD (leakage prevention)
# ──────────────────────────────────────────────────────────────

def extract_video_group(filename: str) -> str:
    """
    Extract the video source group from a RAD filename.
    
    RAD filenames follow pattern: XX_DDMMYYYY_mp4-N_jpg.rf.HASH.jpg
    or: XX_DD-MM-YYYY_mp4-N_jpg.rf.HASH.jpg
    
    The group is the video identifier (XX_DDMMYYYY or XX_DD-MM-YYYY)
    so all frames from the same video stay in the same split.
    """
    # Remove extension and hash parts
    stem = Path(filename).stem
    # Try to find the _mp4- separator
    if '_mp4-' in stem:
        group = stem.split('_mp4-')[0]
        return group
    # Fallback: use the whole stem
    return stem
