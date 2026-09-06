"""
Visualization utilities for the Road Damage AI pipeline.

Provides functions for:
- Drawing bounding boxes with class labels
- Creating annotation preview grids
- Generating dataset comparison visuals
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────────
# Default class names and colors
# ──────────────────────────────────────────────────────────────

DEFAULT_CLASS_NAMES = [
    "pothole", "crack", "manhole",
    "road_damage", "speed_bump", "unsurfaced_road",
]

DEFAULT_COLORS = {
    0: (80, 80, 255),     # pothole — red (BGR)
    1: (255, 180, 80),    # crack — blue (BGR)
    2: (80, 255, 80),     # manhole — green (BGR)
    3: (50, 180, 255),    # road_damage — orange (BGR)
    4: (255, 80, 200),    # speed_bump — purple (BGR)
    5: (80, 255, 255),    # unsurfaced_road — yellow (BGR)
}


# ──────────────────────────────────────────────────────────────
# Draw bounding boxes on an image
# ──────────────────────────────────────────────────────────────

def draw_yolo_boxes(
    image: np.ndarray,
    annotations: List[Dict],
    class_names: List[str] = None,
    colors: Dict[int, Tuple] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw YOLO bounding boxes on an image.

    Args:
        image: BGR image as numpy array
        annotations: List of dicts with class_id, x_center, y_center, width, height
        class_names: List mapping class_id to name
        colors: Dict mapping class_id to BGR color tuple
        thickness: Box line thickness
        font_scale: Text font scale

    Returns:
        Image with boxes drawn
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES
    if colors is None:
        colors = DEFAULT_COLORS

    img = image.copy()
    h, w = img.shape[:2]

    for ann in annotations:
        if "error" in ann and "class_id" not in ann:
            continue

        cls_id = ann["class_id"]
        cx = ann["x_center"] * w
        cy = ann["y_center"] * h
        bw = ann["width"] * w
        bh = ann["height"] * h

        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        color = colors.get(cls_id, (200, 200, 200))
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        label = f"{cls_name}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(img, (x1, y1 - label_h - 8), (x1 + label_w + 4, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )

    return img


# ──────────────────────────────────────────────────────────────
# Draw boxes from a YOLO label file
# ──────────────────────────────────────────────────────────────

def visualize_image_with_labels(
    image_path: Path,
    label_path: Path,
    class_names: List[str] = None,
    colors: Dict[int, Tuple] = None,
) -> Optional[np.ndarray]:
    """
    Load an image and its YOLO labels, draw boxes, return annotated image.
    """
    from .dataset_utils import parse_yolo_label

    img = cv2.imread(str(image_path))
    if img is None:
        return None

    annotations = parse_yolo_label(label_path)
    return draw_yolo_boxes(img, annotations, class_names, colors)


# ──────────────────────────────────────────────────────────────
# Create preview grid
# ──────────────────────────────────────────────────────────────

def create_preview_grid(
    images: List[np.ndarray],
    grid_cols: int = 4,
    cell_size: Tuple[int, int] = (640, 480),
    padding: int = 4,
    bg_color: Tuple[int, int, int] = (40, 40, 40),
) -> np.ndarray:
    """
    Create a grid of images for visual preview.

    Args:
        images: List of BGR images
        grid_cols: Number of columns
        cell_size: (width, height) of each cell
        padding: Padding between cells
        bg_color: Background color

    Returns:
        Grid image as numpy array
    """
    if not images:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)

    n = len(images)
    grid_rows = (n + grid_cols - 1) // grid_cols

    grid_w = grid_cols * (cell_size[0] + padding) + padding
    grid_h = grid_rows * (cell_size[1] + padding) + padding
    grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols

        # Resize maintaining aspect ratio
        h, w = img.shape[:2]
        scale = min(cell_size[0] / w, cell_size[1] / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # Center in cell
        x_offset = padding + col * (cell_size[0] + padding) + (cell_size[0] - new_w) // 2
        y_offset = padding + row * (cell_size[1] + padding) + (cell_size[1] - new_h) // 2

        grid[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return grid


# ──────────────────────────────────────────────────────────────
# Save annotation preview for a dataset
# ──────────────────────────────────────────────────────────────

def save_dataset_preview(
    image_label_pairs: List[Tuple[Path, Path]],
    output_path: Path,
    class_names: List[str] = None,
    colors: Dict[int, Tuple] = None,
    num_samples: int = 16,
    grid_cols: int = 4,
    title: str = "",
) -> None:
    """
    Generate and save an annotation preview grid for a dataset.

    Args:
        image_label_pairs: List of (image_path, label_path) tuples
        output_path: Where to save the preview image
        class_names: Class names for labels
        colors: Class colors for boxes
        num_samples: Number of samples to include
        grid_cols: Grid columns
        title: Optional title text
    """
    import random

    if len(image_label_pairs) == 0:
        print(f"  No image-label pairs found for preview")
        return

    samples = random.sample(
        image_label_pairs,
        min(num_samples, len(image_label_pairs))
    )

    annotated_images = []
    for img_path, lbl_path in samples:
        vis = visualize_image_with_labels(img_path, lbl_path, class_names, colors)
        if vis is not None:
            # Add filename as text at bottom
            h, w = vis.shape[:2]
            cv2.putText(
                vis, img_path.name[:40], (5, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA
            )
            annotated_images.append(vis)

    if not annotated_images:
        print(f"  No valid images found for preview")
        return

    grid = create_preview_grid(annotated_images, grid_cols=grid_cols)

    # Add title if provided
    if title:
        # cv2 fonts are ASCII only — non-ASCII characters render as "???"
        title = title.encode("ascii", "replace").decode("ascii")
        title_bar = np.full((40, grid.shape[1], 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(
            title_bar, title, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
        grid = np.vstack([title_bar, grid])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    print(f"  Saved preview: {output_path}")
