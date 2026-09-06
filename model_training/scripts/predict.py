"""
Prediction / Inference Script
==============================

Runs YOLO inference on images, folders, videos, or webcam.

Features:
- Annotated output with bboxes, class names, confidence
- Video → annotated_video.mp4 with frame numbers
- Structured JSON detection output for future event engine
- FPS measurement

Usage:
    # Single image
    python scripts/predict.py --source image.jpg --model best.pt

    # Folder of images
    python scripts/predict.py --source path/to/images/ --model best.pt

    # Video
    python scripts/predict.py --source road_video.mp4 --model best.pt

    # Webcam
    python scripts/predict.py --source 0 --model best.pt
"""

import sys
import json
import uuid
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    RUNS_DIR, FINAL_CLASS_NAMES,
    get_device, print_hardware_info,
)


def create_detection_event(
    damage_type: str,
    confidence: float,
    bbox: list,
    frame_id: str = "",
    timestamp: str = "",
) -> dict:
    """
    Create a structured detection event for future sensor fusion.

    This format is designed to be consumed by a future temporal
    event engine that will add GPS, IMU, and LiDAR data.
    """
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": timestamp or datetime.now().isoformat(),
        "damage_type": damage_type,
        "confidence": round(confidence, 4),
        "bbox": [round(x, 2) for x in bbox],
        "frame_id": frame_id,
        "source": "camera",
        # Future fields (NOT fabricated):
        # "gps": null,        ← added by GPS module
        # "imu_data": null,   ← added by IMU module
        # "lidar_depth": null, ← added by TF-Luna module
        # "speed_kmh": null,  ← added by GPS/OBD module
    }


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _is_video(source: str) -> bool:
    path = Path(str(source))
    return path.suffix.lower() in VIDEO_SUFFIXES and path.exists()


def run_inference(model_path: str, source: str, device: str,
                  conf: float, iou: float, project: str, name: str,
                  save_json: bool):
    """Run YOLO inference and collect structured detections."""
    from ultralytics import YOLO
    import cv2
    import time

    model = YOLO(model_path)

    # Class names come from the checkpoint itself, so inference stays correct
    # even if the project taxonomy changes later.
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    is_video = _is_video(source)
    output_dir = Path(project) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Running inference on: {source}")
    print(f"  Mode: {'video' if is_video else 'image / folder / stream'}")
    print(f"  Confidence threshold: {conf}")
    print(f"  IoU threshold: {iou}")

    # Source fps, so detections carry a real timestamp and the annotated video
    # plays back at the original speed
    src_fps = None
    if is_video:
        cap = cv2.VideoCapture(str(source))
        probe = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if probe and probe > 0:
            src_fps = probe

    start_time = time.time()

    # stream=True keeps memory flat on long videos (results are yielded rather
    # than accumulated). For video we write the frames ourselves: that puts the
    # frame number and timestamp on the overlay and produces a .mp4, where
    # ultralytics' own writer emits .avi on Windows.
    results = model.predict(
        source=source,
        stream=True,
        device=device,
        conf=conf,
        iou=iou,
        project=project,
        name=name,
        exist_ok=True,
        save=not is_video,
        save_txt=not is_video,
        verbose=not is_video,
    )

    all_detections = []
    total_frames = 0
    total_objects = 0
    writer = None
    video_path = output_dir / "annotated_video.mp4"

    for i, result in enumerate(results):
        total_frames += 1
        frame_id = f"frame_{i:06d}"
        timestamp = f"{i / src_fps:.3f}s" if src_fps else ""

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                event = create_detection_event(
                    damage_type=names.get(cls_id, f"class_{cls_id}"),
                    confidence=conf_val,
                    bbox=xyxy,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
                event["frame_index"] = i
                event["class_id"] = cls_id
                all_detections.append(event)
                total_objects += 1

        if is_video:
            frame = result.plot()  # boxes + class + confidence
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    src_fps or 25.0,
                    (width, height),
                )
            label = f"frame {i}" + (f" | t={timestamp}" if timestamp else "")
            cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            if (i + 1) % 100 == 0:
                print(f"    processed {i + 1} frames, {total_objects} detections so far")

    if writer is not None:
        writer.release()

    elapsed = time.time() - start_time
    fps = total_frames / elapsed if elapsed > 0 else 0

    print(f"\n{'=' * 60}")
    print("INFERENCE RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total frames processed: {total_frames}")
    print(f"  Total detections: {total_objects}")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print(f"  Inference FPS: {fps:.1f}")
    if is_video and writer is not None:
        print(f"  Annotated video: {video_path}")
    print(f"  Results saved to: {output_dir}")

    # Structured detections for the future temporal / sensor-fusion engine
    if save_json:
        json_path = output_dir / "detections.json"
        output_data = {
            "model": model_path,
            "source": str(source),
            "timestamp": datetime.now().isoformat(),
            "source_fps": src_fps,
            "total_frames": total_frames,
            "total_detections": total_objects,
            "inference_fps": round(fps, 2),
            "elapsed_seconds": round(elapsed, 2),
            "detections": all_detections,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"  Detection JSON: {json_path}")

    if all_detections:
        from collections import Counter
        class_counts = Counter(d['damage_type'] for d in all_detections)
        print(f"\n  Detections by class:")
        for cls, count in class_counts.most_common():
            print(f"    {cls}: {count}")

    return all_detections, fps


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO inference for road damage detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Image, folder, video path, or 0 for webcam")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (best.pt)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--project", type=str,
                        default=str(RUNS_DIR / "predict"))
    parser.add_argument("--name", type=str, default="inference")
    parser.add_argument("--no-json", action="store_true",
                        help="Skip saving JSON detections")

    args = parser.parse_args()

    print("=" * 60)
    print("YOLO INFERENCE — Road Damage AI Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print_hardware_info()

    device = args.device if args.device else get_device()
    print(f"\n  Using device: {device}")

    # Validate model
    if not Path(args.model).exists():
        print(f"\n  ERROR: Model not found: {args.model}")
        sys.exit(1)

    # Run inference
    detections, fps = run_inference(
        model_path=args.model,
        source=args.source,
        device=device,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        save_json=not args.no_json,
    )


if __name__ == "__main__":
    main()
