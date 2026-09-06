"""
YOLO Training Script
=====================

Trains a lightweight YOLO model using transfer learning for road damage detection.

Features:
- Auto-detects CUDA GPU / MPS / CPU
- Configurable hyperparameters via CLI
- Realistic road-scene augmentation
- Saves best.pt + last.pt
- Prints hardware info before training

Designed to be portable: runs on CPU for testing, GPU for production training.

Usage:
    # Default training
    python scripts/train.py

    # Custom configuration
    python scripts/train.py --model yolo11n.pt --epochs 100 --batch 16 --imgsz 640

    # Quick test run
    python scripts/train.py --epochs 5 --batch 8 --name quick_test
"""

import sys
import gc
import argparse
import threading
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    CONFIGS_DIR, RUNS_DIR, WEIGHTS_DIR, DEFAULT_TRAIN_CONFIG,
    get_device, print_hardware_info, resolve_data_yaml,
    is_colab, get_total_ram_gb, get_safe_worker_count,
)


class RamGuard:
    """
    Background watchdog for low-RAM hosts (Colab free tier ~12 GB).

    Polls system RAM every few seconds. On a high-water mark it forces a
    Python GC pass + empties the CUDA cache (the two things most likely to
    free reclaimable memory without touching training state), and logs a
    loud warning so a crash isn't a silent mystery. It cannot prevent a
    genuine OOM by itself — the real fix is `--workers`/`--batch` — but it
    buys headroom against transient spikes (dataloader prefetch, matplotlib
    plot buffers, etc.) and always leaves a clear trail in the log.
    """

    def __init__(self, warn_pct: float = 85.0, critical_pct: float = 93.0,
                 interval_s: float = 15.0):
        self.warn_pct = warn_pct
        self.critical_pct = critical_pct
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        import psutil
        while not self._stop.is_set():
            pct = psutil.virtual_memory().percent
            if pct >= self.critical_pct:
                print(f"\n  [RAM guard] CRITICAL: {pct:.0f}% RAM used — "
                      f"freeing what we can (gc + CUDA cache). If this "
                      f"repeats, lower --workers or --batch.\n")
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            elif pct >= self.warn_pct:
                print(f"  [RAM guard] WARNING: {pct:.0f}% RAM used")
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO road damage detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    defaults = DEFAULT_TRAIN_CONFIG

    # Model
    parser.add_argument("--model", type=str, default=defaults["model"],
                        help="Pretrained model checkpoint")
    parser.add_argument("--data", type=str,
                        default=str(CONFIGS_DIR / "data.yaml"),
                        help="Path to data.yaml")

    # Training
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch", type=int, default=defaults["batch"])
    parser.add_argument("--imgsz", type=int, default=defaults["imgsz"])
    parser.add_argument("--lr0", type=float, default=defaults["lr0"],
                        help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=defaults["lrf"],
                        help="Final learning rate factor")
    parser.add_argument("--optimizer", type=str, default=defaults["optimizer"])
    parser.add_argument("--patience", type=int, default=defaults["patience"],
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--workers", type=int, default=defaults["workers"])
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of the train split to use (smoke tests)")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 0, cpu, mps. Auto-detected if not set.")

    # Output
    parser.add_argument("--project", type=str,
                        default=str(RUNS_DIR / "train"))
    parser.add_argument("--name", type=str, default=defaults["name"])
    parser.add_argument("--exist-ok", action="store_true", default=True)

    # Augmentation overrides
    parser.add_argument("--no-mosaic", action="store_true",
                        help="Disable mosaic augmentation")
    parser.add_argument("--no-mixup", action="store_true",
                        help="Disable mixup augmentation")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a checkpoint")
    parser.add_argument("--no-auto-resume", action="store_true",
                        help="Don't auto-resume even if project/name/weights/last.pt "
                             "already exists — start a fresh run instead")

    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────
    # Print banner
    # ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("YOLO TRAINING — Road Damage AI Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if is_colab():
        print("Environment: Google Colab detected")
    print("=" * 70)

    # Hardware info
    print_hardware_info()
    print(f"  System RAM:     {get_total_ram_gb():.1f} GB")

    # Device selection
    device = args.device if args.device else get_device()
    print(f"\n  Using device: {device}")

    # ──────────────────────────────────────────────────────────
    # Auto-resume: if a checkpoint already exists for this project/name
    # (e.g. a Colab session got disconnected mid-run), pick it up instead
    # of silently starting a new run from epoch 0. This is what makes a
    # single unchanged command safe to re-run after any crash.
    # ──────────────────────────────────────────────────────────
    if not args.resume and not args.no_auto_resume:
        auto_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if auto_ckpt.exists():
            print(f"\n  [auto-resume] Found existing checkpoint: {auto_ckpt}")
            print(f"  [auto-resume] Resuming instead of starting fresh "
                  f"(pass --no-auto-resume to force a clean run).")
            args.resume = str(auto_ckpt)

    # ──────────────────────────────────────────────────────────
    # RAM-safe worker count — caps --workers on low-RAM hosts (Colab
    # free tier ~12 GB) so dataloader prefetch buffers don't push the
    # host into an OS-level OOM kill. No-op on a normal 16 GB+ laptop.
    # ──────────────────────────────────────────────────────────
    args.workers = get_safe_worker_count(args.workers)

    # ──────────────────────────────────────────────────────────
    # Validate data.yaml exists
    # ──────────────────────────────────────────────────────────
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"\n  ERROR: data.yaml not found: {data_yaml}")
        print("  Run the data preparation pipeline first!")
        sys.exit(1)

    # Rewrite the dataset root to an absolute path for THIS machine
    data_yaml = resolve_data_yaml(data_yaml)

    print(f"  Data config: {data_yaml}")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Learning rate: {args.lr0}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Patience: {args.patience}")
    print(f"  Seed: {args.seed}")

    # ──────────────────────────────────────────────────────────
    # Build training arguments
    # ──────────────────────────────────────────────────────────
    from ultralytics import YOLO

    # Offline safety: if --model is a bare filename (e.g. "yolo11n.pt") and
    # a local copy already lives in weights/, use that instead of letting
    # ultralytics try to download it. Matters on a machine with no internet
    # (or a Colab session mid-network-hiccup).
    if not args.resume:
        model_arg = Path(args.model)
        if not model_arg.is_absolute() and len(model_arg.parts) == 1:
            local_copy = WEIGHTS_DIR / model_arg.name
            if local_copy.exists():
                args.model = str(local_copy)

    # ──────────────────────────────────────────────────────────
    # Train
    # ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Starting training...")
    print(f"{'=' * 70}\n")

    ram_guard = RamGuard()
    ram_guard.start()

    if args.resume:
        # A true ultralytics resume MUST pass resume=True — this reloads the
        # original run's args.yaml (epoch count, optimizer/EMA state, batch,
        # etc.) and continues from the last completed epoch. Passing a fresh
        # train_args dict here (the old behavior) silently restarts training
        # from epoch 0 using the checkpoint only as pretrained weights.
        print(f"  Resuming from: {args.resume}")
        model = YOLO(args.resume)
        train_call = lambda: model.train(resume=True)
    else:
        print(f"  Loading pretrained model: {args.model}")
        model = YOLO(args.model)

        train_args = {
            "data": str(data_yaml),
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "optimizer": args.optimizer,
            "patience": args.patience,
            "seed": args.seed,
            "workers": args.workers,
            "fraction": args.fraction,
            "device": device,
            "project": args.project,
            "name": args.name,
            "exist_ok": args.exist_ok,
            "pretrained": True,
            "verbose": True,

            # Augmentation — realistic road scenes
            "hsv_h": defaults["hsv_h"],
            "hsv_s": defaults["hsv_s"],
            "hsv_v": defaults["hsv_v"],
            "degrees": defaults["degrees"],
            "translate": defaults["translate"],
            "scale": defaults["scale"],
            "shear": defaults["shear"],
            "perspective": defaults["perspective"],
            "flipud": defaults["flipud"],
            "fliplr": defaults["fliplr"],
            "mosaic": 0.0 if args.no_mosaic else defaults["mosaic"],
            "mixup": 0.0 if args.no_mixup else defaults["mixup"],
            "erasing": defaults["erasing"],
        }

        train_call = lambda: model.train(**train_args)

    last_model = Path(args.project) / args.name / "weights" / "last.pt"
    best_model = Path(args.project) / args.name / "weights" / "best.pt"
    resume_cmd = f"python scripts/train.py --resume {last_model}"

    # A crash (CUDA OOM, host OOM kill, Colab disconnect, network loss
    # downloading the pretrained checkpoint, Ctrl+C) should never look like
    # a dead end. ultralytics writes last.pt after every completed epoch,
    # so the resume command above is valid the moment any epoch has landed.
    try:
        results = train_call()
    except KeyboardInterrupt:
        print(f"\n{'=' * 70}")
        print("Training interrupted by user (Ctrl+C).")
        if last_model.exists():
            print(f"  Last checkpoint saved: {last_model}")
            print(f"  Resume with:\n    {resume_cmd}")
        print(f"{'=' * 70}")
        sys.exit(130)
    except RuntimeError as e:
        print(f"\n{'=' * 70}")
        if "out of memory" in str(e).lower():
            print("Training crashed: GPU ran out of VRAM (CUDA OOM).")
            print(f"  Fix: lower --batch (currently {args.batch}) and re-run.")
        else:
            print(f"Training crashed with a RuntimeError: {e}")
        if last_model.exists():
            print(f"  Last completed-epoch checkpoint is safe: {last_model}")
            print(f"  Same command re-run will auto-resume from it "
                  f"(or explicitly: {resume_cmd})")
        else:
            print("  No checkpoint exists yet (crashed before epoch 1 "
                  "finished) — re-run will start fresh.")
        print(f"{'=' * 70}")
        raise
    finally:
        ram_guard.stop()

    # ──────────────────────────────────────────────────────────
    # Post-training summary
    # ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"{'=' * 70}")
    print(f"  Results directory: {args.project}/{args.name}")

    if best_model.exists():
        print(f"  Best model: {best_model}")
        print(f"  Best model size: {best_model.stat().st_size / (1024*1024):.1f} MB")
    if last_model.exists():
        print(f"  Last model: {last_model}")

    print(f"\n  Next steps:")
    print(f"    python scripts/evaluate.py --model {best_model}")
    print(f"    python scripts/predict.py --source <image_or_video> --model {best_model}")


if __name__ == "__main__":
    main()
