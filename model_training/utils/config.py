"""
Central configuration manager for the Road Damage AI pipeline.

Handles YAML loading, path resolution, default hyperparameters,
and hardware detection. All paths use pathlib for cross-platform compatibility.
"""

import sys
import yaml
import torch
import platform
import psutil
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')



# ──────────────────────────────────────────────────────────────
# Project root detection
# ──────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """Return the road_damage_ai project root directory."""
    # Walk up from this file to find the project root
    current = Path(__file__).resolve().parent.parent
    return current


PROJECT_ROOT = get_project_root()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REPORTS_DIR = PROJECT_ROOT / "reports"
RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"


# ──────────────────────────────────────────────────────────────
# Dataset paths
# ──────────────────────────────────────────────────────────────

RAD_DIR = DATA_RAW / "rad"
ROAD_DAMAGE_DIR = DATA_RAW / "road_damage"
POTHOLE_DIR = DATA_RAW / "pothole_dataset"
COMBINED_DIR = DATA_PROCESSED / "road_damage_combined"


# ──────────────────────────────────────────────────────────────
# YAML utilities
# ──────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ──────────────────────────────────────────────────────────────
# Class mapping
# ──────────────────────────────────────────────────────────────

# Final unified class taxonomy
FINAL_CLASSES = {
    0: "pothole",
    1: "crack",
    2: "manhole",
    3: "road_damage",
    4: "speed_bump",
    5: "unsurfaced_road",
}

FINAL_CLASS_NAMES = list(FINAL_CLASSES.values())
NUM_CLASSES = len(FINAL_CLASSES)


def load_class_mapping(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the class mapping configuration."""
    if path is None:
        path = CONFIGS_DIR / "class_mapping.yaml"
    return load_yaml(path)


def load_data_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the data.yaml configuration for YOLO training."""
    if path is None:
        path = CONFIGS_DIR / "data.yaml"
    return load_yaml(path)


def resolve_data_yaml(path: Optional[Path] = None) -> Path:
    """
    Return a data.yaml whose `path` key is an ABSOLUTE path to this machine.

    Ultralytics resolves a relative `path:` against its own DATASETS_DIR
    setting (not against the yaml file), which breaks when the project is
    copied to another machine. We therefore rewrite the dataset root at
    runtime into configs/data.resolved.yaml and hand THAT to YOLO.

    The committed configs/data.yaml stays machine independent.
    """
    if path is None:
        path = CONFIGS_DIR / "data.yaml"
    path = Path(path)
    cfg = load_yaml(path)

    root = Path(str(cfg.get("path", ""))) if cfg.get("path") else COMBINED_DIR
    if not root.is_absolute():
        # Try relative to the yaml file, then to the project root
        for base in (path.parent, PROJECT_ROOT):
            candidate = (base / root).resolve()
            if candidate.exists():
                root = candidate
                break
        else:
            root = COMBINED_DIR
    cfg["path"] = str(root)

    resolved = path.parent / "data.resolved.yaml"
    save_yaml(cfg, resolved)
    return resolved


# ──────────────────────────────────────────────────────────────
# Hardware detection
# ──────────────────────────────────────────────────────────────

def get_device() -> str:
    """Auto-detect the best available device for training."""
    if torch.cuda.is_available():
        return "0"  # Use first CUDA GPU
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


def print_hardware_info() -> None:
    """Print detailed hardware information."""
    print("=" * 60)
    print("HARDWARE INFORMATION")
    print("=" * 60)
    print(f"  Platform:       {platform.system()} {platform.release()}")
    print(f"  Architecture:   {platform.machine()}")
    print(f"  Python:         {platform.python_version()}")
    print(f"  PyTorch:        {torch.__version__}")

    if torch.cuda.is_available():
        print(f"  CUDA:           {torch.version.cuda}")
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Memory:     {gpu_mem:.1f} GB")
        print(f"  Device:         CUDA (GPU)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"  Device:         MPS (Apple Silicon)")
    else:
        print(f"  CUDA:           Not available")
        print(f"  Device:         CPU")

    print("=" * 60)


def is_colab() -> bool:
    """Detect whether we're running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return Path("/content").exists()


def get_total_ram_gb() -> float:
    """Total system RAM in GB (not VRAM — that's a separate GPU resource)."""
    return psutil.virtual_memory().total / (1024 ** 3)


def get_safe_worker_count(requested: int, low_ram_threshold_gb: float = 13.0,
                           low_ram_workers: int = 4) -> int:
    """
    Cap dataloader workers on low-RAM hosts (e.g. Colab free tier's ~12 GB).

    Each worker buffers its own prefetch queue in system RAM — more workers
    on a small-RAM host raises the odds of an OS-level OOM kill mid-training.
    Does nothing on hosts with plenty of RAM (e.g. a 16 GB+ laptop).
    """
    total_ram = get_total_ram_gb()
    if total_ram <= low_ram_threshold_gb and requested > low_ram_workers:
        print(f"  [RAM guard] {total_ram:.1f} GB system RAM detected (<= "
              f"{low_ram_threshold_gb:.0f} GB) — capping --workers "
              f"{requested} -> {low_ram_workers} to avoid an OOM kill.")
        return low_ram_workers
    return requested


# ──────────────────────────────────────────────────────────────
# Default training hyperparameters
# ──────────────────────────────────────────────────────────────

DEFAULT_TRAIN_CONFIG = {
    # Local copy first so training works offline; ultralytics downloads it
    # by name if the file is missing.
    "model": str(WEIGHTS_DIR / "yolo11n.pt") if (WEIGHTS_DIR / "yolo11n.pt").exists() else "yolo11n.pt",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "lr0": 0.01,
    "lrf": 0.01,
    "optimizer": "auto",
    "patience": 20,
    "seed": 42,
    "workers": 8,
    "project": str(RUNS_DIR / "train"),
    "name": "road_damage_exp",
    "exist_ok": True,
    "pretrained": True,
    "verbose": True,

    # Augmentation — realistic road scene
    "hsv_h": 0.015,
    "hsv_s": 0.5,
    "hsv_v": 0.3,
    "degrees": 5.0,        # Mild rotation only
    "translate": 0.1,
    "scale": 0.3,
    "shear": 2.0,
    "perspective": 0.0005,
    "flipud": 0.0,         # No vertical flip — unrealistic for road
    "fliplr": 0.5,         # Horizontal flip is valid
    "mosaic": 1.0,
    "mixup": 0.1,
    "erasing": 0.1,
}


# ──────────────────────────────────────────────────────────────
# Color palette for visualization
# ──────────────────────────────────────────────────────────────

CLASS_COLORS = {
    0: (255, 80, 80),     # pothole — red
    1: (80, 180, 255),    # crack — blue
    2: (80, 255, 80),     # manhole — green
    3: (255, 180, 50),    # road_damage — orange
    4: (200, 80, 255),    # speed_bump — purple
    5: (255, 255, 80),    # unsurfaced_road — yellow
}
