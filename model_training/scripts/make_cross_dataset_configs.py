"""
Cross-Dataset Experiment Config Generator
==========================================

Generates YOLO data configs for the cross-dataset generalization experiments.

The combined dataset keeps its source in the filename prefix (rad_ / rd_ / pot_),
so a source-specific dataset can be expressed as a plain list of image paths —
no image needs to be copied or moved. Ultralytics accepts a .txt file of image
paths anywhere a split directory is accepted.

Experiments:
    A  train: combined train       test: combined test        (baseline — configs/data.yaml)
    B  train: RAD + Pothole        test: Road Damage
    C  train: RAD + Road Damage    test: Pothole
    D  train: Road Damage + Pothole test: RAD

Train/val come from the combined train/val splits filtered to the training
sources; the test set is every image of the held-out source across ALL splits,
since that source contributes nothing to training in these experiments.

Caveat worth remembering when reading the results: the classes are not shared
evenly. crack/manhole exist only in Road Damage, and road_damage/speed_bump/
unsurfaced_road only in RAD. Experiments B-D are therefore only fully
comparable on `pothole`; the rest measure zero-shot behaviour on classes the
model never saw.

Usage:
    python scripts/make_cross_dataset_configs.py
    python scripts/train.py --data configs/cross/exp_b.yaml --name cross_b
    python scripts/evaluate.py --model runs/train/cross_b/weights/best.pt --data configs/cross/exp_b.yaml --split test
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import COMBINED_DIR, CONFIGS_DIR, FINAL_CLASSES, NUM_CLASSES, save_yaml
from utils.dataset_utils import find_images

SOURCE_PREFIX = {
    "rad": "rad_",
    "road_damage": "rd_",
    "pothole": "pot_",
}

EXPERIMENTS = {
    "exp_b": {"train_on": ["rad", "pothole"], "test_on": "road_damage"},
    "exp_c": {"train_on": ["rad", "road_damage"], "test_on": "pothole"},
    "exp_d": {"train_on": ["road_damage", "pothole"], "test_on": "rad"},
}


def images_of(split: str, sources: list) -> list:
    """Images in `split` belonging to any of `sources`."""
    prefixes = tuple(SOURCE_PREFIX[s] for s in sources)
    img_dir = COMBINED_DIR / "images" / split
    if not img_dir.exists():
        return []
    return [p for p in find_images(img_dir, recursive=False)
            if p.name.startswith(prefixes)]


def write_list(paths: list, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(p.resolve()) for p in paths) + "\n",
                        encoding="utf-8")
    return len(paths)


def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-dataset experiment configs")
    parser.add_argument("--out", type=str, default=str(CONFIGS_DIR / "cross"),
                        help="Output directory for the generated configs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    lists_dir = out_dir / "lists"

    if not (COMBINED_DIR / "images" / "train").exists():
        print("  ERROR: combined dataset not found. Run split_dataset.py first.")
        sys.exit(1)

    print("=" * 60)
    print("CROSS-DATASET EXPERIMENT CONFIGS")
    print("=" * 60)

    for exp_name, spec in EXPERIMENTS.items():
        train_sources = spec["train_on"]
        test_source = spec["test_on"]

        train_imgs = images_of("train", train_sources)
        val_imgs = images_of("val", train_sources)
        # Held-out source contributes nothing to training, so its whole
        # population is available as the test set.
        test_imgs = (images_of("train", [test_source])
                     + images_of("val", [test_source])
                     + images_of("test", [test_source]))

        train_txt = lists_dir / f"{exp_name}_train.txt"
        val_txt = lists_dir / f"{exp_name}_val.txt"
        test_txt = lists_dir / f"{exp_name}_test.txt"

        n_train = write_list(train_imgs, train_txt)
        n_val = write_list(val_imgs, val_txt)
        n_test = write_list(test_imgs, test_txt)

        cfg = {
            "path": str(COMBINED_DIR.resolve()),
            "train": str(train_txt.resolve()),
            "val": str(val_txt.resolve()),
            "test": str(test_txt.resolve()),
            "nc": NUM_CLASSES,
            "names": dict(FINAL_CLASSES),
        }
        cfg_path = out_dir / f"{exp_name}.yaml"
        save_yaml(cfg, cfg_path)

        print(f"\n  {exp_name}: train on {' + '.join(train_sources)} "
              f"-> test on {test_source}")
        print(f"    train: {n_train}  val: {n_val}  test: {n_test}")
        print(f"    config: {cfg_path}")

    print(f"\n{'=' * 60}")
    print("Configs written. Note: these paths are absolute — regenerate this")
    print("on any machine you copy the project to.")
    print("\n  Example run:")
    print("    python scripts/train.py --data configs/cross/exp_b.yaml --name cross_b --epochs 50")
    print("    python scripts/evaluate.py --model runs/train/cross_b/weights/best.pt "
          "--data configs/cross/exp_b.yaml --split test --name cross_b_eval")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
