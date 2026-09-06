"""
Evaluation Script
==================

Evaluates a trained YOLO model on the test set and generates detailed metrics.

Reports:
- Precision, Recall, mAP@50, mAP@50-95
- Per-class precision, recall, mAP
- Confusion matrix
- Saves validation prediction images
- Cross-dataset generalization experiments (optional)

Usage:
    # Standard evaluation
    python scripts/evaluate.py --model runs/train/road_damage_exp/weights/best.pt

    # Cross-dataset experiments
    python scripts/evaluate.py --model runs/train/road_damage_exp/weights/best.pt --cross-dataset
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    CONFIGS_DIR, RUNS_DIR, REPORTS_DIR, COMBINED_DIR,
    get_device, print_hardware_info, FINAL_CLASS_NAMES,
    resolve_data_yaml,
)


def check_taxonomy_match(model, data_yaml: str, force: bool = False) -> None:
    """
    Refuse to evaluate a model against a dataset labelled with a different
    taxonomy.

    A model trained on taxonomy A, validated against labels written in
    taxonomy B, does not fail loudly — it silently reports near-zero recall,
    which reads like "the model is bad" rather than "these two things do not
    belong together". This project has already been burned by exactly that:
    the RDD2022-trained 5-class model scored mAP50 0.067 against the local
    6-class labels while genuinely scoring 0.629 on its own validation set.
    """
    from utils.config import load_yaml

    model_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    data_cfg = load_yaml(Path(data_yaml))
    data_names = data_cfg.get("names") or {}
    if isinstance(data_names, list):
        data_names = dict(enumerate(data_names))

    if not data_names:
        return  # nothing to compare against

    same_count = len(model_names) == len(data_names)
    same_names = {str(v).lower() for v in model_names.values()} == \
                 {str(v).lower() for v in data_names.values()}

    if same_count and same_names:
        return  # taxonomies agree — proceed

    print(f"\n{'!' * 70}")
    print("TAXONOMY MISMATCH — model and dataset use different class lists")
    print(f"{'!' * 70}")
    print(f"  Model ({len(model_names)} classes):")
    for i, n in sorted(model_names.items()):
        print(f"      {i}: {n}")
    print(f"  Dataset ({len(data_names)} classes) — {data_yaml}:")
    for i, n in sorted(data_names.items()):
        print(f"      {i}: {n}")
    print()
    print("  Evaluating across mismatched taxonomies produces meaningless")
    print("  metrics (near-zero recall) that look like model failure.")
    print()
    print("  This project ships a model trained on RDD2022 (5 classes) while")
    print("  data/processed/road_damage_combined/ carries 6-class local labels.")
    print("  See configs/model_classes.yaml and reports/model_sanity_check.md")
    print()
    print("  If you really intend this, re-run with --force-taxonomy-mismatch")
    print(f"{'!' * 70}\n")

    if not force:
        sys.exit(1)
    print("  --force-taxonomy-mismatch given: continuing anyway.\n")


def evaluate_model(model_path: str, data_yaml: str, device: str,
                   project: str, name: str, split: str = "test",
                   force_mismatch: bool = False):
    """Run YOLO validation on the requested split and return results."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    check_taxonomy_match(model, data_yaml, force=force_mismatch)
    results = model.val(
        data=data_yaml,
        split=split,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
        save_json=True,
        plots=True,
    )
    return results


def run_cross_dataset_experiments(model_path: str, device: str):
    """
    Run cross-dataset generalization experiments.

    These test whether the model generalizes across different data sources
    rather than memorizing one dataset's visual style.
    """
    print(f"\n{'=' * 60}")
    print("CROSS-DATASET GENERALIZATION EXPERIMENTS")
    print(f"{'=' * 60}")

    # Check which source prefixes exist in the combined dataset
    test_dir = COMBINED_DIR / "images" / "test"
    if not test_dir.exists():
        print("  Test directory not found — skipping cross-dataset experiments")
        return {}

    from utils.dataset_utils import find_images
    test_images = find_images(test_dir, recursive=False)

    # Group by source
    source_counts = {}
    for img in test_images:
        if img.name.startswith('rad_'):
            source_counts.setdefault('rad', []).append(img)
        elif img.name.startswith('rd_'):
            source_counts.setdefault('road_damage', []).append(img)
        elif img.name.startswith('pot_'):
            source_counts.setdefault('pothole', []).append(img)

    print(f"\n  Test set composition:")
    for src, imgs in source_counts.items():
        print(f"    {src}: {len(imgs)} images")

    print(f"\n  Note: Full cross-dataset experiments (train on subset, test on holdout)")
    print(f"  require retraining. These are documented for the GPU machine.")
    print(f"\n  Recommended experiments to run on GPU:")
    print(f"    Exp A: Train combined → test combined (standard)")
    print(f"    Exp B: Train RAD+Pothole → test Road Damage")
    print(f"    Exp C: Train RAD+Road Damage → test Pothole")
    print(f"    Exp D: Train Road Damage+Pothole → test RAD")

    return source_counts


def generate_report(results, model_path: str, cross_dataset_info: dict):
    """Generate evaluation report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "evaluation_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Model\n")
        f.write(f"- Path: `{model_path}`\n\n")

        f.write(f"## Overall Metrics\n\n")
        f.write("| Metric | Value |\n|---|---|\n")

        # Extract metrics from results
        if hasattr(results, 'results_dict'):
            rd = results.results_dict
            for key, value in rd.items():
                if isinstance(value, float):
                    f.write(f"| {key} | {value:.4f} |\n")

        f.write(f"\n## Per-Class Metrics\n\n")
        f.write("| Class | Precision | Recall | mAP50 | mAP50-95 |\n")
        f.write("|---|---|---|---|---|\n")

        # Per-class rows. Ultralytics fills metric arrays only for classes that
        # actually occur in the split, indexed by box.ap_class_index — so go
        # through that index instead of assuming array position == class id.
        box = getattr(results, "box", None)
        if box is not None and hasattr(box, "ap_class_index"):
            names = getattr(results, "names", None) or dict(enumerate(FINAL_CLASS_NAMES))
            present = [int(c) for c in box.ap_class_index]
            for row, cls_id in enumerate(present):
                cls_name = names.get(cls_id, f"class_{cls_id}") if isinstance(names, dict) else str(cls_id)
                try:
                    prec, rec, ap50, ap = box.class_result(row)
                    f.write(f"| {cls_name} | {prec:.4f} | {rec:.4f} | {ap50:.4f} | {ap:.4f} |\n")
                except (IndexError, AttributeError, ValueError):
                    pass
            # Report missing classes against the MODEL's own class list, not a
            # hardcoded project taxonomy — the shipped model has 5 classes while
            # configs/data.yaml describes a 6-class local dataset.
            all_names = names if isinstance(names, dict) else dict(enumerate(FINAL_CLASS_NAMES))
            missing = [n for i, n in sorted(all_names.items()) if i not in present]
            if missing:
                f.write("\n> Classes with no ground-truth instances in this split: "
                        + ", ".join(missing) + "\n")

        if cross_dataset_info:
            f.write(f"\n## Cross-Dataset Test Set Composition\n\n")
            for src, imgs in cross_dataset_info.items():
                f.write(f"- {src}: {len(imgs)} images\n")

        f.write(f"\n## Recommended Cross-Dataset Experiments\n\n")
        f.write("| Experiment | Training Data | Test Data |\n|---|---|---|\n")
        f.write("| A (standard) | Combined | Combined hold-out |\n")
        f.write("| B | RAD + Pothole | Road Damage |\n")
        f.write("| C | RAD + Road Damage | Pothole |\n")
        f.write("| D | Road Damage + Pothole | RAD |\n")

    print(f"\n  Report saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLO model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (best.pt)")
    parser.add_argument("--data", type=str,
                        default=str(CONFIGS_DIR / "data.yaml"),
                        help="Path to data.yaml")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    parser.add_argument("--project", type=str,
                        default=str(RUNS_DIR / "evaluate"))
    parser.add_argument("--name", type=str, default="eval_results")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--cross-dataset", action="store_true",
                        help="Run cross-dataset analysis")
    parser.add_argument("--force-taxonomy-mismatch", action="store_true",
                        help="Evaluate even when the model's class list differs "
                             "from the dataset's (produces meaningless metrics — "
                             "see configs/model_classes.yaml)")

    args = parser.parse_args()

    print("=" * 60)
    print("MODEL EVALUATION — Road Damage AI Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print_hardware_info()

    device = args.device if args.device else get_device()
    print(f"\n  Using device: {device}")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Split: {args.split}")

    # Validate paths
    if not Path(args.model).exists():
        print(f"\n  ERROR: Model not found: {args.model}")
        sys.exit(1)
    if not Path(args.data).exists():
        print(f"\n  ERROR: Data config not found: {args.data}")
        sys.exit(1)

    # Run evaluation
    print(f"\n{'=' * 60}")
    print("Running evaluation...")
    print(f"{'=' * 60}\n")

    results = evaluate_model(
        args.model, str(resolve_data_yaml(args.data)), device,
        args.project, args.name, split=args.split,
        force_mismatch=args.force_taxonomy_mismatch,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")

    if hasattr(results, 'results_dict'):
        for key, value in results.results_dict.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

    # Cross-dataset analysis
    cross_info = {}
    if args.cross_dataset:
        cross_info = run_cross_dataset_experiments(args.model, device)

    # Generate report
    generate_report(results, args.model, cross_info)

    print(f"\n  Results directory: {args.project}/{args.name}")
    print(f"  Check confusion matrix and prediction images in the results directory.")


if __name__ == "__main__":
    main()
