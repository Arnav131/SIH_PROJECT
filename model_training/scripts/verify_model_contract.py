"""
Verify the shipped model against configs/model_classes.yaml
============================================================

Re-checks that models/best.pt and models/best.onnx still match the inference
contract recorded in configs/model_classes.yaml — class list, tensor shapes,
dtypes, opset, and whether NMS lives inside the graph.

Run this whenever a model file is replaced, re-exported, or quantized. It
reads everything out of the actual files; nothing is assumed. If a value
drifts from the recorded contract, the Android integration built against
that contract is silently wrong, so this exits non-zero on any mismatch.

Usage:
    python scripts/verify_model_contract.py
    python scripts/verify_model_contract.py --pt models/best.pt --onnx models/best.onnx
    python scripts/verify_model_contract.py --skip-consistency   # faster, no inference
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import CONFIGS_DIR, load_yaml

failures = []
warnings_list = []


def check(label: str, actual, expected) -> None:
    """Compare one contract value and record the outcome."""
    ok = actual == expected
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label}")
    if not ok:
        print(f"           expected: {expected}")
        print(f"           actual  : {actual}")
        failures.append(label)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contract", type=str,
                        default=str(CONFIGS_DIR / "model_classes.yaml"))
    parser.add_argument("--pt", type=str, default=None,
                        help="Override the .pt path from the contract")
    parser.add_argument("--onnx", type=str, default=None,
                        help="Override the .onnx path from the contract")
    parser.add_argument("--skip-consistency", action="store_true",
                        help="Skip the PT-vs-ONNX inference comparison")
    args = parser.parse_args()

    contract_path = Path(args.contract)
    if not contract_path.exists():
        print(f"ERROR: contract not found: {contract_path}")
        sys.exit(1)
    c = load_yaml(contract_path)

    pt_path = Path(args.pt or (PROJECT_ROOT / c["model"]["pt_file"]))
    onnx_path = Path(args.onnx or (PROJECT_ROOT / c["model"]["onnx_file"]))

    print("=" * 70)
    print("MODEL CONTRACT VERIFICATION")
    print("=" * 70)
    print(f"  contract : {contract_path}")
    print(f"  .pt      : {pt_path}")
    print(f"  .onnx    : {onnx_path}")

    for p in (pt_path, onnx_path):
        if not p.exists():
            print(f"\nERROR: model file not found: {p}")
            sys.exit(1)

    expected_names = {int(k): str(v) for k, v in c["names"].items()}

    # ── PyTorch checkpoint ────────────────────────────────────────────
    print("\n--- best.pt ---")
    from ultralytics import YOLO
    mpt = YOLO(str(pt_path))
    pt_names = mpt.names if isinstance(mpt.names, dict) else dict(enumerate(mpt.names))
    pt_names = {int(k): str(v) for k, v in pt_names.items()}
    check("loads successfully", True, True)
    check("class count", len(pt_names), c["nc"])
    check("class names", pt_names, expected_names)

    # ── ONNX graph ────────────────────────────────────────────────────
    print("\n--- best.onnx ---")
    import onnx
    import onnxruntime as ort

    m = onnx.load(str(onnx_path))
    try:
        onnx.checker.check_model(m)
        check("onnx.checker", True, True)
    except Exception as e:
        print(f"  [FAIL] onnx.checker -> {e}")
        failures.append("onnx.checker")

    check("IR version", m.ir_version, c["onnx"]["ir_version"])
    opsets = {(op.domain or "ai.onnx"): op.version for op in m.opset_import}
    check("opset (ai.onnx)", opsets.get("ai.onnx"), c["onnx"]["opset"])

    ops = {n.op_type for n in m.graph.node}
    check("NMS inside graph", "NonMaxSuppression" in ops, c["onnx"]["postprocessing"]["nms_in_graph"])

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    i, o = sess.get_inputs()[0], sess.get_outputs()[0]
    check("input name", i.name, c["onnx"]["input"]["name"])
    check("input shape", list(i.shape), list(c["onnx"]["input"]["shape"]))
    check("input dtype", i.type, "tensor(float)")
    check("output count", len(sess.get_outputs()), 1)
    check("output name", o.name, c["onnx"]["output"]["name"])
    check("output shape", list(o.shape), list(c["onnx"]["output"]["shape"]))
    check("output dtype", o.type, "tensor(float)")

    # Output width must equal 4 box terms + nc class scores
    check("output channels == 4 + nc", o.shape[1], 4 + c["nc"])

    # Class names embedded in the ONNX metadata
    meta = {kv.key: kv.value for kv in m.metadata_props}
    if "names" in meta:
        try:
            onnx_names = {int(k): str(v) for k, v in eval(meta["names"], {"__builtins__": {}}).items()}
            check("onnx metadata names", onnx_names, expected_names)
        except Exception:
            warnings_list.append("could not parse ONNX metadata 'names'")
    else:
        warnings_list.append("ONNX carries no 'names' metadata")

    # ── PT vs ONNX numerical consistency ──────────────────────────────
    if not args.skip_consistency:
        print("\n--- PT vs ONNX consistency ---")
        from utils.config import COMBINED_DIR
        test_dir = COMBINED_DIR / "images" / "test"
        imgs = sorted(test_dir.glob("*.jpg"))[:5] if test_dir.exists() else []

        if not imgs:
            print("  [SKIP] no test images found at "
                  f"{test_dir} — cannot run inference comparison")
            warnings_list.append("no test images available for consistency check")
        else:
            mox = YOLO(str(onnx_path), task="detect")
            max_conf_d, max_box_d, mismatches, npt, nox = 0.0, 0.0, 0, 0, 0
            for img in imgs:
                a = mpt.predict(str(img), imgsz=640, conf=0.25, verbose=False)[0].boxes
                b = mox.predict(str(img), imgsz=640, conf=0.25, verbose=False)[0].boxes
                npt += len(a)
                nox += len(b)
                if len(a) != len(b):
                    mismatches += 1
                    continue
                if len(a) == 0:
                    continue
                ac, bc = a.conf.cpu().numpy(), b.conf.cpu().numpy()
                ab, bb = a.xyxy.cpu().numpy(), b.xyxy.cpu().numpy()
                if a.cls.cpu().numpy().tolist() != b.cls.cpu().numpy().tolist():
                    mismatches += 1
                max_conf_d = max(max_conf_d, float(abs(ac - bc).max()))
                max_box_d = max(max_box_d, float(abs(ab - bb).max()))

            print(f"  images compared    : {len(imgs)}")
            print(f"  detections PT/ONNX : {npt} / {nox}")
            print(f"  max conf diff      : {max_conf_d:.6f}")
            print(f"  max bbox diff (px) : {max_box_d:.3f}")
            print(f"  class mismatches   : {mismatches}")

            if mismatches:
                print("  [FAIL] PT and ONNX disagree on classes or detection count")
                failures.append("PT/ONNX consistency")
            elif max_conf_d > 0.01 or max_box_d > 2.0:
                print("  [WARN] larger drift than expected from a float32 export")
                warnings_list.append("PT/ONNX numerical drift above tolerance")
            else:
                print("  [OK  ] export is numerically faithful")

    # ── Verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if failures:
        print(f"RESULT: FAIL — {len(failures)} contract violation(s)")
        for f in failures:
            print(f"  - {f}")
        print("\nThe model no longer matches configs/model_classes.yaml.")
        print("Update the contract, or fix the model — but do not ship an")
        print("Android build against a stale contract.")
    else:
        print("RESULT: PASS — model matches configs/model_classes.yaml")
    for w in warnings_list:
        print(f"  [warn] {w}")
    print("=" * 70)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
