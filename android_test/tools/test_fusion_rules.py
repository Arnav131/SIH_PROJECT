"""
Unit tests for the fusion rules, run against the reference implementation in
evaluate_fusion.py. These encode the two rules the Kotlin
DetectionFusionEngine must also obey — if you change fusion behaviour in
Kotlin, update these and re-run.

    python android_test/tools/test_fusion_rules.py

No models are loaded; this tests fusion logic only, so it runs instantly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_fusion import Detection, fuse, iou, CROSS_MODEL_DUPLICATE_IOU  # noqa: E402

failures = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


def d(dtype, name, conf, box, source="model"):
    return Detection(source, name, dtype, None, conf, box)


# Rule 2: different types are never duplicates, however much they overlap.
r = fuse([d("POTHOLE", "Pothole", 0.9, (0, 0, 100, 100))],
         [d("CRACK", "Alligator Crack", 0.8, (0, 0, 100, 100))])
check("pothole + crack at IoU 1.0 are both kept", len(r) == 2,
      f"kept {[x.class_type for x in r]}")

# Same type, fully overlapping -> collapsed to the more confident one.
r = fuse([d("POTHOLE", "Pothole", 0.9, (0, 0, 100, 100)),
          d("POTHOLE", "Pothole", 0.5, (0, 0, 100, 100))], [])
check("same-type duplicates collapse to the higher confidence",
      len(r) == 1 and abs(r[0].confidence - 0.9) < 1e-6)

# Same type, overlapping below threshold -> both survive.
box_a, box_b = (0, 0, 100, 100), (50, 50, 150, 150)
r = fuse([d("POTHOLE", "Pothole", 0.9, box_a), d("POTHOLE", "Pothole", 0.5, box_b)], [])
check("same-type below duplicate IoU are both kept", len(r) == 2,
      f"IoU={iou(box_a, box_b):.3f} < {CROSS_MODEL_DUPLICATE_IOU}")

# Rule 1: confidences pass through untouched.
r = fuse([d("POTHOLE", "Pothole", 0.94, (0, 0, 50, 50))],
         [d("CRACK", "Longitudinal Crack", 0.88, (200, 200, 260, 260))])
check("confidences are never combined",
      sorted(round(x.confidence, 2) for x in r) == [0.88, 0.94],
      f"got {sorted(round(x.confidence, 2) for x in r)}")

# The survivor is chosen by confidence, not by which model produced it.
r = fuse([d("POTHOLE", "Pothole", 0.3, (0, 0, 100, 100), "pothole_model")],
         [d("POTHOLE", "Pothole", 0.7, (0, 0, 100, 100), "crack_model")])
check("higher-confidence detection wins regardless of source model",
      len(r) == 1 and r[0].source == "crack_model")

# Degenerate input must not throw.
check("empty input returns empty", fuse([], []) == [])

# One model returning nothing still yields the other's detections.
r = fuse([], [d("CRACK", "Transverse Crack", 0.6, (0, 0, 10, 10))])
check("a silent specialist does not suppress the other", len(r) == 1)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all fusion rule tests passed")
