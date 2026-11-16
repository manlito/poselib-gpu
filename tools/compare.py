#!/usr/bin/env python3
"""Metrics of result files against the ground truth of a problem file, and against each other.

    tools/compare.py problems.jpop results_a.txt [results_b.txt ...]

Per result file: success rate, median rotation and translation error, pose AUC at 5, 10 and 20 degrees,
overlap of the inlier set with the generated inliers, and with the inlier set of the first file.
A relative pose succeeds when max(rotation error, translation direction error) is below 5 degrees; an
absolute pose when the rotation error is below 1 degree. Problems without a ground truth pose (no
model) succeed when no model with more than 30 inliers is reported.
"""
import math
import sys

import jpofile


def iou(a, b):
    inter = sum(1 for x, y in zip(a, b) if x and y)
    union = sum(1 for x, y in zip(a, b) if x or y)
    return inter / union if union else 1.0


def metrics(kind, problems, results, first=None):
    errs, rots, trans, overlap, cross = [], [], [], [], []
    success = 0
    for i, (p, r) in enumerate(zip(problems, results)):
        mask = [c == "1" for c in r["mask"]]
        if math.isnan(p["gt_pose"][0]):
            success += 1 if r["n_inliers"] <= 30 else 0
            continue
        rot, tr = 180.0, 180.0
        if r["status"] in (0, 3):
            rot, tr = jpofile.pose_errors(kind, r["pose"], p["gt_pose"])
        err = max(rot, tr) if kind == 0 else rot
        errs.append(err)
        rots.append(rot)
        trans.append(tr)
        success += 1 if err < (5.0 if kind == 0 else 1.0) else 0
        overlap.append(iou(mask, [b == 1 for b in p["gt_inlier"]]))
        if first is not None:
            cross.append(iou(mask, [c == "1" for c in first[i]["mask"]]))
    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    return {
        "success": success / len(problems), "rot": med(rots), "trans": med(trans),
        "auc5": jpofile.auc(errs, 5.0) if errs else float("nan"),
        "auc10": jpofile.auc(errs, 10.0) if errs else float("nan"),
        "auc20": jpofile.auc(errs, 20.0) if errs else float("nan"),
        "iou_gt": mean(overlap), "iou_first": mean(cross),
        "iterations": mean([r["iterations"] for r in results]),
    }


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    kind, problems = jpofile.load_problems(sys.argv[1])
    first = None
    unit = "deg" if kind == 0 else "units"
    print(f"| results | tool | success | rot (deg) | trans ({unit}) | AUC@5 | @10 | @20 | IoU gt | IoU first | iterations | ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for path in sys.argv[2:]:
        seconds, tool, results = jpofile.load_results(path)
        m = metrics(kind, problems, results, first)
        if first is None:
            first = results
        print(f"| {path} | {tool} | {m['success']:.3f} | {m['rot']:.4f} | {m['trans']:.4f} | {m['auc5']:.4f} | "
              f"{m['auc10']:.4f} | {m['auc20']:.4f} | {m['iou_gt']:.4f} | {m['iou_first']:.4f} | "
              f"{m['iterations']:.0f} | {seconds * 1e3:.2f} |")


if __name__ == "__main__":
    main()
