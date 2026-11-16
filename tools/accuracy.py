#!/usr/bin/env python3
"""The accuracy sets: PoseLib against itself across seeds (the noise floor), and jpo against that.

    tools/accuracy.py [--build build/release] [--seeds 5] [--count 200] [--backend cpu] [--float]

Generates the problem sets into <build>/sets (once), runs poselib_ref and `jpo run` with each seed and
prints, per set and tool, the mean and the spread (min .. max over seeds) of the metrics of compare.py.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402
import jpofile  # noqa: E402

MODELS = ["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE"]


def sets(count):
    """name -> jpo gen arguments. The same list for both estimators, plus one set per camera model."""
    out = {}
    for kind in ("relpose", "pnp"):
        base = [kind, "--count", str(count), "--points", "1000"]
        out[f"{kind}_easy"] = base + ["--inliers", "0.7", "--noise", "0.5", "--seed", "101"]
        out[f"{kind}_hard"] = base + ["--inliers", "0.3", "--noise", "0.5", "--seed", "102"]
        out[f"{kind}_harder"] = base + ["--inliers", "0.2", "--noise", "1.0", "--seed", "103"]
        out[f"{kind}_planar"] = base + ["--inliers", "0.6", "--noise", "0.5", "--planar", "--seed", "104"]
        out[f"{kind}_none"] = base + ["--inliers", "0", "--points", "300", "--seed", "105"]
        if kind == "relpose":
            out["relpose_small_baseline"] = base + ["--inliers", "0.6", "--noise", "0.5", "--baseline", "0.02", "--seed", "106"]
        else:
            out["pnp_far_origin"] = base + ["--inliers", "0.6", "--noise", "0.5", "--world-offset", "100000", "--seed", "106"]
        for i, model in enumerate(MODELS):
            out[f"{kind}_{model.lower()}"] = base + ["--inliers", "0.5", "--noise", "0.5", "--model", model, "--seed", str(110 + i)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build/release")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--float", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    jpo = os.path.join(args.build, "jpo")
    ref = os.path.join(args.build, "poselib_ref")
    have_ref = os.path.exists(ref)
    sets_dir = os.path.join(args.build, "sets")
    out_dir = os.path.join(args.build, "accuracy")
    os.makedirs(sets_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    keys = ["success", "rot", "trans", "auc5", "auc10", "auc20", "iou_gt"]
    print("| set | tool | " + " | ".join(keys) + " | iterations | ms |")
    print("|---|---|" + "---|" * (len(keys) + 2))
    worst = 0.0
    for name, gen in sets(args.count).items():
        if args.only and args.only not in name:
            continue
        path = os.path.join(sets_dir, f"{name}_{args.count}.jpop")
        if not os.path.exists(path):
            subprocess.run([jpo, "gen"] + gen + ["-o", path], check=True)
        kind, problems = jpofile.load_problems(path)
        rows = {}
        tools = (["poselib"] if have_ref else []) + ["jpo"]
        for tool in tools:
            runs, times = [], []
            for seed in range(args.seeds):
                res = os.path.join(out_dir, f"{name}_{tool}_{seed}.txt")
                if tool == "poselib":
                    cmd = [ref, path, "--seed", str(seed), "-o", res]
                else:
                    cmd = [jpo, "run", path, "--backend", args.backend, "--seed", str(seed), "-o", res, "--quiet"]
                    if args.float:
                        cmd.append("--float")
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                seconds, _, results = jpofile.load_results(res)
                runs.append(compare.metrics(kind, problems, results))
                times.append(seconds)
            rows[tool] = runs
            cells = []
            for k in keys:
                v = [r[k] for r in runs]
                cells.append(f"{sum(v) / len(v):.4f} ({min(v):.4f} .. {max(v):.4f})")
            it = sum(r["iterations"] for r in runs) / len(runs)
            label = tool if tool == "poselib" else f"jpo {args.backend}{' float' if args.float else ''}"
            print(f"| {name} | {label} | " + " | ".join(cells) + f" | {it:.0f} | {min(times) * 1e3:.1f} |", flush=True)
        if have_ref:
            # How far below PoseLib's worst seed our mean success rate is.
            lo = min(r["success"] for r in rows["poselib"])
            mean = sum(r["success"] for r in rows["jpo"]) / len(rows["jpo"])
            worst = max(worst, lo - mean)
    if have_ref:
        print(f"\nLargest shortfall of jpo's mean success rate below PoseLib's worst seed: {worst:.4f}")


if __name__ == "__main__":
    main()
