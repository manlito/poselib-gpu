#!/usr/bin/env python3
"""The benchmark table: PoseLib on all threads next to jpo's backends, on the same problem sets.

    tools/bench.py [--build build/release] [--backends cpu,metal,auto] [--batches 1,32,300,3000]
                   [--points 200,2000] [--inliers 0.7,0.3,0.1,0] [--repeat 3]

Runs `jpo bench` (which also writes each set to <build>/bench), then poselib_ref on every set with the
same iteration limits, and prints one Markdown table: milliseconds end to end, best of the repeats.
"""
import argparse
import csv
import os
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build/release")
    ap.add_argument("--backends", default="cpu")
    ap.add_argument("--batches", default="1,32,300,3000")
    ap.add_argument("--points", default="200,2000")
    ap.add_argument("--inliers", default="0.7,0.3,0.1,0")
    ap.add_argument("--repeat", default="3")
    args = ap.parse_args()
    out = os.path.join(args.build, "bench")
    os.makedirs(out, exist_ok=True)
    table = os.path.join(out, "jpo.csv")
    subprocess.run([os.path.join(args.build, "jpo"), "bench", "--backends", args.backends, "--batches", args.batches,
                    "--points", args.points, "--inliers", args.inliers, "--repeat", args.repeat, "--save-dir", out,
                    "--csv", table], check=True, stdout=subprocess.DEVNULL)
    rows = {}
    with open(table) as f:
        for r in csv.DictReader(f):
            rows.setdefault((r["kind"], r["batch"], r["points"], r["inliers"]), {})[r["backend"]] = float(r["seconds"])
    ref = os.path.join(args.build, "poselib_ref")
    backends = args.backends.split(",")
    names = (["poselib"] if os.path.exists(ref) else []) + backends
    print("| kind | batch | points | inliers | " + " | ".join(f"{n} ms" for n in names) + " |")
    print("|---|---|---|---|" + "---|" * len(names))
    for key, times in rows.items():
        kind, batch, points, inliers = key
        if "poselib" in names:
            path = os.path.join(out, f"{kind}_b{batch}_n{points}_i{inliers}.jpop")
            text = subprocess.run([ref, path, "--repeat", args.repeat], check=True, capture_output=True, text=True).stdout
            times["poselib"] = float(text.split(" problems in ")[1].split(" ms")[0]) / 1e3
        cells = " | ".join(f"{times[n] * 1e3:.2f}" if n in times else "-" for n in names)
        print(f"| {kind} | {batch} | {points} | {inliers} | {cells} |", flush=True)


if __name__ == "__main__":
    main()
