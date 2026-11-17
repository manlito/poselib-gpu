"""The live demo's data: real matches between KITTI frames, and a small point cloud to register a camera against.

    python3 tools/demo_data.py --jkp ../just-keypoints/build/release/jkp --jmt ../just-matching/build/release/jmt \
        --jmt-weights ../just-matching/weights/flexglue.jmtw --images ../just-matching/web/samples

Keypoints come from just-keypoints (ALIKED, its default model) and matches from just-matching (FlexGlue), both run
here once; the demo only reads what this writes, so it needs neither of them nor their weights:

    web/samples/pairs.json   three pairs: consecutive frames, a corner passed twice, two streets that look alike
    web/samples/cloud.json   points triangulated from frames 100 and 106, and the matches of frame 103 to them
    web/samples/*.jpg        the images, copied from --images

The cloud: the relative pose of frames 100 and 106 from `jpo run` (CPU backend, threshold 1 px), then each inlier
triangulated (midpoint of the two rays) and kept if it lies in front of both cameras, reprojects within 1 px in
both and its rays meet at 1 degree or more. Frame 103 is matched to frame 100; a match whose frame-100 keypoint
has a point becomes a 2D-3D correspondence. The scale is the baseline: frames 100 and 106 are 1 unit apart.

Standard library only. File formats: tools/jpofile.py and cli/problem_file.h (.jpop), just-keypoints' tensor
container (.jkpt, .jmtt).
"""
import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from array import array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jpofile  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# KITTI odometry sequence 00 and raw drive 2011_10_03_drive_0027 (the same drive), rectified: one pinhole camera.
KITTI = {"model": 1, "params": [718.856, 718.856, 607.1928, 185.2157]}
PAIRS = [
    ("consecutive", "Consecutive frames", "kitti-1", "kitti-3",
     "Frames 100 and 106 of KITTI sequence 00, 0.6 s apart: the car turns right into a side street."),
    ("revisit", "A corner passed twice", "kitti-revisit-1", "kitti-revisit-2",
     "Frames 745 and 3679 of the same drive: one street corner, minutes apart, 22 degrees apart in heading."),
    ("alike", "Two streets that look alike", "kitti-alike-1", "kitti-alike-2",
     "Frames 3100 and 1300: two streets 270 m apart that look alike. There is no true pose: whatever the solver fits to these matches is wrong, and a reconstruction drops the pair for its few inliers."),
]
IMAGES = sorted({p[2] for p in PAIRS} | {p[3] for p in PAIRS} | {"kitti-2"})


def read_tensors(path):
    """just-keypoints' tensor container: (meta, {name: (shape, flat list)}). float32 and int32 only."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"JKPW":
        raise ValueError(f"{path}: not a tensor file")
    _, n_meta, n_tensors = struct.unpack_from("<III", data, 4)
    pos = 16

    def s():
        nonlocal pos
        (n,) = struct.unpack_from("<H", data, pos)
        pos += 2 + n
        return data[pos - n:pos].decode()

    meta = {}
    for _ in range(n_meta):
        k = s()
        meta[k] = s()
    tensors = {}
    for _ in range(n_tensors):
        name = s()
        dtype, ndim = struct.unpack_from("<BB", data, pos)
        pos += 2
        shape = struct.unpack_from(f"<{ndim}I", data, pos)
        pos += 4 * ndim
        offset, nbytes = struct.unpack_from("<QQ", data, pos)
        pos += 16
        if dtype not in (0, 2):
            continue
        a = array("f" if dtype == 0 else "i")
        a.frombytes(data[offset:offset + nbytes])
        tensors[name] = (shape, a.tolist())
    return meta, tensors


def keypoints(path):
    shape, flat = read_tensors(path)[1]["keypoints"]
    return [(flat[2 * i], flat[2 * i + 1]) for i in range(shape[0])]


def matches(path):
    shape, flat = read_tensors(path)[1]["matches"]
    return [(int(flat[2 * i]), int(flat[2 * i + 1])) for i in range(shape[0])]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return r.stdout + r.stderr


def write_relpose(path, cam, xy_a, xy_b):
    """One relative-pose problem as .jpop (cli/problem_file.h)."""
    with open(path, "wb") as f:
        f.write(b"JPOP" + struct.pack("<III", 1, 0, 1))
        f.write(struct.pack("<I", len(xy_a)))
        for _ in range(2):
            f.write(struct.pack("<i8d", cam["model"], *(cam["params"] + [0.0] * (8 - len(cam["params"])))))
        f.write(struct.pack("<7d", *([math.nan] * 7)))
        for pts in (xy_a, xy_b):
            f.write(struct.pack(f"<{2 * len(pts)}d", *[v for p in pts for v in p]))
        f.write(bytes([2] * len(xy_a)))


def mat_vec(R, v):
    return [sum(R[r][c] * v[c] for c in range(3)) for r in range(3)]


def mat_t_vec(R, v):
    return [sum(R[c][r] * v[c] for c in range(3)) for r in range(3)]


def triangulate(cam, R, t, xa, xb):
    """Midpoint of the two rays; camera a is the world. None unless in front of both, within 1 px, angle >= 1 deg."""
    fx, fy, cx, cy = cam["params"]
    ra = [(xa[0] - cx) / fx, (xa[1] - cy) / fy, 1.0]
    rb = mat_t_vec(R, [(xb[0] - cx) / fx, (xb[1] - cy) / fy, 1.0])
    cb = [-v for v in mat_t_vec(R, t)]
    dot = lambda u, v: sum(p * q for p, q in zip(u, v))
    a, b, c = dot(ra, ra), dot(ra, rb), dot(rb, rb)
    d, e = dot(ra, cb), dot(rb, cb)
    den = a * c - b * b
    if den < 1e-12:
        return None
    s, u = (d * c - b * e) / den, (b * d - a * e) / den
    X = [(s * ra[i] + cb[i] + u * rb[i]) / 2 for i in range(3)]
    cos = b / math.sqrt(a * c)
    if math.degrees(math.acos(max(-1.0, min(1.0, cos)))) < 1.0:
        return None
    Xb = [p + q for p, q in zip(mat_vec(R, X), t)]
    if X[2] <= 0 or Xb[2] <= 0:
        return None
    for P, x in ((X, xa), (Xb, xb)):
        if math.hypot(fx * P[0] / P[2] + cx - x[0], fy * P[1] / P[2] + cy - x[1]) > 1.0:
            return None
    return X


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jkp", required=True, help="just-keypoints' command-line tool")
    ap.add_argument("--jmt", required=True, help="just-matching's command-line tool")
    ap.add_argument("--jmt-weights", required=True, help="FlexGlue's weights as just-matching reads them")
    ap.add_argument("--images", required=True, help="folder with the KITTI sample images (just-matching's web/samples)")
    ap.add_argument("--jpo", default=os.path.join(ROOT, "build", "release", "jpo"), help="this repository's CLI")
    ap.add_argument("--backend", default="cpu", help="backend for jkp and jmt (cpu, metal, cuda)")
    ap.add_argument("--out", default=os.path.join(ROOT, "web", "samples"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        kp = {}
        for name in IMAGES:
            img = os.path.join(args.images, name + ".jpg")
            run([args.jkp, "extract", img, "--backend", args.backend, "--out", os.path.join(tmp, name + ".jkpt")])
            kp[name] = keypoints(os.path.join(tmp, name + ".jkpt"))
            shutil.copyfile(img, os.path.join(args.out, name + ".jpg"))

        def match(a, b):
            out = os.path.join(tmp, f"{a}--{b}.jmtt")
            run([args.jmt, "match", os.path.join(args.images, a + ".jpg"), os.path.join(args.images, b + ".jpg"),
                 "--kpts-a", os.path.join(tmp, a + ".jkpt"), "--kpts-b", os.path.join(tmp, b + ".jkpt"),
                 "--weights", args.jmt_weights, "--backend", args.backend, "--out", out])
            return matches(out)

        r2 = lambda v: round(v, 2)
        pairs, kept = [], {}
        for pid, title, a, b, note in PAIRS:
            m = match(a, b)
            kept[pid] = m
            pairs.append({
                "id": pid, "title": title, "note": note, "a": a + ".jpg", "b": b + ".jpg",
                "xyA": [r2(v) for i, _ in m for v in kp[a][i]], "xyB": [r2(v) for _, j in m for v in kp[b][j]],
            })
            print(f"{pid}: {len(m)} matches from {len(kp[a])} and {len(kp[b])} keypoints")
        common = {"camera": KITTI, "size": [1241, 376],
                  "source": "ALIKED keypoints (just-keypoints) and FlexGlue matches (just-matching), computed once"}
        with open(os.path.join(args.out, "pairs.json"), "w") as f:
            json.dump({**common, "pairs": pairs}, f, separators=(",", ":"))

        # The cloud from the consecutive pair (frames 100 and 106).
        a, b = "kitti-1", "kitti-3"
        m = kept["consecutive"]
        prob, res = os.path.join(tmp, "pair.jpop"), os.path.join(tmp, "pair.txt")
        write_relpose(prob, KITTI, [kp[a][i] for i, _ in m], [kp[b][j] for _, j in m])
        run([args.jpo, "run", prob, "--backend", "cpu", "-o", res])
        _, _, results = jpofile.load_results(res)
        r = results[0]
        if r["status"] != 0:
            raise SystemExit("the consecutive pair has no pose")
        R, t = jpofile.quat_to_rot(r["pose"][:4]), r["pose"][4:7]
        points = {}
        for k, (i, j) in enumerate(m):
            if r["mask"][k] == "1":
                X = triangulate(KITTI, R, t, kp[a][i], kp[b][j])
                if X is not None:
                    points[i] = X
        print(f"cloud: {r['n_inliers']} inliers of {len(m)}, {len(points)} points triangulated")

        # Frame 103 against frame 100's keypoints that have a point.
        m2 = [(i, j) for i, j in match("kitti-2", a) if j in points]
        r4 = lambda v: round(v, 4)
        cloud = {
            **common, "image": "kitti-2.jpg",
            "note": "Frame 103, between the two frames the points were triangulated from (100 and 106).",
            "xy": [r2(v) for i, _ in m2 for v in kp["kitti-2"][i]],
            "xyz": [r4(v) for _, j in m2 for v in points[j]],
            "cloud": [r4(v) for X in points.values() for v in X],
            "cameras": [{"name": "frame 100", "centre": [0, 0, 0]},
                        {"name": "frame 106", "centre": [r4(v) for v in (-c for c in mat_t_vec(R, t))]}],
        }
        with open(os.path.join(args.out, "cloud.json"), "w") as f:
            json.dump(cloud, f, separators=(",", ":"))
        print(f"frame 103: {len(m2)} matches to points")


if __name__ == "__main__":
    main()
