"""Readers for problem files (.jpop) and result files; see cli/problem_file.h. Standard library only."""
import math
import struct


def load_problems(path):
    """Returns (kind, problems); each problem is a dict with n, gt_pose (7 floats) and gt_inlier (bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version, kind, count = struct.unpack_from("<4sIII", data, 0)
    if magic != b"JPOP" or version != 1:
        raise ValueError(f"{path}: not a .jpop file")
    pos = 16
    problems = []
    for _ in range(count):
        (n,) = struct.unpack_from("<I", data, pos)
        pos += 4 + 68 * (2 if kind == 0 else 1)
        gt_pose = struct.unpack_from("<7d", data, pos)
        pos += 56 + 16 * n + (16 if kind == 0 else 24) * n
        problems.append({"n": n, "gt_pose": gt_pose, "gt_inlier": data[pos:pos + n]})
        pos += n
    return kind, problems


def load_results(path):
    """Returns (seconds, tool, results); each result is a dict."""
    results = []
    with open(path) as f:
        head = f.readline().split()
        seconds, tool = float(head[4]), head[5]
        for line in f:
            p = line.split()
            results.append({
                "status": int(p[0]), "n_inliers": int(p[1]), "iterations": int(p[2]), "score": float(p[3]),
                "pose": [float(v) for v in p[4:11]], "tri_angle": float(p[11]),
                "mask": "" if p[12] == "-" else p[12],
            })
    return seconds, tool, results


def quat_to_rot(q):
    n = math.sqrt(sum(v * v for v in q))
    w, x, y, z = (v / n for v in q)
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def pose_errors(kind, est, gt):
    """(rotation error in degrees, translation error): the angle between directions for a relative pose,
    the distance between camera centres for an absolute pose."""
    Re, Rg = quat_to_rot(est[:4]), quat_to_rot(gt[:4])
    trace = sum(Re[i][j] * Rg[i][j] for i in range(3) for j in range(3))
    rot = math.degrees(math.acos(max(-1.0, min(1.0, 0.5 * (trace - 1.0)))))
    te, tg = est[4:], gt[4:]
    if kind == 0:
        ne, ng = math.sqrt(sum(v * v for v in te)), math.sqrt(sum(v * v for v in tg))
        if ne == 0 or ng == 0:
            return rot, 180.0
        c = sum(a * b for a, b in zip(te, tg)) / (ne * ng)
        return rot, math.degrees(math.acos(max(-1.0, min(1.0, c))))
    ce = [-sum(Re[r][c] * te[r] for r in range(3)) for c in range(3)]
    cg = [-sum(Rg[r][c] * tg[r] for r in range(3)) for c in range(3)]
    return rot, math.sqrt(sum((a - b) ** 2 for a, b in zip(ce, cg)))


def auc(errors, limit):
    """Area under the recall curve up to `limit`, as the pose AUC of the matching literature."""
    errors = sorted(errors)
    n = len(errors)
    area = prev_e = prev_r = 0.0
    for i, e in enumerate(errors):
        if e >= limit:
            break
        r = (i + 1) / n
        area += (e - prev_e) * (prev_r + r) * 0.5
        prev_e, prev_r = e, r
    return (area + (limit - prev_e) * prev_r) / limit
