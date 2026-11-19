"""The Python module on generated problems with known poses: every backend this build has, the GPU ones
skipped when no device is found. Run from the build tree (ctest) or against an installed wheel."""

from __future__ import annotations

import numpy as np
import pytest

import poselib_gpu as pg

W, H = 640, 480
CAM = pg.Camera("PINHOLE", [500.0, 505.0, 319.5, 239.5])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def random_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def project(X: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = CAM.params
    return np.stack([fx * X[:, 0] / X[:, 2] + cx, fy * X[:, 1] / X[:, 2] + cy], axis=1)


def rotation_error_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.rad2deg(np.arccos(np.clip(c, -1, 1))))


def scene(rng: np.random.Generator, n: int, inlier_ratio: float, noise: float = 0.3):
    """Points in front of camera a, a second camera (R, t) with |t| = 1, and matches with outliers."""
    R = random_rotation(rng, 10)
    c = rng.normal(size=3)
    c[2] *= 0.2
    c /= np.linalg.norm(c)
    t = -R @ c * 1.0
    t /= np.linalg.norm(t)
    X = np.stack([rng.uniform(-4, 4, n), rng.uniform(-3, 3, n), rng.uniform(6, 12, n)], axis=1)
    xa = project(X) + rng.normal(scale=noise, size=(n, 2))
    xb = project(X @ R.T + t) + rng.normal(scale=noise, size=(n, 2))
    inlier = rng.uniform(size=n) < inlier_ratio
    out = ~inlier
    xb[out] = np.stack([rng.uniform(0, W, out.sum()), rng.uniform(0, H, out.sum())], axis=1)
    return X, R, t, xa, xb, inlier


def backends() -> list[str]:
    out = []
    for b in pg.compiled_backends():
        try:
            pg.Context(b)
        except RuntimeError:
            continue  # built in, but no device here
        out.append(b)
    return out


def test_build_info():
    assert pg.__version__.count(".") == 2
    assert pg.VARIANT in ("cpu", "cu12", "cu13", "source")
    assert "cpu" in pg.compiled_backends()
    if pg.VARIANT.startswith("cu"):
        assert pg.backend_available("cuda") and pg.CUDA_VERSION.startswith(pg.VARIANT[2:])


def test_cameras():
    assert [pg.camera_param_count(m) for m in pg.CAMERA_MODELS] == [3, 4, 4, 5, 8, 8]
    assert pg.camera_param_count(99) == 0
    assert pg.Camera(1, [1, 1, 0, 0]).model == "PINHOLE"
    with pytest.raises(ValueError):
        pg.Camera("PINHOLE", [1, 2, 3])
    with pytest.raises(ValueError):
        pg.Camera("KANNALA", [1, 2, 3])


def test_bad_arguments():
    ctx = pg.Context("cpu")
    with pytest.raises(ValueError):
        ctx.relative_pose([np.zeros((10, 2))], [np.zeros((9, 2))], CAM)
    with pytest.raises(ValueError):
        ctx.relative_pose([np.zeros((10, 3))], [np.zeros((10, 3))], CAM)
    with pytest.raises(ValueError):  # two cameras, no indices
        ctx.relative_pose([np.zeros((10, 2))], [np.zeros((10, 2))], [CAM, CAM])
    with pytest.raises(RuntimeError, match="confidence"):
        ctx.relative_pose([np.zeros((10, 2))], [np.zeros((10, 2))], CAM, confidence=2.0)
    with pytest.raises(ValueError):
        pg.Context("vulkan")


@pytest.mark.parametrize("backend", ["cpu", "auto"] + [b for b in ("metal", "cuda", "webgpu")])
def test_relative_pose(backend):
    if backend not in backends() and backend != "auto":
        pytest.skip(f"{backend}: not in this build, or no device")
    rng = np.random.default_rng(1)
    problems = [scene(rng, 400, 0.6) for _ in range(8)]
    ctx = pg.Context(backend, seed=3)
    assert ctx.backend.startswith(backend if backend != "auto" else "")
    r = ctx.relative_pose([p[3] for p in problems], [p[4] for p in problems], CAM)
    assert len(r) == 8 and r.pose.shape == (8, 7)
    assert r.ok.all()
    for i, (_, R, t, _, _, inlier) in enumerate(problems):
        Rq = quat_to_matrix(r.pose[i, :4])
        assert rotation_error_deg(Rq, R) < 1.0
        assert np.dot(r.pose[i, 4:], t) > 0.98
        assert r.median_tri_angle_deg[i] > 0
        mask = r.inliers[i]
        assert mask.dtype == bool and mask.shape == inlier.shape
        assert mask.sum() == r.n_inliers[i]
        assert (mask & inlier).sum() >= 0.9 * inlier.sum()


def test_relative_pose_too_few_points():
    ctx = pg.Context("cpu")
    r = ctx.relative_pose([np.zeros((3, 2)), np.zeros((0, 2))], [np.zeros((3, 2)), np.zeros((0, 2))], CAM,
                          inliers=False)
    assert list(r.status) == [2, 2] and r.inliers is None


@pytest.mark.parametrize("backend", ["cpu", "auto", "metal", "cuda", "webgpu"])
def test_absolute_pose(backend):
    if backend not in backends() and backend != "auto":
        pytest.skip(f"{backend}: not in this build, or no device")
    rng = np.random.default_rng(2)
    xy, xyz, truth = [], [], []
    for _ in range(6):
        R = random_rotation(rng, 30)
        t = rng.normal(size=3)
        Xc = np.stack([rng.uniform(-3, 3, 300), rng.uniform(-2, 2, 300), rng.uniform(4, 10, 300)], axis=1)
        Xw = (Xc - t) @ R  # R^T (Xc - t)
        x = project(Xc) + rng.normal(scale=0.5, size=(300, 2))
        out = rng.uniform(size=300) > 0.5
        x[out] = np.stack([rng.uniform(0, W, out.sum()), rng.uniform(0, H, out.sum())], axis=1)
        xy.append(x)
        xyz.append(Xw)
        truth.append((R, t))
    # Two cameras, each image naming its own.
    cams = [CAM, pg.Camera("SIMPLE_PINHOLE", [500.0, 319.5, 239.5])]
    r = pg.Context(backend, seed=5).absolute_pose(xy, xyz, cams, camera=0)
    assert r.ok.all()
    for i, (R, t) in enumerate(truth):
        assert rotation_error_deg(quat_to_matrix(r.pose[i, :4]), R) < 0.5
        assert np.linalg.norm(r.pose[i, 4:] - t) < 0.05
        assert r.iterations[i] >= 100
        assert r.median_tri_angle_deg[i] == 0


def test_same_seed_same_result():
    rng = np.random.default_rng(4)
    _, _, _, xa, xb, _ = scene(rng, 300, 0.4)
    r1 = pg.Context("cpu", seed=7, threads=2).relative_pose([xa], [xb], CAM)
    r2 = pg.Context("cpu", seed=7, threads=1).relative_pose([xa], [xb], CAM)
    np.testing.assert_array_equal(r1.pose, r2.pose)
    assert r1.iterations[0] == r2.iterations[0]


def test_min_inlier_ratio_stops_early():
    rng = np.random.default_rng(6)
    xy = [np.stack([rng.uniform(0, W, 200), rng.uniform(0, H, 200)], axis=1)]
    xyz = [rng.uniform(-3, 3, size=(200, 3)) + [0, 0, 8]]
    ctx = pg.Context("cpu")
    full = ctx.absolute_pose(xy, xyz, CAM, max_iterations=5000)
    early = ctx.absolute_pose(xy, xyz, CAM, max_iterations=5000, min_inlier_ratio=0.25)
    assert early.iterations[0] < full.iterations[0]
