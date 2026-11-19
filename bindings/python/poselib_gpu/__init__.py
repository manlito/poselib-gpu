"""poselib-gpu: robust relative pose and absolute pose (PnP) for calibrated cameras.

The five-point solver and P3P inside LO-RANSAC, with non-linear refinement, ported from PoseLib
(https://github.com/PoseLib/PoseLib), with GPU kernels for Metal and CUDA where a GPU is measured faster.
Use PoseLib for real work: this is a test bed for batching and GPU kernels.

Every call takes a batch of problems: a list of image pairs, or a list of images to register. Conventions
are the C API's (include/jpo/jpo.h): a pose is ``qw qx qy qz tx ty tz`` and maps a point X to R(q) X + t,
world to camera (absolute) or camera a to camera b with |t| = 1 (relative); camera axes are x right, y
down, z forward; cameras are COLMAP's models with COLMAP's parameter order.

>>> import poselib_gpu as pg
>>> ctx = pg.Context()                       # backend "auto": the GPU where it is faster
>>> cam = pg.Camera("PINHOLE", [800, 800, 320, 240])
>>> r = ctx.relative_pose([xy_a], [xy_b], cam)
>>> r.pose[0], r.n_inliers[0], r.inliers[0]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from . import _core
from ._build_info import CUDA_VERSION, VARIANT, __version__

__all__ = [
    "BACKENDS",
    "CAMERA_MODELS",
    "CUDA_VERSION",
    "Camera",
    "Context",
    "PoseResults",
    "STATUS",
    "VARIANT",
    "__version__",
    "backend_available",
    "camera_param_count",
    "compiled_backends",
]

# jpo_backend
BACKENDS = {"auto": 0, "cpu": 1, "metal": 2, "cuda": 3, "webgpu": 4}
# jpo_camera_model: COLMAP's names and ids
CAMERA_MODELS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
}
# jpo_status
STATUS = {0: "ok", 1: "no_model", 2: "too_few_points", 3: "degenerate"}


def _backend_id(backend: Union[str, int]) -> int:
    if isinstance(backend, str):
        try:
            return BACKENDS[backend.lower()]
        except KeyError:
            raise ValueError(f"unknown backend {backend!r}; one of {sorted(BACKENDS)}") from None
    return int(backend)


def _model_id(model: Union[str, int]) -> int:
    if isinstance(model, str):
        try:
            return CAMERA_MODELS[model.upper()]
        except KeyError:
            raise ValueError(f"unknown camera model {model!r}; one of {list(CAMERA_MODELS)}") from None
    return int(model)


def camera_param_count(model: Union[str, int]) -> int:
    """Number of parameters of a camera model (3 .. 8); 0 for an unknown model id."""
    return _core.camera_param_count(_model_id(model))


def backend_available(backend: Union[str, int]) -> bool:
    """Whether this build has the backend ("cpu" and "auto": always). The device is checked by Context."""
    return _core.backend_available(_backend_id(backend))


def compiled_backends() -> list[str]:
    """The backends this build has, besides "auto"."""
    return [name for name, i in BACKENDS.items() if i and _core.backend_available(i)]


@dataclass(frozen=True)
class Camera:
    """A calibrated camera: one of COLMAP's models (name or id) and its parameters in COLMAP's order."""

    model: Union[str, int]
    params: Sequence[float]

    def __post_init__(self) -> None:
        mid = _model_id(self.model)
        count = _core.camera_param_count(mid)
        if count == 0:
            raise ValueError(f"unknown camera model {self.model!r}")
        params = tuple(float(p) for p in self.params)
        if len(params) != count:
            raise ValueError(f"camera model {self.model!r} takes {count} parameters, got {len(params)}")
        object.__setattr__(self, "model", next(k for k, v in CAMERA_MODELS.items() if v == mid))
        object.__setattr__(self, "params", params)


@dataclass
class PoseResults:
    """One entry per problem, in the order given.

    pose: (n, 7) ``qw qx qy qz tx ty tz``. score: MSAC score in pixels^2. median_tri_angle_deg: relative
    pose only, the median triangulation angle of the inliers (0 for PnP). n_inliers, iterations
    (hypotheses drawn), status (0 ok, 1 no model, 2 too few points, 3 degenerate: see STATUS).
    inliers: a boolean mask per problem, or None when not asked for.

    Status "ok" only says that a model with more inliers than a minimal sample was found: accept a pair or
    a registration by your own rule on n_inliers.
    """

    pose: np.ndarray
    score: np.ndarray
    median_tri_angle_deg: np.ndarray
    n_inliers: np.ndarray
    iterations: np.ndarray
    status: np.ndarray
    inliers: list[np.ndarray] | None

    def __len__(self) -> int:
        return len(self.pose)

    @property
    def ok(self) -> np.ndarray:
        """Status ok, per problem."""
        return self.status == 0


def _pack(arrays: Sequence, cols: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenates a list of (m_i, cols) arrays; returns the rows and the offsets (n + 1)."""
    parts = []
    for a in arrays:
        a = np.asarray(a, dtype=np.float64)
        if a.size == 0:
            a = a.reshape(0, cols)
        if a.ndim != 2 or a.shape[1] != cols:
            raise ValueError(f"each array of {name} must have shape (m, {cols})")
        parts.append(a)
    offsets = np.zeros(len(parts) + 1, dtype=np.uint32)
    np.cumsum([len(p) for p in parts], out=offsets[1:])
    rows = np.concatenate(parts) if parts else np.zeros((0, cols))
    return np.ascontiguousarray(rows, dtype=np.float64), offsets


def _cameras(cameras) -> tuple[list[Camera], np.ndarray, np.ndarray]:
    cams = [cameras] if isinstance(cameras, Camera) else list(cameras)
    if not cams:
        raise ValueError("at least one camera is needed")
    for c in cams:
        if not isinstance(c, Camera):
            raise TypeError("cameras must be Camera objects")
    models = np.array([CAMERA_MODELS[c.model] for c in cams], dtype=np.int32)
    params = np.zeros((len(cams), 8), dtype=np.float64)
    for i, c in enumerate(cams):
        params[i, : len(c.params)] = c.params
    return cams, models, params


def _indices(index, n: int, n_cameras: int, name: str) -> np.ndarray:
    if index is None:
        if n_cameras != 1:
            raise ValueError(f"{name} is needed with more than one camera: an index into cameras per problem")
        return np.zeros(n, dtype=np.uint32)
    idx = np.asarray(index)
    if idx.ndim == 0:
        idx = np.full(n, int(idx))
    if idx.shape != (n,):
        raise ValueError(f"{name} needs one camera index per problem ({n})")
    if n and (idx.min() < 0 or idx.max() >= n_cameras):
        raise ValueError(f"{name} has an index outside cameras")
    return np.ascontiguousarray(idx, dtype=np.uint32)


def _results(raw, offsets: np.ndarray) -> PoseResults:
    pose, score, angle, n_inliers, iterations, status, mask = raw
    inliers = None
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        inliers = [m[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    return PoseResults(pose, score, angle, n_inliers, iterations, status, inliers)


class Context:
    """A solver: the backend, its threads and its GPU device, kept for any number of calls (one at a time).

    backend: "auto" (default: every problem starts on the CPU, and those still running move to the GPU
    when that is measured faster; the CPU alone without a GPU), "cpu", "metal", "cuda" or "webgpu".
    seed: the same seed draws the same samples on every backend. threads: CPU threads, 0 = all.
    cpu_float: run the CPU backend in float32, as the GPU kernels do (for testing). gpu_round: hypotheses
    per problem in one GPU dispatch, 0 = 2048. Raises RuntimeError when the backend cannot be created.
    """

    def __init__(
        self,
        backend: Union[str, int] = "auto",
        *,
        seed: int = 0,
        threads: int = 0,
        cpu_float: bool = False,
        gpu_round: int = 0,
    ) -> None:
        self._ctx = _core.Context(_backend_id(backend), seed, threads, cpu_float, gpu_round)

    @property
    def backend(self) -> str:
        """The backend in use: "cpu", "metal", "cuda", "webgpu" or "auto(<gpu>)"."""
        return self._ctx.backend

    def __repr__(self) -> str:
        return f"poselib_gpu.Context(backend={self.backend!r})"

    def relative_pose(
        self,
        xy_a: Sequence,
        xy_b: Sequence,
        cameras,
        camera_a=None,
        camera_b=None,
        *,
        max_error_px: float = 1.0,
        confidence: float = 0.9999,
        min_iterations: int = 100,
        max_iterations: int = 10000,
        refine: bool = True,
        min_inlier_ratio: float = 0.0,
        inliers: bool = True,
    ) -> PoseResults:
        """Relative poses of image pairs (five-point solver in LO-RANSAC).

        xy_a, xy_b: one (m_i, 2) array of matched pixels per pair, in image a and image b. cameras: a
        Camera, or a list of them with camera_a and camera_b giving each pair's index (or one index for
        every pair). Options as jpo_ransac_options. The pose is camera a to camera b with |t| = 1.
        """
        if len(xy_a) != len(xy_b):
            raise ValueError("xy_a and xy_b need one array per pair")
        a, offsets = _pack(xy_a, 2, "xy_a")
        b, offsets_b = _pack(xy_b, 2, "xy_b")
        if not np.array_equal(offsets, offsets_b):
            raise ValueError("each pair needs as many points in xy_a as in xy_b")
        cams, models, params = _cameras(cameras)
        n = len(xy_a)
        ia = _indices(camera_a, n, len(cams), "camera_a")
        ib = _indices(camera_b if camera_b is not None else camera_a, n, len(cams), "camera_b")
        raw = self._ctx.relative(offsets, a, b, models, params, ia, ib, max_error_px, confidence,
                                 min_iterations, max_iterations, refine, min_inlier_ratio, inliers)
        return _results(raw, offsets)

    def absolute_pose(
        self,
        xy: Sequence,
        xyz: Sequence,
        cameras,
        camera=None,
        *,
        max_error_px: float = 4.0,
        confidence: float = 0.9999,
        min_iterations: int = 100,
        max_iterations: int = 10000,
        refine: bool = True,
        min_inlier_ratio: float = 0.0,
        inliers: bool = True,
    ) -> PoseResults:
        """Absolute poses (PnP: P3P in LO-RANSAC) of images to register against known world points.

        xy: one (m_i, 2) array of pixels per image; xyz: the matching (m_i, 3) world points. cameras: a
        Camera, or a list with camera giving each image's index. min_inlier_ratio > 0 lets candidates that
        cannot pass give up early (several candidate images in one call). The pose is world to camera.
        """
        if len(xy) != len(xyz):
            raise ValueError("xy and xyz need one array per image")
        p2, offsets = _pack(xy, 2, "xy")
        p3, offsets_3 = _pack(xyz, 3, "xyz")
        if not np.array_equal(offsets, offsets_3):
            raise ValueError("each image needs as many points in xy as in xyz")
        cams, models, params = _cameras(cameras)
        idx = _indices(camera, len(xy), len(cams), "camera")
        raw = self._ctx.absolute(offsets, p2, p3, models, params, idx, max_error_px, confidence,
                                 min_iterations, max_iterations, refine, min_inlier_ratio, inliers)
        return _results(raw, offsets)
