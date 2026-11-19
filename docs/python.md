# Python

The module `poselib_gpu` wraps the C API: the same batches, the same options, the same results, as NumPy
arrays.

## Install

One of three packages from PyPI, all providing `poselib_gpu`:

| Package | Platforms | Backends |
|---|---|---|
| `poselib-gpu` | Linux x86_64, macOS arm64 (macOS 13 or later), Windows x86_64 | CPU; Metal on macOS |
| `poselib-gpu-cu12` | Linux x86_64, Windows x86_64 | CPU and CUDA, built with CUDA 12 (GPUs from Pascal, sm_60) |
| `poselib-gpu-cu13` | Linux x86_64, Windows x86_64 | CPU and CUDA, built with CUDA 13 (GPUs from Turing, sm_75) |

```bash
pip install poselib-gpu          # or poselib-gpu-cu12, poselib-gpu-cu13
```

Python 3.12 or later; one wheel per platform covers every later Python. The CUDA packages link the CUDA runtime
statically, so they need only an NVIDIA driver recent enough for their CUDA version: no toolkit, no other
package. Install one of the three, not several. WebGPU is not in the wheels; it is in source builds and in the
browser.

## Relative pose

```python
import numpy as np
import poselib_gpu as pg

ctx = pg.Context()                                   # backend "auto"
cam = pg.Camera("PINHOLE", [fx, fy, cx, cy])

r = ctx.relative_pose([xy_a0, xy_a1], [xy_b0, xy_b1], cam)
```

`xy_a` and `xy_b` hold one `(m, 2)` array of matched pixels per pair, in image a and in image b, as
`jpo_pairs` does packed. The result has one entry per pair:

| Field | Shape | Meaning |
|---|---|---|
| `pose` | `(n, 7)` | `qw qx qy qz tx ty tz`, camera a to camera b, with `|t| = 1` |
| `n_inliers` | `(n,)` | Inlier count |
| `inliers` | list of `(m,)` | One boolean mask per pair (`None` with `inliers=False`) |
| `median_tri_angle_deg` | `(n,)` | Median triangulation angle of the inliers |
| `score` | `(n,)` | MSAC score, pixels² |
| `iterations` | `(n,)` | Hypotheses drawn |
| `status` | `(n,)` | 0 ok, 1 no model, 2 too few points, 3 degenerate (`pg.STATUS`); `r.ok` is `status == 0` |

As in C, status ok only says that a model was found: accept a pair by your own rule on `n_inliers`.

## Absolute pose

```python
r = ctx.absolute_pose([xy0, xy1], [xyz0, xyz1], cam, min_inlier_ratio=0.25)
```

One `(m, 2)` array of pixels and the matching `(m, 3)` world points per image; the pose is world to camera.
`min_inlier_ratio` lets candidates that cannot pass give up early, as in
[Candidate images in one call](concepts.md#candidate-images-in-one-call).

## Options and cameras

Both calls take the RANSAC options as keywords: `max_error_px` (1 px for pairs and 4 px for PnP by default),
`confidence`, `min_iterations`, `max_iterations`, `refine`, `min_inlier_ratio`, and `inliers` to leave the
masks out.

`pg.Camera(model, params)` takes a model by COLMAP's name or id and its parameters in COLMAP's order
([Cameras](api.md#cameras)). Pass one camera for every problem, or a list with each problem's index:
`camera_a` and `camera_b` for pairs (`camera_b` defaults to `camera_a`), `camera` for images.

## Backends

`pg.Context(backend, seed=0, threads=0, cpu_float=False, gpu_round=0)` takes the fields of `jpo_config`. The
backend is `"auto"` (default), `"cpu"`, `"metal"`, `"cuda"` or `"webgpu"`; asking for one the package does not
have, or one without a device, raises `RuntimeError`. `ctx.backend` says which one runs, for example
`"auto(cuda)"`, or `"cpu"` when `auto` finds no GPU. `pg.compiled_backends()` lists the package's backends, and
`pg.VARIANT`, `pg.CUDA_VERSION` and `pg.__version__` say which build it is.

The `auto` rule was measured on an M5 Pro (Metal) and on an RTX 2070 Max-Q with a 6-core CPU (CUDA). On other
machines it is a reasonable start rather than a measurement; `jpo bench` from a source build measures yours
([Accuracy and speed](accuracy-and-speed.md)).

A context runs one call at a time; the call releases Python's lock while it works.
