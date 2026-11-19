# poselib-gpu

Robust relative pose and absolute pose (PnP) for calibrated cameras, over batches of problems: the five-point
solver and P3P inside LO-RANSAC, with non-linear refinement, ported from
[PoseLib](https://github.com/PoseLib/PoseLib), with hand-written GPU kernels for Metal and CUDA where a GPU is
measured faster than the CPU.

**Use [PoseLib](https://github.com/PoseLib/PoseLib) for real work.** This project is a test bed for batching and
GPU kernels, not a replacement for it. It adds nothing to PoseLib except speed: no new solvers, models or
features, and only a subset of what PoseLib offers. The methods, and the credit for them, are PoseLib's
(BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; the notice ships in the package).

## Install

One of three packages, all providing the module `poselib_gpu`:

| Package | Platforms | Backends |
|---|---|---|
| `poselib-gpu` | Linux x86_64, macOS arm64, Windows x86_64 | CPU; Metal on macOS |
| `poselib-gpu-cu12` | Linux x86_64, Windows x86_64 | CPU and CUDA, built with CUDA 12 (GPUs from Pascal, sm_60) |
| `poselib-gpu-cu13` | Linux x86_64, Windows x86_64 | CPU and CUDA, built with CUDA 13 (GPUs from Turing, sm_75) |

```bash
pip install poselib-gpu          # or poselib-gpu-cu12 / poselib-gpu-cu13
```

The CUDA packages link the CUDA runtime statically: they need only an NVIDIA driver that supports their CUDA
version, and no CUDA toolkit. Install one package, not several.

## Use

```python
import numpy as np
import poselib_gpu as pg

ctx = pg.Context()                      # "auto": every problem starts on the CPU; those still
                                        # running move to the GPU where that is measured faster
cam = pg.Camera("PINHOLE", [fx, fy, cx, cy])

# Image pairs: one (m, 2) array of matched pixels per pair, in image a and in image b.
r = ctx.relative_pose([xy_a0, xy_a1], [xy_b0, xy_b1], cam, max_error_px=1.0)
r.pose        # (n, 7): qw qx qy qz tx ty tz, camera a to camera b, |t| = 1
r.n_inliers   # (n,)
r.inliers     # one boolean mask per pair
r.median_tri_angle_deg, r.status, r.iterations, r.score

# Images to register: pixels and their world points.
r = ctx.absolute_pose([xy0, xy1], [xyz0, xyz1], cam, max_error_px=4.0, min_inlier_ratio=0.25)
r.pose        # world to camera
```

Several cameras: pass a list and give each problem its index (`camera_a`, `camera_b` for pairs, `camera` for
images). Cameras are COLMAP's models with COLMAP's parameter order: `SIMPLE_PINHOLE`, `PINHOLE`,
`SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, `OPENCV_FISHEYE`. A status of `0` (ok) only says a model was found:
accept a pair or a registration by your own rule on `n_inliers`.

`pg.Context("cpu")`, `"metal"` or `"cuda"` picks a backend; `pg.compiled_backends()` lists the ones the package
has, and `pg.VARIANT`, `pg.CUDA_VERSION` and `pg.__version__` say which build it is. The `auto` rule was
measured on an M5 Pro (Metal) and an RTX 2070 Max-Q (CUDA); on other GPUs it is a reasonable start, not a
measurement.

Documentation, the C API and measurements: <https://poselib-gpu.pages.dev> and
<https://github.com/manlito/poselib-gpu>.
