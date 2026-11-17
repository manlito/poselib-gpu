# poselib-gpu

Robust relative pose and absolute pose (PnP) for calibrated cameras, the geometry an incremental
reconstruction needs between matching and bundle adjustment. The five-point solver and P3P inside LO-RANSAC,
with non-linear refinement, ported from [PoseLib](https://github.com/PoseLib/PoseLib). One C++ library with a
small C API, no dependencies, and hand-written GPU kernels for Metal, CUDA and WebGPU where a GPU is measured
faster than the CPU.

:::{important}
**Use [PoseLib](https://github.com/PoseLib/PoseLib).** This project is a test bed for batching and GPU
kernels, not a replacement for it. It adds nothing to PoseLib except speed: no new solvers, models or
features, and only a subset of what PoseLib offers. The methods, and the credit for them, are PoseLib's. For
real work, PoseLib is the library to depend on.
:::

```
just-keypoints  ->  just-matching  ->  poselib-gpu           ->  just-ba
keypoints           matches per pair    verified matches,         tracks, triangulation,
                                        relative poses, PnP       bundle adjustment
```

- **Two functions, each over a batch**: `jpo_relative_pose` for image pairs, `jpo_absolute_pose` for images to
  register against known 3D points. A call takes every candidate pair, or every candidate image, at once.
- **As accurate as PoseLib**: on generated test sets every estimator stays within PoseLib's own spread over
  random seeds, in double on the CPU and in float32 on the GPU.
- **Fast on the CPU**: all threads, problems that are done drop out early. An easy pair of 2000 matches takes
  0.85 ms on an M5 Pro, a quarter of PoseLib's time on the same machine.
- **The GPU where it wins**: hard problems (low inlier ratios, pairs that do not overlap) in batches. 300 pairs
  of 2000 matches at 30 % inliers take 240 ms on Metal against 512 ms on the CPU, and 473 ms on an RTX 2070
  against 3.6 s on its 6-core CPU. The default backend, `auto`, starts every problem on the CPU and moves only
  the ones still running to the GPU.
- **Everywhere**: Linux, macOS and Windows natively, and the browser through WebAssembly and WebGPU.
- **Six camera models**, with COLMAP's names, ids and parameter order, from `SIMPLE_PINHOLE` to
  `OPENCV_FISHEYE`.

:::{admonition} Try it in your browser
<a href="demo/index.html">The live demo</a> runs the library on real KITTI matches: the relative pose of a
pair with its inliers and epipolar lines, and a camera registered against a point cloud, on the CPU and on
your GPU, with wrong matches and noise you add. <a href="benchmark/index.html">The benchmark</a> times fixed
problem sets in your browser. Both need JavaScript promise integration (Chrome or Edge 137 or later); WebGPU
is optional.
:::

```{toctree}
:maxdepth: 2
:caption: Guide

install
quickstart
concepts
browser
accuracy-and-speed
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
references
```
