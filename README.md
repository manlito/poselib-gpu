# poselib-gpu

Robust relative pose and absolute pose (PnP) for calibrated cameras: the five-point solver and P3P inside
LO-RANSAC, with non-linear refinement. A port of the parts of [PoseLib](https://github.com/PoseLib/PoseLib)
that an incremental reconstruction needs, written like its siblings just-keypoints, just-matching and
just-ba: a C API over a C++ core with no dependencies, and hand-written GPU kernels for Metal, CUDA and
WebGPU where a GPU is measured faster.

> [!IMPORTANT]
> **Use [PoseLib](https://github.com/PoseLib/PoseLib).** This repository is a test bed for batching and GPU
> kernels, not a replacement for it. It adds nothing to PoseLib except speed: no new solvers, models or
> features, and only a subset of what PoseLib offers. The methods, and the credit for them, are PoseLib's. For
> real work, PoseLib is the library to depend on.

```
just-keypoints  ->  just-matching  ->  poselib-gpu          ->  just-ba
keypoints           matches per pair    verified matches,        tracks, triangulation,
                                        relative poses, PnP      bundle adjustment
```

Plan and decisions: [PLAN.md](PLAN.md). Measurements: [RESULTS.md](RESULTS.md). Documentation site, live demo
and benchmark page: [docs/README.md](docs/README.md).

## What it does

Two functions, each over a batch of problems:

```c
#include <jpo/jpo.h>

jpo_config cfg;
jpo_config_init(&cfg);                 /* backend = auto */
jpo_ctx* ctx = jpo_create(&cfg);

/* B image pairs: matched pixels, concatenated; offsets[B + 1] delimits each pair. */
jpo_relative_pose(ctx, &pairs, NULL, results, inliers);

/* B images to register: pixels and world points, concatenated the same way. */
jpo_absolute_pose(ctx, &resections, NULL, results, inliers);

jpo_destroy(ctx);
```

Each result holds the pose (`qw qx qy qz tx ty tz`, world to camera, or camera a to camera b with a unit
translation), the inlier count, the MSAC score, the iterations drawn and a status; a relative result also
holds the median triangulation angle of its inliers, which is what the choice of a first pair needs. Cameras
are COLMAP's models `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV` and `OPENCV_FISHEYE`.
See [include/jpo/jpo.h](include/jpo/jpo.h).

### Several candidate images per round

An incremental reconstruction can try K candidate images in one `jpo_absolute_pose` call and register
every one whose result passes its own test on `n_inliers`. Set `min_inlier_ratio` so that candidates that
cannot pass give up early instead of running to `max_iterations`:

```c
jpo_ransac_options opt;
jpo_ransac_options_init(&opt);     /* min_inlier_ratio = 0: off */
opt.max_error_px = 4.0;
opt.min_inlier_ratio = 0.25;       /* at most log(1 - confidence) / log(1 - 0.25^3) = 585 iterations */
jpo_absolute_pose(ctx, &candidates, &opt, results, inliers);
/* a stopped candidate still returns its best model and real inlier count: compare n_inliers */
```

A problem with at least that inlier ratio has drawn an all-inlier sample with the requested confidence by
then. On the M5 Pro a round of 8 to 32 candidates with a quarter to three quarters failing runs 3.7 to 10
times faster than without the option, with every good candidate still solved, and `auto` sends the
failing candidates of rounds of 16 or more to Metal when that is faster. On the NVIDIA laptop (6 cores,
RTX 2070 Max-Q) the option gains 7 to 14 times on the CPU, and `auto` moves the failing candidates of
every round to CUDA, 1.35 to 4.2 times faster than staying on the CPU (RESULTS.md). In the browser,
where the CPU backend has one thread, `auto` moves rounds of two candidates or more to WebGPU after one CPU
hypothesis each, within 8 % of WebGPU alone. The option also
applies to `jpo_relative_pose`, but with five-point samples the cap is near 10,000 anyway, so leave it at 0
for pairs. CLI: `--min-inlier-ratio r` for `jpo run` and `jpo bench`; JavaScript: `minInlierRatio`.

## Backends

| Backend | Where | Notes |
|---|---|---|
| `cpu` | everywhere | All threads, double. The reference; on par with PoseLib on the same machine. |
| `metal` | Apple | The RANSAC round (2048 hypotheses per problem, solved and scored) as compute kernels. |
| `cuda` | NVIDIA | The same kernels from the same source. |
| `webgpu` | Dawn natively, browsers through Emscripten | A WGSL port of the kernels. |
| `auto` | default | Starts every problem on the CPU and moves the ones that are still running to the GPU. |

A GPU round only pays off on problems the CPU cannot settle quickly: low inlier ratios and pairs that do not
match at all, in batches. Easy problems finish on the CPU in a fraction of a millisecond, so `auto` never
sends them to the GPU. The measured tables are in RESULTS.md.

## Build

```sh
cmake -S . -B build/release -G Ninja
cmake --build build/release
ctest --test-dir build/release
```

Metal is built on Apple platforms and CUDA where `nvcc` is found; `-DJPO_METAL=OFF`, `-DJPO_CUDA=OFF` and
`-DJPO_WEBGPU=OFF` turn backends off. Optional pieces, none of them needed by the library:

```sh
tools/fetch_reference.sh       # PoseLib + Eigen: reference tool and parity tests
tools/fetch_webgpu.sh          # Dawn (native WebGPU) and emdawnwebgpu (browser)
```

The incremental reconstruction test is built when just-ba is found next to this repository
(`-DJPO_JUST_BA_DIR=...`).

From another CMake project, after `cmake --install build/release --prefix <prefix>`:

```cmake
find_package(jpo 0.1 REQUIRED)        # components: the backends, e.g. COMPONENTS cuda
target_link_libraries(my_app PRIVATE jpo::jpo)
```

A project that links the static library needs C++ enabled even when its code is C. `add_subdirectory` gives
the same `jpo::jpo` target.

WebAssembly (Emscripten 7.0.0):

```sh
emcmake cmake -S . -B build/wasm -G Ninja
cmake --build build/wasm          # build/wasm/web: jpo_wasm.js, jpo_wasm.wasm, jpo.js, test.html, bench.html, demo.html, benchmark.html
```

## Python

The module `poselib_gpu` (nanobind, [bindings/python](bindings/python/README.md)) wraps the C API over lists
of NumPy arrays:

```python
import poselib_gpu as pg

ctx = pg.Context()                                       # backend "auto"
cam = pg.Camera("PINHOLE", [fx, fy, cx, cy])
r = ctx.relative_pose([xy_a], [xy_b], cam)               # r.pose, r.n_inliers, r.inliers, r.status, ...
r = ctx.absolute_pose([xy], [xyz], cam, min_inlier_ratio=0.25)
```

From PyPI: `pip install poselib-gpu` (CPU, and Metal on macOS; Linux x86_64, macOS arm64, Windows x86_64),
`poselib-gpu-cu12` or `poselib-gpu-cu13` (CPU and CUDA; Linux and Windows x86_64; the CUDA runtime linked
statically, so only the driver is needed). From this checkout:

```sh
pip install .                                            # the package, through pyproject.toml
# or a CMake build with the module and its tests (needs nanobind==3.1.0 and pytest for that Python):
cmake -S . -B build/py -G Ninja -DJPO_BUILD_PYTHON=ON -DPython_EXECUTABLE=<python>
cmake --build build/py && ctest --test-dir build/py -R python
```

## Releases

`.github/workflows/release.yml` runs on every push to `main`, a merge included (not on changes to Markdown,
`docs/` or `web/` only). It builds and tests seven wheels (the three packages on their platforms) and the sdist
on hosted runners, publishes them to PyPI through trusted publishing, and tags the commit `vX.Y.Z`.
`gh workflow run release` builds and tests without publishing; `-f publish=true` publishes too.

The version is chosen by the workflow (`packaging/version.py next`): the highest of the `vX.Y.Z` tags and the
three packages' versions on PyPI, with its patch bumped, or `project(VERSION)` in CMakeLists.txt when that is
higher, which is how a minor or major release is made (raise it in the pull request that warrants it). With
nothing released, `project(VERSION)` itself (0.1.0). `packaging/version.py set` writes the version into the
checkout before each build; it is never committed. No GPU runs in CI: before merging a GPU change, run the GPU
tests by hand.

First release of each package: PyPI accepts one pending trusted publisher per repository and workflow, so the
packages are created one at a time. Before the first merge, add a pending publisher for `poselib-gpu` (owner
`manlito`, repository `poselib-gpu`, workflow `release.yml`, no environment). The release uploads
`poselib-gpu` and fails at `poselib-gpu-cu12`; add that package's pending publisher, re-run the failed publish
job (`gh run rerun <run> --failed`; files already up are skipped), then do the same for `poselib-gpu-cu13`.
After that all three are ordinary trusted publishers and each release publishes them in one run.

## Command line

```sh
jpo gen relpose -o pairs.jpop --count 200 --points 1000 --inliers 0.3     # generated problems
jpo run pairs.jpop --backend auto -o results.txt                          # poses, masks, accuracy
jpo bench --backends cpu,metal,auto                                       # the timing grid
jpo candidates --k 8,16,32 --failing 0.25,0.5,0.75 --ratio 0.25           # K-candidate rounds
python3 tools/accuracy.py        # the accuracy sets, PoseLib against itself and against jpo
python3 tools/bench.py --backends cpu,metal,auto     # the timing grid with PoseLib alongside
```

## Tests

| Test | What it checks |
|---|---|
| `core_test` | Camera models, both solvers on exact data, both estimators on every camera model, hard cases. |
| `parity_test` | Each solver against PoseLib's on the same samples. |
| `precision_test` | The float32 build of the kernels against the double build. |
| `backend_test <gpu>` | A GPU backend against the float CPU build: the same best hypothesis per round, the same problems solved. |
| `incremental_test` | An incremental reconstruction of three generated scenes with just-ba. |
| `package` | Installs the build, builds a C project against `find_package(jpo)` and registers a camera on every backend. |
| `web/test.html` | The generated sets in the browser, next to `jpo webcheck` natively. |
| `web/bench.html` | `jpo bench` cells and candidate rounds in the browser, for the `auto` rule there (`tools/browser_bench.mjs` runs it in headless Chrome). |

Hosted CI runs the CPU tests and the WebAssembly build; the release workflow runs the Python tests against
every wheel. GPU tests are run by hand on the machine that has
the GPU.

## Licence

Apache-2.0 ([LICENSE](LICENSE)), as just-ba. The solvers are ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson); the notice is in
[LICENSE.PoseLib](LICENSE.PoseLib) and at the top of the ported files.
