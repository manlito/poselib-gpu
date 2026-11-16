# poselib-gpu — plan

Minimal-dependency robust pose estimation for bundle-adjustment pipelines: relative pose and absolute pose
(PnP) with RANSAC and refinement, ported from PoseLib and written the same way as `../just-keypoints`,
`../just-matching` and `../just-ba` (C++ core, no PyTorch or other framework at run time). A fast CPU
library first; hand-written Metal / CUDA / WebGPU kernels only for the work where a GPU is measured faster.

Status: plan only. Nothing is built or measured yet; every number below is an estimate or a target.

Working name: the repository is `poselib-gpu`; the plan uses the C prefix `jpo_` (`jpo.h`, CLI `jpo`),
after `jkp`, `jmt` and `jba`. Rename before M1 if another name is wanted.

## Where it sits

```
just-keypoints  ->  just-matching  ->  poselib-gpu          ->  just-ba
xy (+cov)           matches per pair    verified matches,        tracks, triangulation,
                                        relative poses, PnP      bundle adjustment
```

- just-ba states that matches reaching it are "already geometrically verified by the caller" and that
  relative pose and PnP are "a sizeable amount of code that is not our core" (its document 04). This
  library is that caller-side code.
- just-ba already has track building and robust triangulation (`tracks/`, CPU). They are not repeated here.
- The incremental mapper itself (choose the next image, register, triangulate, adjust) belongs to the final
  project. Here it exists once, as a test (M2), to prove the function set is enough.

## Decisions (proposed)

| Topic | Choice |
|---|---|
| Source | PoseLib (BSD-3-Clause), pinned at one commit in M0. Ported code keeps its copyright notice. PoseLib itself is a test-only reference, as Ceres is in just-ba. |
| Scope | Calibrated cameras only: 5-point relative pose, P3P absolute pose, LO-RANSAC, non-linear refinement. Nothing else until M2 shows it is needed. |
| GPU rule | A GPU path exists only where it is measured faster than our own CPU backend on all threads, on problems a capture really produces. Judged after a path is built and tuned, not before; only work known in advance to be slower (refinement, emulated double) is never written. See "Where the GPU is used". |
| CPU backend | A complete, threaded library in its own right, not only a reference: it is what runs wherever the GPU loses. |
| Batches | Every call takes many problems (pairs, or images to register) at once. The CPU backend spreads them over threads; a GPU path, where one exists, takes the whole batch in one dispatch. |
| GPU work | The RANSAC round, for both estimators: 2048 hypotheses per problem solved and scored against every point in one dispatch, for one problem or a batch. |
| Host work | The stopping rule, refinement, the final polish, and the problems `auto` routes to the CPU. |
| Precision | GPU kernels in float32 (Metal and WGSL have no double). The returned pose is always polished on the host in double. See "Precision". |
| Determinism | Counter-based random numbers: sample = hash(seed, problem, round, hypothesis). Every backend draws the same samples, so backends can be compared hypothesis by hypothesis. |
| Cameras | just-ba's six models, with COLMAP's names, ids and parameter order (`SIMPLE_PINHOLE` ... `OPENCV_FISHEYE`). Keypoints are unprojected to bearings once; solvers never see a camera model. |
| Conventions | Poses as in just-ba: world to camera, `qw qx qy qz tx ty tz`, x right, y down, z forward. Pixels as in just-keypoints: (0,0) is the centre of the top-left pixel. Relative pose: camera a to camera b, as just-ba's relative priors. |
| Backends | `cpu` (reference, double or float), `metal`, `cuda`, `webgpu`; same pluggable structure as just-keypoints. |
| Kernel source | One source for the solvers and residuals, compiled as C++ (CPU), by nvcc and by Metal behind small preludes, as just-ba's `solver.inc` (its D96). WGSL is a port of it, checked operation by operation against the CPU in float. |
| WebGPU toolchain | The same as the other three: Dawn v20261113.214538 natively, emdawnwebgpu and Emscripten 7.0.0 in the browser, single-threaded, JSPI, native wasm exceptions, static library, caller may supply the device. One app links all four libraries into one module. |
| CI | Free hosted runners only: CPU tests and the WebAssembly build. GPU tests are run by hand. |
| Site | Sphinx + MyST with just-ba's theme, built by a workflow started by hand, deployed to Cloudflare Pages with wrangler. Public pages carry no references to this plan, commits or the repository. |

## What an incremental reconstruction needs

| Step | Needs | From |
|---|---|---|
| Verify matches of every pair | relative pose + inlier mask, many pairs per call | here |
| Choose the first pair | inlier count, median triangulation angle of the inliers | here (returned with the relative pose) |
| First two poses | relative pose (translation of unit length) | here |
| First points | triangulation | just-ba `tracks/` |
| Register the next image | absolute pose from 2D-3D matches + inlier mask; several candidate images per call | here |
| New points | triangulation | just-ba `tracks/` |
| Local and global adjustment, filtering | solver | just-ba |

So the whole function set is two estimators and their refinement.

### Left out on purpose

- **Fundamental matrix** (7-point): phones supply intrinsics (just-ba D26).
- **Homography**: with calibrated cameras the 5-point solver handles planar scenes, and pure rotation shows
  as a small triangulation angle. Add it only if M2 picks bad first pairs without it.
- **Unknown focal length, radial distortion solvers, generalized cameras, hybrid pose.**
- **Line features** (PoseLib's point-line PnP solvers). The request read "line relative pose estimation";
  taken as "like relative pose estimation". just-keypoints produces no lines. Say so if lines were meant.
- **Triangulation and track building**: in just-ba.

### Candidates after M2 (own track, like RaCo in just-keypoints)

- **G1 — Gravity-aware solvers**: upright 3-point relative pose and 2-point absolute pose. Phone captures
  carry gravity; fewer points per sample means far fewer rounds at low inlier ratios.
- **G2 — Guided sampling** from just-matching's match scores (PROSAC-like).
- **G3 — Pose with a prior**: score and refine around a tracker pose instead of sampling from nothing.

## Where the GPU is used

The rule: no GPU code for work the CPU does faster. What can be said before measuring anything:

The unit of GPU work is one round: 2048 hypotheses of one problem, solved and scored against all its points
in one dispatch. For a pair with 2000 matches that is 2048 samples x up to 10 essential matrices x 2000
points = 4e7 Sampson errors (estimate); for a resection, 2048 x up to 4 poses x the 2D-3D matches. Nothing
in it depends on anything else in it, which is the shape a GPU is built for, and on Metal the buffers are
shared with the host, so there is no upload or readback copy to pay for.

When that round beats the CPU's serial loop, by estimate (to be measured in M3):

- **Hard problems**: low inlier ratio. The CPU needs thousands of iterations (about 14,000 five-point
  samples at 20 % inliers for 99 % confidence, 145 at 50 %), each scored against every point.
- **Pairs that do not match at all**: the CPU never meets its stopping rule and runs to the iteration
  limit. In pair verification these are a large share of the candidates, so the CPU's worst case is common.
- **Batches**: many problems in one dispatch.
- It also buys quality: 2048 hypotheses where the stopping rule would have accepted the first good one.

When it does not: an easy pair that the CPU settles in a few dozen iterations, and very small problems.
There the GPU computes hypotheses the CPU would have skipped. `auto` has to tell these apart cheaply, for
example by running a short CPU round first and going to the GPU only if it did not finish.

| Work | GPU? | Why |
|---|---|---|
| Relative pose: solve + score + select, one problem or a batch | **Yes, to be proven in M3** | The round above. |
| Absolute pose (PnP): solve + score + select, one problem or a batch | **Yes, to be proven in M3** | Same round, same kernels except the residual. Registration against a noisy map is where low inlier ratios show up. |
| Refinement (6 or 5 unknowns per problem) | No | Microseconds per problem, needs double, threads trivially. |
| Stopping rule, sampling bookkeeping, final polish | No | Tiny. |
| Double emulated with pairs of floats | No | Several times the arithmetic. |
| Triangulation, tracks | No | In just-ba, CPU by its own decision. |

The rule still holds, applied after a fair attempt rather than before one:

- Each GPU path is built, then tuned, then judged (M3). It is judged per estimator and per kind of problem
  (easy, hard, no model; one problem and batches), not as one yes or no.
- `backend = auto` uses the GPU where it measured faster and the CPU elsewhere. A cell where the GPU loses
  after tuning is routed to the CPU; a path is removed only if it wins nowhere.
- In the browser the WebAssembly CPU backend runs on one thread, so the GPU should win over a wider range
  there; measured separately in M6.

## Public API (C, plain buffers)

```c
jpo_ctx* jpo_create(const jpo_config*);        /* backend, device (optional), seed */

/* B pairs. xy_a/xy_b: matched pixels, concatenated; offsets[B+1] delimits each pair. */
int jpo_relative_pose(jpo_ctx*, const jpo_pairs*, const jpo_ransac_options*, jpo_relative_result*);

/* B images. xy: pixels, xyz: world points, concatenated the same way. */
int jpo_absolute_pose(jpo_ctx*, const jpo_resections*, const jpo_ransac_options*, jpo_absolute_result*);

void jpo_destroy(jpo_ctx*);
```

- Inputs name a camera per image (`model`, intrinsics as doubles), shared across problems where equal.
- `jpo_ransac_options`: `max_error_px`, `confidence`, `min_rounds`, `max_rounds`, `refine` (on by default).
- Results per problem: pose (7 doubles), `n_inliers`, `score` (MSAC), `rounds`, `status`; per point an
  inlier byte. Relative results add `median_tri_angle_deg`.
- Host buffers only at first. Device buffers as input (so just-matching's output need not visit the host)
  are added in M5, and only if the copy shows up in the profile.
- `status` separates "no model found", "too few points" and "degenerate" from success.

## RANSAC in rounds

PoseLib's loop is serial: one sample, solve, score, maybe refine, test the stopping rule. Here it runs in
rounds, on every backend, so the CPU and a GPU path do the same work in the same order:

1. **Sample** H minimal sets per problem (small on the CPU, 2048 on a GPU).
2. **Solve** them. 5-point gives up to 10 essential matrices, P3P up to 4 poses.
3. **Score**: Sampson error for relative pose (threshold divided by the focal length, as PoseLib),
   reprojection error through the camera model for PnP; truncated squared error (MSAC) per hypothesis.
4. **Select** the best hypothesis per problem.
5. **Refine** the best on the host in double (the "LO" of LO-RANSAC).
6. **Stop** per problem when the usual confidence bound is met; problems that are done drop out of the
   next round.

On the CPU, rounds are small and a problem stops as soon as it can, which is why an easy pair costs almost
nothing. On a GPU path steps 2-4 are one dispatch for the whole batch and one readback of a few values per
problem; a GPU round computes hypotheses the CPU would have skipped, so the GPU has to win by more than
that waste on easy problems, and wins outright on problems the CPU cannot stop early on. M3 measures both.

## Precision

- Double on the CPU backend is the reference and must agree with PoseLib.
- The same kernel source compiled with `real = float` is the specification for the GPU backends. A GPU
  backend passes when it reproduces that, not when it reproduces double.
- Risk: the 5-point solver (null space, 10x10 elimination, roots of a degree-10 polynomial) loses accuracy
  in float32. M1 measures it before any GPU code exists: share of float hypotheses that land within the
  inlier threshold of their double counterpart, and success rate of the whole estimator.
- Fallback if it is not good enough: solve minimal problems on the host in double (they are a small part
  of the work) and keep only scoring on the GPU. On CUDA, double kernels are also possible.
- Not done: emulating double with pairs of floats on the GPU. It is known to cost several times the
  arithmetic, which is the margin the GPU has to win by.

## Repository layout

```
include/jpo/jpo.h        C API (the only public header)
src/core/                problems, options, sampler, stopping rule, host-side refinement, cameras
src/kernels/             solvers and residuals, one source for CPU, CUDA and Metal
src/backend/cpu/         reference; double or float; threads natively, one thread in WebAssembly
src/backend/metal/       prelude + Objective-C++ glue, shaders compiled at jpo_create
src/backend/cuda/        prelude + glue
src/backend/webgpu/      kernels.wgsl + C++ glue (Dawn native / Emscripten)
cli/                     jpo relpose | pnp | bench
tools/                   dev-only Python: PoseLib reference, synthetic and real problem files, compare, bench
tests/                   solver tests, <backend>_vs_cpu, PoseLib parity, incremental test (M2)
third_party/             PoseLib and Dawn, fetched by script, ignored by git
web/                     WebAssembly entry points, JS wrapper, browser test page
docs/                    documentation site + live demo (Cloudflare Pages)
```

## Milestones

Each milestone is a state that can be run and checked. Numbers are filled in when it closes.

- [ ] **M0 — Reference locked** (M5 laptop)
  - PoseLib pinned and built as a test-only dependency; `tools/` runs it from Python.
  - Problem files: synthetic (noise, outlier ratio, planar, small baseline, each camera model) and real
    (pairs matched by just-keypoints + just-matching; 2D-3D sets from a COLMAP model).
  - Metrics fixed: rotation and translation-direction error, pose AUC at 5/10/20 degrees, inlier-set
    overlap, success rate over seeds.
  - Baseline: PoseLib's time on all CPU threads for one pair, 1000 pairs, and 100 resections. This sets the
    target for M5 and tells whether the GPU can win at all for small batches.
  - **Exit:** `tools/compare.py` reports the metrics for PoseLib against itself across seeds (the noise
    floor every later comparison is read against).

- [ ] **M1 — First poses from our own code** (CPU, C++)
  - C API, batches, counter-based sampler, P3P, 5-point, round-based LO-RANSAC, refinement, six camera models.
  - `jpo relpose` and `jpo pnp` on problem files; tests of each solver against PoseLib's on the same samples.
  - The float build of the same code, and the precision measurement above.
  - **Exit:** on the M0 sets, accuracy and success rate within PoseLib's own seed-to-seed spread, in double.
    A written decision on float32 minimal solvers (keep, or fall back to host solves).

- [ ] **M2 — Enough for an incremental reconstruction** (CPU, with just-ba)
  - A test program, not a product: pair verification, first pair, registration by PnP, just-ba's
    triangulation and solver, on two or three small scenes from exported COLMAP matches.
  - Compared with COLMAP on the same matches: images registered, reprojection error, pose error after alignment.
  - **Exit:** scenes reconstruct; any function found missing is added to the scope here, before it has to be
    written four times. Decides G1-G3 and whether homography comes back.

- [ ] **M3 — Runs on the Mac GPU, and is judged there** (Metal, M5 laptop)
  - The CPU backend made as fast as it should be first (threads over problems, early stopping, no
    allocations per round), timed against PoseLib. This is the number to beat.
  - The Metal round for both estimators: 2048 hypotheses per problem, solve + score + select in one
    dispatch, shared-storage buffers, one problem or a batch. `ops_test` per kernel and `metal_vs_cpu`
    against the float CPU build: same samples, same best hypothesis, same inlier masks except points on the
    threshold.
  - A tuning pass before any verdict: hypotheses per round, threads per hypothesis against per block of
    points, early rejection, what is read back.
  - Measured end to end, from host buffers to poses and masks, for relative pose and PnP: one problem and
    batches of 32, 300 and 3000; 200 and 2000 points; inlier ratios 70 %, 30 %, 10 % and none.
  - **Exit:** M0 metrics as M1 on Metal; the table of where Metal wins and by how much; the rule `auto`
    uses, taken from that table. A path that wins nowhere after tuning is removed, and what was tried is
    written down.

- [ ] **M4 — Runs on NVIDIA** (CUDA, the laptop)
  - The paths that won on Metal, from the same kernel source behind the CUDA prelude; same tests; the same
    table on that machine, with its own `auto` rule (a discrete GPU pays for upload and readback that Metal
    does not).
  - CPU backend built and tested on Linux and Windows.
  - **Exit:** as M3 with `--backend cuda`.

- [ ] **M5 — Fast**
  - Tuning on both machines where the profiles point: rounds for problems of very different sizes in one
    batch, device inputs from just-matching, less waste on easy problems.
  - Benchmark table: problems per second by batch size, points and inlier ratio, for PoseLib on all
    threads, our CPU backend, Metal and CUDA.
  - **Exit:** documented numbers and the final `auto` rule per backend.

- [ ] **M6 — Runs in the browser**
  - The CPU backend in WebAssembly first (one thread): the whole API works there with no GPU.
  - WGSL port of the paths that won, natively through Dawn with the same tests, then Emscripten; device
    supplied by the caller. Measured against the WebAssembly CPU backend in Chrome, with its own `auto` rule.
  - **Exit:** `web/test.html` reproduces the native CPU results on the test sets; timings natively and in
    Chrome.

- [ ] **M7 — Public docs and demo** (Cloudflare Pages)
  - Site: build, quick start, concepts (rounds, precision, conventions), C API, accuracy and speed, in the
    browser, references.
  - Live demo in the visitor's browser: a real image pair with its matches, inliers and epipolar lines drawn,
    outlier and noise sliders, WebGPU against the CPU backend, timings. A second tab: register a camera
    against a small point cloud.
  - Benchmark page with fixed problem sets, numbers copied out as Markdown.
  - **Exit:** published; `sphinx-build -W` passes in the workflow.

- [ ] **M8 — v1.0**
  - CMake package (`find_package`), npm package, optional Python binding as just-ba's, tagged release with
    the benchmark numbers.
  - A link test: one WebAssembly module and one native binary holding just-keypoints, just-matching,
    poselib-gpu and just-ba on one GPU device.

## Development machines

- M0, M1, M2, M3, M6, M7, M8 on the M5 Pro laptop (WebGPU through Dawn uses Metal there).
- M4 and the CUDA columns of M5 on the NVIDIA laptop, after M3. M3 and M4 can swap if that machine is the
  one at hand; each backend is judged on its own machine.

## Main risks

| Risk | Effect | Response |
|---|---|---|
| The GPU wins only on hard problems and loses on easy ones | `auto` must choose well or it makes easy captures slower | Judged per kind of problem in M3 on one backend, after tuning and before anything is ported; a short CPU round first, GPU only for what did not finish |
| float32 5-point solver is too inaccurate | Fewer good hypotheses, more rounds | Measured in M1; host-side double solves as fallback |
| Copies of the solvers drift apart | Backend-specific bugs | One source for CPU, CUDA and Metal; WGSL checked per operation; identical samples on every backend; the two estimators share every kernel but the residual |
| WGSL cannot turn off fused multiply-add or guarantee rounded division | WebGPU picks another best hypothesis on near ties | Compare by metrics there, not bit for bit (as just-keypoints M6) |
| The function set is too small for real captures | Late additions on four backends | M2 comes before any GPU work |

## Open questions

- Name and prefix (`jpo` is a placeholder).
- Licence of this repository (PoseLib's BSD-3-Clause applies to the ported parts either way).
- Whether Sampson error on normalized coordinates is enough for fisheye pairs, or relative pose should be
  scored on bearings (angular error), as just-ba does for triangulation.
- Whether the first-pair choice should live here (a function over the batch of relative results) or in the
  final project.
