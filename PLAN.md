# poselib-gpu — plan

Minimal-dependency robust pose estimation for bundle-adjustment pipelines: relative pose and absolute pose
(PnP) with RANSAC and refinement, ported from PoseLib and written the same way as `../just-keypoints`,
`../just-matching` and `../just-ba` (C++ core, no PyTorch or other framework at run time). A fast CPU
library first; hand-written Metal / CUDA / WebGPU kernels only for the work where a GPU is measured faster.

Status: M0 to M6 are built and measured (see "Milestones" and "Results"). M7 and M8 are not started.
Sections above "Results" are the plan as it was followed; where the implementation differs from the first
plan, "What changed while building" says how and why.

Working name: the repository is `poselib-gpu`; the code uses the C prefix `jpo_` (`jpo.h`, CLI `jpo`),
after `jkp`, `jmt` and `jba`. Still a placeholder (open question).

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
int jpo_relative_pose(jpo_ctx*, const jpo_pairs*, const jpo_ransac_options*, jpo_pose_result*, uint8_t* inliers);

/* B images. xy: pixels, xyz: world points, concatenated the same way. */
int jpo_absolute_pose(jpo_ctx*, const jpo_resections*, const jpo_ransac_options*, jpo_pose_result*, uint8_t* inliers);

void jpo_destroy(jpo_ctx*);
```

- Inputs name a camera per image (`model`, intrinsics as doubles), shared across problems where equal.
- `jpo_ransac_options`: `max_error_px` (default 1 for pairs, 4 for resections), `confidence` (0.9999),
  `min_iterations` (100), `max_iterations` (10000), `refine` (on by default). Iterations, not rounds: the
  numbers mean what they mean in PoseLib, on every backend. `min_inlier_ratio` (off by default, added
  after M6) gives up on problems below a given inlier ratio; see "K candidates per round" below.
- Results per problem, one struct for both estimators: pose (7 doubles), `n_inliers`, `score` (MSAC),
  `iterations`, `status`, and `median_tri_angle_deg` (relative pose; 0 otherwise); per point an inlier byte.
- Host buffers only. Device buffers as input were to be added only if the copy showed up in a profile; it
  did not (the upload is one small buffer per call), so they were not.
- `status` separates "no model found", "too few points" and "degenerate" from success. `JPO_OK` says that
  a model with more inliers than a minimal sample was found, not that the pair is a true match: whether a
  pair or a registration is accepted is the caller's rule on `n_inliers` (found in M2).

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
cli/                     jpo gen | run | bench | webcheck | backends
tools/                   dev-only Python: PoseLib reference, synthetic and real problem files, compare, bench
tests/                   solver tests, <backend>_vs_cpu, PoseLib parity, incremental test (M2)
third_party/             PoseLib and Dawn, fetched by script, ignored by git
web/                     WebAssembly entry points, JS wrapper, browser test page
docs/                    documentation site + live demo (Cloudflare Pages)
```

## Milestones

Each milestone is a state that can be run and checked. Numbers are filled in when it closes.

- [x] **M0 — Reference locked** (M5 laptop)
  - PoseLib pinned and built as a test-only dependency; `tools/` runs it from Python.
  - Problem files: synthetic (noise, outlier ratio, planar, small baseline, each camera model) and real
    (pairs matched by just-keypoints + just-matching; 2D-3D sets from a COLMAP model).
  - Metrics fixed: rotation and translation-direction error, pose AUC at 5/10/20 degrees, inlier-set
    overlap, success rate over seeds.
  - Baseline: PoseLib's time on all CPU threads for one pair, 1000 pairs, and 100 resections. This sets the
    target for M5 and tells whether the GPU can win at all for small batches.
  - **Exit:** `tools/compare.py` reports the metrics for PoseLib against itself across seeds (the noise
    floor every later comparison is read against).
  - **Done, in part.** PoseLib 6a636b1d and Eigen 3.4.0 fetched by `tools/fetch_reference.sh`;
    `poselib_ref` reads the problem files; `tools/compare.py` and `tools/accuracy.py` report the metrics
    over five seeds (RESULTS.md). **Not done:** real problem files. This machine has no image sets, no
    COLMAP and no exported matches, so every set is generated (`jpo gen`: noise, outlier ratio, planar,
    small baseline, far origin, each camera model).

- [x] **M1 — First poses from our own code** (CPU, C++)
  - C API, batches, counter-based sampler, P3P, 5-point, round-based LO-RANSAC, refinement, six camera models.
  - `jpo relpose` and `jpo pnp` on problem files; tests of each solver against PoseLib's on the same samples.
  - The float build of the same code, and the precision measurement above.
  - **Exit:** on the M0 sets, accuracy and success rate within PoseLib's own seed-to-seed spread, in double.
    A written decision on float32 minimal solvers (keep, or fall back to host solves).
  - **Done.** Solvers against PoseLib's on the same samples (`parity_test`): 5-point reproduces 22105 of
    PoseLib's 22119 exact solutions, P3P 10103 of 10111. Success rate within PoseLib's seed spread on every
    set (largest shortfall of the mean below PoseLib's worst seed: 0.1 %). **Short of the exit on one set:**
    on `relpose_harder` (20 % inliers, 1 px noise) the success rate is equal (89.0 % against 89.1 %) but the
    poses are less accurate, AUC@5 0.739 against 0.788 (PoseLib's seeds: 0.759 to 0.821); `relpose_planar`
    is 0.795 against 0.817. Not yet explained; the likely place is the local optimization, which PoseLib
    runs on every new best model and this code runs once per round.
  - **Float32 decision: keep the minimal solvers in float on the GPU.** The float build reproduces about
    92 % of the double 5-point solutions and the estimator solves the same problems (`precision_test`:
    100/100 at 70 % and 30 % inliers in both, 11 against 12 of 100 at 10 %); on `relpose_harder` float
    costs 2 points of success rate (87.0 % against 89.0 %). No host-side double solves.

- [x] **M2 — Enough for an incremental reconstruction** (CPU, with just-ba)
  - A test program, not a product: pair verification, first pair, registration by PnP, just-ba's
    triangulation and solver, on two or three small scenes from exported COLMAP matches.
  - Compared with COLMAP on the same matches: images registered, reprojection error, pose error after alignment.
  - **Exit:** scenes reconstruct; any function found missing is added to the scope here, before it has to be
    written four times. Decides G1-G3 and whether homography comes back.
  - **Done on generated scenes, not against COLMAP.** `tests/incremental` (built when just-ba is next to
    the repository) reconstructs three scenes (orbit, wall, grid) from generated matches with outliers and
    non-overlapping pairs: all images registered (16/16, 14/14, 13/13), mean reprojection 0.46 to 0.51 px,
    median rotation error 0.04 to 0.06 deg, camera centres within 0.03 % of the scene extent.
  - **No function was missing.** Homography does not come back: planar scenes reconstruct with the
    5-point solver, and a first pair with too little parallax shows in `median_tri_angle_deg`. Two findings
    went into `jpo.h`: `JPO_OK` is not "verified" (the caller thresholds `n_inliers`), and the mask of a
    `JPO_DEGENERATE` pair is still usable for tracks. G1 to G3 stay candidates; nothing in M2 needed them.

- [x] **M3 — Runs on the Mac GPU, and is judged there** (Metal, M5 laptop)
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
  - **Done.** CPU first: 1 to 2 times PoseLib's speed on batches of pairs and 0.7 to 3.7 times on
    resections (slower only on small resections that run to the iteration limit; RESULTS.md). Metal passes `backend_test` against the float CPU build (the same best
    hypothesis in every round checked) and the accuracy sets (RESULTS.md).
  - **Verdict.** Metal wins on batches that the CPU cannot stop early on: pairs at 30 % inliers or less
    1.5 to 2.7 times, resections at 10 % or none 2.8 to 7.8 times. It loses on easy problems (0.2 to 0.6
    times) and loses or ties on any single problem (a round costs 2 to 5 ms; 18 threads finish 10,000 hypotheses of one
    pair in 2 ms). Both paths are kept; `auto` matches the CPU where the GPU loses and stays within 3 to
    35 % of Metal where it wins.
  - Tuning that paid: early rejection after the first 64 to 256 points, a bound from the best score so
    far, a start offset per hypothesis, one thread per hypothesis. `ops_test` per kernel was not written:
    the kernels are one source for CPU and Metal, so `backend_test` compares rounds instead.

- [x] **M4 — Runs on NVIDIA** (CUDA, the laptop)
  - The paths that won on Metal, from the same kernel source behind the CUDA prelude; same tests; the same
    table on that machine, with its own `auto` rule (a discrete GPU pays for upload and readback that Metal
    does not).
  - CPU backend built and tested on Linux and Windows.
  - **Exit:** as M3 with `--backend cuda`.
  - **Done on Linux.** Same kernel source through nvcc; `backend_test cuda` passes. RTX 2070 Max-Q against
    a 6-core i7-9750H: pairs at 30 % or less 6 to 9 times on batches and about 2 times on one pair;
    resections at 10 % or none 6 to 17 times; easy problems 0.4 to 0.8 times. Its own `auto` rule (below).
    **Not done:** Windows. The CPU build and tests are in the CI matrix for Windows but that workflow has
    never run (nothing was pushed), and CUDA on Windows was not tried.

- [x] **M5 — Fast**
  - Tuning on both machines where the profiles point: rounds for problems of very different sizes in one
    batch, device inputs from just-matching, less waste on easy problems.
  - Benchmark table: problems per second by batch size, points and inlier ratio, for PoseLib on all
    threads, our CPU backend, Metal and CUDA.
  - **Exit:** documented numbers and the final `auto` rule per backend.
  - **Done, with less tuning than planned.** Tables for PoseLib, CPU, Metal and CUDA in RESULTS.md, and the
    `auto` rule per backend below. Device inputs were not added (no copy showed in a profile). Known gap in
    `auto`: resections of 200 points at 30 % inliers stop near 1000 iterations, under the rule's threshold, so `auto`
    stays on the CPU where the GPU alone is 3 times (Metal) to 7 times (CUDA) faster on batches. On CUDA
    the threshold is now 512 with a 128-iteration CPU phase: those batches run 1.8 to 4.9 times faster,
    1.5 to 1.6 times behind CUDA alone on 200 points and ahead of it on 2000. Metal has the same rule
    since: batches of 300 and more run at Metal alone's speed on 200 points (3000 of them: 145 → 45 ms)
    and beat it on 2000; 32 of them still stay on the CPU (under the 50,000 total).

- [x] **M6 — Runs in the browser**
  - The CPU backend in WebAssembly first (one thread): the whole API works there with no GPU.
  - WGSL port of the paths that won, natively through Dawn with the same tests, then Emscripten; device
    supplied by the caller. Measured against the WebAssembly CPU backend in Chrome, with its own `auto` rule.
  - **Exit:** `web/test.html` reproduces the native CPU results on the test sets; timings natively and in
    Chrome.
  - **Done.** The WebAssembly CPU build reproduces `jpo webcheck --threads 1` exactly and runs about 1.35
    times slower than one native thread. The WGSL port (hand-written, 1600 lines) passes `backend_test
    webgpu` natively through Dawn, including with a device supplied by the caller, and runs in the browser
    through Emscripten: 20 to 40 times faster than the WebAssembly CPU on hard sets (`auto` 13 to 20 times), about equal on easy
    ones (RESULTS.md). The browser's `auto` rule was re-tuned since (`web/bench.html` in headless Chrome,
    `tools/browser_bench.mjs`): a 32-iteration CPU phase, and a single CPU hypothesis for a one-round
    budget, because WebGPU alone wins from about four problems up there; the bench grid takes 1.6 s
    instead of 3.0, candidate rounds come within 8 % of WebGPU alone. Browser timings come from Chrome on
    one machine; Safari and Firefox were not run.

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

## Results

Full tables: [RESULTS.md](RESULTS.md).

### The auto rule

`auto` starts every problem on the CPU. A problem that has drawn `cpu_iterations` hypotheses and still needs
at least `gpu_min_remaining` more becomes a candidate; the candidates move to the GPU together once they
need `gpu_min_total` hypotheses between them (`src/core/auto_rule.cpp`).

| Backend | Measured on | cpu_iterations | gpu_min_remaining | gpu_min_total |
|---|---|---|---|---|
| Metal | M5 Pro, 18 threads | 128 | 512 | 50,000 |
| CUDA | RTX 2070 Max-Q, i7-9750H | 128 | 512 | 5,000 |
| WebGPU, native | M5 Pro through Dawn | 128 | 512 | 2,800 per CPU thread |
| WebGPU, browser (one or two threads) | M5 Pro, headless Chrome | 32 | 32 | 1,000 |

A call whose iteration budget fits in one GPU round (`max_iterations`, or the cap of `min_inlier_ratio`,
at most 2048; resections at r = 0.25 draw at most 585) has its own three numbers, because whatever moves then
takes exactly one dispatch, while the numbers above were measured on problems that take several. They are
used only when the batch could reach the total at all (active problems x (budget − CPU phase) ≥ the total);
otherwise, and on backends without them, the numbers above apply.

| Backend | one-round cpu_iterations | gpu_min_remaining | gpu_min_total |
|---|---|---|---|
| Metal | 64 | 256 | 5,000 |
| CUDA | 16 | 128 | 1,000 |
| WebGPU, native | 64 | 256 | 5,000 |
| WebGPU, browser | 1 | 1 | 1,000 |

The numbers belong to those machines. Another CPU/GPU pair would want its own; nothing calibrates at run time.

### K candidates per round: min_inlier_ratio

The incremental reconstruction this library is for registers images in rounds: K candidate images go to
`jpo_absolute_pose` as one batch, and every one that passes the caller's test is registered. The batch
call already did that; what it lacked was a cheap way to give up on candidates that cannot pass, which ran
to `max_iterations` (10,000) and made up nearly all the time of a round.

- `jpo_ransac_options.min_inlier_ratio` = r, in [0, 1); 0 is off and is what `jpo_ransac_options_init`
  gives. When set, every problem of the call draws at most log(1 - confidence) / log(1 - r^s) iterations
  (s = 3 absolute, 5 relative), clamped to [min_iterations, max_iterations]: 585 for resections at r = 0.25
  and the default confidence, 9427 for pairs, where it saves almost nothing, so callers leave it at 0 there.
- It is one rule in the host driver (`src/core/estimator.cpp`, `iteration_cap`), so every backend has it
  without kernel changes. The cap replaces `max_iterations` everywhere the driver uses it: the stopping
  rule, the remaining work `auto` counts before moving a problem to the GPU, and the early rejection floor.
- A problem stopped by the cap returns the best model it found with its real inlier count and the usual
  status. There is no status code for it; the caller compares `n_inliers`, as it already does.
- A problem just above r may stop before its own stopping rule would (that rule asks three times the
  bound). Problems well above r stop by their own rule first and come back unchanged (`core_test`).
- Measured (RESULTS.md, "K candidates per registration round"): K = 8 to 32 resections with a quarter to
  three quarters failing, 4 to 10 times faster per round on the CPU and 4 to 6 times on Metal, with every
  good candidate still solved. One batch is 1.5 to 5.7 times faster than K calls on the CPU.
- Under the old `auto` rule these capped rounds never left the CPU (329 iterations left per candidate after
  the 256-iteration CPU phase, under `gpu_min_remaining`), where Metal alone was up to 2 times faster. The
  one-round numbers above fix most of that: K = 32 with three quarters failing 2.96 → 1.97 ms (Metal alone
  1.50, CPU 3.02), with `auto` within noise of the CPU or faster elsewhere. Calls with the default budget
  route exactly as before (all 112 cells of the grid checked). Lowering the general thresholds instead was
  tried and dropped: it closed the known gap on 200-point resections at 30 % but slowed single easy
  problems and small hard batches (RESULTS.md, "`auto` with a one-round budget").
- On CUDA (RTX 2070 Max-Q against 6 cores) the option makes a round 7 to 14 times faster on the CPU and 4
  to 7 times on CUDA, and CUDA alone then beats the CPU 1.5 to 4.4 times in every round with failing
  candidates. Its one-round numbers (16 CPU iterations, 128 remaining, 1,000 in total) make `auto` 1.35 to
  4.2 times faster than the old rule there and 1.1 to 1.3 times behind CUDA alone; rounds with no failing
  candidate are 0.3 to 0.6 ms behind the CPU at K = 16 to 32. Default budgets route as before (112 cells).
  On the general resection grid with the option set, hard batches go from the CPU's speed to near CUDA's
  (RESULTS.md, "`auto` with a one-round budget on CUDA").
- Left to the pipeline: ranking candidates, the acceptance threshold, building 2D-3D matches from tracks,
  and stopping after N successes. World points are copied per candidate (under 1 MB per round); there is
  no indexed form.

### What changed while building

- **The CPU phase of `auto` never ends inside the CPU's first round, and only cuts problems that could
  move.** Cutting the widened first round of a small batch left threads idle and cost easy problems a
  second round; a problem that cannot reach `gpu_min_remaining` keeps its whole round (routing unchanged).
  This made a 128-iteration phase pay on CUDA and Metal (RESULTS.md, "CPU phase of 128 on CUDA", "CPU
  phase of 128 and 512 remaining on Metal").
- **The browser has its own `auto` numbers**, not Metal's per thread: on one WebAssembly thread WebGPU
  alone wins from about four problems up, so the CPU phase is its first round and a one-round budget
  moves after one hypothesis (RESULTS.md, "`auto` in the browser"). Timed in headless Chrome
  (`tools/browser_bench.mjs`), because the app's browser pane is throttled whenever its window is out of
  view.
- **Iterations, not rounds, in the options and results.** They mean what they mean in PoseLib on every backend.
- **Hypotheses are scored as poses with a cheirality test**, not as essential matrices (PoseLib's
  behaviour). Scoring E alone lost planar scenes.
- **Two Gauss-Newton steps polish each 5-point solution** on the eliminated constraints; without them the
  solver was less exact than PoseLib's, in float visibly so.
- **Early rejection and a score bound** in the round (M3), which is where most of the GPU's margin came from.
- **CPU rounds double from 32 to 1024 hypotheses** instead of staying small: waking the thread pool for
  tiny rounds made the first CPU version half as fast as PoseLib.
- **Fused multiply-add is left on** (15 to 25 % faster on Apple silicon), so backends agree on the best
  hypothesis except at near ties and tests compare that way, not bit for bit.
- **Relative pose of fisheye pairs is scored with Sampson error on the normalized plane**, as PoseLib.
  Both lose inliers there (inlier IoU 0.79 on the fisheye set against 0.95 elsewhere); scoring on bearings
  is still an open question.
- **`DEGENERATE`** is reported when the median triangulation angle of the inliers is under ten thresholds.

### Not done, or not verified

- Real images, real matches and the comparison with COLMAP (M0, M2): nothing to run them on here.
- The accuracy shortfall on the hardest relative-pose set (M1).
- Windows, and the hosted CI workflow, have never run.
- Browsers other than Chrome; the browser rule was measured in headless Chrome on one machine, and its
  two-thread branch (the same numbers) was not measured at all.
- In the browser `auto` is still up to 1.5 times slower than WebGPU alone on hard resection batches (100 x
  2000 at 30 %: 60.8 against 40.6 ms), and 7 to 22 % on candidate rounds without `min_inlier_ratio`: the
  CPU's first round of 32 iterations is the shortest phase it has on one thread.
- `auto` with `min_inlier_ratio` set is still 0.15 to 0.5 ms per round behind Metal alone at K = 16 to 32
  with failing candidates (the 64-iteration CPU phase), and 0.1 to 0.9 ms behind CUDA alone (the
  16-iteration phase); on CUDA, rounds with no failing candidate at K = 16 to 32 are 0.3 to 0.6 ms behind
  the CPU.
- On CUDA, single or paired easy problems of 2000 points are up to 0.3 ms slower than before the CPU phase
  lasted the CPU's first round (pnp 1 x 2000 at 70 %: 1.37 → 1.69 ms): that round is 384 hypotheses where
  an easy problem needs about 256.
- On Metal, easy batches of 4 to 8 are up to 0.26 ms slower under the 128 / 512 rule (the phase cuts
  problems that may still need 640 iterations), and 32 resections of 200 points at 30 % stay on the CPU,
  2.2 times behind Metal alone: a lower total would move them but slows small pair batches, so the total
  would have to depend on the kind of problem.

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
  scored on bearings (angular error), as just-ba does for triangulation. Measured: it costs inliers.
- Whether "line relative pose estimation" in the request meant line features (left out, see above).
- Whether the first-pair choice should live here (a function over the batch of relative results) or in the
  final project.
