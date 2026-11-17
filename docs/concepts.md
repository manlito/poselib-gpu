# How it works

## Conventions

| | |
|---|---|
| Pose | 7 doubles, `qw qx qy qz tx ty tz`: a point `X` maps to `R(q) X + t`. An absolute pose is world to camera; a relative pose is camera a to camera b, with `\|t\| = 1`. |
| Axes | x right, y down, z forward. |
| Pixels | Whatever convention the camera parameters use; the library only applies the camera to the points it is given. just-keypoints puts (0, 0) at the centre of the top-left pixel. |
| Cameras | COLMAP's six models with their ids and parameter order: `SIMPLE_PINHOLE` (0), `PINHOLE` (1), `SIMPLE_RADIAL` (2), `RADIAL` (3), `OPENCV` (4), `OPENCV_FISHEYE` (5). Points are unprojected to bearings once; the solvers never see a camera model. |

## RANSAC in rounds

Both estimators are LO-RANSAC: draw minimal samples, solve them, score every hypothesis on all the matches,
keep the best, refine it. PoseLib does this one sample at a time. Here it runs in rounds, the same on every
backend:

1. **Sample** a number of minimal sets per problem: five matches for a pair, three for an image to register.
2. **Solve** them: the five-point method gives up to 10 essential matrices, P3P up to 4 poses.
3. **Score** each hypothesis on every match: Sampson error for a pair, reprojection error through the
   camera model for PnP, truncated at the threshold (MSAC).
4. **Select** the best hypothesis of each problem.
5. **Refine** it on the CPU in double (the "LO" of LO-RANSAC).
6. **Stop** each problem as soon as the usual confidence bound is met; problems that are done drop out of the
   next round.

On the CPU, rounds are small, so an easy problem stops after a few hundred hypotheses at most. On a GPU, steps
2 to 4 are one dispatch for the whole batch, 2048 hypotheses per problem, and one small readback. A GPU round
computes hypotheses the CPU would have skipped, so the GPU only wins on problems the CPU cannot stop early:
low inlier ratios, and pairs that do not overlap at all, which run to `max_iterations` (10,000 by default).

Samples come from a counter-based generator: sample = hash(seed, problem, round, hypothesis). Every backend
draws the same samples, so backends can be compared hypothesis by hypothesis.

## The `auto` backend

`auto` (the default) starts every problem on the CPU. A problem that has drawn a first batch of hypotheses
and still needs many more becomes a candidate for the GPU; the candidates move there together once they need
enough hypotheses between them to fill the GPU. Easy problems never leave the CPU, so `auto` costs them
nothing, and hard batches run close to the GPU alone: on an M5 Pro, 3000 pairs of 200 matches at 10 % inliers take
1.99 s with `auto`, 1.94 s on Metal alone and 5.38 s on the CPU. The gap is widest in the browser, where the CPU has one
thread: up to 1.5 times the GPU alone on large batches of hard resections.

The three numbers of the rule (the CPU phase, the hypotheses a problem must still need, the total the
candidates must need) were measured per backend:

| Backend | Measured on | CPU phase | still needed | total |
|---|---|---|---|---|
| Metal | Apple M5 Pro | 128 | 512 | 50,000 |
| CUDA | RTX 2070 Max-Q with an i7-9750H | 128 | 512 | 5,000 |
| WebGPU, native | Apple M5 Pro through Dawn | 128 | 512 | 2,800 per CPU thread |
| WebGPU, browser | Apple M5 Pro, Chrome | 32 | 32 | 1,000 |

They belong to those machines. Another pairing of CPU and GPU may want other numbers; nothing calibrates at
run time. `jpo_backend_name` says which GPU `auto` uses, for example `auto(metal)`.

## Candidate images in one call

An incremental reconstruction can try K candidate images in one `jpo_absolute_pose` call and register every
one that passes its own test on `n_inliers`. Candidates that cannot pass would otherwise run to
`max_iterations`. Setting `min_inlier_ratio` to r caps each problem at
`log(1 - confidence) / log(1 - r^s)` iterations (s = 3 for PnP, 5 for a pair), clamped to the minimum and
maximum: by then a problem with at least that inlier ratio has drawn an all-inlier sample with the requested
confidence.

```c
opt.max_error_px = 4.0;
opt.min_inlier_ratio = 0.25;     /* at most 585 iterations at confidence 0.9999 */
```

A stopped problem still returns its best model with its real inlier count and status. On the M5 Pro, a round
of 8 to 32 candidates with a quarter to three quarters failing runs 3.7 to 10 times faster with the option,
and every good candidate is still solved. With five-point samples the cap is near 10,000 anyway, so leave it
at 0 for pairs.

## Precision

The CPU backend works in double and is the reference, checked against PoseLib. The GPU kernels work in
float32 (Metal and WebGPU have no double): the same kernel source, compiled with `float`, is the
specification a GPU backend must reproduce. The pose returned is always refined on the CPU in double, so
float32 only affects which hypothesis wins a round, not the precision of the result. The
[accuracy tables](accuracy-and-speed.md) show both builds within PoseLib's own spread.

Backends agree on the best hypothesis of a round except at near ties: compilers are free to fuse
multiply-adds (15 to 25 % faster on Apple silicon), so results are compared by their metrics rather
than bit for bit.
