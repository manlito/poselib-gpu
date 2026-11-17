# poselib-gpu — measurements

The tables behind PLAN.md ("Results"). Times are milliseconds end to end, host buffers in, poses and inlier
masks out, best of three runs after a warm-up. Problems are generated (`jpo gen`): 0.5 px noise for pairs,
1 px for resections; "inliers 0" is a set with no model at all. Options are the defaults (confidence 0.9999,
100 to 10000 iterations, 1 px / 4 px). PoseLib is the pinned commit, one problem per thread over all threads
(OpenMP over the batch), so its single-problem rows are one thread against jpo's all threads.

Reproduce: `python3 tools/bench.py --backends cpu,metal,auto`, `python3 tools/accuracy.py [--float | --backend metal]`.

## Apple M5 Pro (18 threads), Metal

The `auto` column is the rule as measured for M5 (256-iteration CPU phase, 1024 remaining); the current
rule is in "CPU phase of 128 and 512 remaining on Metal" below.

| kind | batch | points | inliers | poselib ms | cpu ms | metal ms | auto ms |
|---|---|---|---|---|---|---|---|
| relpose | 1 | 200 | 0.7 | 0.91 | 0.33 | 1.99 | 0.31 |
| relpose | 1 | 200 | 0.3 | 29.79 | 1.99 | 6.00 | 2.04 |
| relpose | 1 | 200 | 0.1 | 28.90 | 1.92 | 5.08 | 1.99 |
| relpose | 1 | 200 | 0 | 28.88 | 1.86 | 4.96 | 1.85 |
| relpose | 1 | 2000 | 0.7 | 3.59 | 0.85 | 4.01 | 0.66 |
| relpose | 1 | 2000 | 0.3 | 40.73 | 2.23 | 7.29 | 2.30 |
| relpose | 1 | 2000 | 0.1 | 38.53 | 1.71 | 4.88 | 1.81 |
| relpose | 1 | 2000 | 0 | 38.56 | 1.71 | 4.84 | 1.81 |
| relpose | 32 | 200 | 0.7 | 2.28 | 2.29 | 8.15 | 2.34 |
| relpose | 32 | 200 | 0.3 | 69.42 | 54.88 | 32.69 | 33.34 |
| relpose | 32 | 200 | 0.1 | 69.29 | 54.57 | 32.21 | 33.50 |
| relpose | 32 | 200 | 0 | 70.03 | 54.50 | 32.26 | 33.40 |
| relpose | 32 | 2000 | 0.7 | 8.79 | 4.66 | 24.25 | 4.44 |
| relpose | 32 | 2000 | 0.3 | 108.20 | 56.44 | 37.63 | 39.77 |
| relpose | 32 | 2000 | 0.1 | 102.19 | 53.45 | 33.68 | 35.27 |
| relpose | 32 | 2000 | 0 | 99.97 | 52.96 | 33.43 | 34.84 |
| relpose | 300 | 200 | 0.7 | 17.88 | 13.80 | 55.13 | 13.83 |
| relpose | 300 | 200 | 0.3 | 610.73 | 500.98 | 198.41 | 209.07 |
| relpose | 300 | 200 | 0.1 | 622.55 | 496.17 | 196.36 | 207.94 |
| relpose | 300 | 200 | 0 | 601.86 | 499.45 | 196.98 | 205.42 |
| relpose | 300 | 2000 | 0.7 | 66.31 | 31.22 | 167.19 | 31.45 |
| relpose | 300 | 2000 | 0.3 | 967.55 | 511.63 | 239.87 | 248.11 |
| relpose | 300 | 2000 | 0.1 | 888.29 | 490.68 | 218.25 | 223.40 |
| relpose | 300 | 2000 | 0 | 902.45 | 489.64 | 214.94 | 222.61 |
| relpose | 3000 | 200 | 0.7 | 178.08 | 133.17 | 516.74 | 134.78 |
| relpose | 3000 | 200 | 0.3 | 6442.55 | 5198.85 | 1926.08 | 2013.24 |
| relpose | 3000 | 200 | 0.1 | 6324.38 | 5281.43 | 1914.75 | 2000.55 |
| relpose | 3000 | 200 | 0 | 6246.65 | 5409.20 | 1901.42 | 1996.63 |
| relpose | 3000 | 2000 | 0.7 | 644.80 | 331.71 | 1636.76 | 333.98 |
| relpose | 3000 | 2000 | 0.3 | 9955.11 | 5649.83 | 2296.57 | 2385.54 |
| relpose | 3000 | 2000 | 0.1 | 9281.58 | 5364.30 | 2081.08 | 2171.57 |
| relpose | 3000 | 2000 | 0 | 9248.79 | 5407.48 | 2064.55 | 2160.62 |
| pnp | 1 | 200 | 0.7 | 0.43 | 0.16 | 0.41 | 0.11 |
| pnp | 1 | 200 | 0.3 | 0.98 | 0.22 | 0.40 | 0.21 |
| pnp | 1 | 200 | 0.1 | 6.94 | 0.54 | 1.88 | 0.63 |
| pnp | 1 | 200 | 0 | 6.67 | 0.57 | 1.96 | 0.63 |
| pnp | 1 | 2000 | 0.7 | 2.86 | 0.59 | 1.59 | 0.55 |
| pnp | 1 | 2000 | 0.3 | 5.27 | 0.55 | 1.43 | 0.60 |
| pnp | 1 | 2000 | 0.1 | 42.95 | 4.11 | 4.11 | 2.42 |
| pnp | 1 | 2000 | 0 | 42.06 | 3.80 | 6.38 | 4.08 |
| pnp | 32 | 200 | 0.7 | 0.83 | 0.73 | 1.27 | 0.93 |
| pnp | 32 | 200 | 0.3 | 2.08 | 2.89 | 1.36 | 2.60 |
| pnp | 32 | 200 | 0.1 | 15.66 | 16.16 | 5.86 | 6.04 |
| pnp | 32 | 200 | 0 | 15.63 | 15.97 | 5.08 | 6.77 |
| pnp | 32 | 2000 | 0.7 | 7.31 | 2.40 | 8.68 | 2.71 |
| pnp | 32 | 2000 | 0.3 | 15.59 | 5.01 | 7.64 | 4.98 |
| pnp | 32 | 2000 | 0.1 | 116.90 | 40.63 | 14.84 | 18.66 |
| pnp | 32 | 2000 | 0 | 116.15 | 129.67 | 30.65 | 34.40 |
| pnp | 300 | 200 | 0.7 | 6.23 | 2.68 | 6.77 | 2.79 |
| pnp | 300 | 200 | 0.3 | 16.53 | 15.14 | 5.17 | 14.89 |
| pnp | 300 | 200 | 0.1 | 130.41 | 142.59 | 21.51 | 25.35 |
| pnp | 300 | 200 | 0 | 130.96 | 143.56 | 21.58 | 25.29 |
| pnp | 300 | 2000 | 0.7 | 52.22 | 14.93 | 62.55 | 14.71 |
| pnp | 300 | 2000 | 0.3 | 128.52 | 34.65 | 51.68 | 34.81 |
| pnp | 300 | 2000 | 0.1 | 1056.00 | 364.48 | 89.06 | 109.43 |
| pnp | 300 | 2000 | 0 | 1068.98 | 1282.02 | 224.21 | 247.69 |
| pnp | 3000 | 200 | 0.7 | 64.94 | 22.25 | 57.89 | 23.09 |
| pnp | 3000 | 200 | 0.3 | 171.77 | 150.87 | 45.94 | 153.84 |
| pnp | 3000 | 200 | 0.1 | 1373.26 | 1510.70 | 193.42 | 225.51 |
| pnp | 3000 | 200 | 0 | 1368.54 | 1490.45 | 196.25 | 227.28 |
| pnp | 3000 | 2000 | 0.7 | 538.62 | 146.65 | 610.65 | 149.16 |
| pnp | 3000 | 2000 | 0.3 | 1368.87 | 340.38 | 500.63 | 343.01 |
| pnp | 3000 | 2000 | 0.1 | 10843.51 | 3793.34 | 851.49 | 1083.69 |
| pnp | 3000 | 2000 | 0 | 10792.89 | 12106.99 | 2176.27 | 2406.95 |

## Intel i7-9750H (6 cores, 12 threads), RTX 2070 Max-Q, CUDA, Linux

| kind | batch | points | inliers | poselib ms | cpu ms | cuda ms | auto ms |
|---|---|---|---|---|---|---|---|
| relpose | 1 | 200 | 0.7 | 2.38 | 0.61 | 1.25 | 0.65 |
| relpose | 1 | 200 | 0.3 | 70.61 | 9.37 | 4.17 | 4.66 |
| relpose | 1 | 200 | 0.1 | 68.92 | 9.23 | 4.14 | 4.61 |
| relpose | 1 | 200 | 0 | 65.42 | 9.48 | 4.16 | 4.62 |
| relpose | 1 | 2000 | 0.7 | 10.86 | 2.67 | 6.28 | 2.57 |
| relpose | 1 | 2000 | 0.3 | 113.36 | 10.22 | 7.44 | 7.92 |
| relpose | 1 | 2000 | 0.1 | 101.28 | 9.35 | 4.34 | 4.78 |
| relpose | 1 | 2000 | 0 | 101.82 | 9.71 | 4.19 | 4.59 |
| relpose | 32 | 200 | 0.7 | 10.48 | 9.05 | 12.59 | 8.84 |
| relpose | 32 | 200 | 0.3 | 388.93 | 352.11 | 51.12 | 57.85 |
| relpose | 32 | 200 | 0.1 | 387.06 | 361.42 | 49.24 | 55.97 |
| relpose | 32 | 200 | 0 | 384.82 | 378.55 | 49.75 | 56.85 |
| relpose | 32 | 2000 | 0.7 | 51.54 | 24.41 | 40.79 | 23.68 |
| relpose | 32 | 2000 | 0.3 | 715.58 | 393.48 | 59.62 | 67.93 |
| relpose | 32 | 2000 | 0.1 | 638.88 | 373.74 | 52.17 | 58.46 |
| relpose | 32 | 2000 | 0 | 641.35 | 377.55 | 51.00 | 57.88 |
| relpose | 300 | 200 | 0.7 | 101.56 | 92.99 | 108.55 | 91.94 |
| relpose | 300 | 200 | 0.3 | 3408.59 | 3375.10 | 401.95 | 468.02 |
| relpose | 300 | 200 | 0.1 | 3333.61 | 3375.59 | 393.42 | 460.51 |
| relpose | 300 | 200 | 0 | 3288.76 | 3386.80 | 394.04 | 462.66 |
| relpose | 300 | 2000 | 0.7 | 439.26 | 199.96 | 354.18 | 200.42 |
| relpose | 300 | 2000 | 0.3 | 6160.95 | 3571.11 | 472.71 | 550.21 |
| relpose | 300 | 2000 | 0.1 | 5546.46 | 3432.07 | 418.88 | 488.84 |
| relpose | 300 | 2000 | 0 | 5528.74 | 3425.09 | 413.97 | 485.82 |
| relpose | 3000 | 200 | 0.7 | 1022.29 | 901.98 | 1101.36 | 901.96 |
| relpose | 3000 | 200 | 0.3 | 33013.43 | 33964.98 | 3968.89 | 4787.71 |
| relpose | 3000 | 200 | 0.1 | 32276.49 | 33704.67 | 3895.86 | 4699.08 |
| relpose | 3000 | 200 | 0 | 32076.36 | 33624.70 | 3887.20 | 4688.45 |
| relpose | 3000 | 2000 | 0.7 | 4245.53 | 2039.04 | 3625.02 | 2070.01 |
| relpose | 3000 | 2000 | 0.3 | 59566.87 | 34822.19 | 4739.22 | 5640.11 |
| relpose | 3000 | 2000 | 0.1 | 54417.56 | 33505.27 | 4109.22 | 4928.82 |
| relpose | 3000 | 2000 | 0 | 53781.48 | 33362.80 | 4082.47 | 4890.39 |
| pnp | 1 | 200 | 0.7 | 1.28 | 0.28 | 0.40 | 0.27 |
| pnp | 1 | 200 | 0.3 | 2.05 | 0.44 | 0.38 | 0.36 |
| pnp | 1 | 200 | 0.1 | 15.36 | 2.61 | 1.53 | 1.60 |
| pnp | 1 | 200 | 0 | 15.66 | 2.70 | 1.45 | 1.53 |
| pnp | 1 | 2000 | 0.7 | 6.40 | 1.68 | 2.57 | 1.49 |
| pnp | 1 | 2000 | 0.3 | 20.00 | 1.81 | 2.25 | 1.82 |
| pnp | 1 | 2000 | 0.1 | 119.00 | 13.53 | 5.41 | 6.26 |
| pnp | 1 | 2000 | 0 | 120.24 | 24.07 | 9.23 | 10.16 |
| pnp | 32 | 200 | 0.7 | 3.32 | 1.61 | 2.31 | 1.64 |
| pnp | 32 | 200 | 0.3 | 10.54 | 9.70 | 1.74 | 10.00 |
| pnp | 32 | 200 | 0.1 | 91.41 | 96.60 | 6.57 | 9.50 |
| pnp | 32 | 200 | 0 | 96.28 | 99.31 | 6.76 | 9.18 |
| pnp | 32 | 2000 | 0.7 | 33.69 | 9.95 | 22.65 | 9.98 |
| pnp | 32 | 2000 | 0.3 | 101.62 | 25.49 | 18.39 | 25.07 |
| pnp | 32 | 2000 | 0.1 | 790.36 | 265.12 | 44.56 | 53.35 |
| pnp | 32 | 2000 | 0 | 769.37 | 856.62 | 78.99 | 97.55 |
| pnp | 300 | 200 | 0.7 | 29.82 | 12.72 | 16.01 | 13.12 |
| pnp | 300 | 200 | 0.3 | 101.86 | 90.32 | 12.15 | 91.41 |
| pnp | 300 | 200 | 0.1 | 827.13 | 905.26 | 54.73 | 70.96 |
| pnp | 300 | 200 | 0 | 825.84 | 922.23 | 56.19 | 71.24 |
| pnp | 300 | 2000 | 0.7 | 284.55 | 91.30 | 209.35 | 90.93 |
| pnp | 300 | 2000 | 0.3 | 887.72 | 214.36 | 171.36 | 213.64 |
| pnp | 300 | 2000 | 0.1 | 7071.86 | 2271.64 | 277.85 | 442.03 |
| pnp | 300 | 2000 | 0 | 6987.80 | 7993.89 | 709.80 | 892.37 |
| pnp | 3000 | 200 | 0.7 | 302.91 | 134.70 | 174.79 | 137.00 |
| pnp | 3000 | 200 | 0.3 | 1029.00 | 905.35 | 133.79 | 906.87 |
| pnp | 3000 | 200 | 0.1 | 8274.17 | 9000.19 | 519.84 | 735.19 |
| pnp | 3000 | 200 | 0 | 8204.38 | 9120.78 | 525.22 | 741.77 |
| pnp | 3000 | 2000 | 0.7 | 2815.19 | 923.56 | 2160.10 | 931.29 |
| pnp | 3000 | 2000 | 0.3 | 8762.65 | 2121.00 | 1763.74 | 2173.03 |
| pnp | 3000 | 2000 | 0.1 | 69709.46 | 23075.69 | 2827.51 | 4667.67 |
| pnp | 3000 | 2000 | 0 | 69258.42 | 76738.37 | 7065.70 | 9167.19 |

The `auto` column above is the rule as it was measured for M5 (1024 remaining); the current CUDA rule moves
problems with 512 or more hypotheses left after a 128-iteration CPU phase (next sections).

### `gpu_min_remaining` 512 on CUDA (the known gap)

Batches of resections at 30 % inliers stop near 1000 iterations: after the 256-iteration CPU phase they
need under 1024 more, so the old rule kept them on the CPU where CUDA alone was 5.8 to 7.3 times faster on
200 points. Searched with `jpo bench --backends auto` and the rule overrides over the whole grid above: CPU
phases of 128 and 256, 256 to 1024 remaining, totals of 2,000 and 5,000. With the 256-iteration phase, 256
or 512 remaining change the routing of exactly six cells, the resection batches at 30 %, and of nothing
else in the grid (CPU and GPU hypotheses, iterations and solved counts compared cell by cell); the total
changes nothing. CUDA now uses 512.

Resections at 30 % inliers, ms; `auto` columns are the best of two runs, all three interleaved in one
window; cpu and cuda from the full grid run before it:

| batch | points | cpu | cuda | auto, 1024 | auto, 512 | auto, 128-iteration phase |
|---|---|---|---|---|---|---|
| 32 | 200 | 10.24 | 1.78 | 9.78 | 3.70 | 2.77 |
| 32 | 2000 | 25.77 | 18.95 | 26.70 | 17.26 | 15.31 |
| 300 | 200 | 89.65 | 12.35 | 94.93 | 31.30 | 20.61 |
| 300 | 2000 | 213.74 | 173.29 | 226.74 | 143.78 | 124.89 |
| 3000 | 200 | 899.00 | 135.94 | 959.05 | 320.65 | 213.52 |
| 3000 | 2000 | 2105.07 | 1766.42 | 2278.14 | 1437.68 | 1271.74 |

- 1.5 to 3.0 times faster than the old rule. On 2000 points `auto` now beats both CPU and CUDA alone (it
  finishes the problems that stop early on the CPU and sends the rest); on 200 points it is still 2.1 to
  2.4 times behind CUDA alone, the cost of the 256-iteration CPU phase.
- In the interleaved runs the old rule ran first in each pair and had a thermal advantage of about 10 %:
  cells that route identically under all three settings, and some never touch the GPU (300 easy pairs:
  83 against 94 to 96 ms), came out that much faster on it. Small differences between the columns
  elsewhere in that run are this, not the rule.
- A 128-iteration CPU phase with 512 remaining (last column) gains more on these cells and about 10 to
  15 % on hard batches (3000 x 2000 at 10 %: 4171 against 4852 ms), but as it was it cost single easy pairs
  an extra CPU round: 0.60 → 0.90 ms on 200 points, 2.54 → 3.23 ms on 2000, as on Metal ("`auto` with a
  one-round budget"). It is the current rule since the change in the next section. 256 remaining routes as
  512 does on this grid; 512 is kept as the smaller change.
- Calls whose budget fits in one GPU round follow CUDA's one-round numbers (below) and are not affected;
  `jpo candidates` with the new rule matched the earlier runs within the laptop's drift.

### CPU phase of 128 on CUDA, and where the phase ends

The 128-iteration phase above lost on single easy pairs because the CPU phase cut the CPU's first round.
With fewer problems than threads, the CPU backend widens a round so every thread has a chunk (one problem
on 12 threads: 384 hypotheses). Ending the phase at 128 inside that round left 8 threads idle for about the
same wall time, and an easy pair that needed a few more iterations paid for a second round. Two changes to
where the phase ends (`src/core/estimator.cpp`, every backend):

- The phase lasts at least the CPU's first round (`max(cpu_iterations, round_size(n_start, 0))`). This
  applies to the general rule only; the one-round numbers were measured with the cut and keep it.
- A round is cut at the phase end only for a problem that could move to the GPU there. A problem's
  remaining need never grows, so one with fewer than `gpu_min_remaining` left at the phase end can never
  move; it keeps its whole round. This cannot change routing. Under the rule of the previous section it
  gave byte-identical results on the grid (batches 1 to 300).

CUDA then moves to a 128-iteration phase. Interleaved, best of two runs each, `jpo bench --backends auto
--batches 1,2,4,8,32,300`: A the rule of the previous section (phase 256), C phase 128 with only the
first change, D phase 128 with both (the current rule); selected cells, ms:

| kind | batch | points | inliers | A, phase 256 | C | D, current |
|---|---|---|---|---|---|---|
| relpose | 1 | 200 | 0.7 | 0.65 | 0.64 | 0.62 |
| relpose | 4 | 2000 | 0.7 | 5.42 | 6.95 | 5.43 |
| relpose | 8 | 2000 | 0.7 | 7.27 | 8.30 | 7.49 |
| relpose | 300 | 200 | 0.1 | 457.15 | 427.24 | 431.65 |
| pnp | 1 | 2000 | 0.7 | 1.37 | 1.56 | 1.69 |
| pnp | 1 | 2000 | 0.1 | 6.25 | 5.11 | 5.16 |
| pnp | 8 | 2000 | 0.1 | 17.02 | 14.55 | 14.91 |
| pnp | 32 | 200 | 0.3 | 3.46 | 2.56 | 2.59 |
| pnp | 300 | 200 | 0.3 | 28.93 | 19.25 | 19.57 |
| pnp | 300 | 2000 | 0.1 | 426.31 | 356.61 | 361.90 |

- Over all 96 cells D is 3.6 % faster than A (geometric mean), with A first in each pair and so warmer
  (see above). Hard resection batches run 8 to 33 % faster and hard pair batches 3 to 6 %; solved counts
  are the same in every cell. Against C, GPU routing and solved counts are identical in every cell; D only
  saves CPU rounds, which is what removes C's losses on easy batches of 4 and 8 (5.43 against 6.95 ms).
- What D still loses: up to 0.3 ms on single or paired easy problems of 2000 points (pnp 1 x 2000 at
  70 %: 1.37 → 1.69 ms; 1.11 times already with the first change alone at phase 256). A single problem's
  first round is now 384 hypotheses where an easy one needs about 256; the same wider round makes a single
  hard resection faster (6.25 → 5.16 ms).
- With 3000 problems the first change does nothing (their first round is 32) and the second only saves
  rounds of easy problems, so the earlier phase-128 run applies: 3000 x 200 resections at 30 % 214 ms
  against 321 (CUDA alone 136), 3000 x 2000 at 10 % 4171 against 4852.
- `jpo candidates`: with the option on, unchanged within 3 %; with it off, 5 to 12 % faster (failing
  candidates move after 128 iterations).
- On Metal the first change touches only batches of 1 and 2 (an 18-thread first round
  is 576 and 288 hypotheses), which never reach the GPU under Metal's rule; easy problems draw 576
  hypotheses instead of 256. Old against new binary, best of two runs of five, batches 1 to 3 (M5 Pro):
  geometric mean 0.996, every cell within 0.1 ms but one (a single hard resection of 2000 points, 2.18
  against 3.49 ms with identical hypothesis counts, which is noise).

## WebGPU

Natively through Dawn on the M5 Pro the WebGPU backend costs what Metal costs (300 pairs of 200 matches with
no model: 211 ms against 197 ms for Metal and 500 ms for the CPU; 300 resections of 2000 points: 216 ms
against 224 ms and 1282 ms), and measured natively it wants Metal's rule too ("`auto` on WebGPU through
Dawn", below).

In the browser (the app's built-in Chromium on the M5 Pro, WebAssembly on one thread, `web/test.html`,
20 problems per set, best of three, ms):

| set | wasm cpu | webgpu | auto |
|---|---|---|---|
| relpose easy (1000 points, 70 %) | 25.7 | 18.8 | 25.5 |
| relpose hard (1000 points, 30 %) | 678.2 | 30.0 | 50.1 |
| relpose none (300 points) | 703.8 | 22.2 | 40.7 |
| pnp easy (1000 points, 70 %) | 9.3 | 7.6 | 9.3 |
| pnp hard (1000 points, 10 %) | 465.2 | 11.9 | 29.2 |
| pnp none (300 points) | 214.6 | 5.0 | 10.7 |

One problem at a time in the browser: an easy pair 1.8 ms on the CPU and 3.5 ms on WebGPU; a hard pair
34.7 ms and 7.9 ms (`auto` 8.1 ms); a hard resection 21.2 ms and 4.8 ms (`auto` 5.3 ms). Every set is solved
20 of 20 on all three, and the WebAssembly CPU rows equal `jpo webcheck --threads 1` natively. Browser
timings were taken once, in one browser; Chrome itself, Safari and Firefox were not run. The `auto` column
is the rule before the browser re-tune below.

### `auto` on WebGPU through Dawn

The native WebGPU build (`-DJPO_METAL=OFF`, so that `auto` picks WebGPU) on the M5 Pro, 18 threads, the
same searches as on Metal: `jpo bench --backends auto --batches 1,2,4,8,32,300,3000` with CPU phases of 128
and 256 and 512 or 1024 remaining (total 2,800 per thread, 50,400), best of two runs, the second in reverse
order. 128 / 512 re-routes 44 cells, 12 % faster (geometric mean), solved counts the same in all 112; the
68 that route the same are 1 % slower; the grid takes 20.3 s instead of 21.5 s. 256 / 512 re-routes only
the four resection batches at 30 % (all faster); 128 / 1024 every hard batch of 8 or more, 4 % faster.
Selected cells, ms (cpu and webgpu from one full run before):

| kind | batch | points | inliers | cpu | webgpu | auto, 256 / 1024 | auto, 128 / 512 |
|---|---|---|---|---|---|---|---|
| relpose | 4 | 2000 | 0.7 | 1.22 | 4.36 | 1.21 | 1.33 |
| relpose | 32 | 200 | 0 | 54.79 | 28.06 | 30.50 | 29.11 |
| relpose | 3000 | 200 | 0.1 | 5228.99 | 2113.02 | 2187.10 | 2084.11 |
| pnp | 8 | 200 | 0.1 | 4.32 | 3.24 | 5.39 | 3.31 |
| pnp | 32 | 200 | 0.3 | 2.61 | 1.10 | 2.73 | 2.70 |
| pnp | 300 | 200 | 0.3 | 14.30 | 5.41 | 14.43 | 5.50 |
| pnp | 300 | 2000 | 0.3 | 33.76 | 50.44 | 34.73 | 26.20 |
| pnp | 3000 | 200 | 0.3 | 148.42 | 49.82 | 150.01 | 47.53 |
| pnp | 3000 | 2000 | 0.3 | 337.25 | 507.93 | 346.30 | 233.52 |
| pnp | 3000 | 2000 | 0.1 | 3757.37 | 854.61 | 1109.80 | 977.27 |

As on Metal, 32 resections of 200 points at 30 % stay on the CPU (2.70 against 1.10 ms): they need under
50,400 hypotheses between them. The total was not searched again; it costs what it costs on Metal.

`jpo candidates` with the one-round overrides: CPU phases of 16, 32, 64 and 128, 128 or 256 remaining,
totals 1,000, 2,000 and 5,000. Single runs scatter (cells of 1.5 ms read 4 to 7 ms at random, the dispatch
latency of rounds this short), so the eight best were run three times, interleaved, best of three. Metal's
64 / 256 / 5,000 is the best of them (ms per round with `min_inlier_ratio` 0.25; without it every
configuration is within 7 %):

| K | failing | cpu | webgpu | auto, no one-round numbers | auto, 64 / 256 / 5,000 |
|---|---|---|---|---|---|
| 8 | 0 | 0.52 | 1.03 | 0.49 | 0.53 |
| 8 | 0.75 | 1.01 | 0.99 | 0.91 | 0.96 |
| 16 | 0 | 0.75 | 1.22 | 0.77 | 0.76 |
| 16 | 0.5 | 1.35 | 1.21 | 1.35 | 1.35 |
| 16 | 0.75 | 1.62 | 1.10 | 1.57 | 1.41 |
| 32 | 0 | 1.29 | 1.50 | 1.32 | 1.32 |
| 32 | 0.25 | 1.84 | 1.50 | 1.95 | 1.91 |
| 32 | 0.5 | 2.27 | 1.47 | 2.29 | 2.09 |
| 32 | 0.75 | 2.99 | 1.40 | 2.98 | 2.00 |

The worst cell against the faster of cpu and webgpu is 1.43 times (2.13 without one-round numbers). 16 /
128 / 5,000 is faster at K = 32 with half or more failing (1.69 to 1.85 ms) but slower with none (1.58).
WebGPU natively now uses Metal's numbers: 128 / 512 / 2,800 per thread, and 64 / 256 / 5,000 for a
one-round budget; the built-in rule routes every grid cell as the overrides did.

### `auto` in the browser

`web/bench.html` runs `jpo bench` cells and candidate rounds in the page, and takes the rule's numbers as a
`route=` parameter, as the CLI overrides. It was run in headless Chrome 157 on the M5 Pro
(`tools/browser_bench.mjs`), because the app's browser pane throttles the page whenever its window is out of
view; WebAssembly on one thread, WebGPU on the same GPU. Grid: batches 1, 4, 32 and 100, best of three runs
of `auto` per configuration, every configuration run twice in opposite orders; cpu and webgpu from one run
before (best of two).

On one WebAssembly thread the CPU is slow enough that WebGPU alone wins from about four problems up, easy
ones included, and the old rule (a 256 phase, 256 remaining, 2,800 total) left most of that on the table.
Shorter phases won throughout: 128 / 256 / 2,800 made the grid 2.4 s instead of 3.0, 64 / 128 / 1,000 1.9 s,
and a 32 phase (the CPU's first round on one thread, so the shortest there is) 1.65 s with 0, 32 or 128
remaining and a total of 200 or 1,000. 32 / 32 / 1,000 has the lowest worst cell (1.53 times the faster
backend); moving everything (16 / 0 / 0) sends single easy problems to the GPU, 4 times slower. Confirmed
with the built-in rule afterwards (same routing as the override in every cell, same solved counts), ms:

| kind | batch | points | inliers | wasm cpu | webgpu | auto, 256 / 256 / 2,800 | auto, 32 / 32 / 1,000 |
|---|---|---|---|---|---|---|---|
| relpose | 1 | 200 | 0.7 | 1.2 | 5.9 | 0.7 | 1.1 |
| relpose | 1 | 200 | 0.3 | 35.0 | 6.5 | 7.1 | 5.8 |
| relpose | 4 | 200 | 0.3 | 141.2 | 7.7 | 11.6 | 8.0 |
| relpose | 32 | 200 | 0.7 | 28.6 | 10.7 | 27.6 | 9.9 |
| relpose | 32 | 200 | 0.3 | 1126.1 | 29.9 | 58.2 | 33.7 |
| relpose | 100 | 200 | 0.7 | 86.8 | 24.4 | 86.0 | 25.5 |
| relpose | 100 | 2000 | 0.3 | 3485.0 | 134.0 | 225.8 | 143.5 |
| pnp | 1 | 200 | 0.7 | 0.1 | 0.7 | 0.1 | under 0.1 |
| pnp | 1 | 2000 | 0.1 | 24.6 | 5.9 | 6.9 | 5.3 |
| pnp | 4 | 2000 | 0.3 | 7.3 | 3.5 | 5.9 | 3.6 |
| pnp | 32 | 2000 | 0.3 | 56.5 | 13.9 | 35.2 | 20.5 |
| pnp | 32 | 2000 | 0.1 | 631.7 | 20.6 | 74.6 | 27.3 |
| pnp | 100 | 200 | 0.3 | 77.6 | 5.6 | 24.3 | 7.8 |
| pnp | 100 | 2000 | 0.1 | 1857.0 | 49.6 | 210.7 | 72.4 |
| pnp | 100 | 2000 | 0 | 6478.7 | 82.9 | 247.9 | 103.7 |

The whole grid by batch size (ms, sum of 16 cells each):

| batch | wasm cpu | webgpu | auto before | auto now |
|---|---|---|---|---|
| 1 | 307 | 82 | 68 | 59 |
| 4 | 1253 | 104 | 126 | 89 |
| 32 | 9950 | 351 | 710 | 393 |
| 100 | 31016 | 952 | 2082 | 1085 |

The worst cell is now 1.50 times the faster backend (100 resections of 2000 points at 30 %), against 4.3
times before; still slower than WebGPU alone on hard resection batches, where the CPU's 32 iterations cost
more than the GPU's whole run.

Candidate rounds (the browser's own draws, not `jpo candidates`', five rounds, best of three, ms per round
with `min_inlier_ratio` 0.25). One-round numbers tried: 1 / 1 / 1, 16 / 64 / 200, 16 / 128 / 1,000, 32 / 64 /
500, 32 / 128 / 1,000 and 64 / 256 / 5,000, each twice in opposite orders. On one thread the one-round
phase is not widened to a first CPU round, so a phase of 1 means a single CPU hypothesis per problem and
then the GPU: that was the best by far (38.7 ms for the 12 cells, WebGPU alone 34.9, the general rule's
numbers 55, the old rule 118). Its total, raised to 1,000, keeps a lone problem on the general rule, where
an easy one stays on the CPU (K = 1, none failing: 0.27 ms against 0.86 on WebGPU). Confirmed with the
built-in rule:

| K | failing | wasm cpu | webgpu | auto |
|---|---|---|---|---|
| 1 | 0 | 0.33 | 0.86 | 0.27 |
| 2 | 0.5 | 1.63 | 1.04 | 0.96 |
| 4 | 0 | 1.05 | 1.37 | 1.36 |
| 8 | 0 | 2.32 | 1.98 | 1.94 |
| 8 | 0.75 | 8.02 | 1.20 | 1.26 |
| 16 | 0 | 4.42 | 3.20 | 3.50 |
| 16 | 0.5 | 13.12 | 2.16 | 2.92 |
| 16 | 0.75 | 17.42 | 1.72 | 1.80 |
| 32 | 0 | 9.42 | 5.64 | 6.26 |
| 32 | 0.5 | 26.32 | 3.96 | 4.22 |
| 32 | 0.75 | 34.64 | 3.38 | 3.56 |

K = 8 to 32: 36.9 ms in all against 34.3 for WebGPU alone, every good candidate solved on all three.
Without the option `auto` is 7 to 22 % slower than WebGPU alone on rounds of 8 to 32 with failing
candidates (the 32-iteration phase), and 7 to 38 times faster than the CPU.

In the browser (one or two CPU threads) `auto` now uses 32 / 32 / 1,000, and 1 / 1 / 1,000 for a one-round
budget. Two threads were not measured. Chrome only, one machine; Safari and Firefox were not run.

## K candidates per registration round (M5 Pro, Metal)

One registration round of an incremental reconstruction: K candidate images tried in one
`jpo_absolute_pose` call, against the same K problems solved by K calls of one problem each. A candidate
is a resection of 300 to 1000 points (uniform); a good one has 50 to 70 % inliers, a failing one 0 to 10 %;
"failing" is the share of failing candidates, shuffled into the batch. 1 px noise, 4 px threshold, other
options the defaults; `min_inlier_ratio` off (0) and at 0.25, which caps a resection at 585 iterations.
ms is the mean over 5 generated rounds of the best of 3 runs after a warm-up. Solved: good candidates with
rotation within 1 degree of the truth, over the 5 rounds; it was the same in all four columns of every row.
No failing candidate reached 25 % inliers in any run. Reproduce: `jpo candidates` and
`jpo candidates --failing 0`. The `auto` rows of this table are the rule before its one-round numbers
were added; with the option on, the current rule is in the next table.

| K | failing | backend | batch, off | batch, 0.25 | one at a time, off | one at a time, 0.25 | good solved |
|---|---|---|---|---|---|---|---|
| 8 | 0 | cpu | 0.53 | 0.51 | 1.93 | 1.93 | 40/40 |
| 8 | 0 | metal | 1.71 | 1.14 | 5.38 | 4.72 | 40/40 |
| 8 | 0 | auto | 0.53 | 0.51 | 1.47 | 1.48 | 40/40 |
| 8 | 0.25 | cpu | 3.27 | 0.81 | 4.12 | 1.77 | 30/30 |
| 8 | 0.25 | metal | 4.21 | 0.97 | 10.58 | 4.75 | 30/30 |
| 8 | 0.25 | auto | 3.26 | 0.79 | 4.24 | 1.77 | 30/30 |
| 8 | 0.5 | cpu | 5.11 | 0.82 | 5.75 | 1.46 | 20/20 |
| 8 | 0.5 | metal | 4.21 | 0.89 | 14.34 | 4.13 | 20/20 |
| 8 | 0.5 | auto | 5.09 | 0.82 | 6.05 | 1.75 | 20/20 |
| 8 | 0.75 | cpu | 7.78 | 1.01 | 8.43 | 1.48 | 10/10 |
| 8 | 0.75 | metal | 4.58 | 0.92 | 19.59 | 4.36 | 10/10 |
| 8 | 0.75 | auto | 5.25 | 1.00 | 9.15 | 2.04 | 10/10 |
| 16 | 0 | cpu | 0.76 | 0.76 | 3.90 | 3.88 | 80/80 |
| 16 | 0 | metal | 2.00 | 1.23 | 10.86 | 9.68 | 80/80 |
| 16 | 0 | auto | 0.78 | 0.76 | 3.15 | 3.16 | 80/80 |
| 16 | 0.25 | cpu | 5.97 | 1.06 | 8.16 | 3.33 | 60/60 |
| 16 | 0.25 | metal | 5.55 | 1.21 | 19.57 | 8.93 | 60/60 |
| 16 | 0.25 | auto | 6.25 | 1.19 | 8.42 | 3.55 | 60/60 |
| 16 | 0.5 | cpu | 10.52 | 1.45 | 12.68 | 3.50 | 40/40 |
| 16 | 0.5 | metal | 6.01 | 1.24 | 28.65 | 8.61 | 40/40 |
| 16 | 0.5 | auto | 7.63 | 1.55 | 13.54 | 4.23 | 40/40 |
| 16 | 0.75 | cpu | 16.21 | 1.71 | 18.22 | 3.56 | 20/20 |
| 16 | 0.75 | metal | 6.77 | 1.15 | 37.60 | 7.92 | 20/20 |
| 16 | 0.75 | auto | 9.65 | 1.88 | 19.89 | 4.06 | 20/20 |
| 32 | 0 | cpu | 1.32 | 1.31 | 7.42 | 7.41 | 160/160 |
| 32 | 0 | metal | 3.19 | 1.63 | 21.33 | 19.11 | 160/160 |
| 32 | 0 | auto | 1.42 | 1.39 | 6.18 | 6.19 | 160/160 |
| 32 | 0.25 | cpu | 10.44 | 2.06 | 16.56 | 7.39 | 120/120 |
| 32 | 0.25 | metal | 7.10 | 1.63 | 39.68 | 18.37 | 120/120 |
| 32 | 0.25 | auto | 7.14 | 1.93 | 15.31 | 6.67 | 120/120 |
| 32 | 0.5 | cpu | 19.70 | 2.24 | 24.28 | 7.34 | 80/80 |
| 32 | 0.5 | metal | 8.17 | 1.62 | 58.26 | 17.09 | 80/80 |
| 32 | 0.5 | auto | 11.01 | 2.41 | 26.48 | 7.90 | 80/80 |
| 32 | 0.75 | cpu | 32.08 | 3.20 | 38.20 | 7.51 | 40/40 |
| 32 | 0.75 | metal | 9.66 | 1.52 | 74.52 | 16.13 | 40/40 |
| 32 | 0.75 | auto | 11.68 | 3.16 | 38.62 | 8.19 | 40/40 |

A second run of the same command gave the same picture, with cells up to about 20 % apart (most within 10 %).

- **The option is what makes a round cheap.** Failing candidates draw 585 iterations instead of 10,000. In
  batches with failing candidates, the CPU gets 4.0 to 10 times faster, Metal 4.3 to 6.4 times and `auto`
  3.7 to 6.2 times. Nothing was lost: every good candidate was solved in every column, and the failing ones came
  back with their real inlier counts (all under 25 %).
- **Where it does not help: rounds with no failing candidate.** Good candidates stop near 100 to 200
  iterations by their own rule, so on the CPU and under `auto` the option changes nothing (0.51 against
  0.53 ms, within noise). On Metal alone it shortens the one GPU round from 2048 to 585 hypotheses (1.14 to
  1.63 ms against 1.71 to 3.19 ms), which is still slower than the CPU.
- **One batch beats one at a time.** With the option on, the batch is 2.0 to 4.5 times faster than K calls
  on `auto` and 1.5 to 5.7 times on the CPU (one small problem cannot fill 18 threads); on Metal one at a
  time is 4 to 12 times slower than the batch, since every call pays for its own dispatch. With the option
  off and failing candidates in the round, the batch gains only 1.1 to 1.6 times on the CPU: the failing
  candidates' 10,000 iterations fill the threads either way.
- **`auto` with the option on stayed on the CPU** under the rule as it was: capped work is 585 − 256 = 329
  iterations per candidate, under `gpu_min_remaining` (1024), so nothing went to the GPU (`jpo run
  --backend auto --min-inlier-ratio 0.25` on 32 resections with no model: 18,720 CPU hypotheses, 0 GPU).
  Metal alone was faster at K = 16 to 32 with half or more failing; the rule was re-tuned for that (next
  table).
- **With the option off**, Metal wins once half or more of the candidates fail (1.2 to 3.3 times the CPU),
  and `auto` sits between the two, as in the main table.
- CUDA: the next section, on the NVIDIA laptop. Not run: WebGPU timings (its rule has no one-round
  numbers, so a one-round budget follows its usual rule there). The option is in the host driver, so every
  backend gets it without kernel changes; it was checked for function, not timed, on WebGPU natively
  (`backend_test webgpu`, Dawn) and in the browser (WebAssembly CPU, WebGPU and `auto` through `web/jpo.js`:
  failing candidates stop at 585 iterations, good ones unchanged on the CPU).

### `auto` with a one-round budget (the rule re-tuned)

With `min_inlier_ratio` = 0.25 a resection's whole budget (585 iterations) fits in one GPU round (2048),
so whatever moves to the GPU takes exactly one dispatch. The rule's numbers (256 CPU iterations, 1024
remaining, 50,000 in total) were measured on problems that need five rounds, and a GPU round costs a fixed
latency: 24 capped candidates with 521 iterations left each (12,500 in all) are worth one dispatch, while one
hard pair with 9,744 left (five dispatches) is not. So a call whose budget fits in one GPU round now has its
own three numbers on Metal: a CPU phase of 64 iterations, 256 remaining, 5,000 in total. The short phase is
taken only when the batch could reach the 5,000 at all (active problems x (budget − 64) ≥ 5,000), because
it costs a CPU round of its own; rounds of 8 candidates keep the old rule. A call with the default budget
(10,000 iterations) runs the old rule unchanged: all 112 cells of `jpo bench --batches
1,2,4,8,32,300,3000` drew the same CPU and GPU hypotheses, iterations and solved counts before and after.

Option on (0.25), batch of K candidates, ms; the same `jpo candidates` sets, all four columns measured in
one quiet window (a reference run before and after agreed within 5 %):

| K | failing | cpu | metal | auto, old rule | auto, one-round rule | good solved |
|---|---|---|---|---|---|---|
| 8 | 0 | 0.55 | 1.14 | 0.52 | 0.52 | 40/40 |
| 8 | 0.25 | 0.79 | 0.96 | 0.80 | 0.80 | 30/30 |
| 8 | 0.5 | 0.83 | 0.94 | 0.81 | 0.86 | 20/20 |
| 8 | 0.75 | 1.01 | 0.92 | 0.98 | 0.93 | 10/10 |
| 16 | 0 | 0.77 | 1.24 | 0.77 | 0.77 | 80/80 |
| 16 | 0.25 | 1.06 | 1.19 | 1.18 | 1.06 | 60/60 |
| 16 | 0.5 | 1.35 | 1.19 | 1.46 | 1.34 | 40/40 |
| 16 | 0.75 | 1.62 | 1.13 | 1.64 | 1.36 | 20/20 |
| 32 | 0 | 1.32 | 1.59 | 1.30 | 1.33 | 160/160 |
| 32 | 0.25 | 1.82 | 1.58 | 1.93 | 1.93 | 120/120 |
| 32 | 0.5 | 2.26 | 1.67 | 2.33 | 2.02 | 80/80 |
| 32 | 0.75 | 3.02 | 1.50 | 2.96 | 1.97 | 40/40 |

- `auto` is now within noise of the CPU or faster in every cell but one (K = 32 at 25 %: 1.93 against
  1.82 ms, the same as under the old rule), and faster than the old rule where the GPU wins: K = 32 at 75 % failing 2.96 → 1.97 ms, at 50 % 2.33 → 2.02 ms, K = 16 at 75 % 1.64 → 1.36 ms. The
  failing candidates move to Metal there; K = 16 at 25 to 50 % and K = 32 at 25 % stay on the CPU (under
  5,000 hypotheses) and gain a little from the shorter CPU phase alone. K = 8 rows run the same code as
  before, so their differences (up to 0.05 ms) are noise.
- Metal alone is still faster in five cells (K = 16 at 50 and 75 %, K = 32 at 25 to 75 %): `auto` is 0.15
  to 0.5 ms behind it there, against 0.3 to 1.5 ms before. The rest is the CPU phase, which is what keeps
  `auto` at the CPU's speed where Metal loses (every round with no failing candidate, and K = 8).
- Option off: `auto` is unchanged (the budget is 10,000 iterations), within 3 % of the old rule in every row.

What was tried and dropped, measured on the same sets:

- Lowering `gpu_min_remaining` to 256 with a threshold per GPU round, keeping the 256-iteration CPU phase:
  the candidates reached Metal, but `auto` got slower (K = 32: 2.5 to 3.1 ms against 2.0 to 3.0); the CPU
  phase alone costs about what the CPU needs to finish.
- The same with a 32- or 64-iteration CPU phase for every call, not only one-round budgets: it closed the
  known gap on 200-point resections at 30 % inliers (3000 of them: 147 → 41 ms), but cost 1.5 to 2 times
  on single easy problems (an extra CPU round: pnp 1 x 2000 points at 70 %, 0.54 → 0.82 ms) and sent
  small hard batches to five Metal rounds (pnp 4 x 200 at 10 %: 2.5 → 5.1 ms). Timings of that grid were
  taken under load from another job and are not used beyond these directions.
- A 32-iteration phase for one-round budgets: about the same at K = 32, slower at K = 16 (0.84 against
  0.77 ms with no failing candidate, 1.58 against 1.46 at 50 %).
- No short-phase gate: 0.05 to 0.15 ms slower on K = 8 rounds that never reach the GPU.

### CPU phase of 128 and 512 remaining on Metal

With the phase no longer cutting rounds it need not ("CPU phase of 128 on CUDA" below), the shorter phase
that was dropped on Metal ("`auto` with a one-round budget", above) was searched again with `jpo bench
--backends auto --batches 1,2,4,8,32,300,3000` and the rule overrides: CPU phases of 64, 128 and 256 with
256, 512 and 1024 remaining, total 50,000. With the 256 phase, 256 or 512 remaining re-route only four
cells (resection batches at 30 %), all faster; a 128 or 64 phase re-routes 44 to 46 (every hard batch of 8
or more). Batches of 1 and 2 are untouched by any of them: their first CPU round (576 and 288) is longer
than the phase. Metal now uses a 128-iteration phase and 512 remaining, as CUDA.

Confirmed in one window, best of two runs each, the second run in reverse order; cpu and metal from a full
grid run before it; selected cells, ms:

| kind | batch | points | inliers | cpu | metal | auto, 256 / 1024 | auto, 128 / 512 |
|---|---|---|---|---|---|---|---|
| relpose | 4 | 2000 | 0.7 | 1.18 | 4.66 | 1.16 | 1.42 |
| relpose | 8 | 200 | 0.7 | 0.80 | 2.93 | 0.78 | 0.90 |
| relpose | 32 | 200 | 0 | 54.65 | 32.51 | 33.84 | 32.41 |
| relpose | 3000 | 200 | 0.1 | 5381.31 | 1944.65 | 2027.44 | 1988.93 |
| pnp | 8 | 200 | 0.1 | 4.17 | 3.45 | 3.56 | 4.30 |
| pnp | 8 | 2000 | 0.1 | 12.73 | 6.29 | 8.54 | 7.51 |
| pnp | 32 | 200 | 0.3 | 2.64 | 1.21 | 2.78 | 2.64 |
| pnp | 300 | 200 | 0.3 | 14.16 | 5.61 | 14.75 | 5.64 |
| pnp | 300 | 2000 | 0.3 | 32.57 | 52.76 | 33.54 | 24.77 |
| pnp | 300 | 2000 | 0.1 | 344.92 | 87.54 | 108.69 | 99.87 |
| pnp | 3000 | 200 | 0.3 | 144.49 | 46.19 | 144.56 | 44.88 |
| pnp | 3000 | 2000 | 0.3 | 330.84 | 504.37 | 329.58 | 218.79 |
| pnp | 3000 | 2000 | 0.1 | 3685.49 | 848.01 | 1072.55 | 971.87 |

- The 46 re-routed cells are 9 % faster (geometric mean), solved counts the same in all 112. The known gap
  is closed for batches of 300 and more: 200-point resections at 30 % run at Metal alone's speed (3000 of
  them: 145 → 45 ms), and 2000-point ones beat both the CPU and Metal alone (330 → 219 ms).
- The 66 cells that route the same are 1.9 % slower, up to 0.26 ms on easy batches of 4 to 8 (4 x 2000
  pairs at 70 %: 1.16 → 1.42 ms). With 512 remaining the phase cuts the round of any problem that still
  needs 640 iterations, so one that has not found a good model by then pays for a round it would not have
  under 1024. The worst re-routed cell is 8 x 200 resections at 10 %, 3.56 → 4.30 ms.
- 32 resections of 200 points at 30 % still stay on the CPU (2.64 against 1.21 ms on Metal): together
  they need under 50,000 hypotheses. A total of 20,000 moved them (1.20 ms) and 4 hard resections of 2000
  points (8.3 → 5.3 ms), but slowed 4 pairs of 2000 points at 30 % (7.7 → 9.7 ms, where Metal alone is
  slower than the CPU); 10,000 and 5,000 sent single and paired hard problems to Metal (1.8 → 5.2 ms). The
  total stays at 50,000.
- Also tried: a 64-iteration phase (1.5 % faster overall than 128, but 8 x 200 resections at 10 % 5.1 ms and
  3000 x 2000 at 30 % 252 ms) and 256 remaining with the 128 phase (within noise of 512).
- `jpo candidates` (old and new binary, interleaved): every cell within 7 %, with the option on or off.

## K candidates per registration round (RTX 2070 Max-Q, CUDA)

The same `jpo candidates` sets as on the M5 Pro (above), on the NVIDIA laptop: i7-9750H (6 cores, 12
threads), RTX 2070 Max-Q, Linux, on mains power. `auto` here is the rule before CUDA had one-round numbers:
with the option on, nothing left the CPU. Reproduce: `jpo candidates --backends cpu,cuda,auto --failing
0,0.25,0.5,0.75`.

| K | failing | backend | batch, off | batch, 0.25 | one at a time, off | one at a time, 0.25 | good solved |
|---|---|---|---|---|---|---|---|
| 8 | 0 | cpu | 1.55 | 1.45 | 4.48 | 4.42 | 40/40 |
| 8 | 0 | cuda | 3.15 | 2.13 | 8.69 | 7.98 | 40/40 |
| 8 | 0 | auto | 1.55 | 1.48 | 4.20 | 4.23 | 40/40 |
| 8 | 0.25 | cpu | 17.58 | 2.38 | 20.36 | 5.27 | 30/30 |
| 8 | 0.25 | cuda | 5.80 | 1.56 | 12.32 | 6.19 | 30/30 |
| 8 | 0.25 | auto | 6.28 | 2.65 | 11.13 | 4.95 | 30/30 |
| 8 | 0.5 | cpu | 29.79 | 2.80 | 32.11 | 5.35 | 20/20 |
| 8 | 0.5 | cuda | 5.78 | 1.43 | 15.03 | 4.90 | 20/20 |
| 8 | 0.5 | auto | 6.77 | 2.98 | 15.11 | 5.00 | 20/20 |
| 8 | 0.75 | cpu | 46.30 | 4.10 | 49.02 | 6.26 | 10/10 |
| 8 | 0.75 | cuda | 8.33 | 1.75 | 21.03 | 5.05 | 10/10 |
| 8 | 0.75 | auto | 8.62 | 3.98 | 22.67 | 6.05 | 10/10 |
| 16 | 0 | cpu | 2.29 | 2.32 | 8.84 | 8.78 | 80/80 |
| 16 | 0 | cuda | 3.74 | 2.31 | 13.64 | 12.51 | 80/80 |
| 16 | 0 | auto | 2.28 | 2.33 | 8.66 | 8.51 | 80/80 |
| 16 | 0.25 | cpu | 34.02 | 4.54 | 38.91 | 9.91 | 60/60 |
| 16 | 0.25 | cuda | 9.07 | 2.16 | 27.04 | 11.29 | 60/60 |
| 16 | 0.25 | auto | 8.66 | 4.39 | 20.86 | 9.38 | 60/60 |
| 16 | 0.5 | cpu | 60.01 | 6.26 | 65.98 | 11.09 | 40/40 |
| 16 | 0.5 | cuda | 10.86 | 2.37 | 35.34 | 11.40 | 40/40 |
| 16 | 0.5 | auto | 11.18 | 5.86 | 32.55 | 10.92 | 40/40 |
| 16 | 0.75 | cpu | 95.70 | 7.67 | 100.62 | 12.26 | 20/20 |
| 16 | 0.75 | cuda | 12.68 | 1.95 | 42.19 | 9.73 | 20/20 |
| 16 | 0.75 | auto | 13.44 | 7.21 | 45.13 | 12.08 | 20/20 |
| 32 | 0 | cpu | 3.55 | 3.59 | 17.45 | 17.47 | 160/160 |
| 32 | 0 | cuda | 5.97 | 3.22 | 26.01 | 23.69 | 160/160 |
| 32 | 0 | auto | 3.59 | 3.58 | 17.07 | 16.90 | 160/160 |
| 32 | 0.25 | cpu | 60.02 | 6.87 | 72.29 | 19.84 | 120/120 |
| 32 | 0.25 | cuda | 13.46 | 3.49 | 45.45 | 23.35 | 120/120 |
| 32 | 0.25 | auto | 11.65 | 6.64 | 39.84 | 19.35 | 120/120 |
| 32 | 0.5 | cpu | 126.91 | 10.46 | 138.84 | 21.67 | 80/80 |
| 32 | 0.5 | cuda | 17.38 | 3.13 | 64.59 | 21.55 | 80/80 |
| 32 | 0.5 | auto | 17.37 | 10.03 | 65.85 | 21.42 | 80/80 |
| 32 | 0.75 | cpu | 194.56 | 13.92 | 206.69 | 23.44 | 40/40 |
| 32 | 0.75 | cuda | 22.18 | 3.16 | 83.29 | 19.57 | 40/40 |
| 32 | 0.75 | auto | 23.14 | 13.41 | 90.45 | 24.26 | 40/40 |

- **The option does more here than on the M5 Pro**, because the CPU has 6 cores instead of 18: in batches
  with failing candidates it makes the CPU 7.4 to 14 times faster and CUDA 3.7 to 7.0 times. Every good
  candidate was solved in every column; no failing candidate reached 25 % inliers.
- **With the option on, CUDA alone beats the CPU in every round with failing candidates**, 1.5 to 4.4
  times, and ties it with none failing at K = 16 and 32 (2.31 against 2.32 ms, 3.22 against 3.59). Only
  K = 8 with none failing is faster on the CPU (1.45 against 2.13 ms). The old `auto` stayed on the CPU
  (329 iterations left per candidate, under `gpu_min_remaining`) and was as slow as the CPU there.
- **One batch beats one at a time**, with the option on, 1.5 to 4.9 times on the CPU and 2.9 to 7.4 times
  on CUDA (every call pays for its own upload, launch and readback).

### `auto` with a one-round budget on CUDA

Searched with the rule overrides of `jpo candidates` (`--one-round-cpu-iter`, `--one-round-gpu-min-remaining`,
`--one-round-gpu-min-total`), option on, all twelve cells per setting: CPU phases of 8, 16, 32, 64 and 128
iterations, 128 or 256 remaining, totals of 500 to 5,000. The CPU phase is the cost: with totals of 2,000
or less, 128 iterations put `auto` 2.0 to 2.2 times behind the faster of CPU and CUDA in its worst cell, 64
iterations 1.5 to 1.6 times, 8 to 32 iterations 1.3 to 1.7 times. Totals of 5,000 left K = 8 and some K = 16
rounds on the CPU where CUDA was 2.0 to 2.4 times faster; 500 to 1,500 were within noise of each other. CUDA's one-round numbers are now a CPU phase of 16 iterations,
128 remaining and 1,000 in total (`src/core/auto_rule.cpp`); the short-phase gate is the same as Metal's
(active problems x (budget − 16) ≥ 1,000), so a single capped problem keeps the old rule.

Option on (0.25), batch of K candidates, ms; all four columns measured in one window, then again in a second
one. The new rule agreed within 6 % between the two; the CPU column slowed by up to 12 % as the laptop
warmed:

| K | failing | cpu | cuda | auto, old rule | auto, one-round rule | good solved |
|---|---|---|---|---|---|---|
| 8 | 0 | 1.59 | 1.91 | 1.56 | 1.61 | 40/40 |
| 8 | 0.25 | 2.37 | 1.59 | 2.71 | 2.01 | 30/30 |
| 8 | 0.5 | 2.82 | 1.44 | 2.94 | 1.84 | 20/20 |
| 8 | 0.75 | 4.07 | 1.73 | 3.98 | 1.86 | 10/10 |
| 16 | 0 | 2.46 | 2.34 | 2.45 | 2.72 | 80/80 |
| 16 | 0.25 | 4.52 | 2.15 | 4.44 | 2.62 | 60/60 |
| 16 | 0.5 | 6.12 | 2.36 | 5.94 | 2.55 | 40/40 |
| 16 | 0.75 | 8.01 | 2.03 | 7.74 | 2.30 | 20/20 |
| 32 | 0 | 3.75 | 3.23 | 3.86 | 4.36 | 160/160 |
| 32 | 0.25 | 7.39 | 3.45 | 6.89 | 4.30 | 120/120 |
| 32 | 0.5 | 11.47 | 3.25 | 10.99 | 3.69 | 80/80 |
| 32 | 0.75 | 15.29 | 3.20 | 14.57 | 3.48 | 40/40 |

- **`auto` is 1.35 to 4.2 times faster than under the old rule** in every round with failing candidates,
  and 0.1 to 0.9 ms behind CUDA alone there (1.1 to 1.3 times). The failing candidates move after 16 CPU
  iterations; the rest of the gap is that phase.
- **Rounds with no failing candidate lose a little.** K = 8 is within noise of the CPU (1.61 against 1.59 ms).
  At K = 16 and 32, `auto` is 0.3 and 0.6 ms behind the CPU and 0.4 and 1.1 ms behind CUDA alone: after 16
  iterations a few good candidates (4 of 32 in `backend_test`) have not found their model yet and move with
  the rest, so the round pays for both a CPU phase and a dispatch. The runner-up, a 32-iteration phase
  with 256 remaining, was faster there (K = 32 with none failing: 3.5 to 4.1 ms in three windows) but up
  to 0.7 ms slower in rounds with failing candidates (the two windows above); neither was faster
  everywhere.
- Also within noise of the choice: 8 iterations with 128 remaining (the 12 cells summed to 32.6 ms against
  33.5 in the same sweep); 16 was kept and confirmed in the two windows above. Dropped: any total of 5,000,
  and 64 or 128 iterations.

Calls with the default budget (10,000 iterations) route exactly as before: all 112 cells of `jpo bench
--batches 1,2,4,8,32,300,3000 --backends auto` drew the same CPU and GPU hypotheses, iterations and solved
counts with the new numbers and with `--one-round-cpu-iter 0` (the old rule).

The rule applies to any call whose budget fits in one GPU round, not only to candidate rounds. Resections
of the `jpo bench` grid with the option on (`jpo bench --kinds pnp --batches 1,8,32,300,3000
--min-inlier-ratio 0.25 --backends cpu,cuda,auto`, against `--one-round-cpu-iter 0` for the old rule), ms:

| batch | points | inliers | cpu | cuda | auto, old rule | auto, one-round rule |
|---|---|---|---|---|---|---|
| 1 | 200 | 0.7 | 0.25 | 0.52 | 0.27 | 0.29 |
| 1 | 200 | 0.3 | 0.40 | 0.47 | 0.30 | 0.41 |
| 1 | 200 | 0.1 | 0.35 | 0.29 | 0.28 | 0.32 |
| 1 | 200 | 0 | 0.28 | 0.44 | 0.29 | 0.37 |
| 1 | 2000 | 0.7 | 1.59 | 4.15 | 1.49 | 1.49 |
| 1 | 2000 | 0.3 | 1.68 | 3.82 | 1.63 | 1.60 |
| 1 | 2000 | 0.1 | 2.04 | 2.87 | 2.15 | 2.05 |
| 1 | 2000 | 0 | 1.93 | 2.06 | 2.06 | 1.97 |
| 8 | 200 | 0.7 | 0.67 | 0.94 | 0.58 | 0.59 |
| 8 | 200 | 0.3 | 1.46 | 0.70 | 1.73 | 0.98 |
| 8 | 200 | 0.1 | 1.44 | 0.65 | 1.55 | 0.70 |
| 8 | 200 | 0 | 1.42 | 0.60 | 1.70 | 0.68 |
| 8 | 2000 | 0.7 | 3.41 | 7.02 | 3.53 | 3.70 |
| 8 | 2000 | 0.3 | 5.04 | 5.90 | 5.33 | 6.41 |
| 8 | 2000 | 0.1 | 12.61 | 5.29 | 14.10 | 5.76 |
| 8 | 2000 | 0 | 11.81 | 5.36 | 13.83 | 5.71 |
| 32 | 200 | 0.7 | 1.54 | 1.91 | 1.77 | 1.68 |
| 32 | 200 | 0.3 | 5.23 | 1.54 | 6.11 | 1.75 |
| 32 | 200 | 0.1 | 5.28 | 1.17 | 6.38 | 1.36 |
| 32 | 200 | 0 | 5.23 | 1.28 | 6.32 | 1.42 |
| 32 | 2000 | 0.7 | 8.68 | 17.28 | 10.34 | 9.14 |
| 32 | 2000 | 0.3 | 16.51 | 14.52 | 18.89 | 14.53 |
| 32 | 2000 | 0.1 | 42.15 | 11.57 | 48.99 | 13.39 |
| 32 | 2000 | 0 | 44.76 | 11.66 | 52.54 | 11.87 |
| 300 | 200 | 0.7 | 11.60 | 10.98 | 13.69 | 12.18 |
| 300 | 200 | 0.3 | 46.62 | 8.43 | 52.90 | 9.87 |
| 300 | 200 | 0.1 | 48.11 | 6.98 | 55.33 | 8.35 |
| 300 | 200 | 0 | 47.54 | 6.70 | 54.92 | 7.94 |
| 300 | 2000 | 0.7 | 80.26 | 84.67 | 90.18 | 82.74 |
| 300 | 2000 | 0.3 | 139.77 | 63.85 | 156.97 | 70.86 |
| 300 | 2000 | 0.1 | 374.88 | 54.91 | 420.73 | 67.06 |
| 300 | 2000 | 0 | 427.94 | 52.63 | 474.38 | 63.82 |
| 3000 | 200 | 0.7 | 121.50 | 77.37 | 135.26 | 133.63 |
| 3000 | 200 | 0.3 | 476.34 | 70.99 | 520.94 | 73.05 |
| 3000 | 200 | 0.1 | 505.47 | 68.67 | 547.70 | 57.52 |
| 3000 | 200 | 0 | 502.66 | 47.55 | 542.10 | 54.53 |
| 3000 | 2000 | 0.7 | 851.72 | 919.91 | 905.28 | 910.57 |
| 3000 | 2000 | 0.3 | 1480.44 | 711.52 | 1533.41 | 772.35 |
| 3000 | 2000 | 0.1 | 4180.69 | 589.53 | 4218.10 | 729.18 |
| 3000 | 2000 | 0 | 4719.50 | 564.19 | 4653.97 | 710.37 |

- Hard batches gain the most: `auto` goes from the CPU's speed to within 1.0 to 1.3 times of CUDA alone,
  for example 300 x 2000 points at 10 % 420.7 → 67.1 ms, 3000 x 2000 with no model 4654 → 710 ms. With the
  option set, this also closes the known gap on 200-point resections at 30 % (3000 of them: 520.9 → 73.0
  ms). With the default budget, `gpu_min_remaining` 512 narrows it (above: 959 → 321 ms).
- One cell is worse: 8 x 2000 points at 30 %, 5.33 → 6.41 ms (CPU 5.04, CUDA alone 5.90). Single problems
  cannot take the one-round path (1 x 569 < 1,000), so the differences in batch 1 rows are noise.
- Solved counts are the same as under the old rule in all 40 cells. The median error differs only in three
  sets at 10 %, where the median problem is unsolved anyway.

## Accuracy against PoseLib (M5 Pro)

200 problems per set, 1000 points (300 for "none"), five seeds each. Success: rotation and translation
within 5 degrees (pairs), rotation within 1 degree (resections); for "none", no pose with more than 30
inliers. PoseLib's row is its own spread over seeds: the noise floor.

| set | tool | success (min .. max over 5 seeds) | rotation deg | AUC@5 | AUC@10 | inlier IoU | iterations |
|---|---|---|---|---|---|---|---|
| relpose_easy | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0198 | 0.9910 | 0.9955 | 0.9532 | 197 |
| relpose_easy | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0205 | 0.9911 | 0.9955 | 0.9533 | 197 |
| relpose_easy | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0205 | 0.9911 | 0.9955 | 0.9533 | 197 |
| relpose_easy | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0203 | 0.9910 | 0.9955 | 0.9533 | 2048 |
| relpose_hard | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0333 | 0.9854 | 0.9927 | 0.9527 | 10000 |
| relpose_hard | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0335 | 0.9856 | 0.9928 | 0.9526 | 10000 |
| relpose_hard | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0335 | 0.9856 | 0.9928 | 0.9526 | 10000 |
| relpose_hard | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0336 | 0.9856 | 0.9928 | 0.9527 | 10000 |
| relpose_harder | poselib | 0.8910 (0.8750 .. 0.9150) | 0.2025 | 0.7876 | 0.8481 | 0.5895 | 10000 |
| relpose_harder | jpo cpu | 0.9070 (0.8850 .. 0.9250) | 0.2004 | 0.7934 | 0.8586 | 0.5952 | 10000 |
| relpose_harder | jpo cpu float | 0.8890 (0.8600 .. 0.9050) | 0.2125 | 0.7756 | 0.8425 | 0.5846 | 10000 |
| relpose_harder | jpo metal | 0.8890 (0.8650 .. 0.9100) | 0.2128 | 0.7753 | 0.8423 | 0.5840 | 10000 |
| relpose_planar | poselib | 0.8550 (0.8450 .. 0.8650) | 0.0477 | 0.8173 | 0.8392 | 0.9531 | 440 |
| relpose_planar | jpo cpu | 0.8400 (0.8300 .. 0.8550) | 0.0483 | 0.8032 | 0.8242 | 0.9531 | 442 |
| relpose_planar | jpo cpu float | 0.8550 (0.8450 .. 0.8600) | 0.0481 | 0.8138 | 0.8366 | 0.9527 | 445 |
| relpose_planar | jpo metal | 0.8440 (0.8350 .. 0.8550) | 0.0512 | 0.7960 | 0.8236 | 0.9518 | 2048 |
| relpose_none | poselib | 1.0000 (1.0000 .. 1.0000) | nan | nan | nan | nan | 10000 |
| relpose_none | jpo cpu | 1.0000 (1.0000 .. 1.0000) | nan | nan | nan | nan | 10000 |
| relpose_none | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | nan | nan | nan | nan | 10000 |
| relpose_none | jpo metal | 1.0000 (1.0000 .. 1.0000) | nan | nan | nan | nan | 10000 |
| relpose_small_baseline | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0211 | 0.9414 | 0.9707 | 0.9533 | 439 |
| relpose_small_baseline | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0213 | 0.9408 | 0.9704 | 0.9534 | 440 |
| relpose_small_baseline | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0215 | 0.9397 | 0.9699 | 0.9533 | 442 |
| relpose_small_baseline | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0214 | 0.9357 | 0.9678 | 0.9529 | 2048 |
| relpose_simple_pinhole | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0221 | 0.9895 | 0.9948 | 0.9517 | 1124 |
| relpose_simple_pinhole | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0211 | 0.9897 | 0.9949 | 0.9517 | 1125 |
| relpose_simple_pinhole | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0212 | 0.9897 | 0.9949 | 0.9517 | 1125 |
| relpose_simple_pinhole | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0210 | 0.9897 | 0.9949 | 0.9516 | 2048 |
| relpose_pinhole | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0262 | 0.9893 | 0.9946 | 0.9534 | 1114 |
| relpose_pinhole | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0254 | 0.9894 | 0.9947 | 0.9534 | 1114 |
| relpose_pinhole | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0255 | 0.9894 | 0.9947 | 0.9534 | 1114 |
| relpose_pinhole | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0252 | 0.9894 | 0.9947 | 0.9534 | 2048 |
| relpose_simple_radial | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0198 | 0.9903 | 0.9951 | 0.9522 | 1123 |
| relpose_simple_radial | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0204 | 0.9904 | 0.9952 | 0.9522 | 1123 |
| relpose_simple_radial | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0205 | 0.9904 | 0.9952 | 0.9522 | 1123 |
| relpose_simple_radial | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0204 | 0.9904 | 0.9952 | 0.9522 | 2048 |
| relpose_radial | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0208 | 0.9897 | 0.9949 | 0.9520 | 1123 |
| relpose_radial | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0198 | 0.9898 | 0.9949 | 0.9521 | 1123 |
| relpose_radial | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0198 | 0.9898 | 0.9949 | 0.9521 | 1123 |
| relpose_radial | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0198 | 0.9898 | 0.9949 | 0.9521 | 2048 |
| relpose_opencv | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0231 | 0.9896 | 0.9948 | 0.9519 | 1122 |
| relpose_opencv | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0224 | 0.9897 | 0.9949 | 0.9519 | 1122 |
| relpose_opencv | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0225 | 0.9897 | 0.9949 | 0.9520 | 1122 |
| relpose_opencv | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0219 | 0.9897 | 0.9949 | 0.9519 | 2048 |
| relpose_opencv_fisheye | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0275 | 0.9794 | 0.9897 | 0.7868 | 3365 |
| relpose_opencv_fisheye | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0285 | 0.9792 | 0.9896 | 0.7863 | 3380 |
| relpose_opencv_fisheye | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0286 | 0.9789 | 0.9895 | 0.7863 | 3380 |
| relpose_opencv_fisheye | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0289 | 0.9784 | 0.9892 | 0.7860 | 3437 |
| pnp_easy | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0024 | 0.9995 | 0.9997 | 1.0000 | 101 |
| pnp_easy | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0025 | 0.9995 | 0.9997 | 1.0000 | 101 |
| pnp_easy | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0025 | 0.9995 | 0.9997 | 1.0000 | 101 |
| pnp_easy | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0025 | 0.9995 | 0.9997 | 1.0000 | 2048 |
| pnp_hard | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0041 | 0.9992 | 0.9996 | 1.0000 | 1018 |
| pnp_hard | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0037 | 0.9992 | 0.9996 | 1.0000 | 1018 |
| pnp_hard | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0037 | 0.9992 | 0.9996 | 1.0000 | 1018 |
| pnp_hard | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0037 | 0.9992 | 0.9996 | 1.0000 | 2048 |
| pnp_harder | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0106 | 0.9978 | 0.9989 | 0.9995 | 3484 |
| pnp_harder | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0090 | 0.9981 | 0.9991 | 0.9995 | 3484 |
| pnp_harder | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0090 | 0.9981 | 0.9991 | 0.9995 | 3484 |
| pnp_harder | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0090 | 0.9981 | 0.9991 | 0.9995 | 3484 |
| pnp_planar | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0092 | 0.9980 | 0.9990 | 1.0000 | 115 |
| pnp_planar | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0090 | 0.9981 | 0.9990 | 1.0000 | 115 |
| pnp_planar | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0090 | 0.9981 | 0.9990 | 1.0000 | 115 |
| pnp_planar | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0090 | 0.9981 | 0.9990 | 1.0000 | 2048 |
| pnp_none | poselib | 1.0000 (1.0000 .. 1.0000) | nan | nan | nan | nan | 10000 |
| pnp_none | jpo cpu | 1.0000 (1.0000 .. 1.0000) | nan | nan | nan | nan | 10000 |
| pnp_none | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | nan | nan | nan | nan | 10000 |
| pnp_none | jpo metal | 1.0000 (1.0000 .. 1.0000) | nan | nan | nan | nan | 10000 |
| pnp_far_origin | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0028 | 0.9994 | 0.9997 | 1.0000 | 115 |
| pnp_far_origin | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0027 | 0.9994 | 0.9997 | 1.0000 | 115 |
| pnp_far_origin | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0027 | 0.9994 | 0.9997 | 1.0000 | 115 |
| pnp_far_origin | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0027 | 0.9994 | 0.9997 | 1.0000 | 2048 |
| pnp_simple_pinhole | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0030 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_simple_pinhole | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0030 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_simple_pinhole | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0030 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_simple_pinhole | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0030 | 0.9994 | 0.9997 | 1.0000 | 2048 |
| pnp_pinhole | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0032 | 0.9993 | 0.9997 | 1.0000 | 209 |
| pnp_pinhole | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0029 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_pinhole | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0029 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_pinhole | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0029 | 0.9994 | 0.9997 | 1.0000 | 2048 |
| pnp_simple_radial | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0029 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_simple_radial | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0027 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_simple_radial | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0027 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_simple_radial | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0027 | 0.9994 | 0.9997 | 1.0000 | 2048 |
| pnp_radial | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0029 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_radial | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0027 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_radial | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0027 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_radial | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0027 | 0.9994 | 0.9997 | 1.0000 | 2048 |
| pnp_opencv | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0031 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_opencv | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0029 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_opencv | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0029 | 0.9994 | 0.9997 | 1.0000 | 209 |
| pnp_opencv | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0029 | 0.9994 | 0.9997 | 1.0000 | 2048 |
| pnp_opencv_fisheye | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0059 | 0.9987 | 0.9993 | 0.9238 | 286 |
| pnp_opencv_fisheye | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0048 | 0.9990 | 0.9995 | 1.0000 | 287 |
| pnp_opencv_fisheye | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0048 | 0.9990 | 0.9995 | 1.0000 | 287 |
| pnp_opencv_fisheye | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0048 | 0.9990 | 0.9995 | 1.0000 | 2048 |

### The `relpose_harder` gap (closed)

The pair rows above are after one change: the points a local optimization refines on are those near the
model **that pass the cheirality test**, as PoseLib's `get_inliers` takes them. Before, an outlier behind a
camera that happened to lie within sqrt(5) thresholds of its epipolar line was refined on, and at 20 %
inliers there are enough of them to pull the pose. Before → after, AUC@5 (PoseLib 0.7876, seeds 0.7589 ..
0.8212):

| set | cpu | cpu float | metal | webgpu (Dawn) |
|---|---|---|---|---|
| relpose_harder | 0.7387 → 0.7934 | 0.7199 → 0.7756 | 0.7180 → 0.7753 | → 0.7744 |
| relpose_planar | 0.7948 → 0.8032 | 0.8004 → 0.8138 | 0.7710 → 0.7960 | 0.7606 → 0.7836 |
| relpose_small_baseline | 0.9390 → 0.9408 | 0.9377 → 0.9397 | 0.9268 → 0.9357 | → 0.9358 |

The success rate on `relpose_harder` rises with it (cpu 0.890 → 0.907, PoseLib 0.891); PnP is unchanged
(its refinement already dropped points behind the camera). Speed is unchanged: on the relative-pose grid of
`jpo bench` (batches 1, 32, 300; 200 and 2000 points; cpu, metal, auto) the median ratio new / old is 1.01,
and the cells that differed by up to 10 % went either way when run again alternately.

How it was found: from the same start poses on the same points, our local optimization and PoseLib's
reached the same MSAC score once the cheirality test was in the point selection, and a slightly worse one
without it; PoseLib's final poses scored better than ours by PoseLib's own MSAC function in 180 of the 187
problems where ours were more than 0.5 degrees worse. Tried and dropped:

- PoseLib's final bundle (Cauchy loss at 1 px over the inliers, after RANSAC): +0.0045 AUC@5 on
  `relpose_harder` with the fix, nothing elsewhere, and 25 to 30 % slower on easy 2000-point pairs. On PnP
  it was slightly worse than the truncated refinement alone (`pnp_harder` 0.9978 against 0.9981).
- PoseLib's Levenberg-Marquardt (additive damping, Nielsen's update): a worse score from the same starts.
- A local optimization per hypothesis instead of per round, and PoseLib's trigger (more inliers or a better
  score than the best minimal model so far): no change on `relpose_harder`; on `relpose_planar` one
  hypothesis per round reaches 0.8088 and a success rate of 0.845 at several times the run time, and one
  local optimization per 32 or 64 hypotheses gives 0.805, so neither was kept.

Left: `relpose_planar` stays under PoseLib's seed spread, AUC@5 0.803 against 0.817 (0.809 .. 0.826), and
its success rate is 0.840 against PoseLib's worst seed 0.845 on the CPU (0.844 before; Metal 0.844, WebGPU
0.832 before and after). The difference is how often the local optimization runs, above.
