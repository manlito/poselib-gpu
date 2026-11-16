# poselib-gpu — measurements

The tables behind PLAN.md ("Results"). Times are milliseconds end to end, host buffers in, poses and inlier
masks out, best of three runs after a warm-up. Problems are generated (`jpo gen`): 0.5 px noise for pairs,
1 px for resections; "inliers 0" is a set with no model at all. Options are the defaults (confidence 0.9999,
100 to 10000 iterations, 1 px / 4 px). PoseLib is the pinned commit, one problem per thread over all threads
(OpenMP over the batch), so its single-problem rows are one thread against jpo's all threads.

Reproduce: `python3 tools/bench.py --backends cpu,metal,auto`, `python3 tools/accuracy.py [--float | --backend metal]`.

## Apple M5 Pro (18 threads), Metal

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

## WebGPU

Natively through Dawn on the M5 Pro the WebGPU backend costs what Metal costs (300 pairs of 200 matches with
no model: 211 ms against 197 ms for Metal and 500 ms for the CPU; 300 resections of 2000 points: 216 ms
against 224 ms and 1282 ms), so natively it follows Metal's rule.

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
timings were taken once, in one browser; Chrome itself, Safari and Firefox were not run.

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
- Not run: CUDA (it is on the NVIDIA laptop; its rule has no one-round numbers, so a one-round budget
  follows its usual rule there) and WebGPU timings (likewise). The option is in the host driver, so every
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

## Accuracy against PoseLib (M5 Pro)

200 problems per set, 1000 points (300 for "none"), five seeds each. Success: rotation and translation
within 5 degrees (pairs), rotation within 1 degree (resections); for "none", no pose with more than 30
inliers. PoseLib's row is its own spread over seeds: the noise floor.

| set | tool | success (min .. max over 5 seeds) | rotation deg | AUC@5 | AUC@10 | inlier IoU | iterations |
|---|---|---|---|---|---|---|---|
| relpose_easy | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0198 | 0.9910 | 0.9955 | 0.9532 | 197 |
| relpose_easy | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0208 | 0.9910 | 0.9955 | 0.9533 | 197 |
| relpose_easy | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0208 | 0.9910 | 0.9955 | 0.9533 | 197 |
| relpose_easy | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0206 | 0.9909 | 0.9955 | 0.9533 | 2048 |
| relpose_hard | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0333 | 0.9854 | 0.9927 | 0.9527 | 10000 |
| relpose_hard | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0345 | 0.9851 | 0.9926 | 0.9525 | 10000 |
| relpose_hard | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0340 | 0.9852 | 0.9926 | 0.9525 | 10000 |
| relpose_hard | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0344 | 0.9851 | 0.9926 | 0.9525 | 10000 |
| relpose_harder | poselib | 0.8910 (0.8750 .. 0.9150) | 0.2025 | 0.7876 | 0.8481 | 0.5895 | 10000 |
| relpose_harder | jpo cpu | 0.8900 (0.8750 .. 0.9150) | 0.2455 | 0.7387 | 0.8277 | 0.5691 | 10000 |
| relpose_harder | jpo cpu float | 0.8700 (0.8600 .. 0.8800) | 0.2652 | 0.7199 | 0.8102 | 0.5578 | 10000 |
| relpose_harder | jpo metal | 0.8690 (0.8550 .. 0.8800) | 0.2629 | 0.7180 | 0.8087 | 0.5560 | 10000 |
| relpose_planar | poselib | 0.8550 (0.8450 .. 0.8650) | 0.0477 | 0.8173 | 0.8392 | 0.9531 | 440 |
| relpose_planar | jpo cpu | 0.8440 (0.8300 .. 0.8650) | 0.0512 | 0.7948 | 0.8216 | 0.9524 | 444 |
| relpose_planar | jpo cpu float | 0.8550 (0.8350 .. 0.8700) | 0.0535 | 0.8004 | 0.8305 | 0.9519 | 447 |
| relpose_planar | jpo metal | 0.8440 (0.8300 .. 0.8550) | 0.0654 | 0.7710 | 0.8106 | 0.9507 | 2048 |
| relpose_none | poselib | 1.0000 (1.0000 .. 1.0000) | nan | nan | nan | nan | 10000 |
| relpose_none | jpo cpu | 1.0000 (1.0000 .. 1.0000) | nan | nan | nan | nan | 10000 |
| relpose_none | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | nan | nan | nan | nan | 10000 |
| relpose_none | jpo metal | 1.0000 (1.0000 .. 1.0000) | nan | nan | nan | nan | 10000 |
| relpose_small_baseline | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0211 | 0.9414 | 0.9707 | 0.9533 | 439 |
| relpose_small_baseline | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0219 | 0.9390 | 0.9695 | 0.9533 | 441 |
| relpose_small_baseline | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0221 | 0.9377 | 0.9688 | 0.9531 | 443 |
| relpose_small_baseline | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0223 | 0.9268 | 0.9634 | 0.9525 | 2048 |
| relpose_simple_pinhole | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0221 | 0.9895 | 0.9948 | 0.9517 | 1124 |
| relpose_simple_pinhole | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0215 | 0.9896 | 0.9948 | 0.9517 | 1125 |
| relpose_simple_pinhole | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0215 | 0.9896 | 0.9948 | 0.9516 | 1125 |
| relpose_simple_pinhole | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0223 | 0.9895 | 0.9948 | 0.9515 | 2048 |
| relpose_pinhole | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0262 | 0.9893 | 0.9946 | 0.9534 | 1114 |
| relpose_pinhole | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0253 | 0.9893 | 0.9946 | 0.9533 | 1115 |
| relpose_pinhole | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0254 | 0.9893 | 0.9946 | 0.9533 | 1115 |
| relpose_pinhole | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0253 | 0.9892 | 0.9946 | 0.9533 | 2048 |
| relpose_simple_radial | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0198 | 0.9903 | 0.9951 | 0.9522 | 1123 |
| relpose_simple_radial | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0203 | 0.9903 | 0.9952 | 0.9522 | 1123 |
| relpose_simple_radial | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0203 | 0.9903 | 0.9952 | 0.9522 | 1123 |
| relpose_simple_radial | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0194 | 0.9902 | 0.9951 | 0.9522 | 2048 |
| relpose_radial | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0208 | 0.9897 | 0.9949 | 0.9520 | 1123 |
| relpose_radial | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0195 | 0.9898 | 0.9949 | 0.9521 | 1123 |
| relpose_radial | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0195 | 0.9898 | 0.9949 | 0.9520 | 1123 |
| relpose_radial | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0194 | 0.9898 | 0.9949 | 0.9520 | 2048 |
| relpose_opencv | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0231 | 0.9896 | 0.9948 | 0.9519 | 1122 |
| relpose_opencv | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0228 | 0.9897 | 0.9948 | 0.9519 | 1122 |
| relpose_opencv | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0228 | 0.9896 | 0.9948 | 0.9519 | 1122 |
| relpose_opencv | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0226 | 0.9894 | 0.9947 | 0.9519 | 2048 |
| relpose_opencv_fisheye | poselib | 1.0000 (1.0000 .. 1.0000) | 0.0275 | 0.9794 | 0.9897 | 0.7868 | 3365 |
| relpose_opencv_fisheye | jpo cpu | 1.0000 (1.0000 .. 1.0000) | 0.0288 | 0.9787 | 0.9894 | 0.7863 | 3381 |
| relpose_opencv_fisheye | jpo cpu float | 1.0000 (1.0000 .. 1.0000) | 0.0290 | 0.9785 | 0.9892 | 0.7863 | 3381 |
| relpose_opencv_fisheye | jpo metal | 1.0000 (1.0000 .. 1.0000) | 0.0298 | 0.9775 | 0.9888 | 0.7857 | 3442 |
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
