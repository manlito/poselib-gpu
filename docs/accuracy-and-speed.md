# Accuracy and speed

All numbers come from generated problems, where the true pose is known, with a share of the matches
replaced by random ones ("inliers"). A set with no inliers has no true pose at all, like two images that do
not overlap. Options are the defaults: confidence 0.9999, 100 to 10,000 iterations, thresholds 1 px for pairs
and 4 px for PnP.

## Accuracy against PoseLib

200 problems per set, 1000 matches (300 for the sets with no true pose), 0.5 px of noise (1 px for the
"harder" sets), five random seeds each, on an Apple M5 Pro. Success: rotation and direction of travel within
5° of the truth for a pair, rotation within 1° for a resection; for the sets with no true pose, no pose with
more than 30 inliers. PoseLib against itself over five seeds gives the spread to expect from randomness
alone.

| set | inliers | PoseLib | CPU (double) | CPU (float32) | Metal |
|---|---|---|---|---|---|
| pairs | 70 % | 1.000 | 1.000 | 1.000 | 1.000 |
| pairs | 30 % | 1.000 | 1.000 | 1.000 | 1.000 |
| pairs, 1 px noise | 20 % | 0.891 (0.875 .. 0.915) | 0.907 (0.885 .. 0.925) | 0.889 (0.860 .. 0.905) | 0.889 (0.865 .. 0.910) |
| pairs, planar scene | 60 % | 0.855 (0.845 .. 0.865) | 0.840 (0.830 .. 0.855) | 0.855 (0.845 .. 0.860) | 0.844 (0.835 .. 0.855) |
| pairs, very short baseline | 60 % | 1.000 | 1.000 | 1.000 | 1.000 |
| pairs, fisheye camera | 50 % | 1.000 | 1.000 | 1.000 | 1.000 |
| pairs, no true pose | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| PnP | 70 % | 1.000 | 1.000 | 1.000 | 1.000 |
| PnP | 30 % | 1.000 | 1.000 | 1.000 | 1.000 |
| PnP, 1 px noise | 20 % | 1.000 | 1.000 | 1.000 | 1.000 |
| PnP, planar scene | 60 % | 1.000 | 1.000 | 1.000 | 1.000 |
| PnP, world far from the origin | 60 % | 1.000 | 1.000 | 1.000 | 1.000 |
| PnP, no true pose | 0 | 1.000 | 1.000 | 1.000 | 1.000 |

Success rate, mean over five seeds (lowest .. highest). Every camera model at 50 % inliers gives 1.000 on all
four. Rotation errors agree with PoseLib's to within a few thousandths of a degree: on pairs at 30 % inliers,
0.0333° for PoseLib and 0.0335° here. On the planar pairs the double CPU build is just below PoseLib's lowest
seed (0.840 against 0.845); the other builds are within its spread.

## Speed

Milliseconds end to end, host buffers in, poses and inlier masks out, best of three runs after a warm-up.
0.5 px of noise on pairs, 1 px on resections. PoseLib runs one problem per thread over all threads.

### Apple M5 Pro (18 threads), Metal

| problem | per call | matches | inliers | PoseLib | CPU | Metal |
|---|---|---|---|---|---|---|
| pair | 1 | 2000 | 70 % | 3.59 | 0.85 | 4.01 |
| pair | 1 | 2000 | 30 % | 40.73 | 2.23 | 7.29 |
| pair | 300 | 200 | 70 % | 17.88 | 13.80 | 55.13 |
| pair | 300 | 2000 | 30 % | 967.55 | 511.63 | 239.87 |
| pair | 300 | 2000 | none | 902.45 | 489.64 | 214.94 |
| resection | 1 | 2000 | 70 % | 2.86 | 0.59 | 1.59 |
| resection | 300 | 2000 | 70 % | 52.22 | 14.93 | 62.55 |
| resection | 300 | 2000 | 10 % | 1056.00 | 364.48 | 89.06 |
| resection | 300 | 2000 | none | 1068.98 | 1282.02 | 224.21 |

### Intel i7-9750H (6 cores), RTX 2070 Max-Q, CUDA

| problem | per call | matches | inliers | PoseLib | CPU | CUDA |
|---|---|---|---|---|---|---|
| pair | 1 | 2000 | 70 % | 10.86 | 2.67 | 6.28 |
| pair | 1 | 2000 | 30 % | 113.36 | 10.22 | 7.44 |
| pair | 300 | 200 | 70 % | 101.56 | 92.99 | 108.55 |
| pair | 300 | 2000 | 30 % | 6160.95 | 3571.11 | 472.71 |
| pair | 300 | 2000 | none | 5528.74 | 3425.09 | 413.97 |
| resection | 1 | 2000 | 70 % | 6.40 | 1.68 | 2.57 |
| resection | 300 | 2000 | 70 % | 284.55 | 91.30 | 209.35 |
| resection | 300 | 2000 | 10 % | 7071.86 | 2271.64 | 277.85 |
| resection | 300 | 2000 | none | 6987.80 | 7993.89 | 709.80 |

The pattern is the same on both machines. Easy problems are fastest on the CPU, which stops them after a few
hundred hypotheses, where a GPU round always computes 2048. Hard problems, and pairs that do not overlap, run
to thousands of hypotheses: there the GPU is 2 to 6 times faster than the CPU on the M5 Pro and 7 to 11 times
on the laptop. `auto` keeps the first kind on the CPU and moves the second to the GPU (see
[How it works](concepts.md#the-auto-backend)).

### In the browser

See [In the browser](browser.md#speed-in-the-browser), and run <a href="benchmark/index.html">the
benchmark</a> on your own machine.

## Not measured

- Real captures against COLMAP's reconstructions: the incremental test uses generated scenes.
- CUDA on Windows (the CPU build is tested there), and browsers other than Chrome.
