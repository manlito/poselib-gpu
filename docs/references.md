# References and licences

poselib-gpu is a port of other people's solvers and estimators, checked against their code. This page credits
that work, the methods it implements, the data the demo uses, and the software it is built with.

## PoseLib

Viktor Larsson and contributors. PoseLib: minimal solvers for camera pose estimation.
<https://github.com/PoseLib/PoseLib>.

The solvers, the estimators and the refinement are ported from PoseLib, and every solver is tested against
PoseLib's on the same samples. PoseLib's licence applies to the ported code, in source and in compiled form,
including the WebAssembly module this site serves:

```text
BSD 3-Clause License

Copyright (c) 2020, Viktor Larsson
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Methods

- **Five-point relative pose.** David Nistér. "An Efficient Solution to the Five-Point Relative Pose Problem."
  IEEE Transactions on Pattern Analysis and Machine Intelligence, 2004.
- **P3P.** Yaqing Ding, Jian Yang, Viktor Larsson, Carl Olsson and Kalle Åström. "Revisiting the P3P
  Problem." CVPR 2023.
- **RANSAC.** Martin A. Fischler and Robert C. Bolles. "Random Sample Consensus: A Paradigm for Model Fitting
  with Applications to Image Analysis and Automated Cartography." Communications of the ACM, 1981.
- **MSAC scoring.** Philip H. S. Torr and Andrew Zisserman. "MLESAC: A New Robust Estimator with Application
  to Estimating Image Geometry." Computer Vision and Image Understanding, 2000.
- **Local optimization.** Ondřej Chum, Jiří Matas and Josef Kittler. "Locally Optimized RANSAC." DAGM 2003.
- **Camera models.** COLMAP's models, names and parameter order. Johannes L. Schönberger and Jan-Michael
  Frahm. "Structure-from-Motion Revisited." CVPR 2016. <https://colmap.github.io>.

## Data in the demo

- Three frames of the KITTI odometry benchmark (sequence 00, frames 100, 103 and 106). Andreas Geiger,
  Philip Lenz and Raquel Urtasun. "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite."
  CVPR 2012. <https://www.cvlibs.net/datasets/kitti/>. CC BY-NC-SA 3.0.
- Four frames of the same drive in the KITTI raw data (`2011_10_03_drive_0027`, left colour camera, frames
  745, 3679, 3100 and 1300): one street corner passed twice, and two streets that look alike. Andreas Geiger,
  Philip Lenz, Christoph Stiller and Raquel Urtasun. "Vision meets Robotics: The KITTI Dataset." IJRR 2013.
  <https://www.cvlibs.net/datasets/kitti/raw_data.php>. CC BY-NC-SA 3.0.

The images were re-encoded as JPEG; they, and these copies, are for non-commercial use only
(<https://creativecommons.org/licenses/by-nc-sa/3.0/>). The matches were computed once, ahead of time:
ALIKED keypoints with [just-keypoints](https://just-keypoints.pages.dev) (Xiaoming Zhao, Xingming Wu, Weihai
Chen, Peter C. Y. Chen, Qingsong Xu and Zhengguo Li, "ALIKED: A Lighter Keypoint and Descriptor Extraction
Network via Deformable Transformation", IEEE Transactions on Instrumentation and Measurement, 2023), matched
with FlexGlue through just-matching (Philipp Lindenberger, Paul-Edouard Sarlin, Ondrej Miksik and Marc
Pollefeys, "FlexGlue: Sparse-to-Dense Feature Matching and Visual Disambiguation", 2026). The point cloud of
the second tab was triangulated from frames 100 and 106 with this library's relative pose; its scale is the
distance between those two frames.

## Software

- **Dawn** and **emdawnwebgpu**, Google's WebGPU implementation and its Emscripten bindings: the WebGPU
  backend, natively and in the browser. <https://dawn.googlesource.com/dawn>. BSD 3-Clause licence.
- **Emscripten**: the browser build. <https://emscripten.org>. MIT and University of Illinois/NCSA licences.
- **Sphinx** and **MyST-Parser**: this site. The theme follows the documentation site of just-ba, a bundle
  adjustment library by the same authors.
- **Inter** and **IBM Plex Mono**, from Google Fonts: the site's typefaces. SIL Open Font License.
