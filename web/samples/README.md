# Demo data

What `demo.html` loads, written by `tools/demo_data.py` (its docstring has the command):

| File | What |
|---|---|
| `pairs.json` | The camera, and three pairs with their matches: `xyA`, `xyB` (pixels, (0, 0) the centre of the top-left pixel). |
| `cloud.json` | Points triangulated from frames 100 and 106 (`cloud`), the centres of those two cameras, and frame 103's matches to the points (`xy`, `xyz`). Scale: frames 100 and 106 are 1 apart. |
| `*.jpg` | The images, copied from just-matching's `web/samples`. |

The matches are ALIKED keypoints (just-keypoints, its default model) matched with FlexGlue (just-matching), run
once. The camera is KITTI's rectified one for this drive: `PINHOLE` 718.856, 718.856, 607.1928, 185.2157.

| Files | Source | Licence |
|---|---|---|
| `kitti-1`, `kitti-2`, `kitti-3` | KITTI odometry benchmark, sequence 00, frames 100, 103 and 106 (0.3 s apart), from the sample sequence in the ALIKED repository. Andreas Geiger, Philip Lenz and Raquel Urtasun, "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite", CVPR 2012. <https://www.cvlibs.net/datasets/kitti/> | CC BY-NC-SA 3.0 (non-commercial) |
| `kitti-revisit-1`, `kitti-revisit-2` | KITTI raw data, drive `2011_10_03_drive_0027` (odometry sequence 00), left colour camera (`image_02`), frames 745 and 3679: the same corner, minutes apart and 22° apart in heading. Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun, "Vision meets Robotics: The KITTI Dataset", IJRR 2013. <https://www.cvlibs.net/datasets/kitti/raw_data.php> | CC BY-NC-SA 3.0 (non-commercial) |
| `kitti-alike-1`, `kitti-alike-2` | The same drive and camera, frames 3100 and 1300: two streets 270 m apart. Same citation. | CC BY-NC-SA 3.0 (non-commercial) |

The images were re-encoded from PNG as JPEG (quality 88), otherwise unchanged. The KITTI images, and these JPEG
copies of them, are under CC BY-NC-SA 3.0 <https://creativecommons.org/licenses/by-nc-sa/3.0/>: they may be used
for non-commercial purposes only, with credit to the authors above, and shared under the same licence.
