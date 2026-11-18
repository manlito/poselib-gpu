# Quick start

One context, then any number of calls. Each call takes a batch: every pair to verify, or every image to
register, with their matches concatenated and an `offsets` array saying where each problem's matches start.

## Relative pose of image pairs

```c
#include <stdio.h>
#include <jpo/jpo.h>

int main(void) {
  jpo_config cfg;
  jpo_config_init(&cfg);              /* backend = auto, all CPU threads */
  jpo_ctx* ctx = jpo_create(&cfg);
  if (!ctx) {
    fprintf(stderr, "jpo_create: %s\n", jpo_last_error());
    return 1;
  }
  printf("backend: %s\n", jpo_backend_name(ctx));   /* for example "auto(metal)" */

  /* One pinhole camera shared by every image: fx fy cx cy (COLMAP's PINHOLE). */
  jpo_camera cam = {JPO_PINHOLE, 0, {718.856, 718.856, 607.19, 185.22}};

  /* Two pairs with 100 and 80 matches. xy_a[2 i], xy_a[2 i + 1] is match i's pixel in image a;
   * pair p owns matches offsets[p] .. offsets[p + 1] - 1. */
  uint32_t offsets[3] = {0, 100, 180};
  double xy_a[2 * 180], xy_b[2 * 180];   /* your matches */
  uint32_t camera_of_pair[2] = {0, 0};   /* index into the cameras, for image a and for image b */

  jpo_pairs pairs = {
    .n_pairs = 2, .n_cameras = 1, .offsets = offsets, .xy_a = xy_a, .xy_b = xy_b,
    .cameras = &cam, .camera_a = camera_of_pair, .camera_b = camera_of_pair,
  };
  jpo_pose_result results[2];
  uint8_t inliers[180];                  /* one byte per match; may be NULL */
  if (jpo_relative_pose(ctx, &pairs, NULL, results, inliers) != 0) {   /* NULL: default options */
    fprintf(stderr, "jpo_relative_pose: %s\n", jpo_last_error());
    return 1;
  }
  for (int p = 0; p < 2; ++p) {
    const jpo_pose_result* r = &results[p];
    printf("pair %d: status %d, %u inliers, median triangulation angle %.2f deg\n",
           p, r->status, r->n_inliers, r->median_tri_angle_deg);
  }
  jpo_destroy(ctx);
  return 0;
}
```

Each result holds the pose of camera b relative to camera a: `pose[0..3]` is the rotation as a quaternion
`qw qx qy qz`, `pose[4..6]` the translation, of unit length, so that a point `X` in camera a's frame is
`R X + t` in camera b's. The inlier mask says which matches agree with it.

## Deciding which pairs to keep

`JPO_OK` only says that some pose has more inliers than a minimal sample. Matches between images that do not
overlap come back `JPO_OK` with a handful of inliers, because a few random matches always agree with some
pose. Keep a pair by your own rule on `n_inliers`, for example at least 30 inliers and a quarter of its
matches. The median triangulation angle tells how well the baseline is observed; it is what the choice of
the first pair of a reconstruction needs. A pair whose baseline cannot be observed (as under a pure rotation)
comes back `JPO_DEGENERATE`: its rotation and inlier mask are still usable, its translation is not.

## Registering images against 3D points

```c
/* Three images to register: for each, its 2D-3D matches. */
uint32_t offsets[4] = {0, 500, 900, 1600};
double xy[2 * 1600];                  /* pixels */
double xyz[3 * 1600];                 /* the matched world points */
uint32_t camera_of_image[3] = {0, 0, 1};

jpo_resections images = {
  .n_images = 3, .n_cameras = 2, .offsets = offsets, .xy = xy, .xyz = xyz,
  .cameras = cams, .camera = camera_of_image,
};
jpo_ransac_options opt;
jpo_ransac_options_init(&opt);        /* 1 px; PnP's default (options = NULL) is 4 px */
opt.max_error_px = 4.0;
jpo_pose_result results[3];
uint8_t inliers[1600];
jpo_absolute_pose(ctx, &images, &opt, results, inliers);
/* results[i].pose: world to camera, X_cam = R X_world + t */
```

When a reconstruction tries several candidate images at once and keeps only those that pass, set
`opt.min_inlier_ratio` (for example 0.25): candidates that cannot reach that ratio give up early instead of
running to `max_iterations`. See [How it works](concepts.md#candidate-images-in-one-call).

## Linking

Link through CMake, from the installed package or the added folder
(`target_link_libraries(my_app PRIVATE jpo::jpo)`, see [Build](install.md#using-it-from-another-cmake-project)):
the target carries what each backend needs, such as Metal.framework or the CUDA runtime, and Dawn for native
WebGPU. Linking `libjpo.a` by hand means listing those as well.
