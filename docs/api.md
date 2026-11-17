# C API

One header, `jpo/jpo.h`. Every call takes a batch of problems; all buffers belong to the caller, and the
library keeps nothing after a call returns.

## Context

```c
void jpo_config_init(jpo_config* config);
jpo_ctx* jpo_create(const jpo_config* config);   /* NULL on failure: see jpo_last_error() */
void jpo_destroy(jpo_ctx* ctx);

int jpo_backend_available(int32_t backend);      /* whether this build has it (CPU and auto: always) */
const char* jpo_backend_name(const jpo_ctx* ctx);  /* "cpu", "metal", "cuda", "webgpu" or "auto(<gpu>)" */
const char* jpo_last_error(void);                /* the message of the last failed call on this thread */
```

| `jpo_config` field | Meaning | Default |
|---|---|---|
| `backend` | `JPO_BACKEND_AUTO`, `_CPU`, `_METAL`, `_CUDA` or `_WEBGPU` | `AUTO` |
| `seed` | Seed of the samples; the same seed draws the same samples on every backend | 0 |
| `threads` | CPU threads; 0 = all | 0 |
| `cpu_float` | Run the CPU backend in float32, as the GPU kernels do (for testing) | 0 |
| `gpu_round` | Hypotheses per problem in one GPU dispatch; 0 = 2048 | 0 |
| `webgpu_instance`, `webgpu_device` | The caller's `WGPUInstance` and `WGPUDevice`, to share one GPU device; NULL to create them | NULL |

A context may be used for any number of calls, one at a time.

## Cameras

```c
typedef struct {
  int32_t model;      /* jpo_camera_model */
  int32_t reserved;
  double params[8];   /* COLMAP's parameter order */
} jpo_camera;

int jpo_camera_param_count(int32_t model);   /* 3 .. 8, or 0 for an unknown model */
```

| Model | Id | Parameters |
|---|---|---|
| `JPO_SIMPLE_PINHOLE` | 0 | f cx cy |
| `JPO_PINHOLE` | 1 | fx fy cx cy |
| `JPO_SIMPLE_RADIAL` | 2 | f cx cy k |
| `JPO_RADIAL` | 3 | f cx cy k1 k2 |
| `JPO_OPENCV` | 4 | fx fy cx cy k1 k2 p1 p2 |
| `JPO_OPENCV_FISHEYE` | 5 | fx fy cx cy k1 k2 k3 k4 |

## Options

```c
void jpo_ransac_options_init(jpo_ransac_options* options);
```

| Field | Meaning | Default |
|---|---|---|
| `max_error_px` | Inlier threshold in pixels | 1.0 after `jpo_ransac_options_init`. Passing NULL options gives 1 px for pairs and 4 px for PnP. |
| `confidence` | Stop when a better model is this unlikely | 0.9999 |
| `min_iterations`, `max_iterations` | Bounds on the hypotheses drawn per problem | 100, 10000 |
| `refine` | Non-linear refinement of the result | 1 |
| `min_inlier_ratio` | Give up on a problem whose inlier ratio is below this, in [0, 1); 0 = off. See [Candidate images in one call](concepts.md#candidate-images-in-one-call). | 0 |

## Relative pose

```c
typedef struct {
  uint32_t n_pairs, n_cameras;
  const uint32_t* offsets;    /* n_pairs + 1: pair i owns matches offsets[i] .. offsets[i + 1] - 1 */
  const double* xy_a;         /* 2 per match: the pixel in image a */
  const double* xy_b;         /* 2 per match: the pixel in image b */
  const jpo_camera* cameras;  /* n_cameras */
  const uint32_t* camera_a;   /* n_pairs: index into cameras */
  const uint32_t* camera_b;   /* n_pairs */
} jpo_pairs;

int jpo_relative_pose(jpo_ctx* ctx, const jpo_pairs* pairs, const jpo_ransac_options* options,
                      jpo_pose_result* results, uint8_t* inliers);
```

`results` has one entry per pair; `inliers` (may be NULL) one byte per match. The pose is camera a to camera
b, with a translation of unit length.

## Absolute pose

```c
typedef struct {
  uint32_t n_images, n_cameras;
  const uint32_t* offsets;    /* n_images + 1 */
  const double* xy;           /* 2 per match: the pixel */
  const double* xyz;          /* 3 per match: the world point */
  const jpo_camera* cameras;  /* n_cameras */
  const uint32_t* camera;     /* n_images: index into cameras */
} jpo_resections;

int jpo_absolute_pose(jpo_ctx* ctx, const jpo_resections* resections, const jpo_ransac_options* options,
                      jpo_pose_result* results, uint8_t* inliers);
```

The pose is world to camera. Both functions return 0, or a negative value on bad arguments or a backend
failure (see `jpo_last_error`). A problem without a pose is not an error: it says so in its status.

## Results

```c
typedef struct {
  double pose[7];               /* qw qx qy qz tx ty tz */
  double score;                 /* MSAC: sum over matches of min(error^2, threshold^2), in pixels^2 */
  double median_tri_angle_deg;  /* relative pose: median triangulation angle of the inliers; else 0 */
  uint32_t n_inliers;
  uint32_t iterations;          /* hypotheses drawn */
  int32_t status;               /* jpo_status */
  int32_t reserved;
} jpo_pose_result;
```

| Status | Meaning |
|---|---|
| `JPO_OK` (0) | A model with more inliers than a minimal sample was found. Not a verdict: random matches between images that do not overlap come back OK with a handful of inliers. Accept by your own rule on `n_inliers`. |
| `JPO_NO_MODEL` (1) | No hypothesis had more inliers than a minimal sample. |
| `JPO_TOO_FEW_POINTS` (2) | Fewer matches than a minimal sample: 5 for a pair, 3 for PnP. |
| `JPO_DEGENERATE` (3) | Relative pose only: the baseline is not observable (median triangulation angle below ten inlier thresholds, as under a pure rotation). The rotation and the inlier mask are still usable; the translation is not. |
