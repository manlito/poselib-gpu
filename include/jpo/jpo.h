/* poselib-gpu: robust relative pose and absolute pose (PnP) for calibrated cameras.
 *
 * Conventions
 *   Pose      7 doubles, qw qx qy qz tx ty tz: a point X maps to R(q) X + t. For an absolute pose that is
 *             world to camera; for a relative pose, camera a to camera b, with |t| = 1.
 *   Axes      x right, y down, z forward.
 *   Pixels    whatever convention the camera parameters use; the library only ever applies the camera
 *             to the points it is given. (just-keypoints: (0,0) is the centre of the top-left pixel.)
 *   Cameras   COLMAP's models, ids and parameter order.
 *
 * Every call takes a batch of problems. All buffers are the caller's; nothing is kept after a call.
 */
#ifndef JPO_H
#define JPO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jpo_ctx jpo_ctx;

typedef enum {
  JPO_BACKEND_AUTO = 0, /* the measured rule of this machine: GPU where it is faster, CPU elsewhere */
  JPO_BACKEND_CPU = 1,
  JPO_BACKEND_METAL = 2,
  JPO_BACKEND_CUDA = 3,
  JPO_BACKEND_WEBGPU = 4
} jpo_backend;

typedef enum {
  JPO_SIMPLE_PINHOLE = 0, /* f cx cy */
  JPO_PINHOLE = 1,        /* fx fy cx cy */
  JPO_SIMPLE_RADIAL = 2,  /* f cx cy k */
  JPO_RADIAL = 3,         /* f cx cy k1 k2 */
  JPO_OPENCV = 4,         /* fx fy cx cy k1 k2 p1 p2 */
  JPO_OPENCV_FISHEYE = 5  /* fx fy cx cy k1 k2 k3 k4 */
} jpo_camera_model;

typedef struct {
  int32_t model; /* jpo_camera_model */
  int32_t reserved;
  double params[8];
} jpo_camera;

typedef struct {
  int32_t backend;       /* jpo_backend */
  uint32_t seed;         /* same seed, same samples, on every backend */
  int32_t threads;       /* CPU threads; 0 = all */
  int32_t cpu_float;     /* CPU backend in float32 (what the GPU backends are tested against) */
  uint32_t gpu_round;    /* hypotheses per problem in one GPU dispatch; 0 = 2048 */
  uint32_t reserved;
  void* webgpu_instance; /* WGPUInstance and WGPUDevice of the caller, or NULL to create them */
  void* webgpu_device;
} jpo_config;

/* Number of parameters of a camera model (3 .. 8), 0 for an unknown model. */
int jpo_camera_param_count(int32_t model);

void jpo_config_init(jpo_config* config);

/* NULL on failure; jpo_last_error() says why. */
jpo_ctx* jpo_create(const jpo_config* config);
void jpo_destroy(jpo_ctx* ctx);

/* Whether this build has the backend (CPU: always). */
int jpo_backend_available(int32_t backend);
/* "cpu", "metal", "cuda", "webgpu", or "auto(<gpu>)". */
const char* jpo_backend_name(const jpo_ctx* ctx);
/* The message of the last failed call on this thread. */
const char* jpo_last_error(void);

typedef struct {
  double max_error_px;     /* inlier threshold in pixels (default 1.0 relative, see init functions) */
  double confidence;       /* stop when a better model is this unlikely (default 0.9999) */
  uint32_t min_iterations; /* default 100 */
  uint32_t max_iterations; /* default 10000 */
  int32_t refine;          /* non-linear refinement of the result (default 1) */
  int32_t reserved;
} jpo_ransac_options;

void jpo_ransac_options_init(jpo_ransac_options* options);

/* n_pairs image pairs. Pair i owns matches offsets[i] .. offsets[i + 1] - 1 of xy_a and xy_b. */
typedef struct {
  uint32_t n_pairs;
  uint32_t n_cameras;
  const uint32_t* offsets;   /* n_pairs + 1 */
  const double* xy_a;        /* 2 per match: the pixel in image a */
  const double* xy_b;        /* 2 per match: the pixel in image b */
  const jpo_camera* cameras; /* n_cameras */
  const uint32_t* camera_a;  /* n_pairs: index into cameras */
  const uint32_t* camera_b;  /* n_pairs */
} jpo_pairs;

/* n_images images to register. Image i owns matches offsets[i] .. offsets[i + 1] - 1 of xy and xyz. */
typedef struct {
  uint32_t n_images;
  uint32_t n_cameras;
  const uint32_t* offsets;   /* n_images + 1 */
  const double* xy;          /* 2 per match: the pixel */
  const double* xyz;         /* 3 per match: the world point */
  const jpo_camera* cameras; /* n_cameras */
  const uint32_t* camera;    /* n_images: index into cameras */
} jpo_resections;

/* JPO_OK only says that a model with more inliers than a minimal sample was found: random matches of
 * images that do not overlap come back OK with a handful of inliers. Whether a pair or a registration
 * is accepted is the caller's rule on n_inliers (for example at least 30 inliers and a quarter of the
 * matches for a pair). */
typedef enum {
  JPO_OK = 0,
  JPO_NO_MODEL = 1,        /* no hypothesis had more inliers than a minimal sample */
  JPO_TOO_FEW_POINTS = 2,  /* fewer points than a minimal sample (5 relative, 3 absolute) */
  JPO_DEGENERATE = 3       /* relative pose only: a pose was found but the baseline is not observable
                              (median triangulation angle below ten inlier thresholds, as in a pure
                              rotation); the rotation and the inlier mask are still usable (the
                              matches can go into tracks), the translation is not */
} jpo_status;

typedef struct {
  double pose[7];
  double score;                /* MSAC: sum over points of min(error^2, threshold^2), in pixels^2 */
  double median_tri_angle_deg; /* relative pose: median triangulation angle of the inliers; else 0 */
  uint32_t n_inliers;
  uint32_t iterations;         /* hypotheses drawn */
  int32_t status;              /* jpo_status */
  int32_t reserved;
} jpo_pose_result;

/* results: n_pairs entries. inliers: one byte per match (may be NULL). Returns 0, or a negative value
 * on bad arguments or a backend failure (jpo_last_error). A problem that fails is not an error: see
 * jpo_pose_result.status. */
int jpo_relative_pose(jpo_ctx* ctx, const jpo_pairs* pairs, const jpo_ransac_options* options,
                      jpo_pose_result* results, uint8_t* inliers);

/* results: n_images entries. inliers: one byte per match (may be NULL). */
int jpo_absolute_pose(jpo_ctx* ctx, const jpo_resections* resections, const jpo_ransac_options* options,
                      jpo_pose_result* results, uint8_t* inliers);

#ifdef __cplusplus
}
#endif

#endif /* JPO_H */
