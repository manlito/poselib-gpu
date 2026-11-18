/* Built against the installed package only (tests/package/run.cmake), in C: one camera registered
 * against 200 world points, a quarter of them wrong, on the CPU backend and on every other backend the
 * package lists. The pose must come back within a millimetre and a hundredth of a degree. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <jpo/jpo.h>

#define N 200

static uint64_t rng_state = 12345;
static double uniform(double a, double b) {
  uint64_t z = (rng_state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return a + (b - a) * (double)((z ^ (z >> 31)) >> 11) * (1.0 / 9007199254740992.0);
}

static void rotate(const double q[4], const double v[3], double out[3]) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  const double R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
                       2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                       2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)};
  for (int i = 0; i < 3; ++i) out[i] = R[3 * i] * v[0] + R[3 * i + 1] * v[1] + R[3 * i + 2] * v[2];
}

static int solve(int32_t backend, const jpo_resections* problem, const double gt[7]) {
  jpo_config config;
  jpo_config_init(&config);
  config.backend = backend;
  jpo_ctx* ctx = jpo_create(&config);
  if (!ctx) {
    fprintf(stderr, "jpo_create(%d): %s\n", backend, jpo_last_error());
    return 1;
  }
  jpo_ransac_options options;
  jpo_ransac_options_init(&options);
  jpo_pose_result result;
  uint8_t inliers[N];
  if (jpo_absolute_pose(ctx, problem, &options, &result, inliers) != 0) {
    fprintf(stderr, "jpo_absolute_pose: %s\n", jpo_last_error());
    jpo_destroy(ctx);
    return 1;
  }
  const char* name = jpo_backend_name(ctx);
  double dot = 0.0, dt = 0.0;
  for (int i = 0; i < 4; ++i) dot += result.pose[i] * gt[i];
  for (int i = 4; i < 7; ++i) dt += (result.pose[i] - gt[i]) * (result.pose[i] - gt[i]);
  const double angle_deg = 2.0 * acos(fmin(1.0, fabs(dot))) * 180.0 / 3.14159265358979323846;
  dt = sqrt(dt);
  printf("%-14s status %d, %u inliers, rotation %.2e deg, translation %.2e\n", name, result.status,
         result.n_inliers, angle_deg, dt);
  jpo_destroy(ctx);
  return result.status != JPO_OK || angle_deg > 1e-2 || dt > 1e-3 || result.n_inliers < 140;
}

int main(void) {
  /* World to camera: a turn of about 11 degrees around y and a shift. */
  const double gt[7] = {0.9952, 0.0, 0.0978, 0.0, 0.1, -0.2, 0.3};
  const double norm = sqrt(gt[0] * gt[0] + gt[2] * gt[2]);
  const double q[4] = {gt[0] / norm, 0.0, gt[2] / norm, 0.0};
  const double pose[7] = {q[0], q[1], q[2], q[3], gt[4], gt[5], gt[6]};

  jpo_camera camera;
  memset(&camera, 0, sizeof camera);
  camera.model = JPO_PINHOLE;
  camera.params[0] = camera.params[1] = 1000.0;
  camera.params[2] = 640.0, camera.params[3] = 480.0;

  static double xy[2 * N], xyz[3 * N];
  for (int i = 0; i < N; ++i) {
    double X[3] = {uniform(-2, 2), uniform(-1.5, 1.5), uniform(4, 8)}, Xc[3];
    rotate(q, X, Xc);
    for (int k = 0; k < 3; ++k) Xc[k] += pose[4 + k];
    memcpy(&xyz[3 * i], X, sizeof X);
    if (i % 4 == 3) { /* a wrong match */
      xy[2 * i] = uniform(0, 1280), xy[2 * i + 1] = uniform(0, 960);
    } else {
      xy[2 * i] = 1000.0 * Xc[0] / Xc[2] + 640.0 + uniform(-0.3, 0.3);
      xy[2 * i + 1] = 1000.0 * Xc[1] / Xc[2] + 480.0 + uniform(-0.3, 0.3);
    }
  }
  const uint32_t offsets[2] = {0, N}, camera_index[1] = {0};
  jpo_resections problem = {1, 1, offsets, xy, xyz, &camera, camera_index};

  int failed = solve(JPO_BACKEND_CPU, &problem, pose);
  const int32_t gpus[3] = {JPO_BACKEND_METAL, JPO_BACKEND_CUDA, JPO_BACKEND_WEBGPU};
  for (int i = 0; i < 3; ++i)
    if (jpo_backend_available(gpus[i])) failed |= solve(gpus[i], &problem, pose);
  failed |= solve(JPO_BACKEND_AUTO, &problem, pose);
  printf(failed ? "FAILED\n" : "ok\n");
  return failed;
}
