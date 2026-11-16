// WebAssembly entry points: the C API with flat arguments, so that web/jpo.js can call it with typed
// arrays on the module's heap. jpow_create and the two estimators may wait for the GPU (JSPI).
#include <cstdint>
#include <cstring>
#include <vector>

#include <emscripten/emscripten.h>

#include "cli/synth.h"
#include "jpo/jpo.h"

extern "C" {

EMSCRIPTEN_KEEPALIVE int jpow_backend_available(int backend) { return jpo_backend_available(backend); }
EMSCRIPTEN_KEEPALIVE const char* jpow_error() { return jpo_last_error(); }

// device: a WGPUDevice handle imported by the page (emdawnwebgpu's WebGPU.importJsDevice), or 0.
EMSCRIPTEN_KEEPALIVE jpo_ctx* jpow_create(int backend, uint32_t seed, void* instance, void* device) {
  jpo_config cfg;
  jpo_config_init(&cfg);
  cfg.backend = backend;
  cfg.seed = seed;
  cfg.webgpu_instance = instance;
  cfg.webgpu_device = device;
  return jpo_create(&cfg);
}

EMSCRIPTEN_KEEPALIVE void jpow_destroy(jpo_ctx* ctx) { jpo_destroy(ctx); }

// cameras: 9 doubles each (model, then 8 parameters). options: max_error_px, confidence,
// min_iterations, max_iterations, refine, min_inlier_ratio. results: 12 doubles per problem (pose 7, score,
// median_tri_angle_deg, n_inliers, iterations, status).
static void unpack(const double* cameras, uint32_t n, std::vector<jpo_camera>* out) {
  out->resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    std::memset(&(*out)[i], 0, sizeof(jpo_camera));
    (*out)[i].model = static_cast<int32_t>(cameras[9 * i]);
    for (int k = 0; k < 8; ++k) (*out)[i].params[k] = cameras[9 * i + 1 + k];
  }
}
static jpo_ransac_options options(const double* o) {
  jpo_ransac_options opt;
  jpo_ransac_options_init(&opt);
  opt.max_error_px = o[0];
  opt.confidence = o[1];
  opt.min_iterations = static_cast<uint32_t>(o[2]);
  opt.max_iterations = static_cast<uint32_t>(o[3]);
  opt.refine = o[4] != 0.0;
  opt.min_inlier_ratio = o[5];
  return opt;
}
static void pack(const std::vector<jpo_pose_result>& res, double* out) {
  for (size_t i = 0; i < res.size(); ++i) {
    double* r = out + 12 * i;
    for (int k = 0; k < 7; ++k) r[k] = res[i].pose[k];
    r[7] = res[i].score;
    r[8] = res[i].median_tri_angle_deg;
    r[9] = res[i].n_inliers;
    r[10] = res[i].iterations;
    r[11] = res[i].status;
  }
}

EMSCRIPTEN_KEEPALIVE int jpow_relative(jpo_ctx* ctx, uint32_t n_pairs, const uint32_t* offsets, const double* xy_a,
                                       const double* xy_b, uint32_t n_cameras, const double* cameras,
                                       const uint32_t* camera_a, const uint32_t* camera_b, const double* opts,
                                       double* results, uint8_t* inliers) {
  std::vector<jpo_camera> cams;
  unpack(cameras, n_cameras, &cams);
  jpo_pairs p;
  p.n_pairs = n_pairs;
  p.n_cameras = n_cameras;
  p.offsets = offsets;
  p.xy_a = xy_a;
  p.xy_b = xy_b;
  p.cameras = cams.data();
  p.camera_a = camera_a;
  p.camera_b = camera_b;
  const jpo_ransac_options opt = options(opts);
  std::vector<jpo_pose_result> res(n_pairs);
  const int rc = jpo_relative_pose(ctx, &p, &opt, res.data(), inliers);
  if (rc == 0) pack(res, results);
  return rc;
}

EMSCRIPTEN_KEEPALIVE int jpow_absolute(jpo_ctx* ctx, uint32_t n_images, const uint32_t* offsets, const double* xy,
                                       const double* xyz, uint32_t n_cameras, const double* cameras,
                                       const uint32_t* camera, const double* opts, double* results, uint8_t* inliers) {
  std::vector<jpo_camera> cams;
  unpack(cameras, n_cameras, &cams);
  jpo_resections r;
  r.n_images = n_images;
  r.n_cameras = n_cameras;
  r.offsets = offsets;
  r.xy = xy;
  r.xyz = xyz;
  r.cameras = cams.data();
  r.camera = camera;
  const jpo_ransac_options opt = options(opts);
  std::vector<jpo_pose_result> res(n_images);
  const int rc = jpo_absolute_pose(ctx, &r, &opt, res.data(), inliers);
  if (rc == 0) pack(res, results);
  return rc;
}

// The generator of the native tools (cli/synth.h), so that the test page runs the same problems as
// `jpo gen` + `jpo run`. Fills the buffers the two functions above take; gt: 7 doubles per problem.
// Call with null buffers first: returns the total number of matches.
EMSCRIPTEN_KEEPALIVE uint32_t jpow_synth(uint32_t kind, uint32_t count, uint32_t points, double inlier_ratio,
                                         double noise_px, int model, uint32_t seed, uint32_t* offsets, double* xy_a,
                                         double* xy_b_or_xyz, double* cameras, uint32_t* camera_a, uint32_t* camera_b,
                                         double* gt) {
  jposynth::Spec spec;
  spec.kind = kind;
  spec.count = count;
  spec.points = points;
  spec.inlier_ratio = inlier_ratio;
  spec.noise_px = noise_px;
  spec.model = model;
  spec.seed = seed;
  const jpofile::ProblemSet set = jposynth::make_set(spec);
  const jpofile::Flat flat = jpofile::flatten(set);
  const uint32_t total = flat.offsets.back();
  if (!offsets) return total;
  std::memcpy(offsets, flat.offsets.data(), flat.offsets.size() * 4);
  std::memcpy(xy_a, flat.xy_a.data(), flat.xy_a.size() * 8);
  if (kind == 0)
    std::memcpy(xy_b_or_xyz, flat.xy_b.data(), flat.xy_b.size() * 8);
  else
    std::memcpy(xy_b_or_xyz, flat.xyz.data(), flat.xyz.size() * 8);
  for (size_t i = 0; i < flat.cameras.size(); ++i) {
    cameras[9 * i] = flat.cameras[i].model;
    for (int k = 0; k < 8; ++k) cameras[9 * i + 1 + k] = flat.cameras[i].params[k];
  }
  std::memcpy(camera_a, flat.cam_a.data(), flat.cam_a.size() * 4);
  if (kind == 0) std::memcpy(camera_b, flat.cam_b.data(), flat.cam_b.size() * 4);
  for (size_t i = 0; i < set.problems.size(); ++i) std::memcpy(gt + 7 * i, set.problems[i].gt_pose, 56);
  return total;
}

}  // extern "C"
