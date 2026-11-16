#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "backend/backend.h"
#include "core/api_internal.h"
#include "core/estimator.h"
#include "core/geometry.h"
#include "core/pool.h"
#include "jpo/jpo.h"

namespace {
thread_local std::string g_error;

bool gpu_built(int32_t backend) {
  switch (backend) {
#ifdef JPO_HAVE_METAL
    case JPO_BACKEND_METAL: return true;
#endif
#ifdef JPO_HAVE_CUDA
    case JPO_BACKEND_CUDA: return true;
#endif
#ifdef JPO_HAVE_WEBGPU
    case JPO_BACKEND_WEBGPU: return true;
#endif
    default: return false;
  }
}

std::unique_ptr<jpo::Backend> make_gpu(int32_t backend, const jpo_config& cfg) {
  const uint32_t round = cfg.gpu_round ? cfg.gpu_round : 2048;
  (void)round;
  switch (backend) {
#ifdef JPO_HAVE_METAL
    case JPO_BACKEND_METAL: return jpo::make_metal_backend(cfg.seed, round);
#endif
#ifdef JPO_HAVE_CUDA
    case JPO_BACKEND_CUDA: return jpo::make_cuda_backend(cfg.seed, round);
#endif
#ifdef JPO_HAVE_WEBGPU
    case JPO_BACKEND_WEBGPU: return jpo::make_webgpu_backend(cfg.seed, round, cfg.webgpu_instance, cfg.webgpu_device);
#endif
    default: throw std::runtime_error("backend not in this build");
  }
}

void check_options(const jpo_ransac_options& o) {
  if (!(o.max_error_px > 0.0)) throw std::invalid_argument("max_error_px must be positive");
  if (!(o.confidence > 0.0 && o.confidence < 1.0)) throw std::invalid_argument("confidence must be in (0, 1)");
  if (!(o.min_inlier_ratio >= 0.0 && o.min_inlier_ratio < 1.0))
    throw std::invalid_argument("min_inlier_ratio must be in [0, 1)");
}

void check_cameras(const jpo_camera* cams, uint32_t n) {
  if (n && !cams) throw std::invalid_argument("cameras is null");
  for (uint32_t i = 0; i < n; ++i) {
    if (jpo::camera_param_count(cams[i].model) == 0) throw std::invalid_argument("unknown camera model");
    if (!(jpo::camera_focal(cams[i]) > 0.0)) throw std::invalid_argument("camera focal length must be positive");
  }
}
}  // namespace

struct jpo_ctx {
  jpo_config config;
  std::unique_ptr<jpo::ThreadPool> pool;
  std::unique_ptr<jpo::Backend> cpu;
  std::unique_ptr<jpo::Backend> gpu;
  jpo::Route route;
  jpo::BatchStats stats;
  std::string name;
};

namespace jpo {
const BatchStats& ctx_stats(const jpo_ctx* ctx) { return ctx->stats; }
Route& ctx_route(jpo_ctx* ctx) { return ctx->route; }
Backend* ctx_cpu(jpo_ctx* ctx) { return ctx->cpu.get(); }
Backend* ctx_gpu(jpo_ctx* ctx) { return ctx->gpu.get(); }
}  // namespace jpo

extern "C" {

int jpo_camera_param_count(int32_t model) { return jpo::camera_param_count(model); }

void jpo_config_init(jpo_config* config) {
  if (!config) return;
  std::memset(config, 0, sizeof(*config));
  config->backend = JPO_BACKEND_AUTO;
}

void jpo_ransac_options_init(jpo_ransac_options* o) {
  if (!o) return;
  std::memset(o, 0, sizeof(*o));
  o->max_error_px = 1.0;
  o->confidence = 0.9999;
  o->min_iterations = 100;
  o->max_iterations = 10000;
  o->refine = 1;
}

int jpo_backend_available(int32_t backend) {
  return backend == JPO_BACKEND_CPU || backend == JPO_BACKEND_AUTO || gpu_built(backend) ? 1 : 0;
}

const char* jpo_last_error(void) { return g_error.c_str(); }

jpo_ctx* jpo_create(const jpo_config* config) {
  try {
    jpo_config cfg;
    if (config)
      cfg = *config;
    else
      jpo_config_init(&cfg);
    std::unique_ptr<jpo_ctx> ctx(new jpo_ctx());
    ctx->config = cfg;
    ctx->pool.reset(new jpo::ThreadPool(cfg.threads));
    ctx->cpu = jpo::make_cpu_backend(ctx->pool.get(), cfg.seed, cfg.cpu_float != 0);
    ctx->route.cpu = ctx->cpu.get();
    ctx->name = ctx->cpu->name();
    if (cfg.backend == JPO_BACKEND_AUTO) {
      // The GPU of this build, if it can be created; the rule comes from the measurements of each
      // backend (PLAN.md, "The auto rule").
      const int32_t order[3] = {JPO_BACKEND_METAL, JPO_BACKEND_CUDA, JPO_BACKEND_WEBGPU};
      for (int32_t b : order) {
        if (!gpu_built(b)) continue;
        try {
          ctx->gpu = make_gpu(b, cfg);
        } catch (const std::exception&) {
          continue;  // no device: the CPU alone
        }
        ctx->route.gpu = ctx->gpu.get();
        jpo::auto_rule(b, ctx->pool->size(), &ctx->route);
        ctx->name = std::string("auto(") + ctx->gpu->name() + ")";
        break;
      }
    } else if (cfg.backend != JPO_BACKEND_CPU) {
      if (!gpu_built(cfg.backend)) throw std::runtime_error("backend not in this build");
      ctx->gpu = make_gpu(cfg.backend, cfg);
      ctx->route.gpu = ctx->gpu.get();
      ctx->route.cpu = nullptr;
      ctx->name = ctx->gpu->name();
    }
    return ctx.release();
  } catch (const std::exception& e) {
    g_error = e.what();
    return nullptr;
  }
}

void jpo_destroy(jpo_ctx* ctx) { delete ctx; }

const char* jpo_backend_name(const jpo_ctx* ctx) { return ctx ? ctx->name.c_str() : ""; }

int jpo_relative_pose(jpo_ctx* ctx, const jpo_pairs* pairs, const jpo_ransac_options* options,
                      jpo_pose_result* results, uint8_t* inliers) {
  try {
    if (!ctx || !pairs || !results) throw std::invalid_argument("null argument");
    jpo_ransac_options opt;
    if (options)
      opt = *options;
    else
      jpo_ransac_options_init(&opt);
    check_options(opt);
    if (pairs->n_pairs == 0) return 0;
    if (!pairs->offsets || !pairs->camera_a || !pairs->camera_b) throw std::invalid_argument("null argument");
    check_cameras(pairs->cameras, pairs->n_cameras);
    for (uint32_t i = 0; i < pairs->n_pairs; ++i) {
      if (pairs->offsets[i + 1] < pairs->offsets[i]) throw std::invalid_argument("offsets must not decrease");
      if (pairs->camera_a[i] >= pairs->n_cameras || pairs->camera_b[i] >= pairs->n_cameras)
        throw std::invalid_argument("camera index out of range");
    }
    if (pairs->offsets[pairs->n_pairs] && (!pairs->xy_a || !pairs->xy_b)) throw std::invalid_argument("null points");
    ctx->stats = jpo::BatchStats();
    jpo::estimate_relative(ctx->route, ctx->pool.get(), *pairs, opt, results, inliers, &ctx->stats);
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

int jpo_absolute_pose(jpo_ctx* ctx, const jpo_resections* res, const jpo_ransac_options* options,
                      jpo_pose_result* results, uint8_t* inliers) {
  try {
    if (!ctx || !res || !results) throw std::invalid_argument("null argument");
    jpo_ransac_options opt;
    if (options) {
      opt = *options;
    } else {
      jpo_ransac_options_init(&opt);
      opt.max_error_px = 4.0;
    }
    check_options(opt);
    if (res->n_images == 0) return 0;
    if (!res->offsets || !res->camera) throw std::invalid_argument("null argument");
    check_cameras(res->cameras, res->n_cameras);
    for (uint32_t i = 0; i < res->n_images; ++i) {
      if (res->offsets[i + 1] < res->offsets[i]) throw std::invalid_argument("offsets must not decrease");
      if (res->camera[i] >= res->n_cameras) throw std::invalid_argument("camera index out of range");
    }
    if (res->offsets[res->n_images] && (!res->xy || !res->xyz)) throw std::invalid_argument("null points");
    ctx->stats = jpo::BatchStats();
    jpo::estimate_absolute(ctx->route, ctx->pool.get(), *res, opt, results, inliers, &ctx->stats);
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

}  // extern "C"
