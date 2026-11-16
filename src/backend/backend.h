// What a backend does: one RANSAC round. For each listed problem it draws the samples of iterations
// [iter0, iter0 + count), solves them, scores every solution against all points of the problem and
// returns the best one. Everything else (stopping rule, refinement) is host work (core/estimator.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace jpo {

enum class Kind { Relative, Absolute };

inline int kind_stride(Kind k) { return k == Kind::Relative ? 4 : 5; }
inline int kind_model_size(Kind) { return 12; }  // a pose: R row major, t
inline int kind_sample_size(Kind k) { return k == Kind::Relative ? 5 : 3; }

struct RoundItem {
  uint32_t problem;   // index in the batch: part of the sample counter
  uint32_t first;     // index of the problem's first point in the point buffer
  uint32_t n;         // points
  uint32_t iter0;     // first iteration of this round
  uint32_t count;     // hypotheses
  double inv_sq_thr;  // 1 / squared threshold, on the normalized image plane
  double bound;       // scores above this cannot win: the scoring loop may stop there
  uint32_t pre_k;     // early rejection: a hypothesis with fewer than pre_min inliers among its first
  uint32_t pre_min;   // pre_k points is dropped (pre_min = 0: no test; pre_k a multiple of 16)
};

struct RoundResult {
  double score;  // in units of the squared threshold; >= 3e38 if no hypothesis had a solution
  uint32_t iter;
  int32_t sol;   // -1: none
  double model[12];
};

const double kNoScore = 3.0e38;

class Backend {
 public:
  virtual ~Backend() = default;
  virtual const char* name() const = 0;
  // The points of the whole batch: stride 4 (relative) or 5 (absolute) doubles per point.
  virtual void set_points(Kind kind, const double* pts, size_t n_points) = 0;
  // Hypotheses the next round of a problem should hold, when n_active problems are still running and
  // this one has drawn `iters` so far.
  virtual uint32_t round_size(size_t n_active, uint32_t iters) const = 0;
  virtual void run_round(const RoundItem* items, size_t n_items, RoundResult* out) = 0;
};

class ThreadPool;

std::unique_ptr<Backend> make_cpu_backend(ThreadPool* pool, uint32_t seed, bool use_float);
#ifdef JPO_HAVE_METAL
std::unique_ptr<Backend> make_metal_backend(uint32_t seed, uint32_t round);
#endif
#ifdef JPO_HAVE_CUDA
std::unique_ptr<Backend> make_cuda_backend(uint32_t seed, uint32_t round);
#endif
#ifdef JPO_HAVE_WEBGPU
std::unique_ptr<Backend> make_webgpu_backend(uint32_t seed, uint32_t round, void* instance, void* device);
#endif

}  // namespace jpo
