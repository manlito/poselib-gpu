// CPU backend: the kernels on all threads. A round is cut into chunks of kChunk hypotheses; chunks of
// all problems go to the pool, so one hard problem uses every core and a large batch spreads out.
#include <algorithm>
#include <vector>

#include "backend/backend.h"
#include "backend/cpu/kernels_cpu.h"
#include "core/pool.h"

namespace jpo {

namespace {

const uint32_t kChunk = 32;
const uint32_t kMaxRound = 1024;

struct Double {
  typedef double real;
  static real rel_hyp(const real* p, uint32_t off, uint32_t n, uint32_t seed, uint32_t pr, uint32_t it, real ist,
                      real bound, uint32_t pk, uint32_t pm, int* sol) {
    return kd::jpo_rel_hypothesis(p, off, n, seed, pr, it, ist, bound, pk, pm, sol);
  }
  static real abs_hyp(const real* p, uint32_t off, uint32_t n, uint32_t seed, uint32_t pr, uint32_t it, real ist,
                      real bound, uint32_t pk, uint32_t pm, int* sol) {
    return kd::jpo_abs_hypothesis(p, off, n, seed, pr, it, ist, bound, pk, pm, sol);
  }
  static int rel_model(const real* p, uint32_t off, uint32_t n, uint32_t seed, uint32_t pr, uint32_t it, int sol,
                       real* m) {
    return kd::jpo_rel_model(p, off, n, seed, pr, it, sol, m);
  }
  static int abs_model(const real* p, uint32_t off, uint32_t n, uint32_t seed, uint32_t pr, uint32_t it, int sol,
                       real* m) {
    return kd::jpo_abs_model(p, off, n, seed, pr, it, sol, m);
  }
};

struct Float {
  typedef float real;
  static real rel_hyp(const real* p, uint32_t off, uint32_t n, uint32_t seed, uint32_t pr, uint32_t it, real ist,
                      real bound, uint32_t pk, uint32_t pm, int* sol) {
    return kf::jpo_rel_hypothesis(p, off, n, seed, pr, it, ist, bound, pk, pm, sol);
  }
  static real abs_hyp(const real* p, uint32_t off, uint32_t n, uint32_t seed, uint32_t pr, uint32_t it, real ist,
                      real bound, uint32_t pk, uint32_t pm, int* sol) {
    return kf::jpo_abs_hypothesis(p, off, n, seed, pr, it, ist, bound, pk, pm, sol);
  }
  static int rel_model(const real* p, uint32_t off, uint32_t n, uint32_t seed, uint32_t pr, uint32_t it, int sol,
                       real* m) {
    return kf::jpo_rel_model(p, off, n, seed, pr, it, sol, m);
  }
  static int abs_model(const real* p, uint32_t off, uint32_t n, uint32_t seed, uint32_t pr, uint32_t it, int sol,
                       real* m) {
    return kf::jpo_abs_model(p, off, n, seed, pr, it, sol, m);
  }
};

template <class K>
class CpuBackend final : public Backend {
  typedef typename K::real real;

 public:
  CpuBackend(ThreadPool* pool, uint32_t seed, const char* name) : pool_(pool), seed_(seed), name_(name) {}
  const char* name() const override { return name_; }

  void set_points(Kind kind, const double* pts, size_t n_points) override {
    kind_ = kind;
    const size_t n = n_points * kind_stride(kind);
    pts_.resize(n);
    for (size_t i = 0; i < n; ++i) pts_[i] = static_cast<real>(pts[i]);
  }

  // Rounds double in size (32, 32, 64, 128, ... kMaxRound): an easy problem stops after little work and
  // a hard one is not slowed by a synchronization every few microseconds. With fewer problems than
  // threads, a round is widened so that every thread has a chunk.
  uint32_t round_size(size_t n_active, uint32_t iters) const override {
    const size_t threads = static_cast<size_t>(pool_->size());
    const size_t spread = n_active >= threads ? 1 : (threads + n_active - 1) / std::max<size_t>(n_active, 1);
    const uint32_t base = std::min(kMaxRound, std::max(kChunk, iters / kChunk * kChunk));
    return base * static_cast<uint32_t>(spread);
  }

  void run_round(const RoundItem* items, size_t n_items, RoundResult* out) override {
    std::vector<Chunk>& chunks = chunks_;
    chunks.clear();
    for (size_t i = 0; i < n_items; ++i) {
      for (uint32_t s = 0; s < items[i].count; s += kChunk)
        chunks.push_back({static_cast<uint32_t>(i), items[i].iter0 + s, std::min(kChunk, items[i].count - s),
                          static_cast<real>(kNoScore), 0u, -1, {}});
    }
    const int stride = kind_stride(kind_);
    const real* pts = pts_.data();
    const Kind kind = kind_;
    const uint32_t seed = seed_;
    pool_->parallel_for(chunks.size(), [&](size_t c) {
      Chunk& ch = chunks[c];
      const RoundItem& it = items[ch.item];
      const real ist = static_cast<real>(it.inv_sq_thr);
      const real bound = it.bound >= kNoScore ? static_cast<real>(kNoScore) : static_cast<real>(it.bound);
      const uint32_t off = it.first * stride;
      for (uint32_t h = 0; h < ch.count; ++h) {
        int sol = -1;
        const real b = ch.score < bound ? ch.score : bound;
        const real sc = kind == Kind::Relative ? K::rel_hyp(pts, off, it.n, seed, it.problem, ch.iter0 + h, ist, b, it.pre_k, it.pre_min, &sol)
                                               : K::abs_hyp(pts, off, it.n, seed, it.problem, ch.iter0 + h, ist, b, it.pre_k, it.pre_min, &sol);
        if (sol >= 0 && sc < ch.score) {
          ch.score = sc;
          ch.iter = ch.iter0 + h;
          ch.sol = sol;
        }
      }
      if (ch.sol >= 0) {  // the model of the chunk's best hypothesis, solved once more
        const int ok = kind == Kind::Relative ? K::rel_model(pts, off, it.n, seed, it.problem, ch.iter, ch.sol, ch.model)
                                              : K::abs_model(pts, off, it.n, seed, it.problem, ch.iter, ch.sol, ch.model);
        if (!ok) ch.sol = -1;
      }
    });
    for (size_t i = 0; i < n_items; ++i) {
      out[i].score = kNoScore;
      out[i].iter = 0;
      out[i].sol = -1;
    }
    for (const Chunk& ch : chunks) {  // in iteration order: a tie keeps the earlier hypothesis
      RoundResult& r = out[ch.item];
      if (ch.sol >= 0 && static_cast<double>(ch.score) < r.score) {
        r.score = ch.score;
        r.iter = ch.iter;
        r.sol = ch.sol;
        for (int k = 0; k < 12; ++k) r.model[k] = ch.model[k];
      }
    }
  }

 private:
  struct Chunk {
    uint32_t item, iter0, count;
    real score;
    uint32_t iter;
    int sol;
    real model[12];
  };
  ThreadPool* pool_;
  uint32_t seed_;
  const char* name_;
  Kind kind_ = Kind::Relative;
  std::vector<real> pts_;
  std::vector<Chunk> chunks_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(ThreadPool* pool, uint32_t seed, bool use_float) {
  if (use_float) return std::unique_ptr<Backend>(new CpuBackend<Float>(pool, seed, "cpu-float"));
  return std::unique_ptr<Backend>(new CpuBackend<Double>(pool, seed, "cpu"));
}

}  // namespace jpo
