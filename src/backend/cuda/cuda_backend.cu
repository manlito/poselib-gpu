// CUDA backend: the round of 2048 hypotheses per problem as two kernels, from the shared kernel source
// (src/kernels/kernels.inc) behind this prelude. The same structure as the Metal backend; a discrete
// GPU also pays for the upload of the points (once per call) and of the items, and for the read-back
// of one small result per problem (every round).
#include <cuda_runtime.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backend/backend.h"

namespace jpo {

namespace {

typedef float real;
typedef uint32_t jpo_u32;
#define JPO_FN __device__ static inline
#define JPO_P
#define JPO_PTS const float*
#define JPO_C(x) x##f
#define JPO_SQRT(x) sqrtf(x)
#define JPO_ABS(x) fabsf(x)
#define JPO_CBRT(x) cbrtf(x)
#define JPO_ACOS(x) acosf(x)
#define JPO_COS(x) cosf(x)
#define JPO_TOL 1.0e-6f
#define JPO_STURM_DEPTH 36
#include "kernels/kernels.inc"

struct Item {
  uint32_t first, n, problem, iter0, count;
  float inv_sq_thr, bound;
  uint32_t pre_k, pre_min, pad;
};
struct Result {
  float score;
  uint32_t iter;
  int32_t sol;
  uint32_t pad;
  float model[12];
};

// One thread per hypothesis (x) and item (y).
__global__ void round_kernel(const float* pts, const Item* items, float* score, int* sol, uint32_t seed, uint32_t round,
                             int relative) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y;
  const Item it = items[y];
  if (x >= it.count) return;
  int s = -1;
  const float sc = relative ? jpo_rel_hypothesis(pts, it.first * JPO_REL_STRIDE, it.n, seed, it.problem, it.iter0 + x,
                                                 it.inv_sq_thr, it.bound, it.pre_k, it.pre_min, &s)
                            : jpo_abs_hypothesis(pts, it.first * JPO_ABS_STRIDE, it.n, seed, it.problem, it.iter0 + x,
                                                 it.inv_sq_thr, it.bound, it.pre_k, it.pre_min, &s);
  const uint32_t o = y * round + x;
  score[o] = sc;
  sol[o] = s;
}

// One thread per item: the best hypothesis (a tie keeps the earlier one), solved once more for its model.
__global__ void select_kernel(const float* pts, const Item* items, const float* score, const int* sol, Result* out,
                              uint32_t seed, uint32_t round, uint32_t n_items, int relative) {
  const uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= n_items) return;
  const Item it = items[y];
  float best = JPO_BIG;
  uint32_t bh = 0;
  int bs = -1;
  const uint32_t o = y * round;
  for (uint32_t h = 0; h < it.count; ++h) {
    const int s = sol[o + h];
    const float sc = score[o + h];
    if (s >= 0 && sc < best) {
      best = sc;
      bh = h;
      bs = s;
    }
  }
  float m[12];
  for (int i = 0; i < 12; ++i) m[i] = 0.0f;
  if (bs >= 0) {
    const int ok = relative ? jpo_rel_model(pts, it.first * JPO_REL_STRIDE, it.n, seed, it.problem, it.iter0 + bh, bs, m)
                            : jpo_abs_model(pts, it.first * JPO_ABS_STRIDE, it.n, seed, it.problem, it.iter0 + bh, bs, m);
    if (!ok) bs = -1;
  }
  out[y].score = bs >= 0 ? best : JPO_BIG;
  out[y].iter = it.iter0 + bh;
  out[y].sol = bs;
  out[y].pad = 0;
  for (int i = 0; i < 12; ++i) out[y].model[i] = m[i];
}

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(e));
}

// A device buffer that only grows.
template <class T>
struct DeviceBuffer {
  T* ptr = nullptr;
  size_t cap = 0;
  ~DeviceBuffer() {
    if (ptr) cudaFree(ptr);
  }
  void ensure(size_t n) {
    if (n <= cap) return;
    if (ptr) cudaFree(ptr);
    ptr = nullptr;
    size_t c = cap ? cap : 1024;
    while (c < n) c *= 2;
    check(cudaMalloc(&ptr, c * sizeof(T)), "allocation");
    cap = c;
  }
};

const uint32_t kMaxItemsPerLaunch = 1024;
const uint32_t kBlock = 64;

// Whether NVIDIA's driver is installed. On Windows the static CUDA 13 runtime crashes in its first call
// when nvcuda.dll is missing (a hosted runner with CUDA 13.4: an access violation in cudaGetDeviceCount,
// where CUDA 12.9 returns an error), so the driver is looked for first. Elsewhere the runtime reports it.
bool driver_present() {
#ifdef _WIN32
  HMODULE h = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!h) return false;
  FreeLibrary(h);
#endif
  return true;
}

class CudaBackend final : public Backend {
 public:
  CudaBackend(uint32_t seed, uint32_t round) : seed_(seed), round_(round) {
    if (!driver_present()) throw std::runtime_error("CUDA: no NVIDIA driver (nvcuda.dll not found)");
    int count = 0;
    check(cudaGetDeviceCount(&count), "device count");
    if (count < 1) throw std::runtime_error("CUDA: no device");
    check(cudaSetDevice(0), "set device");
  }
  const char* name() const override { return "cuda"; }

  void set_points(Kind kind, const double* pts, size_t n_points) override {
    kind_ = kind;
    const size_t n = n_points * kind_stride(kind);
    host_.resize(std::max<size_t>(n, 1));
    for (size_t i = 0; i < n; ++i) host_[i] = static_cast<float>(pts[i]);
    pts_.ensure(host_.size());
    check(cudaMemcpy(pts_.ptr, host_.data(), host_.size() * sizeof(float), cudaMemcpyHostToDevice), "upload");
  }

  uint32_t round_size(size_t, uint32_t) const override { return round_; }

  void run_round(const RoundItem* items, size_t n_items, RoundResult* out) override {
    if (n_items == 0) return;
    hitems_.resize(n_items);
    uint32_t round = 1;
    for (size_t i = 0; i < n_items; ++i) {
      const RoundItem& it = items[i];
      Item& d = hitems_[i];
      d.first = it.first;
      d.n = it.n;
      d.problem = it.problem;
      d.iter0 = it.iter0;
      d.count = it.count;
      d.inv_sq_thr = static_cast<float>(it.inv_sq_thr);
      d.bound = it.bound >= kNoScore ? 3.0e38f : static_cast<float>(it.bound);
      d.pre_k = it.pre_k;
      d.pre_min = it.pre_min;
      d.pad = 0;
      round = std::max(round, it.count);
    }
    items_.ensure(n_items);
    results_.ensure(n_items);
    check(cudaMemcpy(items_.ptr, hitems_.data(), n_items * sizeof(Item), cudaMemcpyHostToDevice), "upload");
    const size_t scratch = std::min<size_t>(n_items, kMaxItemsPerLaunch);
    score_.ensure(scratch * round);
    sol_.ensure(scratch * round);
    const int relative = kind_ == Kind::Relative ? 1 : 0;
    for (size_t begin = 0; begin < n_items; begin += kMaxItemsPerLaunch) {
      const uint32_t count = static_cast<uint32_t>(std::min<size_t>(kMaxItemsPerLaunch, n_items - begin));
      const dim3 grid((round + kBlock - 1) / kBlock, count);
      round_kernel<<<grid, kBlock>>>(pts_.ptr, items_.ptr + begin, score_.ptr, sol_.ptr, seed_, round, relative);
      select_kernel<<<(count + kBlock - 1) / kBlock, kBlock>>>(pts_.ptr, items_.ptr + begin, score_.ptr, sol_.ptr,
                                                               results_.ptr + begin, seed_, round, count, relative);
    }
    check(cudaGetLastError(), "launch");
    hresults_.resize(n_items);
    check(cudaMemcpy(hresults_.data(), results_.ptr, n_items * sizeof(Result), cudaMemcpyDeviceToHost), "read-back");
    for (size_t i = 0; i < n_items; ++i) {
      const Result& r = hresults_[i];
      out[i].sol = r.sol;
      out[i].iter = r.iter;
      out[i].score = r.sol >= 0 ? static_cast<double>(r.score) : kNoScore;
      for (int m = 0; m < 12; ++m) out[i].model[m] = r.model[m];
    }
  }

 private:
  uint32_t seed_, round_;
  Kind kind_ = Kind::Relative;
  std::vector<float> host_;
  std::vector<Item> hitems_;
  std::vector<Result> hresults_;
  DeviceBuffer<float> pts_, score_;
  DeviceBuffer<int> sol_;
  DeviceBuffer<Item> items_;
  DeviceBuffer<Result> results_;
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend(uint32_t seed, uint32_t round) {
  return std::unique_ptr<Backend>(new CudaBackend(seed, round));
}

}  // namespace jpo
