// Metal backend: the round of 2048 hypotheses per problem as two compute dispatches (entry.metal).
// Buffers use shared storage, so points, items and results are never copied to or from the GPU.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "backend/backend.h"

extern const char kJpoMetalSource[];

namespace jpo {

namespace {

struct Item {
  uint32_t first, n, problem, iter0, count;
  float inv_sq_thr, bound;
  uint32_t pre_k, pre_min, pad;
};
struct Params {
  uint32_t seed, round;
};
struct Result {
  float score;
  uint32_t iter;
  int32_t sol;
  uint32_t pad;
  float model[12];
};

// Upper bound of point evaluations (hypotheses x points) in one command buffer: keeps a dispatch well
// under the time after which the system cancels GPU work, and the scratch buffers small.
const uint64_t kWorkPerCommandBuffer = 6ull << 30;
const uint32_t kMaxItemsPerDispatch = 1024;

class MetalBackend final : public Backend {
 public:
  MetalBackend(uint32_t seed, uint32_t round) : seed_(seed), round_(round) {
    @autoreleasepool {
      device_ = MTLCreateSystemDefaultDevice();
      if (!device_) throw std::runtime_error("Metal: no device");
      queue_ = [device_ newCommandQueue];
      MTLCompileOptions* opt = [MTLCompileOptions new];
      // The kernels test for NaN and rely on comparisons being false then.
      if (@available(macOS 15.0, iOS 18.0, *)) {
        opt.mathMode = MTLMathModeSafe;
      } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opt.fastMathEnabled = NO;
#pragma clang diagnostic pop
      }
      NSError* err = nil;
      id<MTLLibrary> lib = [device_ newLibraryWithSource:[NSString stringWithUTF8String:kJpoMetalSource]
                                                 options:opt
                                                   error:&err];
      if (!lib) throw std::runtime_error(std::string("Metal: ") + (err ? err.localizedDescription.UTF8String : "compile failed"));
      const char* names[4] = {"jpo_round_rel", "jpo_round_abs", "jpo_select_rel", "jpo_select_abs"};
      for (int i = 0; i < 4; ++i) {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:names[i]]];
        if (!fn) throw std::runtime_error(std::string("Metal: missing function ") + names[i]);
        pipe_[i] = [device_ newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe_[i]) throw std::runtime_error(std::string("Metal: ") + (err ? err.localizedDescription.UTF8String : "pipeline failed"));
      }
    }
  }

  const char* name() const override { return "metal"; }

  void set_points(Kind kind, const double* pts, size_t n_points) override {
    kind_ = kind;
    const size_t n = n_points * kind_stride(kind);
    ensure(pts_, std::max<size_t>(n, 1) * sizeof(float));
    float* dst = static_cast<float*>(pts_.contents);
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(pts[i]);
  }

  uint32_t round_size(size_t, uint32_t) const override { return round_; }

  void run_round(const RoundItem* items, size_t n_items, RoundResult* out) override {
    if (n_items == 0) return;
    @autoreleasepool {
      ensure(items_, n_items * sizeof(Item));
      ensure(results_, n_items * sizeof(Result));
      Item* dst = static_cast<Item*>(items_.contents);
      uint32_t round = 1;
      for (size_t i = 0; i < n_items; ++i) {
        const RoundItem& it = items[i];
        dst[i].first = it.first;
        dst[i].n = it.n;
        dst[i].problem = it.problem;
        dst[i].iter0 = it.iter0;
        dst[i].count = it.count;
        dst[i].inv_sq_thr = static_cast<float>(it.inv_sq_thr);
        dst[i].bound = it.bound >= kNoScore ? 3.0e38f : static_cast<float>(it.bound);
        dst[i].pre_k = it.pre_k;
        dst[i].pre_min = it.pre_min;
        dst[i].pad = 0;
        round = std::max(round, it.count);
      }
      Params prm = {seed_, round};
      const int k = kind_ == Kind::Relative ? 0 : 1;
      id<MTLCommandBuffer> last = nil;
      std::vector<id<MTLCommandBuffer>> all;
      size_t begin = 0;
      size_t scratch = 0;
      // Chunks of items: each its own command buffer, all committed before the first wait.
      std::vector<std::pair<size_t, size_t>> chunks;
      while (begin < n_items) {
        uint64_t work = 0;
        size_t end = begin;
        while (end < n_items && end - begin < kMaxItemsPerDispatch) {
          const uint64_t w = static_cast<uint64_t>(items[end].count) * std::max<uint32_t>(items[end].n, 64);
          if (end > begin && work + w > kWorkPerCommandBuffer) break;
          work += w;
          ++end;
        }
        chunks.emplace_back(begin, end);
        scratch = std::max(scratch, end - begin);
        begin = end;
      }
      ensure(score_, scratch * round * sizeof(float));
      ensure(sol_, scratch * round * sizeof(int32_t));
      for (const auto& ch : chunks) {
        const size_t count = ch.second - ch.first;
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setBuffer:pts_ offset:0 atIndex:0];
        [enc setBuffer:items_ offset:ch.first * sizeof(Item) atIndex:1];
        [enc setBuffer:score_ offset:0 atIndex:2];
        [enc setBuffer:sol_ offset:0 atIndex:3];
        [enc setBytes:&prm length:sizeof(prm) atIndex:4];
        [enc setBuffer:results_ offset:ch.first * sizeof(Result) atIndex:5];
        id<MTLComputePipelineState> p = pipe_[k];
        [enc setComputePipelineState:p];
        const NSUInteger w = p.threadExecutionWidth;
        const NSUInteger h = std::max<NSUInteger>(1, p.maxTotalThreadsPerThreadgroup / w);
        [enc dispatchThreads:MTLSizeMake(round, count, 1) threadsPerThreadgroup:MTLSizeMake(w, std::min<NSUInteger>(h, count), 1)];
        id<MTLComputePipelineState> s = pipe_[2 + k];
        [enc setComputePipelineState:s];
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(s.threadExecutionWidth, count), 1, 1)];
        [enc endEncoding];
        [cb commit];
        all.push_back(cb);
        last = cb;
      }
      [last waitUntilCompleted];
      for (id<MTLCommandBuffer> cb : all) {
        if (cb.status != MTLCommandBufferStatusCompleted || cb.error)
          throw std::runtime_error(std::string("Metal: ") + (cb.error ? cb.error.localizedDescription.UTF8String : "command buffer failed"));
      }
      const Result* res = static_cast<const Result*>(results_.contents);
      for (size_t i = 0; i < n_items; ++i) {
        out[i].sol = res[i].sol;
        out[i].iter = res[i].iter;
        out[i].score = res[i].sol >= 0 ? static_cast<double>(res[i].score) : kNoScore;
        for (int m = 0; m < 12; ++m) out[i].model[m] = res[i].model[m];
      }
    }
  }

 private:
  void ensure(id<MTLBuffer> __strong& buf, size_t bytes) {
    if (buf && buf.length >= bytes) return;
    size_t cap = buf ? buf.length : 4096;
    while (cap < bytes) cap *= 2;
    buf = [device_ newBufferWithLength:cap options:MTLResourceStorageModeShared];
    if (!buf) throw std::runtime_error("Metal: out of memory");
  }

  uint32_t seed_, round_;
  Kind kind_ = Kind::Relative;
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLComputePipelineState> pipe_[4];
  id<MTLBuffer> pts_ = nil, items_ = nil, score_ = nil, sol_ = nil, results_ = nil;
};

}  // namespace

std::unique_ptr<Backend> make_metal_backend(uint32_t seed, uint32_t round) {
  return std::unique_ptr<Backend>(new MetalBackend(seed, round));
}

}  // namespace jpo
