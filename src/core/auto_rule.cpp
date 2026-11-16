// The rule `backend = auto` follows (core/estimator.h, Route): how long a problem stays on the CPU and
// how much work must be left before the GPU is worth a dispatch. Each backend's numbers come from its
// own measurements (PLAN.md, "The auto rule"); a backend that has not been measured keeps everything
// on the CPU.
#include <cstdint>

#include "core/api_internal.h"

namespace jpo {

void auto_rule(int32_t gpu_backend, int cpu_threads, Route* route) {
  route->cpu_iterations = 0xffffffffu;  // never
  route->gpu_min_remaining = 0;
  route->gpu_min_total = 0;
  route->one_round_cpu_iterations = 0;  // a one-round budget follows the numbers above
  route->one_round_gpu_min_remaining = 0;
  route->one_round_gpu_min_total = 0;
  switch (gpu_backend) {
    case JPO_BACKEND_METAL:
      // M5 Pro, 18 threads: a GPU round costs about what the CPU needs for 50,000 hypotheses. A CPU phase
      // of 128 with 512 remaining (since the phase stopped cutting rounds it need not, core/estimator.cpp)
      // moves resection batches at 30 % inliers, which stop near 1000 iterations, and hard batches earlier.
      // A lower total helped small resection batches but slowed small pair batches; it stays.
      route->cpu_iterations = 128;
      route->gpu_min_remaining = 512;
      route->gpu_min_total = 50000;
      // A budget that fits in one GPU round (min_inlier_ratio = 0.25 on resections: 585 iterations):
      // one dispatch of what is left after 64 CPU iterations beats the CPU once it is 5,000 hypotheses
      // (K = 16 to 32 candidates, half or more failing); a short CPU phase still lets the good ones finish.
      route->one_round_cpu_iterations = 64;
      route->one_round_gpu_min_remaining = 256;
      route->one_round_gpu_min_total = 5000;
      break;
    case JPO_BACKEND_CUDA:
      // RTX 2070 Max-Q against a 6-core i7-9750H: the GPU already wins on one hard problem, so the
      // candidates move as soon as they need a few thousand hypotheses between them. 512 remaining, not
      // 1024: batches of resections at 30 % inliers stop near 1000 iterations and run 1.5 to 3 times
      // faster once they move; on the bench grid nothing else routes differently. A CPU phase of 128,
      // not 256: hard batches move earlier (5 to 33 % faster); with fewer problems than threads the phase
      // still lasts the CPU's first round, and easy problems are not cut (core/estimator.cpp).
      route->cpu_iterations = 128;
      route->gpu_min_remaining = 512;
      route->gpu_min_total = 5000;
      // A budget that fits in one GPU round: one dispatch ties the CPU on K = 16 to 32 candidates with
      // none failing and is 1.5 to 4.4 times faster with failing ones, so the CPU phase is short (16
      // iterations; most good candidates finish in it) and a thousand hypotheses are worth the dispatch.
      route->one_round_cpu_iterations = 16;
      route->one_round_gpu_min_remaining = 128;
      route->one_round_gpu_min_total = 1000;
      break;
    case JPO_BACKEND_WEBGPU:
      // Dawn on the M5 Pro's GPU costs what the Metal backend costs, so the same rule per CPU thread
      // (50,000 / 18). In a browser the CPU backend has one thread and the GPU is 20 to 40 times
      // faster on anything that does not stop at once: a few thousand hypotheses are enough.
      route->cpu_iterations = 256;
      route->gpu_min_remaining = cpu_threads <= 2 ? 256 : 1024;
      route->gpu_min_total = 2800ull * static_cast<uint64_t>(cpu_threads > 0 ? cpu_threads : 1);
      break;
    default:
      break;
  }
}

}  // namespace jpo
