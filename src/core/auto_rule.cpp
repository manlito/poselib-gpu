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
      if (cpu_threads > 2) {
        // Dawn on the M5 Pro's GPU costs what the Metal backend costs, and measured natively it wants
        // Metal's rule, the total per CPU thread (50,000 / 18): 128 / 512 moved the same resection
        // batches at 30 % inliers (3000 of 200 points: 150 -> 48 ms), and Metal's one-round numbers
        // were the best of nine on candidate rounds.
        route->cpu_iterations = 128;
        route->gpu_min_remaining = 512;
        route->gpu_min_total = 2800ull * static_cast<uint64_t>(cpu_threads);
        route->one_round_cpu_iterations = 64;
        route->one_round_gpu_min_remaining = 256;
        route->one_round_gpu_min_total = 5000;
        break;
      }
      // In a browser the CPU backend has one WebAssembly thread, and WebGPU alone wins from about four
      // problems up, easy ones included (100 easy pairs: 87 ms on the CPU, 24 ms). So the CPU phase is
      // the CPU's first round (32), and whatever still needs 32 more moves once a thousand hypotheses
      // are left (Chrome on the M5 Pro; the bench grid 2.0 -> 1.1 s at 100 problems). A budget that fits
      // in one GPU round moves after a single CPU hypothesis once there are two problems or more: within
      // 10 % of WebGPU alone on candidate rounds, where the numbers above are 1.5 times slower.
      route->cpu_iterations = 32;
      route->gpu_min_remaining = 32;
      route->gpu_min_total = 1000;
      route->one_round_cpu_iterations = 1;
      route->one_round_gpu_min_remaining = 1;
      route->one_round_gpu_min_total = 1000;
      break;
    default:
      break;
  }
}

}  // namespace jpo
