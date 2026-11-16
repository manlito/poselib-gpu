// The rule `backend = auto` follows (core/estimator.h, Route): how long a problem stays on the CPU and
// how much work must be left before the GPU is worth a dispatch. Each backend's numbers come from its
// own measurements (PLAN.md, "The auto rule"); a backend that has not been measured keeps everything
// on the CPU.
#include <cstdint>

#include "core/api_internal.h"

namespace jpo {

void auto_rule(int32_t gpu_backend, int cpu_threads, Route* route) {
  (void)cpu_threads;
  route->cpu_iterations = 0xffffffffu;  // never
  route->gpu_min_remaining = 0;
  route->gpu_min_total = 0;
  switch (gpu_backend) {
    case JPO_BACKEND_METAL:
      // M5 Pro, 18 threads: a GPU round costs about what the CPU needs for 50,000 hypotheses, and
      // problems that stop within about a thousand iterations are faster where they are.
      route->cpu_iterations = 256;
      route->gpu_min_remaining = 1024;
      route->gpu_min_total = 50000;
      break;
    case JPO_BACKEND_CUDA:
      // RTX 2070 Max-Q against a 6-core i7-9750H: the GPU already wins on one hard problem, so the
      // candidates move as soon as they need a few thousand hypotheses between them.
      route->cpu_iterations = 256;
      route->gpu_min_remaining = 1024;
      route->gpu_min_total = 5000;
      break;
    default:
      break;
  }
}

}  // namespace jpo
