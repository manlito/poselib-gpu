// The rule `backend = auto` follows: how many hypotheses a problem draws on the CPU before it moves to
// the GPU. Taken from the measurements of each backend (PLAN.md, "The auto rule"); until a backend has
// been measured its rule keeps everything on the CPU.
#include <cstdint>

#include "core/api_internal.h"

namespace jpo {

void auto_rule(int32_t gpu_backend, int cpu_threads, Route* route) {
  (void)cpu_threads;
  (void)gpu_backend;
  route->cpu_iterations = 0xffffffffu;  // never
  route->gpu_min_points = 0;
}

}  // namespace jpo
