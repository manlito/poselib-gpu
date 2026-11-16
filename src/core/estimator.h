// The host side of RANSAC in rounds: preparation of a batch, the loop over rounds, the stopping rule,
// local optimization and the final refinement. Backends only run rounds.
#pragma once

#include <cstdint>

#include "backend/backend.h"
#include "core/pool.h"
#include "jpo/jpo.h"

namespace jpo {

// Where rounds run. With both backends set, a problem starts on the CPU and moves to the GPU once it
// has drawn cpu_iterations hypotheses without meeting its stopping rule (the `auto` rule).
struct Route {
  Backend* cpu = nullptr;
  Backend* gpu = nullptr;
  uint32_t cpu_iterations = 0;
  // Problems with fewer points than this never move to the GPU.
  uint32_t gpu_min_points = 0;
};

struct BatchStats {
  uint64_t cpu_hypotheses = 0;
  uint64_t gpu_hypotheses = 0;
  uint32_t rounds = 0;
};

void estimate_relative(const Route& route, ThreadPool* pool, const jpo_pairs& pairs, const jpo_ransac_options& opt,
                       jpo_pose_result* results, uint8_t* inliers, BatchStats* stats);
void estimate_absolute(const Route& route, ThreadPool* pool, const jpo_resections& res, const jpo_ransac_options& opt,
                       jpo_pose_result* results, uint8_t* inliers, BatchStats* stats);

}  // namespace jpo
