// The host side of RANSAC in rounds: preparation of a batch, the loop over rounds, the stopping rule,
// local optimization and the final refinement. Backends only run rounds.
#pragma once

#include <cstdint>

#include "backend/backend.h"
#include "core/pool.h"
#include "jpo/jpo.h"

namespace jpo {

// Where rounds run. With only one backend set, everything runs there. With both (the `auto` rule), a
// problem starts on the CPU; once it has drawn cpu_iterations hypotheses it becomes a candidate for the
// GPU if it still needs at least gpu_min_remaining more, and the candidates move together as soon as
// they need gpu_min_total hypotheses between them. Easy problems therefore never leave the CPU, and a
// GPU round starts with a bound and an early rejection test from the CPU phase.
struct Route {
  Backend* cpu = nullptr;
  Backend* gpu = nullptr;
  uint32_t cpu_iterations = 0;
  uint32_t gpu_min_remaining = 0;
  uint64_t gpu_min_total = 0;
  // The same three numbers for a call whose iteration budget (max_iterations, or the cap that
  // min_inlier_ratio sets) fits in one GPU round; 0 keeps the numbers above. Whatever moves then takes
  // exactly one dispatch, while the numbers above were measured on problems that take several: a GPU
  // round costs a fixed latency, so 24 resections that need 500 more hypotheses each are worth one
  // dispatch where one that needs 10,000 (five rounds) is not, at about the same total.
  uint32_t one_round_cpu_iterations = 0;
  uint32_t one_round_gpu_min_remaining = 0;
  uint64_t one_round_gpu_min_total = 0;
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
