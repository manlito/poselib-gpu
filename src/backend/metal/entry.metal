
// ---------------------------------------------------------------------------------------------------
// Entry points. A round is two dispatches: jpo_round_* runs one thread per hypothesis (x) and item (y)
// and writes its score and solution index; jpo_select_* runs one thread per item, finds the best
// hypothesis and solves it once more for the model.

struct JpoItem {
  uint first;      // first point of the problem
  uint n;          // points
  uint problem;    // sample counter
  uint iter0;
  uint count;      // hypotheses of this item (<= round)
  float inv_sq_thr;
  float bound;
  uint pre_k;      // early rejection (kernels.inc, "Scores")
  uint pre_min;
  uint pad;
};

struct JpoParams {
  uint seed;
  uint round;      // hypotheses per item in the grid
};

struct JpoResult {
  float score;
  uint iter;
  int sol;
  uint pad;
  float model[12];
};

kernel void jpo_round_rel(const device float* pts [[buffer(0)]], const device JpoItem* items [[buffer(1)]],
                          device float* score [[buffer(2)]], device int* sol [[buffer(3)]],
                          constant JpoParams& prm [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  JpoItem it = items[gid.y];
  if (gid.x >= it.count) return;
  int s = -1;
  float sc = jpo_rel_hypothesis(pts, it.first * JPO_REL_STRIDE, it.n, prm.seed, it.problem, it.iter0 + gid.x,
                                it.inv_sq_thr, it.bound, it.pre_k, it.pre_min, &s);
  uint o = gid.y * prm.round + gid.x;
  score[o] = sc;
  sol[o] = s;
}

kernel void jpo_round_abs(const device float* pts [[buffer(0)]], const device JpoItem* items [[buffer(1)]],
                          device float* score [[buffer(2)]], device int* sol [[buffer(3)]],
                          constant JpoParams& prm [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  JpoItem it = items[gid.y];
  if (gid.x >= it.count) return;
  int s = -1;
  float sc = jpo_abs_hypothesis(pts, it.first * JPO_ABS_STRIDE, it.n, prm.seed, it.problem, it.iter0 + gid.x,
                                it.inv_sq_thr, it.bound, it.pre_k, it.pre_min, &s);
  uint o = gid.y * prm.round + gid.x;
  score[o] = sc;
  sol[o] = s;
}

static inline void jpo_best(const device float* score, const device int* sol, uint o, uint count, thread float* best,
                            thread uint* best_h, thread int* best_sol) {
  best[0] = JPO_BIG;
  best_h[0] = 0;
  best_sol[0] = -1;
  for (uint h = 0; h < count; ++h) {  // in iteration order: a tie keeps the earlier hypothesis
    int s = sol[o + h];
    float sc = score[o + h];
    if (s >= 0 && sc < best[0]) {
      best[0] = sc;
      best_h[0] = h;
      best_sol[0] = s;
    }
  }
}

kernel void jpo_select_rel(const device float* pts [[buffer(0)]], const device JpoItem* items [[buffer(1)]],
                           const device float* score [[buffer(2)]], const device int* sol [[buffer(3)]],
                           constant JpoParams& prm [[buffer(4)]], device JpoResult* out [[buffer(5)]],
                           uint gid [[thread_position_in_grid]]) {
  JpoItem it = items[gid];
  float best;
  uint h;
  int s;
  jpo_best(score, sol, gid * prm.round, it.count, &best, &h, &s);
  float m[12];
  for (int i = 0; i < 12; ++i) m[i] = 0.0f;
  if (s >= 0) {
    if (jpo_rel_model(pts, it.first * JPO_REL_STRIDE, it.n, prm.seed, it.problem, it.iter0 + h, s, m) == 0) s = -1;
  }
  out[gid].score = s >= 0 ? best : JPO_BIG;
  out[gid].iter = it.iter0 + h;
  out[gid].sol = s;
  out[gid].pad = 0;
  for (int i = 0; i < 12; ++i) out[gid].model[i] = m[i];
}

kernel void jpo_select_abs(const device float* pts [[buffer(0)]], const device JpoItem* items [[buffer(1)]],
                           const device float* score [[buffer(2)]], const device int* sol [[buffer(3)]],
                           constant JpoParams& prm [[buffer(4)]], device JpoResult* out [[buffer(5)]],
                           uint gid [[thread_position_in_grid]]) {
  JpoItem it = items[gid];
  float best;
  uint h;
  int s;
  jpo_best(score, sol, gid * prm.round, it.count, &best, &h, &s);
  float m[12];
  for (int i = 0; i < 12; ++i) m[i] = 0.0f;
  if (s >= 0) {
    if (jpo_abs_model(pts, it.first * JPO_ABS_STRIDE, it.n, prm.seed, it.problem, it.iter0 + h, s, m) == 0) s = -1;
  }
  out[gid].score = s >= 0 ? best : JPO_BIG;
  out[gid].iter = it.iter0 + h;
  out[gid].sol = s;
  out[gid].pad = 0;
  for (int i = 0; i < 12; ++i) out[gid].model[i] = m[i];
}
