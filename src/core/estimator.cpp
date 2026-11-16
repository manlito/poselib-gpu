#include "core/estimator.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/geometry.h"

namespace jpo {

namespace {

const double kDynTrialsMult = 3.0;  // as PoseLib
const int kLoIterations = 25;       // refinement steps of a local optimization
const int kFinalIterations = 100;

struct State {
  uint32_t first = 0, n = 0;
  double sq_thr = 1.0;  // squared threshold on the normalized image plane
  bool active = false;
  bool has_model = false;
  bool on_gpu = false;
  uint32_t iters = 0;
  uint32_t dyn_max = 0;
  double best_minimal = kNoScore;  // best score a minimal sample reached: the bound of the next rounds
  double best_score = kNoScore;
  size_t best_inliers = 0;
  Pose best;
};

uint32_t dynamic_max_iter(size_t inliers, size_t n, int sample, double log_missing, uint32_t min_it, uint32_t max_it) {
  if (inliers < static_cast<size_t>(sample) || n < static_cast<size_t>(sample)) return max_it;
  double p = 1.0;
  for (int i = 0; i < sample; ++i) p *= static_cast<double>(inliers - i) / static_cast<double>(n - i);
  if (p >= 0.9999) return min_it;
  if (p <= 0.0001) return max_it;
  const double it = std::ceil(log_missing / std::log(1.0 - p) * kDynTrialsMult);
  if (!(it < static_cast<double>(max_it))) return max_it;
  return std::max(min_it, static_cast<uint32_t>(it));
}

// The points a refinement can use: those within sqrt(5) thresholds of the model (PoseLib's choice).
void near_points_rel(const Pose& pose, const double* pts, size_t n, double sq_thr, std::vector<double>* out) {
  out->clear();
  double E[9];
  essential_from_pose(pose, E);
  for (size_t i = 0; i < n; ++i) {
    const double* p = pts + 4 * i;
    const double a0 = E[0] * p[0] + E[1] * p[1] + E[2];
    const double a1 = E[3] * p[0] + E[4] * p[1] + E[5];
    const double a2 = E[6] * p[0] + E[7] * p[1] + E[8];
    const double b0 = E[0] * p[2] + E[3] * p[3] + E[6];
    const double b1 = E[1] * p[2] + E[4] * p[3] + E[7];
    const double C = p[2] * a0 + p[3] * a1 + a2;
    if (C * C < 5.0 * sq_thr * (a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1)) out->insert(out->end(), p, p + 4);
  }
}

void near_points_abs(const Pose& pose, const double* pts, size_t n, double sq_thr, std::vector<double>* out) {
  out->clear();
  for (size_t i = 0; i < n; ++i) {
    const double* p = pts + 5 * i;
    const double* X = p + 2;
    const double z = pose.R[6] * X[0] + pose.R[7] * X[1] + pose.R[8] * X[2] + pose.t[2];
    if (!(z > 0.0)) continue;
    const double d0 = (pose.R[0] * X[0] + pose.R[1] * X[1] + pose.R[2] * X[2] + pose.t[0]) / z - p[0];
    const double d1 = (pose.R[3] * X[0] + pose.R[4] * X[1] + pose.R[5] * X[2] + pose.t[1]) / z - p[1];
    if (d0 * d0 + d1 * d1 < 5.0 * sq_thr) out->insert(out->end(), p, p + 5);
  }
}

struct Scratch {
  std::vector<double> near;
};
Scratch& scratch() {
  static thread_local Scratch s;
  return s;
}

// A model from a backend, scored in double and locally optimized.
void consider_relative(State& st, const double* pts, const double* model) {
  const double* p = pts + 4 * static_cast<size_t>(st.first);
  for (int i = 0; i < 12; ++i)
    if (!std::isfinite(model[i])) return;
  Pose cand;
  for (int i = 0; i < 9; ++i) cand.R[i] = model[i];
  for (int i = 0; i < 3; ++i) cand.t[i] = model[9 + i];
  double q[4];
  rotmat_to_quat(cand.R, q);  // a float model is only nearly a rotation
  quat_to_rotmat(q, cand.R);
  size_t ci = 0;
  const double cs = score_relpose(cand, p, st.n, st.sq_thr, &ci, nullptr);
  if (cs < st.best_score) {
    st.best_score = cs;
    st.best_inliers = ci;
    st.best = cand;
    st.has_model = true;
  }
  std::vector<double>& near = scratch().near;
  near_points_rel(cand, p, st.n, st.sq_thr, &near);
  if (near.size() / 4 > 5) {
    refine_relpose(&cand, near.data(), near.size() / 4, st.sq_thr, kLoIterations);
    size_t inl = 0;
    const double s = score_relpose(cand, p, st.n, st.sq_thr, &inl, nullptr);
    if (s < st.best_score) {
      st.best_score = s;
      st.best_inliers = inl;
      st.best = cand;
      st.has_model = true;
    }
  }
}

void consider_absolute(State& st, const double* pts, const double* model) {
  const double* p = pts + 5 * static_cast<size_t>(st.first);
  Pose cand;
  for (int i = 0; i < 9; ++i) cand.R[i] = model[i];
  for (int i = 0; i < 3; ++i) cand.t[i] = model[9 + i];
  for (int i = 0; i < 12; ++i)
    if (!std::isfinite(model[i])) return;
  double q[4];
  rotmat_to_quat(cand.R, q);  // a float model is only nearly a rotation
  quat_to_rotmat(q, cand.R);
  size_t inl = 0;
  double s = score_abspose(cand, p, st.n, st.sq_thr, &inl, nullptr);
  if (s < st.best_score) {
    st.best_score = s;
    st.best_inliers = inl;
    st.best = cand;
    st.has_model = true;
  }
  std::vector<double>& near = scratch().near;
  near_points_abs(cand, p, st.n, st.sq_thr, &near);
  if (near.size() / 5 > 3) {
    refine_abspose(&cand, near.data(), near.size() / 5, st.sq_thr, kLoIterations);
    s = score_abspose(cand, p, st.n, st.sq_thr, &inl, nullptr);
    if (s < st.best_score) {
      st.best_score = s;
      st.best_inliers = inl;
      st.best = cand;
      st.has_model = true;
    }
  }
}

// The loop over rounds, the same for both estimators.
void run_rounds(Kind kind, const Route& route, ThreadPool* pool, std::vector<State>& states,
                const std::vector<double>& pts, size_t n_points, const jpo_ransac_options& opt, BatchStats* stats) {
  const int sample = kind_sample_size(kind);
  const double log_missing = std::log(std::max(1.0 - opt.confidence, 1e-300));
  const uint32_t min_it = std::min(opt.min_iterations, opt.max_iterations);
  const uint32_t max_it = opt.max_iterations;
  Backend* cpu = route.cpu;
  Backend* gpu = route.gpu;
  if (cpu) cpu->set_points(kind, pts.data(), n_points);
  bool gpu_has_points = false;

  std::vector<RoundItem> items[2];
  std::vector<RoundResult> results[2];
  std::vector<uint32_t> index[2];
  for (;;) {
    size_t n_active = 0;
    for (const State& st : states) n_active += st.active ? 1 : 0;
    if (n_active == 0) break;
    for (int b = 0; b < 2; ++b) {
      items[b].clear();
      index[b].clear();
    }
    // Where each active problem runs this round.
    size_t count_b[2] = {0, 0};
    for (State& st : states) {
      if (!st.active) continue;
      if (!st.on_gpu && gpu && (!cpu || (st.iters >= route.cpu_iterations && st.n >= route.gpu_min_points)))
        st.on_gpu = true;
      count_b[st.on_gpu ? 1 : 0] += 1;
    }
    for (size_t i = 0; i < states.size(); ++i) {
      State& st = states[i];
      if (!st.active) continue;
      const int b = st.on_gpu ? 1 : 0;
      const uint32_t need = std::min(max_it, std::max(min_it, st.dyn_max) + 1);
      uint32_t count = std::min((b == 0 ? cpu : gpu)->round_size(count_b[b], st.iters), need - st.iters);
      if (b == 0 && gpu && st.n >= route.gpu_min_points && st.iters < route.cpu_iterations)
        count = std::min(count, route.cpu_iterations - st.iters);
      RoundItem it;
      it.problem = static_cast<uint32_t>(i);
      it.first = st.first;
      it.n = st.n;
      it.iter0 = st.iters;
      it.count = count;
      it.inv_sq_thr = 1.0 / st.sq_thr;
      it.bound = st.best_minimal;
      items[b].push_back(it);
      index[b].push_back(static_cast<uint32_t>(i));
    }
    for (int b = 0; b < 2; ++b) {
      if (items[b].empty()) continue;
      Backend* be = b == 0 ? cpu : gpu;
      if (b == 1 && !gpu_has_points) {
        gpu->set_points(kind, pts.data(), n_points);
        gpu_has_points = true;
      }
      results[b].resize(items[b].size());
      be->run_round(items[b].data(), items[b].size(), results[b].data());
      if (stats) {
        uint64_t h = 0;
        for (const RoundItem& it : items[b]) h += it.count;
        (b == 0 ? stats->cpu_hypotheses : stats->gpu_hypotheses) += h;
      }
      const std::vector<RoundItem>& its = items[b];
      const std::vector<RoundResult>& res = results[b];
      pool->parallel_for(its.size(), [&](size_t k) {
        State& st = states[its[k].problem];
        const RoundResult& r = res[k];
        st.iters += its[k].count;
        if (r.sol >= 0 && r.score < st.best_minimal) {
          st.best_minimal = r.score;
          if (kind == Kind::Relative)
            consider_relative(st, pts.data(), r.model);
          else
            consider_absolute(st, pts.data(), r.model);
          st.dyn_max = dynamic_max_iter(st.best_inliers, st.n, sample, log_missing, min_it, max_it);
        }
        if (st.iters >= max_it || (st.iters > min_it && st.iters > st.dyn_max)) st.active = false;
      });
    }
    if (stats) stats->rounds += 1;
  }
}

void fail_result(jpo_pose_result* r, int status) {
  for (int i = 0; i < 7; ++i) r->pose[i] = 0.0;
  r->pose[0] = 1.0;
  r->score = 0.0;
  r->median_tri_angle_deg = 0.0;
  r->n_inliers = 0;
  r->iterations = 0;
  r->status = status;
  r->reserved = 0;
}

}  // namespace

void estimate_relative(const Route& route, ThreadPool* pool, const jpo_pairs& pairs, const jpo_ransac_options& opt,
                       jpo_pose_result* results, uint8_t* inliers, BatchStats* stats) {
  const size_t B = pairs.n_pairs;
  const size_t total = B ? pairs.offsets[B] : 0;
  std::vector<double> pts(total * 4);
  std::vector<State> states(B);
  pool->parallel_for(B, [&](size_t i) {
    State& st = states[i];
    const uint32_t o0 = pairs.offsets[i], o1 = pairs.offsets[i + 1];
    const jpo_camera& ca = pairs.cameras[pairs.camera_a[i]];
    const jpo_camera& cb = pairs.cameras[pairs.camera_b[i]];
    st.first = o0;
    st.n = o1 - o0;
    const double scale = 0.5 * (1.0 / camera_focal(ca) + 1.0 / camera_focal(cb));
    st.sq_thr = (opt.max_error_px * scale) * (opt.max_error_px * scale);
    for (uint32_t k = o0; k < o1; ++k) {
      double* p = &pts[4 * static_cast<size_t>(k)];
      camera_unproject(ca, pairs.xy_a[2 * k], pairs.xy_a[2 * k + 1], p, p + 1);
      camera_unproject(cb, pairs.xy_b[2 * k], pairs.xy_b[2 * k + 1], p + 2, p + 3);
    }
    st.active = st.n >= 5 && opt.max_iterations > 0;
    st.dyn_max = opt.max_iterations;
  });

  run_rounds(Kind::Relative, route, pool, states, pts, total, opt, stats);

  pool->parallel_for(B, [&](size_t i) {
    State& st = states[i];
    jpo_pose_result* r = &results[i];
    uint8_t* mask = inliers ? inliers + st.first : nullptr;
    if (mask) std::fill(mask, mask + st.n, static_cast<uint8_t>(0));
    if (st.n < 5) return fail_result(r, JPO_TOO_FEW_POINTS);
    if (!st.has_model) {
      fail_result(r, JPO_NO_MODEL);
      r->iterations = st.iters;
      return;
    }
    const double* p = pts.data() + 4 * static_cast<size_t>(st.first);
    Pose pose = st.best;
    double score = st.best_score;
    size_t n_inl = st.best_inliers;
    if (opt.refine) {
      std::vector<double>& near = scratch().near;
      near_points_rel(pose, p, st.n, st.sq_thr, &near);
      if (near.size() / 4 > 5) {
        Pose cand = pose;
        refine_relpose(&cand, near.data(), near.size() / 4, st.sq_thr, kFinalIterations);
        size_t inl = 0;
        const double s = score_relpose(cand, p, st.n, st.sq_thr, &inl, nullptr);
        if (s <= score) {
          pose = cand;
          score = s;
          n_inl = inl;
        }
      }
    }
    std::vector<uint8_t> local;
    uint8_t* m = mask;
    if (!m) {
      local.resize(st.n);
      m = local.data();
    }
    score = score_relpose(pose, p, st.n, st.sq_thr, &n_inl, m);
    rotmat_to_quat(pose.R, r->pose);
    for (int k = 0; k < 3; ++k) r->pose[4 + k] = pose.t[k];
    r->score = score * opt.max_error_px * opt.max_error_px;
    r->n_inliers = static_cast<uint32_t>(n_inl);
    r->iterations = st.iters;
    r->median_tri_angle_deg = median_triangulation_angle(pose, p, st.n, m);
    r->reserved = 0;
    r->status = JPO_OK;
    if (n_inl <= 5) {
      r->status = JPO_NO_MODEL;
    } else {
      // The baseline is not observable when the two rays of a typical inlier are parallel to within a
      // few thresholds: ten, since the inliers that pass the cheirality test are the ones noise spread.
      const double limit = 10.0 * std::atan(std::sqrt(st.sq_thr)) * 180.0 / 3.14159265358979323846;
      if (r->median_tri_angle_deg < limit) r->status = JPO_DEGENERATE;
    }
  });
}

void estimate_absolute(const Route& route, ThreadPool* pool, const jpo_resections& res, const jpo_ransac_options& opt,
                       jpo_pose_result* results, uint8_t* inliers, BatchStats* stats) {
  const size_t B = res.n_images;
  const size_t total = B ? res.offsets[B] : 0;
  std::vector<double> pts(total * 5);
  std::vector<State> states(B);
  // World points are centred and scaled per problem, so the float kernels see well-conditioned numbers.
  std::vector<double> centre(3 * B), scale(B);
  pool->parallel_for(B, [&](size_t i) {
    State& st = states[i];
    const uint32_t o0 = res.offsets[i], o1 = res.offsets[i + 1];
    const jpo_camera& cam = res.cameras[res.camera[i]];
    st.first = o0;
    st.n = o1 - o0;
    const double thr = opt.max_error_px / camera_focal(cam);
    st.sq_thr = thr * thr;
    double c[3] = {0.0, 0.0, 0.0};
    for (uint32_t k = o0; k < o1; ++k)
      for (int d = 0; d < 3; ++d) c[d] += res.xyz[3 * static_cast<size_t>(k) + d];
    if (st.n)
      for (int d = 0; d < 3; ++d) c[d] /= st.n;
    double spread = 0.0;
    for (uint32_t k = o0; k < o1; ++k)
      for (int d = 0; d < 3; ++d) {
        const double v = res.xyz[3 * static_cast<size_t>(k) + d] - c[d];
        spread += v * v;
      }
    spread = st.n ? std::sqrt(spread / st.n) : 0.0;
    const double s = spread > 0.0 && std::isfinite(spread) ? 1.0 / spread : 1.0;
    for (int d = 0; d < 3; ++d) centre[3 * i + d] = std::isfinite(c[d]) ? c[d] : 0.0;
    scale[i] = s;
    for (uint32_t k = o0; k < o1; ++k) {
      double* p = &pts[5 * static_cast<size_t>(k)];
      camera_unproject(cam, res.xy[2 * k], res.xy[2 * k + 1], p, p + 1);
      for (int d = 0; d < 3; ++d) p[2 + d] = (res.xyz[3 * static_cast<size_t>(k) + d] - centre[3 * i + d]) * s;
    }
    st.active = st.n >= 3 && opt.max_iterations > 0;
    st.dyn_max = opt.max_iterations;
  });

  run_rounds(Kind::Absolute, route, pool, states, pts, total, opt, stats);

  pool->parallel_for(B, [&](size_t i) {
    State& st = states[i];
    jpo_pose_result* r = &results[i];
    uint8_t* mask = inliers ? inliers + st.first : nullptr;
    if (mask) std::fill(mask, mask + st.n, static_cast<uint8_t>(0));
    if (st.n < 3) return fail_result(r, JPO_TOO_FEW_POINTS);
    if (!st.has_model) {
      fail_result(r, JPO_NO_MODEL);
      r->iterations = st.iters;
      return;
    }
    // Back to the caller's world: Z ~ R X' + t' with X' = s (X - c), so t = t' / s - R c.
    Pose pose = st.best;
    const double* c = &centre[3 * i];
    for (int d = 0; d < 3; ++d)
      pose.t[d] = pose.t[d] / scale[i] - (pose.R[3 * d] * c[0] + pose.R[3 * d + 1] * c[1] + pose.R[3 * d + 2] * c[2]);
    const jpo_camera& cam = res.cameras[res.camera[i]];
    const double* xy = res.xy + 2 * static_cast<size_t>(st.first);
    const double* xyz = res.xyz + 3 * static_cast<size_t>(st.first);
    const double sq_px = opt.max_error_px * opt.max_error_px;
    size_t n_inl = 0;
    double score = score_abspose_px(pose, cam, xy, xyz, st.n, sq_px, &n_inl, nullptr);
    if (opt.refine && n_inl > 3) {
      // In pixels through the camera model, on the points near the model.
      std::vector<double> nxy, nxyz;
      for (uint32_t k = 0; k < st.n; ++k) {
        double Z[3], px, py;
        for (int d = 0; d < 3; ++d)
          Z[d] = pose.R[3 * d] * xyz[3 * k] + pose.R[3 * d + 1] * xyz[3 * k + 1] + pose.R[3 * d + 2] * xyz[3 * k + 2] +
                 pose.t[d];
        if (!camera_project(cam, Z, &px, &py)) continue;
        const double d0 = px - xy[2 * k], d1 = py - xy[2 * k + 1];
        if (d0 * d0 + d1 * d1 < 5.0 * sq_px) {
          nxy.insert(nxy.end(), xy + 2 * k, xy + 2 * k + 2);
          nxyz.insert(nxyz.end(), xyz + 3 * k, xyz + 3 * k + 3);
        }
      }
      if (nxy.size() / 2 > 3) {
        Pose cand = pose;
        refine_abspose_px(&cand, cam, nxy.data(), nxyz.data(), nxy.size() / 2, sq_px, kFinalIterations);
        size_t inl = 0;
        const double s = score_abspose_px(cand, cam, xy, xyz, st.n, sq_px, &inl, nullptr);
        if (s <= score) {
          pose = cand;
          score = s;
          n_inl = inl;
        }
      }
    }
    score = score_abspose_px(pose, cam, xy, xyz, st.n, sq_px, &n_inl, mask);
    rotmat_to_quat(pose.R, r->pose);
    for (int k = 0; k < 3; ++k) r->pose[4 + k] = pose.t[k];
    r->score = score * sq_px;
    r->n_inliers = static_cast<uint32_t>(n_inl);
    r->iterations = st.iters;
    r->median_tri_angle_deg = 0.0;
    r->reserved = 0;
    r->status = n_inl > 3 ? JPO_OK : JPO_NO_MODEL;
  });
}

}  // namespace jpo
