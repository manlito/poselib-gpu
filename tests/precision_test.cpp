// The float build of the kernels against the double build: how much the minimal solvers lose in
// float32, measured before any GPU code depends on it (PLAN.md, "Precision").
#include <cmath>
#include <cstdio>
#include <vector>

#include "backend/cpu/kernels_cpu.h"
#include "cli/synth.h"
#include "jpo/jpo.h"

int main() {
  // 1. Solver level: the same samples of noisy problems, solved in both precisions. A double solution
  //    counts as reproduced if a float solution lies within 1e-3 of it (unit Frobenius norm / rotation).
  jposynth::Spec spec;
  spec.count = 50;
  spec.points = 500;
  spec.inlier_ratio = 1.0;
  spec.noise_px = 0.5;
  size_t d_sols = 0, f_sols = 0, matched = 0, samples = 0;
  size_t pd_sols = 0, pf_sols = 0, p_matched = 0;
  // Score level: the best MSAC score among 256 hypotheses, float against double.
  double worst_rel_score = 0.0;
  for (uint32_t kind = 0; kind < 2; ++kind) {
    spec.kind = kind;
    const jpofile::ProblemSet set = jposynth::make_set(spec);
    for (size_t pi = 0; pi < set.problems.size(); ++pi) {
      const jpofile::Problem& p = set.problems[pi];
      const uint32_t n = p.n();
      std::vector<double> pd(n * (kind == 0 ? 4 : 5));
      for (uint32_t k = 0; k < n; ++k) {
        if (kind == 0) {
          jpo::camera_unproject(p.cam_a, p.xy_a[2 * k], p.xy_a[2 * k + 1], &pd[4 * k], &pd[4 * k + 1]);
          jpo::camera_unproject(p.cam_b, p.xy_b[2 * k], p.xy_b[2 * k + 1], &pd[4 * k + 2], &pd[4 * k + 3]);
        } else {
          jpo::camera_unproject(p.cam_a, p.xy_a[2 * k], p.xy_a[2 * k + 1], &pd[5 * k], &pd[5 * k + 1]);
          for (int d = 0; d < 3; ++d) pd[5 * k + 2 + d] = p.xyz[3 * k + d] * 0.2;
        }
      }
      std::vector<float> pf(pd.begin(), pd.end());
      const double ist = 1.0e6;  // threshold 1e-3 on the normalized plane
      double best_d = 3e38, best_f = 3e38;
      for (uint32_t it = 0; it < 256; ++it) {
        ++samples;
        if (kind == 0) {
          double b1[15], b2[15], Ed[90];
          float c1[15], c2[15], Ef[90];
          jpo::kd::jpo_rel_sample(pd.data(), 0, n, 1, pi, it, b1, b2);
          jpo::kf::jpo_rel_sample(pf.data(), 0, n, 1, pi, it, c1, c2);
          const int nd = jpo::kd::jpo_relpose_5pt(b1, b2, Ed);
          const int nf = jpo::kf::jpo_relpose_5pt(c1, c2, Ef);
          d_sols += nd;
          f_sols += nf;
          for (int s = 0; s < nd; ++s) {
            double best = 1e9;
            for (int t = 0; t < nf; ++t) {
              double dp = 0, dm = 0;
              for (int i = 0; i < 9; ++i) {
                dp += std::pow(Ed[9 * s + i] - Ef[9 * t + i], 2);
                dm += std::pow(Ed[9 * s + i] + Ef[9 * t + i], 2);
              }
              best = std::min(best, std::sqrt(std::min(dp, dm)));
            }
            matched += best < 1e-3 ? 1 : 0;
          }
          int sol;
          best_d = std::min(best_d, jpo::kd::jpo_rel_hypothesis(pd.data(), 0, n, 1, pi, it, ist, 3e38, &sol));
          best_f = std::min<double>(best_f, jpo::kf::jpo_rel_hypothesis(pf.data(), 0, n, 1, pi, it, (float)ist, 3e38f, &sol));
        } else {
          double xs[9], Xs[9], Pd[48];
          float ys[9], Ys[9], Pf[48];
          jpo::kd::jpo_abs_sample(pd.data(), 0, n, 1, pi, it, xs, Xs);
          jpo::kf::jpo_abs_sample(pf.data(), 0, n, 1, pi, it, ys, Ys);
          const int nd = jpo::kd::jpo_p3p(xs, Xs, Pd);
          const int nf = jpo::kf::jpo_p3p(ys, Ys, Pf);
          pd_sols += nd;
          pf_sols += nf;
          for (int s = 0; s < nd; ++s) {
            double best = 1e9;
            for (int t = 0; t < nf; ++t) {
              double d = 0;
              for (int i = 0; i < 12; ++i) d += std::pow(Pd[12 * s + i] - Pf[12 * t + i], 2);
              best = std::min(best, std::sqrt(d));
            }
            p_matched += best < 1e-3 ? 1 : 0;
          }
          int sol;
          best_d = std::min(best_d, jpo::kd::jpo_abs_hypothesis(pd.data(), 0, n, 1, pi, it, ist, 3e38, &sol));
          best_f = std::min<double>(best_f, jpo::kf::jpo_abs_hypothesis(pf.data(), 0, n, 1, pi, it, (float)ist, 3e38f, &sol));
        }
      }
      worst_rel_score = std::max(worst_rel_score, (best_f - best_d) / best_d);
    }
  }
  const double rel_share = static_cast<double>(matched) / d_sols;
  const double abs_share = static_cast<double>(p_matched) / pd_sols;
  std::printf("5pt: double %zu solutions, float %zu, float reproduces %.2f%% of the double ones\n", d_sols, f_sols,
              100.0 * rel_share);
  std::printf("p3p: double %zu solutions, float %zu, float reproduces %.2f%% of the double ones\n", pd_sols, pf_sols,
              100.0 * abs_share);
  std::printf("best score of 256 hypotheses: float is at most %.2f%% above double\n", 100.0 * worst_rel_score);

  // 2. Estimator level: the whole estimator with float kernels against double kernels.
  int fail = 0;
  for (uint32_t kind = 0; kind < 2; ++kind) {
    for (double ratio : {0.7, 0.3, 0.1}) {
      spec.kind = kind;
      spec.count = 100;
      spec.points = 1000;
      spec.inlier_ratio = ratio;
      spec.seed = 21;
      const jpofile::ProblemSet set = jposynth::make_set(spec);
      const jpofile::Flat flat = jpofile::flatten(set);
      double med[2] = {0, 0};
      int ok[2] = {0, 0};
      double iters[2] = {0, 0};
      for (int f = 0; f < 2; ++f) {
        jpo_config cfg;
        jpo_config_init(&cfg);
        cfg.backend = JPO_BACKEND_CPU;
        cfg.cpu_float = f;
        jpo_ctx* ctx = jpo_create(&cfg);
        std::vector<jpo_pose_result> res(set.problems.size());
        if (kind == 0) {
          const jpo_pairs p = flat.pairs();
          jpo_relative_pose(ctx, &p, nullptr, res.data(), nullptr);
        } else {
          const jpo_resections r = flat.resections();
          jpo_absolute_pose(ctx, &r, nullptr, res.data(), nullptr);
        }
        std::vector<double> errs;
        for (size_t i = 0; i < res.size(); ++i) {
          double rot = 180, tr = 180;
          if (res[i].status == JPO_OK) jposynth::pose_errors(kind, res[i].pose, set.problems[i].gt_pose, &rot, &tr);
          const double e = kind == 0 ? std::max(rot, tr) : rot;
          errs.push_back(e);
          ok[f] += e < (kind == 0 ? 5.0 : 1.0) ? 1 : 0;
          iters[f] += res[i].iterations;
        }
        std::sort(errs.begin(), errs.end());
        med[f] = errs[errs.size() / 2];
        jpo_destroy(ctx);
      }
      std::printf("%s inliers %.0f%%: double %d/100 solved, median %.4f deg, %.0f iterations | float %d/100, %.4f deg, %.0f\n",
                  kind ? "pnp    " : "relpose", 100 * ratio, ok[0], med[0], iters[0] / 100, ok[1], med[1], iters[1] / 100);
      if (ok[1] < ok[0] - 3) ++fail;
    }
  }
  if (rel_share < 0.9 || abs_share < 0.9) ++fail;
  std::printf(fail ? "precision_test: FAILED\n" : "precision_test: passed\n");
  return fail ? 1 : 0;
}
