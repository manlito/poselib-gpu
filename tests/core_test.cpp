// Cameras, solvers, refinement and the two estimators on generated data with known answers.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "backend/cpu/kernels_cpu.h"
#include "cli/synth.h"
#include "core/geometry.h"
#include "jpo/jpo.h"

static int g_fail = 0;
#define CHECK(cond, ...)                 \
  do {                                   \
    if (!(cond)) {                       \
      ++g_fail;                          \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);          \
      std::printf("\n");                 \
    }                                    \
  } while (0)

using jposynth::Rng;

static void test_cameras() {
  Rng rng(3);
  for (int model = 0; model <= JPO_OPENCV_FISHEYE; ++model) {
    double worst = 0.0;
    for (int t = 0; t < 2000; ++t) {
      const jpo_camera cam = jposynth::make_camera(model, rng, 1600, 1200);
      const double px = rng.uniform(0, 1599), py = rng.uniform(0, 1199);
      double x, y, u, v;
      if (!jpo::camera_unproject(cam, px, py, &x, &y)) continue;
      const double X[3] = {x * 3.0, y * 3.0, 3.0};
      CHECK(jpo::camera_project(cam, X, &u, &v), "project failed, model %d", model);
      worst = std::max(worst, std::hypot(u - px, v - py));
    }
    CHECK(worst < 1e-6, "camera model %d round trip %.3g px", model, worst);
  }
}

// Five exact correspondences of a known motion: one of the solutions is that motion.
static void test_solvers() {
  Rng rng(5);
  int rel_found = 0, abs_found = 0;
  const int trials = 2000;
  for (int t = 0; t < trials; ++t) {
    double R[9], tr[3] = {rng.normal(), rng.normal(), rng.normal()};
    jposynth::random_rotation(rng, 40.0, R);
    double b1[15], b2[15], xs[9], Xs[9];
    for (int k = 0; k < 5; ++k) {
      const double X[3] = {rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(3, 8)};
      double Y[3];
      jposynth::apply(R, tr, X, Y);
      const double n1 = std::sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);
      const double n2 = std::sqrt(Y[0] * Y[0] + Y[1] * Y[1] + Y[2] * Y[2]);
      for (int d = 0; d < 3; ++d) {
        b1[3 * k + d] = X[d] / n1;
        b2[3 * k + d] = Y[d] / n2;
        if (k < 3) {
          Xs[3 * k + d] = X[d];
          xs[3 * k + d] = Y[d] / n2;
        }
      }
    }
    jpo::Pose gt;
    std::memcpy(gt.R, R, sizeof(R));
    const double tn = std::sqrt(tr[0] * tr[0] + tr[1] * tr[1] + tr[2] * tr[2]);
    for (int d = 0; d < 3; ++d) gt.t[d] = tr[d] / tn;
    double Egt[9];
    jpo::essential_from_pose(gt, Egt);
    double en = 0.0;
    for (int i = 0; i < 9; ++i) en += Egt[i] * Egt[i];
    en = std::sqrt(en);
    double E[90];
    const int ns = jpo::kd::jpo_relpose_5pt(b1, b2, E);
    bool found = false;
    for (int s = 0; s < ns; ++s) {
      double dp = 0.0, dm = 0.0;
      for (int i = 0; i < 9; ++i) {
        dp += std::pow(E[9 * s + i] - Egt[i] / en, 2);
        dm += std::pow(E[9 * s + i] + Egt[i] / en, 2);
      }
      found = found || std::min(dp, dm) < 1e-10;
      for (int k = 0; k < 5; ++k) {  // every solution satisfies the five constraints
        double r = 0.0;
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) r += b2[3 * k + a] * E[9 * s + 3 * a + b] * b1[3 * k + b];
        CHECK(std::abs(r) < 1e-7, "5pt residual %.3g", r);
      }
    }
    rel_found += found ? 1 : 0;

    double P[48];
    const int np = jpo::kd::jpo_p3p(xs, Xs, P);
    found = false;
    for (int s = 0; s < np; ++s) {
      double d = 0.0;
      for (int i = 0; i < 9; ++i) d += std::pow(P[12 * s + i] - R[i], 2);
      for (int i = 0; i < 3; ++i) d += std::pow(P[12 * s + 9 + i] - tr[i], 2);
      found = found || d < 1e-12;
    }
    abs_found += found ? 1 : 0;
  }
  CHECK(rel_found >= trials - 2, "5pt found the true motion in %d of %d", rel_found, trials);
  CHECK(abs_found >= trials - 2, "p3p found the true pose in %d of %d", abs_found, trials);
  std::printf("solvers: 5pt %d/%d, p3p %d/%d\n", rel_found, trials, abs_found, trials);
}

struct Summary {
  double median_err = 0.0, worst_err = 0.0, mask_agree = 0.0;
  int ok = 0;
};

static Summary run(const jpofile::ProblemSet& set, jpo_ctx* ctx, std::vector<jpo_pose_result>* out = nullptr) {
  const jpofile::Flat flat = jpofile::flatten(set);
  std::vector<jpo_pose_result> res(set.problems.size());
  std::vector<uint8_t> inl(flat.offsets.back());
  int rc;
  if (set.kind == 0) {
    const jpo_pairs p = flat.pairs();
    rc = jpo_relative_pose(ctx, &p, nullptr, res.data(), inl.data());
  } else {
    const jpo_resections r = flat.resections();
    rc = jpo_absolute_pose(ctx, &r, nullptr, res.data(), inl.data());
  }
  CHECK(rc == 0, "estimator failed: %s", jpo_last_error());
  Summary s;
  std::vector<double> errs;
  size_t agree = 0, total = 0;
  for (size_t i = 0; i < res.size(); ++i) {
    if (res[i].status == JPO_OK) s.ok += 1;
    double rot = 180, tr = 180;
    if (res[i].status == JPO_OK && std::isfinite(set.problems[i].gt_pose[0])) {
      jposynth::pose_errors(set.kind, res[i].pose, set.problems[i].gt_pose, &rot, &tr);
      errs.push_back(set.kind == 0 ? std::max(rot, tr) : rot);
    }
    uint32_t count = 0;
    for (uint32_t k = 0; k < set.problems[i].n(); ++k) {
      const uint8_t m = inl[flat.offsets[i] + k];
      count += m;
      // Outliers are uniform, so a few fall on the model by chance; inliers are noisy, so a few fall off.
      agree += (m != 0) == (set.problems[i].gt_inlier[k] != 0) ? 1 : 0;
      ++total;
    }
    CHECK(count == res[i].n_inliers, "mask has %u inliers, result says %u", count, res[i].n_inliers);
  }
  std::sort(errs.begin(), errs.end());
  if (!errs.empty()) {
    s.median_err = errs[errs.size() / 2];
    s.worst_err = errs.back();
  }
  s.mask_agree = total ? static_cast<double>(agree) / total : 1.0;
  if (out) *out = res;
  return s;
}

static void test_estimators() {
  jpo_config cfg;
  jpo_config_init(&cfg);
  cfg.backend = JPO_BACKEND_CPU;
  jpo_ctx* ctx = jpo_create(&cfg);
  for (uint32_t kind = 0; kind < 2; ++kind) {
    for (int model = 0; model <= JPO_OPENCV_FISHEYE; ++model) {
      jposynth::Spec spec;
      spec.kind = kind;
      spec.count = 20;
      spec.points = 400;
      spec.inlier_ratio = 0.5;
      spec.noise_px = 0.3;
      spec.model = model;
      spec.world_offset = 1000.0;
      spec.seed = 11 + model;
      const Summary s = run(jposynth::make_set(spec), ctx);
      std::printf("%s %-15s ok %d/20 median %.4f worst %.4f deg, mask agreement %.4f\n", kind ? "pnp    " : "relpose",
                  jpofile::model_name(model), s.ok, s.median_err, s.worst_err, s.mask_agree);
      CHECK(s.ok == 20, "kind %u model %d: %d of 20 solved", kind, model, s.ok);
      CHECK(s.median_err < (kind == 0 ? 0.5 : 0.1), "kind %u model %d: median error %.4f deg", kind, model, s.median_err);
      CHECK(s.worst_err < (kind == 0 ? 5.0 : 1.0), "kind %u model %d: worst error %.4f deg", kind, model, s.worst_err);
      // A relative pose is scored on the normalized image plane, where a pixel of noise far from the
      // centre of a fisheye image is several thresholds: some true inliers are left out there.
      const double agree = kind == 0 && model == JPO_OPENCV_FISHEYE ? 0.93 : 0.97;
      CHECK(s.mask_agree > agree, "kind %u model %d: mask agreement %.4f", kind, model, s.mask_agree);
    }
  }
  // Hard cases: few inliers, a plane, no model, too few points.
  {
    jposynth::Spec spec;
    spec.count = 20;
    spec.points = 1000;
    spec.inlier_ratio = 0.25;
    spec.seed = 31;
    Summary s = run(jposynth::make_set(spec), ctx);
    CHECK(s.ok >= 19 && s.median_err < 1.0, "25%% inliers: ok %d median %.3f", s.ok, s.median_err);
    spec.kind = 1;
    s = run(jposynth::make_set(spec), ctx);
    CHECK(s.ok >= 19 && s.median_err < 0.2, "pnp 25%% inliers: ok %d median %.3f", s.ok, s.median_err);
    spec.kind = 0;
    spec.inlier_ratio = 0.6;
    spec.planar = true;
    s = run(jposynth::make_set(spec), ctx);
    CHECK(s.ok >= 19 && s.median_err < 1.0, "planar: ok %d median %.3f", s.ok, s.median_err);
    spec.kind = 1;
    s = run(jposynth::make_set(spec), ctx);
    CHECK(s.ok >= 19 && s.median_err < 0.2, "pnp planar: ok %d median %.3f", s.ok, s.median_err);
    spec.planar = false;
    for (uint32_t kind = 0; kind < 2; ++kind) {
      spec.kind = kind;
      spec.inlier_ratio = 0.0;
      spec.points = 300;
      std::vector<jpo_pose_result> res;
      run(jposynth::make_set(spec), ctx, &res);
      int found = 0;
      for (const jpo_pose_result& r : res) found += r.n_inliers > 30 ? 1 : 0;
      CHECK(found == 0, "kind %u: %d of 20 random problems reported more than 30 inliers", kind, found);
      spec.points = kind == 0 ? 4 : 2;
      run(jposynth::make_set(spec), ctx, &res);
      for (const jpo_pose_result& r : res) CHECK(r.status == JPO_TOO_FEW_POINTS, "status %d", r.status);
    }
    // Pure rotation: a pose comes back, flagged as degenerate.
    spec.kind = 0;
    spec.points = 500;
    spec.inlier_ratio = 0.8;
    spec.baseline = 0.0;
    std::vector<jpo_pose_result> res;
    run(jposynth::make_set(spec), ctx, &res);
    int flagged = 0;
    for (const jpo_pose_result& r : res) flagged += r.status == JPO_DEGENERATE ? 1 : 0;
    CHECK(flagged >= 18, "pure rotation flagged in %d of 20", flagged);
  }
  // The same seed gives the same result, whatever the number of threads.
  {
    jposynth::Spec spec;
    spec.count = 8;
    spec.points = 600;
    spec.inlier_ratio = 0.3;
    const jpofile::ProblemSet set = jposynth::make_set(spec);
    std::vector<jpo_pose_result> a, b, c;
    run(set, ctx, &a);
    run(set, ctx, &b);
    cfg.threads = 1;
    jpo_ctx* one = jpo_create(&cfg);
    run(set, one, &c);
    jpo_destroy(one);
    for (size_t i = 0; i < a.size(); ++i) {
      CHECK(std::memcmp(a[i].pose, b[i].pose, sizeof(a[i].pose)) == 0, "two runs differ");
      // One thread draws rounds of another size, so it may stop at another iteration; the answer agrees.
      double rot, tr;
      jposynth::pose_errors(0, a[i].pose, c[i].pose, &rot, &tr);
      CHECK(rot < 0.2 && tr < 2.0, "1 thread against all: %.4f %.4f deg", rot, tr);
    }
  }
  jpo_destroy(ctx);
}

int main() {
  test_cameras();
  test_solvers();
  test_estimators();
  if (g_fail) {
    std::printf("%d checks failed\n", g_fail);
    return 1;
  }
  std::printf("core_test: all passed\n");
  return 0;
}
