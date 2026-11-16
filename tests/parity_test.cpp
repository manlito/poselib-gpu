// Each minimal solver against PoseLib's on the same samples: the same set of solutions.
#include <cmath>
#include <cstdio>
#include <vector>

#include <PoseLib/solvers/p3p.h>
#include <PoseLib/solvers/relpose_5pt.h>

#include "backend/cpu/kernels_cpu.h"
#include "cli/synth.h"

int main() {
  jposynth::Rng rng(17);
  const int trials = 5000;
  size_t ref5 = 0, ours5 = 0, match5 = 0, ref_exact = 0, ours_exact = 0, ref3 = 0, ours3 = 0, match3 = 0;
  for (int t = 0; t < trials; ++t) {
    double R[9], tr[3] = {rng.normal(), rng.normal(), rng.normal()};
    jposynth::random_rotation(rng, 40.0, R);
    const double noise = (t % 2) ? 1e-3 : 0.0;  // exact and noisy samples
    std::vector<Eigen::Vector3d> x1(5), x2(5), xs(3), Xs(3);
    double b1[15], b2[15], ys[9], Ys[9];
    for (int k = 0; k < 5; ++k) {
      const double X[3] = {rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(3, 8)};
      double Y[3];
      jposynth::apply(R, tr, X, Y);
      Eigen::Vector3d a(X[0], X[1], X[2]), b(Y[0], Y[1], Y[2]);
      a = (a / a.z() + noise * Eigen::Vector3d(rng.normal(), rng.normal(), 0)).normalized();
      b = (b / b.z() + noise * Eigen::Vector3d(rng.normal(), rng.normal(), 0)).normalized();
      x1[k] = a;
      x2[k] = b;
      for (int d = 0; d < 3; ++d) {
        b1[3 * k + d] = a(d);
        b2[3 * k + d] = b(d);
      }
      if (k < 3) {
        xs[k] = b;
        Xs[k] = Eigen::Vector3d(X[0], X[1], X[2]);
        for (int d = 0; d < 3; ++d) {
          ys[3 * k + d] = b(d);
          Ys[3 * k + d] = X[d];
        }
      }
    }
    std::vector<Eigen::Matrix3d> Eref;
    poselib::relpose_5pt(x1, x2, &Eref);
    double E[90];
    const int ns = jpo::kd::jpo_relpose_5pt(b1, b2, E);
    ref5 += Eref.size();
    ours5 += ns;
    for (int s = 0; s < ns; ++s) {
      Eigen::Matrix3d Eo;
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) Eo(r, c) = E[9 * s + 3 * r + c];
      ours_exact += (2 * Eo * Eo.transpose() * Eo - (Eo * Eo.transpose()).trace() * Eo).norm() < 1e-8 ? 1 : 0;
    }
    for (const Eigen::Matrix3d& Er : Eref) {
      const Eigen::Matrix3d En = Er / Er.norm();
      // PoseLib's own solution can be off where the problem is badly conditioned: it is only a
      // reference where it satisfies the constraints of an essential matrix.
      if ((2 * En * En.transpose() * En - (En * En.transpose()).trace() * En).norm() > 1e-8) continue;
      ref_exact += 1;
      double best = 1e9;
      for (int s = 0; s < ns; ++s) {
        double dp = 0, dm = 0;
        for (int r = 0; r < 3; ++r)
          for (int c = 0; c < 3; ++c) {
            dp += std::pow(E[9 * s + 3 * r + c] - En(r, c), 2);
            dm += std::pow(E[9 * s + 3 * r + c] + En(r, c), 2);
          }
        best = std::min(best, std::sqrt(std::min(dp, dm)));
      }
      match5 += best < 1e-6 ? 1 : 0;
    }
    std::vector<poselib::CameraPose> Pref;
    poselib::p3p(xs, Xs, &Pref);
    double P[48];
    const int np = jpo::kd::jpo_p3p(ys, Ys, P);
    ref3 += Pref.size();
    ours3 += np;
    for (const poselib::CameraPose& pr : Pref) {
      const Eigen::Matrix3d Rr = pr.R();
      double best = 1e9;
      for (int s = 0; s < np; ++s) {
        double d = 0;
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) d += std::pow(P[12 * s + 3 * r + c] - Rr(r, c), 2);
          d += std::pow(P[12 * s + 9 + r] - pr.t(r), 2);
        }
        best = std::min(best, std::sqrt(d));
      }
      match3 += best < 1e-6 ? 1 : 0;
    }
  }
  std::printf("5pt: PoseLib %zu solutions (%zu exact to 1e-8), ours %zu (%zu exact), %zu of PoseLib's exact ones reproduced\n",
              ref5, ref_exact, ours5, ours_exact, match5);
  std::printf("p3p: PoseLib %zu solutions, ours %zu, %zu of PoseLib's reproduced\n", ref3, ours3, match3);
  // A root can be lost or gained where two roots meet, and which side of that a sample falls on
  // depends on the compiler's rounding; allow two in a thousand.
  const bool ok = match5 * 1000 >= ref_exact * 999 && ours_exact >= ref_exact && match3 * 1000 >= ref3 * 998 &&
                  std::llabs((long long)ours5 - (long long)ref5) * 1000 <= (long long)ref5 * 2 &&
                  std::llabs((long long)ours3 - (long long)ref3) * 1000 <= (long long)ref3 * 2;
  std::printf(ok ? "parity_test: passed\n" : "parity_test: FAILED\n");
  return ok ? 0 : 1;
}
