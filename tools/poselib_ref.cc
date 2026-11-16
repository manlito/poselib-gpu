// PoseLib on problem files: the reference every accuracy and speed number is read against.
// Built only when third_party/poselib and third_party/eigen exist (tools/fetch_reference.sh).
//
//   poselib_ref file.jpop [--seed s] [--max-error px] [--confidence c] [--min-iter n] [--max-iter n]
//               [--threads n] [--repeat r] [-o results.txt]
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <PoseLib/robust.h>

#include "cli/problem_file.h"

namespace {
poselib::Camera camera(const jpo_camera& c) {
  static const int count[6] = {3, 4, 4, 5, 8, 8};
  std::vector<double> params(c.params, c.params + count[c.model]);
  return poselib::Camera(jpofile::model_name(c.model), params, 1600, 1200);
}
}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> opt;
  std::string file;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s[0] == '-' && i + 1 < argc)
      opt[s] = argv[++i];
    else
      file = s;
  }
  auto num = [&](const char* k, double d) { return opt.count(k) ? std::atof(opt[k].c_str()) : d; };
  if (file.empty()) {
    std::fprintf(stderr, "usage: poselib_ref file.jpop [options]\n");
    return 2;
  }
  const jpofile::ProblemSet set = jpofile::load(file);
  const size_t B = set.problems.size();
  const double max_error = num("--max-error", set.kind == 0 ? 1.0 : 4.0);
  poselib::RansacOptions ransac;
  ransac.seed = static_cast<unsigned long>(num("--seed", 0));
  ransac.min_iterations = static_cast<size_t>(num("--min-iter", 100));
  ransac.max_iterations = static_cast<size_t>(num("--max-iter", 10000));
  ransac.success_prob = num("--confidence", 0.9999);
  int threads = static_cast<int>(num("--threads", 0));
  if (threads <= 0) threads = static_cast<int>(std::thread::hardware_concurrency());
  const int repeat = std::max(1, static_cast<int>(num("--repeat", 1)));

  // PoseLib's own containers, built outside the timed part.
  std::vector<std::vector<poselib::Point2D>> xa(B), xb(B);
  std::vector<std::vector<poselib::Point3D>> X(B);
  for (size_t i = 0; i < B; ++i) {
    const jpofile::Problem& p = set.problems[i];
    for (uint32_t k = 0; k < p.n(); ++k) {
      xa[i].emplace_back(p.xy_a[2 * k], p.xy_a[2 * k + 1]);
      if (set.kind == 0)
        xb[i].emplace_back(p.xy_b[2 * k], p.xy_b[2 * k + 1]);
      else
        X[i].emplace_back(p.xyz[3 * k], p.xyz[3 * k + 1], p.xyz[3 * k + 2]);
    }
  }

  std::vector<jpo_pose_result> results(B);
  std::vector<uint32_t> offsets(1, 0);
  for (size_t i = 0; i < B; ++i) offsets.push_back(offsets.back() + set.problems[i].n());
  std::vector<uint8_t> inliers(offsets.back());
  double best = 1e300;
  for (int r = 0; r < repeat; ++r) {
    std::atomic<size_t> next(0);
    const auto t0 = std::chrono::steady_clock::now();
    auto work = [&] {
      for (;;) {
        const size_t i = next.fetch_add(1);
        if (i >= B) break;
        const jpofile::Problem& p = set.problems[i];
        jpo_pose_result& res = results[i];
        res = jpo_pose_result();
        std::vector<char> inl;
        poselib::CameraPose pose;
        poselib::RansacStats stats;
        if (set.kind == 0) {
          poselib::RelativePoseOptions o;
          o.ransac = ransac;
          o.ransac.seed = ransac.seed + i;
          o.max_error = max_error;
          stats = poselib::estimate_relative_pose(xa[i], xb[i], camera(p.cam_a), camera(p.cam_b), o, &pose, &inl);
        } else {
          poselib::AbsolutePoseOptions o;
          o.ransac = ransac;
          o.ransac.seed = ransac.seed + i;
          o.max_error = max_error;
          poselib::Image image;
          image.camera = camera(p.cam_a);
          stats = poselib::estimate_absolute_pose(xa[i], X[i], o, &image, &inl);
          pose = image.pose;
        }
        for (int k = 0; k < 4; ++k) res.pose[k] = pose.q(k);
        for (int k = 0; k < 3; ++k) res.pose[4 + k] = pose.t(k);
        res.score = stats.model_score;
        res.n_inliers = static_cast<uint32_t>(stats.num_inliers);
        res.iterations = static_cast<uint32_t>(stats.iterations);
        const uint32_t sample = set.kind == 0 ? 5 : 3;
        res.status = stats.num_inliers > sample ? JPO_OK : (p.n() < sample ? JPO_TOO_FEW_POINTS : JPO_NO_MODEL);
        uint32_t count = 0;
        for (size_t k = 0; k < inl.size() && k < p.n(); ++k) {
          inliers[offsets[i] + k] = inl[k] ? 1 : 0;
          count += inl[k] ? 1 : 0;
        }
        res.n_inliers = count;
      }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    best = std::min(best, dt);
  }
  if (opt.count("-o")) jpofile::save_results(opt["-o"], set.kind, results, offsets, inliers, best, "poselib");
  double iters = 0.0, inl = 0.0;
  for (size_t i = 0; i < B; ++i) {
    iters += results[i].iterations;
    inl += results[i].n_inliers;
  }
  std::printf("poselib: %zu problems in %.3f ms (%.1f /s)  mean iterations %.0f  mean inliers %.1f\n", B, best * 1e3,
              B / best, iters / B, inl / B);
  return 0;
}
