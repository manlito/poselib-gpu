// jpo: generate problem files, run them, report accuracy against the ground truth in the file.
//
//   jpo gen relpose|pnp -o file.jpop [--count B] [--points N] [--inliers r] [--noise px] [--model NAME]
//                       [--planar] [--baseline b] [--rotation deg] [--world-offset d] [--seed s]
//   jpo run file.jpop [--backend cpu|metal|cuda|webgpu|auto] [--float] [--threads n] [--seed s]
//                     [--max-error px] [--confidence c] [--min-iter n] [--max-iter n] [--no-refine]
//                     [--repeat r] [-o results.txt] [--quiet]
//   jpo backends
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "cli/problem_file.h"
#include "cli/synth.h"
#include "core/api_internal.h"
#include "jpo/jpo.h"

namespace {

struct Args {
  std::vector<std::string> pos;
  std::map<std::string, std::string> opt;
  bool has(const std::string& k) const { return opt.count(k) != 0; }
  std::string str(const std::string& k, const std::string& d) const { return has(k) ? opt.at(k) : d; }
  double num(const std::string& k, double d) const { return has(k) ? std::atof(opt.at(k).c_str()) : d; }
};

Args parse(int argc, char** argv, int from) {
  static const char* flags[] = {"--planar", "--float", "--no-refine", "--quiet"};
  Args a;
  for (int i = from; i < argc; ++i) {
    std::string s = argv[i];
    if (s.size() > 1 && s[0] == '-' && !(s[1] >= '0' && s[1] <= '9')) {
      bool flag = false;
      for (const char* f : flags) flag = flag || s == f;
      if (flag || i + 1 >= argc)
        a.opt[s] = "1";
      else
        a.opt[s] = argv[++i];
    } else {
      a.pos.push_back(s);
    }
  }
  return a;
}

int backend_from_name(const std::string& n) {
  if (n == "auto") return JPO_BACKEND_AUTO;
  if (n == "cpu") return JPO_BACKEND_CPU;
  if (n == "metal") return JPO_BACKEND_METAL;
  if (n == "cuda") return JPO_BACKEND_CUDA;
  if (n == "webgpu") return JPO_BACKEND_WEBGPU;
  return -1;
}

double auc(std::vector<double> errors, double limit) {
  // Area under the recall curve up to `limit` degrees, as the pose AUC of the matching literature.
  std::sort(errors.begin(), errors.end());
  const double n = static_cast<double>(errors.size());
  double area = 0.0, prev_e = 0.0, prev_r = 0.0;
  for (size_t i = 0; i < errors.size(); ++i) {
    const double e = errors[i];
    if (!(e < limit)) break;
    const double r = static_cast<double>(i + 1) / n;
    area += (e - prev_e) * (prev_r + r) * 0.5;
    prev_e = e;
    prev_r = r;
  }
  area += (limit - prev_e) * prev_r;
  return area / limit;
}

int cmd_gen(const Args& a) {
  if (a.pos.size() < 1 || !a.has("-o")) {
    std::fprintf(stderr, "usage: jpo gen relpose|pnp -o file.jpop [options]\n");
    return 2;
  }
  jposynth::Spec spec;
  spec.kind = a.pos[0] == "pnp" ? 1 : 0;
  spec.count = static_cast<uint32_t>(a.num("--count", 1));
  spec.points = static_cast<uint32_t>(a.num("--points", 1000));
  spec.inlier_ratio = a.num("--inliers", 0.5);
  spec.noise_px = a.num("--noise", 0.5);
  spec.model = jpofile::model_from_name(a.str("--model", "PINHOLE"));
  if (spec.model < 0) {
    std::fprintf(stderr, "unknown camera model\n");
    return 2;
  }
  spec.planar = a.has("--planar");
  spec.baseline = a.num("--baseline", 0.15);
  spec.max_rotation_deg = a.num("--rotation", 25.0);
  spec.world_offset = a.num("--world-offset", 0.0);
  spec.seed = static_cast<uint64_t>(a.num("--seed", 1));
  jpofile::save(a.str("-o", ""), jposynth::make_set(spec));
  return 0;
}

int cmd_run(const Args& a) {
  if (a.pos.size() < 1) {
    std::fprintf(stderr, "usage: jpo run file.jpop [options]\n");
    return 2;
  }
  const jpofile::ProblemSet set = jpofile::load(a.pos[0]);
  const jpofile::Flat flat = jpofile::flatten(set);
  jpo_config cfg;
  jpo_config_init(&cfg);
  cfg.backend = backend_from_name(a.str("--backend", "auto"));
  if (cfg.backend < 0) {
    std::fprintf(stderr, "unknown backend\n");
    return 2;
  }
  cfg.seed = static_cast<uint32_t>(a.num("--seed", 0));
  cfg.threads = static_cast<int32_t>(a.num("--threads", 0));
  cfg.cpu_float = a.has("--float") ? 1 : 0;
  cfg.gpu_round = static_cast<uint32_t>(a.num("--gpu-round", 0));
  jpo_ctx* ctx = jpo_create(&cfg);
  if (!ctx) {
    std::fprintf(stderr, "jpo_create: %s\n", jpo_last_error());
    return 1;
  }
  if (a.has("--cpu-iter")) jpo::ctx_route(ctx).cpu_iterations = static_cast<uint32_t>(a.num("--cpu-iter", 0));
  jpo_ransac_options opt;
  jpo_ransac_options_init(&opt);
  opt.max_error_px = a.num("--max-error", set.kind == 0 ? 1.0 : 4.0);
  opt.confidence = a.num("--confidence", opt.confidence);
  opt.min_iterations = static_cast<uint32_t>(a.num("--min-iter", opt.min_iterations));
  opt.max_iterations = static_cast<uint32_t>(a.num("--max-iter", opt.max_iterations));
  opt.refine = a.has("--no-refine") ? 0 : 1;

  const size_t B = set.problems.size();
  std::vector<jpo_pose_result> results(B);
  std::vector<uint8_t> inliers(flat.offsets.back());
  const int repeat = std::max(1, static_cast<int>(a.num("--repeat", 1)));
  double best = 1e300;
  for (int r = 0; r < repeat; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    int rc;
    if (set.kind == 0) {
      const jpo_pairs pairs = flat.pairs();
      rc = jpo_relative_pose(ctx, &pairs, &opt, results.data(), inliers.data());
    } else {
      const jpo_resections res = flat.resections();
      rc = jpo_absolute_pose(ctx, &res, &opt, results.data(), inliers.data());
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (rc != 0) {
      std::fprintf(stderr, "error: %s\n", jpo_last_error());
      jpo_destroy(ctx);
      return 1;
    }
    best = std::min(best, dt);
  }
  if (a.has("-o"))
    jpofile::save_results(a.str("-o", ""), set.kind, results, flat.offsets, inliers, best, jpo_backend_name(ctx));

  if (!a.has("--quiet")) {
    std::vector<double> err;
    size_t ok = 0, with_gt = 0;
    double iters = 0.0, inl = 0.0;
    for (size_t i = 0; i < B; ++i) {
      iters += results[i].iterations;
      inl += results[i].n_inliers;
      ok += results[i].status == JPO_OK ? 1 : 0;
      if (!std::isfinite(set.problems[i].gt_pose[0])) continue;
      ++with_gt;
      double rot = 180.0, tr = 180.0;
      if (results[i].status == JPO_OK || results[i].status == JPO_DEGENERATE)
        jposynth::pose_errors(set.kind, results[i].pose, set.problems[i].gt_pose, &rot, &tr);
      err.push_back(set.kind == 0 ? std::max(rot, tr) : rot);
    }
    const jpo::BatchStats& st = jpo::ctx_stats(ctx);
    std::printf("%s: %zu problems in %.3f ms (%.1f /s)  ok %zu  mean iterations %.0f  mean inliers %.1f\n",
                jpo_backend_name(ctx), B, best * 1e3, B / best, ok, iters / B, inl / B);
    std::printf("  rounds %u  hypotheses cpu %llu gpu %llu\n", st.rounds, static_cast<unsigned long long>(st.cpu_hypotheses),
                static_cast<unsigned long long>(st.gpu_hypotheses));
    if (with_gt) {
      std::vector<double> sorted = err;
      std::sort(sorted.begin(), sorted.end());
      std::printf("  pose error (deg): median %.4f  AUC@5 %.4f  @10 %.4f  @20 %.4f  (%zu with ground truth)\n",
                  sorted[sorted.size() / 2], auc(err, 5.0), auc(err, 10.0), auc(err, 20.0), with_gt);
    }
  }
  jpo_destroy(ctx);
  return 0;
}

}  // namespace

int cmd_bench(int argc, char** argv);

int main(int argc, char** argv) {
  try {
    const std::string cmd = argc > 1 ? argv[1] : "";
    if (cmd == "gen") return cmd_gen(parse(argc, argv, 2));
    if (cmd == "run") return cmd_run(parse(argc, argv, 2));
    if (cmd == "bench") return cmd_bench(argc, argv);
    if (cmd == "backends") {
      const char* names[] = {"auto", "cpu", "metal", "cuda", "webgpu"};
      for (int b = 1; b <= 4; ++b)
        if (jpo_backend_available(b)) std::printf("%s\n", names[b]);
      return 0;
    }
    std::fprintf(stderr, "usage: jpo gen|run|bench|backends ...  (see cli/main.cpp)\n");
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "jpo: %s\n", e.what());
    return 1;
  }
}
