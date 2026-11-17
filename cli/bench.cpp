// jpo bench: end-to-end time (host buffers in, poses and masks out) over a grid of problems, for
// several backends side by side.
//
//   jpo bench [--kinds relpose,pnp] [--batches 1,32,300,3000] [--points 200,2000]
//             [--inliers 0.7,0.3,0.1,0] [--backends cpu,metal,auto] [--repeat 3]
//             [--min-iter n] [--max-iter n] [--min-inlier-ratio r] [--gpu-round n] [--threads n]
//             [--cpu-iter n] [--gpu-min-remaining n] [--gpu-min-total n] (auto only)
//             [--one-round-cpu-iter n] [--one-round-gpu-min-remaining n] [--one-round-gpu-min-total n]
//             [--save-dir dir] [--csv file]
//
// --save-dir writes each problem set as <dir>/<kind>_b<B>_n<N>_i<ratio>.jpop so that the PoseLib
// reference tool can be timed on the same data (tools/bench.py).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "cli/problem_file.h"
#include "cli/synth.h"
#include "core/api_internal.h"
#include "jpo/jpo.h"

namespace {

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(item);
  return out;
}

int backend_id(const std::string& n) {
  if (n == "auto") return JPO_BACKEND_AUTO;
  if (n == "cpu") return JPO_BACKEND_CPU;
  if (n == "metal") return JPO_BACKEND_METAL;
  if (n == "cuda") return JPO_BACKEND_CUDA;
  if (n == "webgpu") return JPO_BACKEND_WEBGPU;
  return -1;
}

}  // namespace

int cmd_bench(int argc, char** argv) {
  std::map<std::string, std::string> opt;
  for (int i = 2; i + 1 < argc; i += 2) opt[argv[i]] = argv[i + 1];
  auto str = [&](const char* k, const char* d) { return opt.count(k) ? opt[k] : std::string(d); };
  auto num = [&](const char* k, double d) { return opt.count(k) ? std::atof(opt[k].c_str()) : d; };

  const std::vector<std::string> kinds = split(str("--kinds", "relpose,pnp"));
  const std::vector<std::string> batches = split(str("--batches", "1,32,300,3000"));
  const std::vector<std::string> points = split(str("--points", "200,2000"));
  const std::vector<std::string> ratios = split(str("--inliers", "0.7,0.3,0.1,0"));
  std::vector<std::string> backends = split(str("--backends", "cpu"));
  const int repeat = std::max(1, static_cast<int>(num("--repeat", 3)));
  const std::string save_dir = str("--save-dir", "");
  std::FILE* csv = opt.count("--csv") ? std::fopen(opt["--csv"].c_str(), "w") : nullptr;
  if (csv) std::fprintf(csv, "kind,batch,points,inliers,backend,seconds,ok,mean_iterations,cpu_hypotheses,gpu_hypotheses,median_error_deg\n");

  std::vector<jpo_ctx*> ctxs;
  std::vector<std::string> names;
  for (const std::string& b : backends) {
    jpo_config cfg;
    jpo_config_init(&cfg);
    cfg.backend = backend_id(b);
    cfg.threads = static_cast<int32_t>(num("--threads", 0));
    cfg.gpu_round = static_cast<uint32_t>(num("--gpu-round", 0));
    jpo_ctx* ctx = cfg.backend < 0 ? nullptr : jpo_create(&cfg);
    if (!ctx) {
      std::fprintf(stderr, "bench: backend %s unavailable (%s)\n", b.c_str(), jpo_last_error());
      continue;
    }
    if (cfg.backend == JPO_BACKEND_AUTO) {  // try another rule than the built-in one
      jpo::Route& route = jpo::ctx_route(ctx);
      if (opt.count("--cpu-iter")) route.cpu_iterations = static_cast<uint32_t>(num("--cpu-iter", 0));
      if (opt.count("--gpu-min-remaining")) route.gpu_min_remaining = static_cast<uint32_t>(num("--gpu-min-remaining", 0));
      if (opt.count("--gpu-min-total")) route.gpu_min_total = static_cast<uint64_t>(num("--gpu-min-total", 0));
      if (opt.count("--one-round-cpu-iter")) route.one_round_cpu_iterations = static_cast<uint32_t>(num("--one-round-cpu-iter", 0));
      if (opt.count("--one-round-gpu-min-remaining")) route.one_round_gpu_min_remaining = static_cast<uint32_t>(num("--one-round-gpu-min-remaining", 0));
      if (opt.count("--one-round-gpu-min-total")) route.one_round_gpu_min_total = static_cast<uint64_t>(num("--one-round-gpu-min-total", 0));
    }
    ctxs.push_back(ctx);
    names.push_back(b);
  }

  std::printf("| kind | batch | points | inliers |");
  for (const std::string& n : names) std::printf(" %s ms |", n.c_str());
  for (size_t i = 1; i < names.size(); ++i) std::printf(" %s / %s |", names[0].c_str(), names[i].c_str());
  std::printf("\n|---|---|---|---|");
  for (size_t i = 0; i + 1 < 2 * names.size(); ++i) std::printf("---|");
  std::printf("\n");

  for (const std::string& kind : kinds) {
    for (const std::string& bs : batches) {
      for (const std::string& ps : points) {
        for (const std::string& rs : ratios) {
          jposynth::Spec spec;
          spec.kind = kind == "pnp" ? 1 : 0;
          spec.count = static_cast<uint32_t>(std::atoi(bs.c_str()));
          spec.points = static_cast<uint32_t>(std::atoi(ps.c_str()));
          spec.inlier_ratio = std::atof(rs.c_str());
          spec.noise_px = spec.kind == 0 ? 0.5 : 1.0;
          spec.seed = 7;
          const jpofile::ProblemSet set = jposynth::make_set(spec);
          if (!save_dir.empty())
            jpofile::save(save_dir + "/" + kind + "_b" + bs + "_n" + ps + "_i" + rs + ".jpop", set);
          const jpofile::Flat flat = jpofile::flatten(set);
          jpo_ransac_options o;
          jpo_ransac_options_init(&o);
          o.max_error_px = spec.kind == 0 ? 1.0 : 4.0;
          o.min_iterations = static_cast<uint32_t>(num("--min-iter", o.min_iterations));
          o.max_iterations = static_cast<uint32_t>(num("--max-iter", o.max_iterations));
          o.min_inlier_ratio = num("--min-inlier-ratio", 0.0);
          std::vector<jpo_pose_result> results(spec.count);
          std::vector<uint8_t> inl(flat.offsets.back());
          std::vector<double> secs;
          std::printf("| %s | %s | %s | %s |", kind.c_str(), bs.c_str(), ps.c_str(), rs.c_str());
          for (size_t c = 0; c < ctxs.size(); ++c) {
            double best = 1e300;
            for (int r = 0; r < repeat + 1; ++r) {  // the first run warms up (pipelines, buffers)
              const auto t0 = std::chrono::steady_clock::now();
              int rc;
              if (spec.kind == 0) {
                const jpo_pairs pairs = flat.pairs();
                rc = jpo_relative_pose(ctxs[c], &pairs, &o, results.data(), inl.data());
              } else {
                const jpo_resections res = flat.resections();
                rc = jpo_absolute_pose(ctxs[c], &res, &o, results.data(), inl.data());
              }
              const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
              if (rc != 0) {
                std::fprintf(stderr, "bench: %s\n", jpo_last_error());
                return 1;
              }
              if (r > 0 || repeat == 1) best = std::min(best, dt);
            }
            secs.push_back(best);
            std::printf(" %.3f |", best * 1e3);
            size_t ok = 0;
            double iters = 0.0;
            std::vector<double> errs;
            for (uint32_t i = 0; i < spec.count; ++i) {
              ok += results[i].status == JPO_OK ? 1 : 0;
              iters += results[i].iterations;
              if (std::isfinite(set.problems[i].gt_pose[0])) {
                double rot = 180.0, tr = 180.0;
                if (results[i].status == JPO_OK)
                  jposynth::pose_errors(spec.kind, results[i].pose, set.problems[i].gt_pose, &rot, &tr);
                errs.push_back(spec.kind == 0 ? std::max(rot, tr) : rot);
              }
            }
            std::sort(errs.begin(), errs.end());
            if (csv) {
              const jpo::BatchStats& st = jpo::ctx_stats(ctxs[c]);
              std::fprintf(csv, "%s,%s,%s,%s,%s,%.9g,%zu,%.1f,%llu,%llu,%.6g\n", kind.c_str(), bs.c_str(), ps.c_str(),
                           rs.c_str(), names[c].c_str(), best, ok, iters / spec.count,
                           static_cast<unsigned long long>(st.cpu_hypotheses),
                           static_cast<unsigned long long>(st.gpu_hypotheses), errs.empty() ? -1.0 : errs[errs.size() / 2]);
            }
          }
          for (size_t c = 1; c < secs.size(); ++c) std::printf(" %.2fx |", secs[0] / secs[c]);
          std::printf("\n");
          std::fflush(stdout);
        }
      }
    }
  }
  for (jpo_ctx* c : ctxs) jpo_destroy(c);
  if (csv) std::fclose(csv);
  return 0;
}

// jpo candidates: one registration round of an incremental reconstruction, K candidate images tried
// together. Each candidate is a resection of 300 to 1000 points; a good one has 50 to 70 % inliers, a
// failing one 0 to 10 %. Every cell is timed as one batch call and as K calls of one problem, with
// min_inlier_ratio off and on.
//
//   jpo candidates [--k 8,16,32] [--failing 0.25,0.5,0.75] [--backends cpu,metal,auto] [--ratio 0.25]
//                  [--rounds 5] [--repeat 3] [--threads n] [auto rule overrides as jpo bench]
//
// ms: mean over the rounds of the best of --repeat runs (after a warm-up). solved: good candidates
// whose rotation is within 1 degree of the truth, over all rounds. passed: failing candidates whose
// inlier count reaches ratio x points (a caller's test would accept them; should be 0).
int cmd_candidates(int argc, char** argv) {
  std::map<std::string, std::string> opt;
  for (int i = 2; i + 1 < argc; i += 2) opt[argv[i]] = argv[i + 1];
  auto str = [&](const char* k, const char* d) { return opt.count(k) ? opt[k] : std::string(d); };
  auto num = [&](const char* k, double d) { return opt.count(k) ? std::atof(opt[k].c_str()) : d; };
  const std::vector<std::string> ks = split(str("--k", "8,16,32"));
  const std::vector<std::string> shares = split(str("--failing", "0.25,0.5,0.75"));
  const std::vector<std::string> backends = split(str("--backends", "cpu,metal,auto"));
  const double ratio = num("--ratio", 0.25);
  const int rounds = std::max(1, static_cast<int>(num("--rounds", 5)));
  const int repeat = std::max(1, static_cast<int>(num("--repeat", 3)));

  std::vector<jpo_ctx*> ctxs;
  std::vector<std::string> names;
  for (const std::string& b : backends) {
    jpo_config cfg;
    jpo_config_init(&cfg);
    cfg.backend = backend_id(b);
    cfg.threads = static_cast<int32_t>(num("--threads", 0));
    jpo_ctx* ctx = cfg.backend < 0 ? nullptr : jpo_create(&cfg);
    if (!ctx) {
      std::fprintf(stderr, "candidates: backend %s unavailable (%s)\n", b.c_str(), jpo_last_error());
      continue;
    }
    if (cfg.backend == JPO_BACKEND_AUTO) {
      jpo::Route& route = jpo::ctx_route(ctx);
      if (opt.count("--cpu-iter")) route.cpu_iterations = static_cast<uint32_t>(num("--cpu-iter", 0));
      if (opt.count("--gpu-min-remaining")) route.gpu_min_remaining = static_cast<uint32_t>(num("--gpu-min-remaining", 0));
      if (opt.count("--gpu-min-total")) route.gpu_min_total = static_cast<uint64_t>(num("--gpu-min-total", 0));
      if (opt.count("--one-round-cpu-iter")) route.one_round_cpu_iterations = static_cast<uint32_t>(num("--one-round-cpu-iter", 0));
      if (opt.count("--one-round-gpu-min-remaining")) route.one_round_gpu_min_remaining = static_cast<uint32_t>(num("--one-round-gpu-min-remaining", 0));
      if (opt.count("--one-round-gpu-min-total")) route.one_round_gpu_min_total = static_cast<uint64_t>(num("--one-round-gpu-min-total", 0));
    }
    ctxs.push_back(ctx);
    names.push_back(b);
  }

  std::printf("| K | failing | backend | batch off ms | batch %.2f ms | one at a time off ms | one at a time %.2f ms |"
              " solved (batch off / on, single off / on) | mean iterations of failing (off / on) | passed |\n",
              ratio, ratio);
  std::printf("|---|---|---|---|---|---|---|---|---|---|\n");
  for (const std::string& kstr : ks) {
    for (const std::string& sstr : shares) {
      const uint32_t K = static_cast<uint32_t>(std::atoi(kstr.c_str()));
      const uint32_t n_fail = static_cast<uint32_t>(std::lround(std::atof(sstr.c_str()) * K));
      // The rounds of this cell: the same problems for every backend.
      std::vector<jpofile::ProblemSet> sets(rounds);
      for (int r = 0; r < rounds; ++r) {
        jposynth::Rng rng(1000 * K + 37 * n_fail + r);
        sets[r].kind = 1;
        std::vector<bool> failing(K, false);
        for (uint32_t i = 0; i < n_fail; ++i) failing[i] = true;
        for (uint32_t i = K; i > 1; --i) {
          const uint32_t j = static_cast<uint32_t>(rng.next() % i);
          const bool t = failing[i - 1];
          failing[i - 1] = failing[j];
          failing[j] = t;
        }
        for (uint32_t i = 0; i < K; ++i) {
          jposynth::Spec spec;
          spec.kind = 1;
          spec.count = 1;
          spec.points = 300 + static_cast<uint32_t>(rng.next() % 701);
          spec.inlier_ratio = failing[i] ? rng.uniform(0.0, 0.1) : rng.uniform(0.5, 0.7);
          spec.noise_px = 1.0;
          spec.seed = rng.next() % 1000000007ull;
          sets[r].problems.push_back(jposynth::make_set(spec).problems[0]);
        }
      }
      for (size_t c = 0; c < ctxs.size(); ++c) {
        double ms[2][2] = {{0, 0}, {0, 0}};  // [single][option]
        int solved[2][2] = {{0, 0}, {0, 0}}, good = 0, passed = 0;
        double fail_iters[2] = {0, 0};
        int fail_count = 0;
        for (int r = 0; r < rounds; ++r) {
          const jpofile::ProblemSet& set = sets[r];
          const jpofile::Flat flat = jpofile::flatten(set);
          std::vector<jpofile::Flat> one(K);
          for (uint32_t i = 0; i < K; ++i) {
            jpofile::ProblemSet s1;
            s1.kind = 1;
            s1.problems.push_back(set.problems[i]);
            one[i] = jpofile::flatten(s1);
          }
          for (int single = 0; single < 2; ++single) {
            for (int o = 0; o < 2; ++o) {
              jpo_ransac_options opts;
              jpo_ransac_options_init(&opts);
              opts.max_error_px = 4.0;
              opts.min_inlier_ratio = o ? ratio : 0.0;
              std::vector<jpo_pose_result> res(K);
              std::vector<uint8_t> inl(flat.offsets.back());
              double best = 1e300;
              for (int rep = 0; rep < repeat + 1; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                if (!single) {
                  const jpo_resections rs = flat.resections();
                  if (jpo_absolute_pose(ctxs[c], &rs, &opts, res.data(), inl.data()) != 0) {
                    std::fprintf(stderr, "candidates: %s\n", jpo_last_error());
                    return 1;
                  }
                } else {
                  for (uint32_t i = 0; i < K; ++i) {
                    const jpo_resections rs = one[i].resections();
                    if (jpo_absolute_pose(ctxs[c], &rs, &opts, &res[i], inl.data() + flat.offsets[i]) != 0) {
                      std::fprintf(stderr, "candidates: %s\n", jpo_last_error());
                      return 1;
                    }
                  }
                }
                const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (rep > 0 || repeat == 1) best = std::min(best, dt);
              }
              ms[single][o] += best * 1e3 / rounds;
              for (uint32_t i = 0; i < K; ++i) {
                const jpofile::Problem& p = set.problems[i];
                const bool is_good = p.gt_pose[0] == p.gt_pose[0] &&
                                     std::count(p.gt_inlier.begin(), p.gt_inlier.end(), 1) >= 0.4 * p.n();
                if (is_good) {
                  if (single == 0 && o == 0) ++good;
                  double rot = 180.0, tr = 0.0;
                  if (res[i].status == JPO_OK) jposynth::pose_errors(1, res[i].pose, p.gt_pose, &rot, &tr);
                  solved[single][o] += rot < 1.0 ? 1 : 0;
                } else {
                  if (single == 0) {
                    fail_iters[o] += res[i].iterations;
                    if (o == 0) ++fail_count;
                  }
                  passed += res[i].n_inliers >= ratio * p.n() ? 1 : 0;
                }
              }
            }
          }
        }
        std::printf("| %u | %s | %s | %.2f | %.2f | %.2f | %.2f | %d / %d, %d / %d of %d | %.0f / %.0f | %d |\n", K,
                    sstr.c_str(), jpo_backend_name(ctxs[c]), ms[0][0], ms[0][1], ms[1][0], ms[1][1], solved[0][0],
                    solved[0][1], solved[1][0], solved[1][1], good, fail_count ? fail_iters[0] / fail_count : 0.0,
                    fail_count ? fail_iters[1] / fail_count : 0.0, passed);
        std::fflush(stdout);
      }
    }
  }
  for (jpo_ctx* c : ctxs) jpo_destroy(c);
  return 0;
}
