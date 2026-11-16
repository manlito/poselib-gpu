// jpo bench: end-to-end time (host buffers in, poses and masks out) over a grid of problems, for
// several backends side by side.
//
//   jpo bench [--kinds relpose,pnp] [--batches 1,32,300,3000] [--points 200,2000]
//             [--inliers 0.7,0.3,0.1,0] [--backends cpu,metal,auto] [--repeat 3]
//             [--min-iter n] [--max-iter n] [--cpu-iter n] [--gpu-round n] [--threads n]
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
