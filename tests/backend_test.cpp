// A GPU backend against the float CPU build of the same kernels.
//
//   backend_test metal|cuda|webgpu
//
// 1. Round level: the same items on both; the best hypothesis of each problem must be the same
//    (iteration and solution), or a near tie (scores within 1e-3 of each other), for at least nine
//    problems in ten.
// 2. Estimator level: the whole estimator on generated problems; the same problems solved.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "cli/synth.h"
#include "core/api_internal.h"
#include "core/geometry.h"
#include "core/pool.h"
#include "jpo/jpo.h"

static int backend_id(const std::string& n) {
  if (n == "metal") return JPO_BACKEND_METAL;
  if (n == "cuda") return JPO_BACKEND_CUDA;
  if (n == "webgpu") return JPO_BACKEND_WEBGPU;
  return -1;
}

static std::unique_ptr<jpo::Backend> make(int id, uint32_t seed, uint32_t round) {
  (void)seed;
  (void)round;
  switch (id) {
#ifdef JPO_HAVE_METAL
    case JPO_BACKEND_METAL: return jpo::make_metal_backend(seed, round);
#endif
#ifdef JPO_HAVE_CUDA
    case JPO_BACKEND_CUDA: return jpo::make_cuda_backend(seed, round);
#endif
#ifdef JPO_HAVE_WEBGPU
    case JPO_BACKEND_WEBGPU: return jpo::make_webgpu_backend(seed, round, nullptr, nullptr);
#endif
    default: return nullptr;
  }
}

int main(int argc, char** argv) {
  const std::string name = argc > 1 ? argv[1] : "";
  const int id = backend_id(name);
  int fail = 0;
  const uint32_t seed = 5, round = 2048;
  jpo::ThreadPool pool(0);
  std::unique_ptr<jpo::Backend> gpu;
  try {
    gpu = make(id, seed, round);
  } catch (const std::exception& e) {
    std::printf("backend_test: cannot create %s: %s\n", name.c_str(), e.what());
    return 1;
  }
  if (!gpu) {
    std::printf("usage: backend_test metal|cuda|webgpu (a backend of this build)\n");
    return 2;
  }
  std::unique_ptr<jpo::Backend> cpu = jpo::make_cpu_backend(&pool, seed, true);

  // ---- 1. rounds ----
  for (uint32_t kind = 0; kind < 2; ++kind) {
    jposynth::Spec spec;
    spec.kind = kind;
    spec.count = 24;
    spec.points = 700;
    spec.inlier_ratio = 0.4;
    spec.seed = 3 + kind;
    const jpofile::ProblemSet set = jposynth::make_set(spec);
    const jpo::Kind k = kind == 0 ? jpo::Kind::Relative : jpo::Kind::Absolute;
    const int stride = jpo::kind_stride(k);
    std::vector<double> pts;
    std::vector<jpo::RoundItem> items;
    uint32_t first = 0;
    for (size_t i = 0; i < set.problems.size(); ++i) {
      const jpofile::Problem& p = set.problems[i];
      // Problems of different sizes in one batch.
      const uint32_t n = p.n() - static_cast<uint32_t>(i) * 20;
      for (uint32_t j = 0; j < n; ++j) {
        double v[5] = {0, 0, 0, 0, 0};
        jpo::camera_unproject(p.cam_a, p.xy_a[2 * j], p.xy_a[2 * j + 1], &v[0], &v[1]);
        if (kind == 0) {
          jpo::camera_unproject(p.cam_b, p.xy_b[2 * j], p.xy_b[2 * j + 1], &v[2], &v[3]);
        } else {
          for (int d = 0; d < 3; ++d) v[2 + d] = p.xyz[3 * j + d] * 0.2;
        }
        pts.insert(pts.end(), v, v + stride);
      }
      jpo::RoundItem it;
      it.problem = static_cast<uint32_t>(i);
      it.first = first;
      it.n = n;
      it.iter0 = static_cast<uint32_t>(i) * 100;
      it.count = i % 5 == 4 ? 777 : round;  // some items are shorter than the round
      it.inv_sq_thr = 1.0e6;
      it.bound = i % 3 == 2 ? 0.9 * n : jpo::kNoScore;  // some with a bound that stops scoring early
      it.pre_k = i % 4 == 1 ? 64 : 0;  // some with the early rejection test
      it.pre_min = i % 4 == 1 ? 6 : 0;
      items.push_back(it);
      first += n;
    }
    cpu->set_points(k, pts.data(), first);
    gpu->set_points(k, pts.data(), first);
    std::vector<jpo::RoundResult> rc(items.size()), rg(items.size());
    cpu->run_round(items.data(), items.size(), rc.data());
    try {
      gpu->run_round(items.data(), items.size(), rg.data());
    } catch (const std::exception& e) {
      std::printf("FAIL %s round: %s\n", name.c_str(), e.what());
      return 1;
    }
    int same = 0, tie = 0, both_none = 0;
    for (size_t i = 0; i < items.size(); ++i) {
      const bool c_none = rc[i].sol < 0 || rc[i].score > items[i].bound;
      const bool g_none = rg[i].sol < 0 || rg[i].score > items[i].bound;
      if (c_none && g_none) {  // nothing beat the bound: what is reported then is not specified
        ++both_none;
        continue;
      }
      if (!c_none && !g_none && rc[i].iter == rg[i].iter && rc[i].sol == rg[i].sol) {
        double d = 0.0;
        for (int m = 0; m < 12; ++m) d = std::max(d, std::abs(rc[i].model[m] - rg[i].model[m]));
        if (d < 2e-3 && std::abs(rc[i].score - rg[i].score) < 1e-2 * rc[i].score + 1e-3) {  // float, different rounding
          ++same;
          continue;
        }
        std::printf("FAIL %s kind %u item %zu: same hypothesis, model differs by %.3g, score %.6f against %.6f\n",
                    name.c_str(), kind, i, d, rc[i].score, rg[i].score);
        ++fail;
      } else if (!c_none && !g_none && std::abs(rc[i].score - rg[i].score) < 1e-3 * rc[i].score) {
        ++tie;
      } else {
        // The two builds round differently (fused multiply-add, library functions), and a root of the
        // degree-10 polynomial found by one can be missed by the other: reported, allowed for a few.
        std::printf("note %s kind %u item %zu: cpu iter %u sol %d score %.6f, gpu iter %u sol %d score %.6f\n", name.c_str(),
                    kind, i, rc[i].iter, rc[i].sol, rc[i].score, rg[i].iter, rg[i].sol, rg[i].score);
      }
    }
    std::printf("%s rounds, %s: %d of %zu the same hypothesis, %d near ties, %d without a hypothesis under the bound\n",
                name.c_str(), kind ? "pnp" : "relpose", same, items.size(), tie, both_none);
    if (same + tie + both_none < static_cast<int>(items.size()) * 9 / 10) {
      std::printf("FAIL: too few identical hypotheses\n");
      ++fail;
    }
  }

  // ---- 2. estimators ----
  for (uint32_t kind = 0; kind < 2; ++kind) {
    for (double ratio : {0.6, 0.2}) {
      jposynth::Spec spec;
      spec.kind = kind;
      spec.count = 40;
      spec.points = 800;
      spec.inlier_ratio = ratio;
      spec.model = kind == 0 ? JPO_OPENCV : JPO_SIMPLE_RADIAL;
      spec.world_offset = 500.0;
      spec.seed = 40 + kind;
      const jpofile::ProblemSet set = jposynth::make_set(spec);
      const jpofile::Flat flat = jpofile::flatten(set);
      int ok[2] = {0, 0};
      double worst[2] = {0, 0};
      for (int b = 0; b < 2; ++b) {
        jpo_config cfg;
        jpo_config_init(&cfg);
        cfg.backend = b == 0 ? JPO_BACKEND_CPU : id;
        cfg.cpu_float = 1;
        jpo_ctx* ctx = jpo_create(&cfg);
        if (!ctx) {
          std::printf("FAIL jpo_create: %s\n", jpo_last_error());
          return 1;
        }
        std::vector<jpo_pose_result> res(set.problems.size());
        std::vector<uint8_t> inl(flat.offsets.back());
        int rcode;
        if (kind == 0) {
          const jpo_pairs p = flat.pairs();
          rcode = jpo_relative_pose(ctx, &p, nullptr, res.data(), inl.data());
        } else {
          const jpo_resections r = flat.resections();
          rcode = jpo_absolute_pose(ctx, &r, nullptr, res.data(), inl.data());
        }
        if (rcode != 0) {
          std::printf("FAIL estimator: %s\n", jpo_last_error());
          return 1;
        }
        for (size_t i = 0; i < res.size(); ++i) {
          double rot = 180, tr = 180;
          if (res[i].status == JPO_OK) jposynth::pose_errors(kind, res[i].pose, set.problems[i].gt_pose, &rot, &tr);
          const double e = kind == 0 ? std::max(rot, tr) : rot;
          if (e < (kind == 0 ? 5.0 : 1.0)) {
            ok[b] += 1;
            worst[b] = std::max(worst[b], e);
          }
        }
        jpo_destroy(ctx);
      }
      std::printf("%s estimator, %s, %.0f%% inliers: cpu-float solved %d/40, %s %d/40 (worst error %.3f, %.3f deg)\n",
                  name.c_str(), kind ? "pnp" : "relpose", 100 * ratio, ok[0], name.c_str(), ok[1], worst[0], worst[1]);
      if (ok[1] < ok[0] - 2) {
        std::printf("FAIL: the GPU backend solved fewer problems\n");
        ++fail;
      }
    }
  }
  // ---- 3. min_inlier_ratio, on the GPU alone and under auto ----
  // Candidates with no model stop at the cap; the good ones are solved as without the option.
  for (uint32_t kind = 0; kind < 2; ++kind) {
    jposynth::Spec spec;
    spec.kind = kind;
    spec.count = 12;
    spec.points = 600;
    spec.noise_px = kind == 0 ? 0.5 : 1.0;
    spec.inlier_ratio = 0.6;
    spec.seed = 90 + kind;
    jpofile::ProblemSet set = jposynth::make_set(spec);
    spec.inlier_ratio = 0.0;
    spec.seed += 10;
    const jpofile::ProblemSet none = jposynth::make_set(spec);
    set.problems.insert(set.problems.end(), none.problems.begin(), none.problems.end());
    const jpofile::Flat flat = jpofile::flatten(set);
    jpo_ransac_options opt[2];
    jpo_ransac_options_init(&opt[0]);
    opt[0].max_error_px = kind == 0 ? 1.0 : 4.0;
    opt[1] = opt[0];
    opt[1].min_inlier_ratio = 0.25;
    const uint32_t cap = static_cast<uint32_t>(
        std::ceil(std::log(1.0 - opt[1].confidence) / std::log1p(-std::pow(0.25, kind == 0 ? 5 : 3))));
    for (int route = 0; route < 2; ++route) {
      jpo_config cfg;
      jpo_config_init(&cfg);
      cfg.backend = route == 0 ? id : JPO_BACKEND_AUTO;
      jpo_ctx* ctx = jpo_create(&cfg);
      if (!ctx) {
        std::printf("FAIL jpo_create: %s\n", jpo_last_error());
        return 1;
      }
      int solved[2] = {0, 0}, stopped = 0;
      for (int o = 0; o < 2; ++o) {
        std::vector<jpo_pose_result> res(set.problems.size());
        std::vector<uint8_t> inl(flat.offsets.back());
        int rcode;
        if (kind == 0) {
          const jpo_pairs p = flat.pairs();
          rcode = jpo_relative_pose(ctx, &p, &opt[o], res.data(), inl.data());
        } else {
          const jpo_resections r = flat.resections();
          rcode = jpo_absolute_pose(ctx, &r, &opt[o], res.data(), inl.data());
        }
        if (rcode != 0) {
          std::printf("FAIL estimator: %s\n", jpo_last_error());
          return 1;
        }
        for (size_t i = 0; i < res.size(); ++i) {
          if (!std::isfinite(set.problems[i].gt_pose[0])) {
            if (o == 1) stopped += res[i].iterations == cap ? 1 : 0;
            continue;
          }
          double rot = 180, tr = 180;
          if (res[i].status == JPO_OK) jposynth::pose_errors(kind, res[i].pose, set.problems[i].gt_pose, &rot, &tr);
          solved[o] += (kind == 0 ? std::max(rot, tr) : rot) < (kind == 0 ? 5.0 : 1.0) ? 1 : 0;
        }
      }
      std::printf("%s min_inlier_ratio, %s, %s: good solved %d/12 off, %d/12 on; %d of 12 without a model stopped at %u\n",
                  name.c_str(), jpo_backend_name(ctx), kind ? "pnp" : "relpose", solved[0], solved[1], stopped, cap);
      if (solved[1] != solved[0] || stopped != 12) {
        std::printf("FAIL: min_inlier_ratio\n");
        ++fail;
      }
      jpo_destroy(ctx);
    }
  }

  // ---- 4. auto with a one-round budget (Metal's rule, PLAN.md "The auto rule") ----
  // min_inlier_ratio 0.25 on resections caps them at 585 iterations, one GPU round. Under auto, a round
  // of 32 candidates with 24 failing moves the failing ones to the GPU after 64 CPU iterations; 8
  // candidates, or 32 good ones, never could need 5,000 hypotheses there and stay on the CPU.
  if (id == JPO_BACKEND_METAL) {
    jpo_config cfg;
    jpo_config_init(&cfg);
    jpo_ctx* ctx = jpo_create(&cfg);
    jpo_ransac_options opt;
    jpo_ransac_options_init(&opt);
    opt.max_error_px = 4.0;
    opt.min_inlier_ratio = 0.25;
    struct Case {
      uint32_t good, failing;
      bool gpu;
    };
    for (const Case& c : {Case{8, 24, true}, Case{2, 6, false}, Case{32, 0, false}}) {
      jposynth::Spec spec;
      spec.kind = 1;
      spec.points = 600;
      spec.noise_px = 1.0;
      spec.count = c.good;
      spec.inlier_ratio = 0.6;
      spec.seed = 120;
      jpofile::ProblemSet set = jposynth::make_set(spec);
      spec.count = c.failing;
      spec.inlier_ratio = 0.0;
      spec.seed = 121;
      const jpofile::ProblemSet none = jposynth::make_set(spec);
      set.problems.insert(set.problems.end(), none.problems.begin(), none.problems.end());
      const jpofile::Flat flat = jpofile::flatten(set);
      const jpo_resections r = flat.resections();
      std::vector<jpo_pose_result> res(set.problems.size());
      if (jpo_absolute_pose(ctx, &r, &opt, res.data(), nullptr) != 0) {
        std::printf("FAIL estimator: %s\n", jpo_last_error());
        return 1;
      }
      const jpo::BatchStats& st = jpo::ctx_stats(ctx);
      int solved = 0, capped = 0;
      for (size_t i = 0; i < res.size(); ++i) {
        if (i >= c.good) {
          capped += res[i].iterations == 585 ? 1 : 0;
          continue;
        }
        double rot = 180, tr = 180;
        if (res[i].status == JPO_OK) jposynth::pose_errors(1, res[i].pose, set.problems[i].gt_pose, &rot, &tr);
        solved += rot < 1.0 ? 1 : 0;
      }
      std::printf("%s auto, %u good + %u failing capped resections: gpu hypotheses %llu, solved %d/%u, capped %d/%u\n",
                  name.c_str(), c.good, c.failing, static_cast<unsigned long long>(st.gpu_hypotheses), solved, c.good,
                  capped, c.failing);
      if ((st.gpu_hypotheses > 0) != c.gpu || solved != static_cast<int>(c.good) || capped != static_cast<int>(c.failing)) {
        std::printf("FAIL: one-round routing\n");
        ++fail;
      }
    }
    jpo_destroy(ctx);
  }

  std::printf(fail ? "backend_test %s: FAILED\n" : "backend_test %s: passed\n", name.c_str());
  return fail ? 1 : 0;
}
