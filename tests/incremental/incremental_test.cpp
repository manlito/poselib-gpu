// An incremental reconstruction on synthetic captures (PLAN.md, M2): jpo verifies the matches of every
// pair, gives the first two poses and registers each further image; just-ba builds the tracks,
// triangulates and runs the bundle adjustment. A test, not a product: it exists to show that jpo's two
// functions are all a mapper needs from this library, and to find what is missing.
//
// What the mapper had to write itself, beyond calling the two libraries, is marked "MAPPER:" below.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "cli/synth.h"
#include "core/geometry.h"
#include "jpo/jpo.h"

#include "jba/problem.h"
#include "jba/report.h"
#include "jba/solver.h"
#include "jba/tracks.h"
#include "jba/triangulation.h"

#include "scenes.h"

namespace {

using inc::kPi;
using inc::Scene;

bool g_verbose = false;
// How the first pair is chosen. 0: the rule of the test. The others are experiments run by hand
// (--first-pair inliers|degenerate) to see what the angle and the DEGENERATE flag are worth:
// 1: the OK pair with the most inliers, whatever its angle; 2: the DEGENERATE pair with the most inliers,
// as a mapper that ignored the flag could pick (the usual rule where the scene has no such pair).
int g_first_rule = 0;

// ---- thresholds of the mapper ------------------------------------------------------------------------
// COLMAP's defaults where it has one.
const uint32_t kMinPairInliers = 30;     // a pair's matches are kept from this many inliers
const double kMinPairInlierRatio = 0.25; // and this share of its matches
const uint32_t kMinFirstPairInliers = 100;
const uint32_t kMinPnpInliers = 15;      // COLMAP abs_pose_min_num_inliers is 30; these scenes are small
const double kMinPnpInlierRatio = 0.25;
const double kMaxReprojPx = 4.0;         // filter after each adjustment
const double kMinTriAngleDeg = 1.5;
const size_t kPnpBatch = 4;              // images tried in one jpo_absolute_pose call

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---- the map ------------------------------------------------------------------------------------------

struct Map {
  jpo_camera camera;
  jba::Keypoints keypoints;
  jba::Tracks tracks;
  jba::Cameras cams;                         // poses of every image; meaningful where registered
  std::vector<char> registered;
  std::vector<std::vector<int64_t>> image_obs;  // per image: its observations (indices into tracks.image)
  std::vector<int32_t> obs_track;            // per observation: its track
  std::vector<char> has_point;               // per track
  std::vector<double> X;                     // 3 per track
  std::vector<char> obs_in;                  // per observation: it is an inlier of its track's point
  size_t n_images() const { return registered.size(); }
};

// Reprojection error in pixels through jpo's camera model (the solver below uses just-ba's: if the two
// libraries disagreed on a model or a parameter order this test would not pass). Negative: not projectable.
double reprojection_error(const Map& m, int image, const double* xy, const double* X) {
  const double* pose = &m.cams.poses[7 * static_cast<size_t>(image)];
  double R[9], Y[3], u, v;
  jpo::quat_to_rotmat(pose, R);
  jposynth::apply(R, pose + 4, X, Y);
  if (!(Y[2] > 0.0) || !jpo::camera_project(m.camera, Y, &u, &v)) return -1.0;
  return std::hypot(u - xy[0], v - xy[1]);
}

// New points for tracks that have none, and new observations for points that exist, from the images
// registered so far. Run after every registration and every adjustment, so a track that failed with two
// views is tried again with three, and an observation dropped under a poor pose can come back.
void update_points(Map& m) {
  jba::TriangulationOptions tri;
  // just-ba's default (2 degrees) is for noisy tracker poses. Here poses come from PnP and adjustment,
  // so the angle is the reprojection threshold seen through the camera.
  tri.max_angle_error_deg = std::atan(kMaxReprojPx / jpo::camera_focal(m.camera)) * 180.0 / kPi;
  tri.min_tri_angle_deg = kMinTriAngleDeg;
  tri.min_views = 2;
  std::vector<jba::Ray> rays;
  std::vector<int64_t> idx;
  for (size_t t = 0; t < m.tracks.size(); ++t) {
    const int64_t b = m.tracks.begin[t], e = m.tracks.begin[t + 1];
    if (m.has_point[t]) {
      for (int64_t o = b; o < e; ++o) {
        const int image = m.tracks.image[o];
        if (m.obs_in[o] || !m.registered[image]) continue;
        const double err = reprojection_error(m, image, m.keypoints.at(image, m.tracks.keypoint[o]), &m.X[3 * t]);
        if (err >= 0.0 && err <= kMaxReprojPx) m.obs_in[o] = 1;
      }
      continue;
    }
    rays.clear();
    idx.clear();
    for (int64_t o = b; o < e; ++o) {
      const int image = m.tracks.image[o];
      jba::Ray r;
      if (!m.registered[image] || !jba::keypoint_ray(m.cams, image, m.keypoints.at(image, m.tracks.keypoint[o]), r)) continue;
      rays.push_back(r);
      idx.push_back(o);
    }
    if (rays.size() < 2) continue;
    const jba::TriangulationResult res = jba::triangulate_robust(rays, tri);
    if (!res.ok) continue;
    m.has_point[t] = 1;
    for (int d = 0; d < 3; ++d) m.X[3 * t + d] = res.X[d];
    for (size_t k = 0; k < idx.size(); ++k) m.obs_in[idx[k]] = res.inlier[k];
  }
}

struct Adjusted {
  jba::Problem problem;           // registered images only
  std::vector<int> image_of;      // problem image -> scene image
  std::vector<int> track_of;      // problem point -> track
  double mean_error_px = 0.0;     // of the observations kept by the filter
  size_t removed = 0;             // observations dropped by the filter
};

// Bundle adjustment of everything registered (just-ba), then the usual filter: observations with a large
// reprojection error or behind the camera are dropped, and points left with fewer than two views or a
// small triangulation angle are removed (their tracks can be triangulated again later).
Adjusted adjust(Map& m, int max_iterations) {
  Adjusted a;
  jba::Problem& p = a.problem;
  p.model = m.cams.model;
  p.intrinsics = m.cams.intrinsics;  // calibrated: every intrinsic stays fixed
  std::vector<int> dense(m.n_images(), -1);
  for (size_t i = 0; i < m.n_images(); ++i)
    if (m.registered[i]) {
      dense[i] = static_cast<int>(a.image_of.size());
      a.image_of.push_back(static_cast<int>(i));
      p.poses.insert(p.poses.end(), &m.cams.poses[7 * i], &m.cams.poses[7 * i] + 7);
    }
  std::vector<int64_t> obs_of;  // problem observation -> observation of the map
  for (size_t t = 0; t < m.tracks.size(); ++t) {
    if (!m.has_point[t]) continue;
    int n = 0;
    for (int64_t o = m.tracks.begin[t]; o < m.tracks.begin[t + 1]; ++o) n += m.obs_in[o] && m.registered[m.tracks.image[o]];
    if (n < 2) continue;
    const int j = static_cast<int>(a.track_of.size());
    a.track_of.push_back(static_cast<int>(t));
    p.points.insert(p.points.end(), &m.X[3 * t], &m.X[3 * t] + 3);
    for (int64_t o = m.tracks.begin[t]; o < m.tracks.begin[t + 1]; ++o) {
      const int image = m.tracks.image[o];
      if (!m.obs_in[o] || !m.registered[image]) continue;
      const double* xy = m.keypoints.at(image, m.tracks.keypoint[o]);
      p.obs_image.push_back(dense[image]);
      p.obs_point.push_back(j);
      p.obs_xy.push_back(xy[0]);
      p.obs_xy.push_back(xy[1]);
      obs_of.push_back(o);
    }
  }

  jba::SolverOptions so;
  so.loss = {jba::LossType::Cauchy, 2.0};  // just-ba's own choice in refine()
  so.backend = jba::BackendKind::Cpu;
  so.max_iterations = max_iterations;
  // The gauge is free (no priors): just-ba holds one pose and one translation coordinate (GaugeMode::Auto),
  // so the frame and scale set by the first pair carry through.
  jba::solve(p, so);

  for (size_t k = 0; k < a.image_of.size(); ++k)
    std::copy(&p.poses[7 * k], &p.poses[7 * k] + 7, &m.cams.poses[7 * static_cast<size_t>(a.image_of[k])]);
  for (size_t j = 0; j < a.track_of.size(); ++j)
    std::copy(&p.points[3 * j], &p.points[3 * j] + 3, &m.X[3 * static_cast<size_t>(a.track_of[j])]);

  const jba::ResidualReport rep = jba::evaluate(p, so);
  double sum = 0.0;
  size_t kept = 0;
  for (size_t k = 0; k < obs_of.size(); ++k) {
    if (rep.in_front[k] && rep.reprojection_error[k] <= kMaxReprojPx) {
      sum += rep.reprojection_error[k];
      ++kept;
    } else {
      m.obs_in[obs_of[k]] = 0;
      ++a.removed;
    }
  }
  a.mean_error_px = kept ? sum / kept : 0.0;
  std::vector<jba::Ray> rays;
  for (int t : a.track_of) {
    rays.clear();
    for (int64_t o = m.tracks.begin[t]; o < m.tracks.begin[t + 1]; ++o) {
      const int image = m.tracks.image[o];
      jba::Ray r;
      if (m.obs_in[o] && m.registered[image] && jba::keypoint_ray(m.cams, image, m.keypoints.at(image, m.tracks.keypoint[o]), r))
        rays.push_back(r);
    }
    if (rays.size() >= 2 && jba::triangulation_angle(rays.data(), rays.size(), &m.X[3 * t]) * 180.0 / kPi >= kMinTriAngleDeg)
      continue;
    m.has_point[t] = 0;
    for (int64_t o = m.tracks.begin[t]; o < m.tracks.begin[t + 1]; ++o) m.obs_in[o] = 0;
  }
  return a;
}

// ---- the mapper ---------------------------------------------------------------------------------------

struct Run {
  Map map;
  Adjusted final;
  std::vector<jpo_pose_result> pair_results;
  std::vector<uint8_t> pair_inliers;       // one byte per match, all pairs
  std::vector<uint32_t> pair_offsets;
  std::vector<char> pair_kept;
  int first_a = -1, first_b = -1;
  double first_angle_deg = 0.0;
  uint32_t first_inliers = 0;
  double first_angle_floor = 0.0;          // the level of the ladder the first pair came from
  int n_ok = 0, n_degenerate = 0, n_failed = 0, n_kept = 0, n_kept_degenerate = 0;
  int first_status = 0;
  int pnp_calls = 0, pnp_problems = 0, pnp_rejected = 0;
  int adjustments = 0;
  double t_verify = 0.0, t_tracks = 0.0, t_pnp = 0.0, t_points = 0.0, t_adjust = 0.0;
  std::string error;
};

// MAPPER: choosing the first pair. jpo returns what the choice needs (inlier count, median triangulation
// angle, the DEGENERATE flag) but not the choice. The rule here: among pairs with status OK and at least
// 100 inliers, the one with the most inliers whose median triangulation angle is at least 16 degrees
// (COLMAP's init_min_tri_angle); if there is none, the same with 8, 4, then 2 degrees. Many inliers keep
// the first adjustment well constrained; the angle keeps the first points well conditioned.
bool choose_first_pair(const Scene& scene, Run& run) {
  bool use_degenerate = false;
  if (g_first_rule == 2)
    for (const jpo_pose_result& r : run.pair_results) use_degenerate |= r.status == JPO_DEGENERATE && r.n_inliers >= kMinFirstPairInliers;
  const int rule = g_first_rule == 2 && !use_degenerate ? 0 : g_first_rule;
  for (const double floor_deg : {16.0, 8.0, 4.0, 2.0, 0.0}) {
    if (floor_deg == 0.0 && rule == 0) break;
    if (floor_deg > 0.0 && rule != 0) continue;
    int best = -1;
    for (size_t i = 0; i < scene.pairs.size(); ++i) {
      const jpo_pose_result& r = run.pair_results[i];
      if (r.status != (use_degenerate ? JPO_DEGENERATE : JPO_OK)) continue;
      if (r.n_inliers < kMinFirstPairInliers || r.median_tri_angle_deg < floor_deg) continue;
      if (best < 0 || r.n_inliers > run.pair_results[best].n_inliers) best = static_cast<int>(i);
    }
    if (best < 0) continue;
    run.first_a = scene.pairs[best].a;
    run.first_b = scene.pairs[best].b;
    run.first_angle_deg = run.pair_results[best].median_tri_angle_deg;
    run.first_inliers = run.pair_results[best].n_inliers;
    run.first_angle_floor = floor_deg;
    run.first_status = run.pair_results[best].status;
    // The first image is the origin; the second is the relative pose as returned (camera a to camera b
    // is then world to camera b). Its translation has length 1, which fixes the scale of the map.
    double* pa = &run.map.cams.poses[7 * static_cast<size_t>(run.first_a)];
    const double identity[7] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::copy(identity, identity + 7, pa);
    std::copy(run.pair_results[best].pose, run.pair_results[best].pose + 7, &run.map.cams.poses[7 * static_cast<size_t>(run.first_b)]);
    run.map.registered[run.first_a] = run.map.registered[run.first_b] = 1;
    return true;
  }
  return false;
}

Run reconstruct(const Scene& scene, jpo_ctx* ctx) {
  Run run;
  Map& m = run.map;
  const size_t n_img = scene.n_images();
  m.camera = scene.camera;
  m.registered.assign(n_img, 0);

  // -- 1. Verify every pair: one call for the whole capture.
  auto t0 = std::chrono::steady_clock::now();
  std::vector<uint32_t>& offsets = run.pair_offsets;
  std::vector<double> xy_a, xy_b;
  std::vector<uint32_t> cam_index(scene.pairs.size(), 0);
  offsets.push_back(0);
  for (const inc::Pair& p : scene.pairs) {
    for (size_t k = 0; k < p.ka.size(); ++k) {
      xy_a.insert(xy_a.end(), &scene.xy[p.a][2 * p.ka[k]], &scene.xy[p.a][2 * p.ka[k]] + 2);
      xy_b.insert(xy_b.end(), &scene.xy[p.b][2 * p.kb[k]], &scene.xy[p.b][2 * p.kb[k]] + 2);
    }
    offsets.push_back(static_cast<uint32_t>(xy_a.size() / 2));
  }
  jpo_pairs pairs;
  std::memset(&pairs, 0, sizeof(pairs));
  pairs.n_pairs = static_cast<uint32_t>(scene.pairs.size());
  pairs.n_cameras = 1;
  pairs.offsets = offsets.data();
  pairs.xy_a = xy_a.data();
  pairs.xy_b = xy_b.data();
  pairs.cameras = &scene.camera;
  pairs.camera_a = cam_index.data();
  pairs.camera_b = cam_index.data();
  run.pair_results.resize(scene.pairs.size());
  run.pair_inliers.assign(offsets.back(), 0);
  if (jpo_relative_pose(ctx, &pairs, nullptr, run.pair_results.data(), run.pair_inliers.data()) != 0) {
    run.error = std::string("jpo_relative_pose: ") + jpo_last_error();
    return run;
  }
  run.t_verify = seconds_since(t0);

  // The matches that go on to track building: the inliers of every pair with enough of them.
  // A DEGENERATE pair (no observable baseline) is kept too: its inliers are still consistent with one
  // rotation, they are good matches, and they connect images that the pairs with a baseline may not.
  std::vector<int32_t> verified;
  run.pair_kept.assign(scene.pairs.size(), 0);
  for (size_t i = 0; i < scene.pairs.size(); ++i) {
    const inc::Pair& p = scene.pairs[i];
    const jpo_pose_result& r = run.pair_results[i];
    if (r.status == JPO_OK) ++run.n_ok;
    else if (r.status == JPO_DEGENERATE) ++run.n_degenerate;
    else ++run.n_failed;
    if (g_verbose && r.status == JPO_DEGENERATE)
      std::printf("  pair %d-%d DEGENERATE: %u inliers of %zu matches, median angle %.2f deg\n", p.a, p.b, r.n_inliers,
                  p.ka.size(), r.median_tri_angle_deg);
    if (r.status != JPO_OK && r.status != JPO_DEGENERATE) continue;
    if (r.n_inliers < kMinPairInliers || r.n_inliers < kMinPairInlierRatio * p.ka.size()) continue;
    run.pair_kept[i] = 1;
    ++run.n_kept;
    if (r.status == JPO_DEGENERATE) ++run.n_kept_degenerate;
    for (size_t k = 0; k < p.ka.size(); ++k)
      if (run.pair_inliers[offsets[i] + k]) verified.insert(verified.end(), {p.a, p.ka[k], p.b, p.kb[k]});
  }

  // -- 2. Tracks (just-ba). No poses yet, so only its combinatorial stage runs: a component with two
  // keypoints in one image is dropped.
  t0 = std::chrono::steady_clock::now();
  m.keypoints.begin.push_back(0);
  for (size_t i = 0; i < n_img; ++i) {
    m.keypoints.xy.insert(m.keypoints.xy.end(), scene.xy[i].begin(), scene.xy[i].end());
    m.keypoints.begin.push_back(static_cast<int64_t>(m.keypoints.xy.size() / 2));
  }
  jba::TrackOptions track_options;
  track_options.min_views = 2;
  track_options.conflict_policy = jba::ConflictPolicy::Drop;
  m.tracks = jba::build_tracks(m.keypoints, verified, track_options);
  run.t_tracks = seconds_since(t0);
  const size_t n_obs = m.tracks.image.size();
  m.image_obs.assign(n_img, {});
  m.obs_track.assign(n_obs, 0);
  for (size_t t = 0; t < m.tracks.size(); ++t)
    for (int64_t o = m.tracks.begin[t]; o < m.tracks.begin[t + 1]; ++o) {
      m.obs_track[o] = static_cast<int32_t>(t);
      m.image_obs[m.tracks.image[o]].push_back(o);
    }
  m.has_point.assign(m.tracks.size(), 0);
  m.X.assign(3 * m.tracks.size(), 0.0);
  m.obs_in.assign(n_obs, 0);

  // The same camera for just-ba: model ids and parameter order are COLMAP's in both libraries.
  m.cams.model = static_cast<jba::CameraModel>(scene.camera.model);
  m.cams.intrinsics.assign(scene.camera.params, scene.camera.params + jpo::camera_param_count(scene.camera.model));
  m.cams.poses.assign(7 * n_img, 0.0);
  for (size_t i = 0; i < n_img; ++i) m.cams.poses[7 * i] = 1.0;

  // -- 3. First pair, first points, first adjustment.
  if (!choose_first_pair(scene, run)) {
    run.error = "no pair is fit to start from";
    return run;
  }
  t0 = std::chrono::steady_clock::now();
  update_points(m);
  run.t_points += seconds_since(t0);
  t0 = std::chrono::steady_clock::now();
  run.final = adjust(m, 25);
  ++run.adjustments;
  run.t_adjust += seconds_since(t0);
  if (g_verbose)
    std::printf("  first pair %d-%d: %u inliers, median angle %.1f deg, %zu points\n", run.first_a, run.first_b,
                run.first_inliers, run.first_angle_deg, run.final.track_of.size());

  // -- 4. Register the remaining images.
  // MAPPER: choosing the next images. Those that see the most triangulated points go first; up to four
  // are resected in one call (all against the same map), which is how the batch is meant to be used.
  std::vector<int> failed_with(n_img, -1);  // points an image saw when its resection last failed
  for (;;) {
    t0 = std::chrono::steady_clock::now();
    update_points(m);
    run.t_points += seconds_since(t0);

    std::vector<std::pair<int, int>> seen;  // (points seen, image)
    for (size_t i = 0; i < n_img; ++i) {
      if (m.registered[i]) continue;
      int n = 0;
      for (int64_t o : m.image_obs[i]) n += m.has_point[m.obs_track[o]];
      // An image that failed is tried again only once it sees more points than it did then.
      if (n >= static_cast<int>(kMinPnpInliers) && n > failed_with[i]) seen.emplace_back(n, static_cast<int>(i));
    }
    if (seen.empty()) break;
    std::sort(seen.begin(), seen.end(), [](const auto& x, const auto& y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
    // Only images that see at least half as many points as the best one: a weakly connected image is
    // better registered later, against a larger map.
    size_t n_cand = 1;
    while (n_cand < seen.size() && n_cand < kPnpBatch && 2 * seen[n_cand].first >= seen[0].first) ++n_cand;

    t0 = std::chrono::steady_clock::now();
    std::vector<uint32_t> off = {0}, cam(n_cand, 0);
    std::vector<double> xy, xyz;
    for (size_t c = 0; c < n_cand; ++c) {
      const int image = seen[c].second;
      for (int64_t o : m.image_obs[image]) {
        const int32_t t = m.obs_track[o];
        if (!m.has_point[t]) continue;
        const double* k = m.keypoints.at(image, m.tracks.keypoint[o]);
        xy.insert(xy.end(), k, k + 2);
        xyz.insert(xyz.end(), &m.X[3 * t], &m.X[3 * t] + 3);
      }
      off.push_back(static_cast<uint32_t>(xy.size() / 2));
    }
    jpo_resections res;
    std::memset(&res, 0, sizeof(res));
    res.n_images = static_cast<uint32_t>(n_cand);
    res.n_cameras = 1;
    res.offsets = off.data();
    res.xy = xy.data();
    res.xyz = xyz.data();
    res.cameras = &scene.camera;
    res.camera = cam.data();
    std::vector<jpo_pose_result> out(n_cand);
    if (jpo_absolute_pose(ctx, &res, nullptr, out.data(), nullptr) != 0) {
      run.error = std::string("jpo_absolute_pose: ") + jpo_last_error();
      return run;
    }
    ++run.pnp_calls;
    run.pnp_problems += static_cast<int>(n_cand);
    run.t_pnp += seconds_since(t0);

    int added = 0;
    for (size_t c = 0; c < n_cand; ++c) {
      const int image = seen[c].second;
      const uint32_t n = off[c + 1] - off[c];
      const bool ok = out[c].status == JPO_OK && out[c].n_inliers >= kMinPnpInliers && out[c].n_inliers >= kMinPnpInlierRatio * n;
      if (g_verbose)
        std::printf("  image %2d: %4u points seen, %4u inliers, status %d -> %s\n", image, n, out[c].n_inliers,
                    out[c].status, ok ? "registered" : "rejected");
      if (!ok) {
        failed_with[image] = seen[c].first;
        ++run.pnp_rejected;
        continue;
      }
      std::copy(out[c].pose, out[c].pose + 7, &m.cams.poses[7 * static_cast<size_t>(image)]);
      m.registered[image] = 1;
      ++added;
    }
    if (!added) continue;  // the rejected images are now barred until the map grows; others get their turn

    t0 = std::chrono::steady_clock::now();
    update_points(m);
    run.t_points += seconds_since(t0);
    // Every registered image and point is adjusted each time: at this size a local window buys nothing.
    t0 = std::chrono::steady_clock::now();
    run.final = adjust(m, 25);
    ++run.adjustments;
    run.t_adjust += seconds_since(t0);
  }

  // -- 5. Final adjustment: complete the tracks, adjust, filter, and adjust what is left.
  for (int pass = 0; pass < 2; ++pass) {
    t0 = std::chrono::steady_clock::now();
    if (pass == 0) update_points(m);
    run.t_points += seconds_since(t0);
    t0 = std::chrono::steady_clock::now();
    run.final = adjust(m, 50);
    ++run.adjustments;
    run.t_adjust += seconds_since(t0);
  }
  return run;
}

// ---- comparison with the ground truth -----------------------------------------------------------------

// Largest eigenvector of a symmetric 4 x 4 matrix (cyclic Jacobi).
void largest_eigenvector4(double A[4][4], double v[4]) {
  double V[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  for (int sweep = 0; sweep < 60; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < 4; ++p)
      for (int q = p + 1; q < 4; ++q) off += A[p][q] * A[p][q];
    if (off < 1e-30) break;
    for (int p = 0; p < 4; ++p)
      for (int q = p + 1; q < 4; ++q) {
        if (std::abs(A[p][q]) < 1e-300) continue;
        const double theta = 0.5 * (A[q][q] - A[p][p]) / A[p][q];
        const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k < 4; ++k) {
          const double akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 4; ++k) {
          const double apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 4; ++k) {
          const double vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
  }
  int best = 0;
  for (int k = 1; k < 4; ++k)
    if (A[k][k] > A[best][best]) best = k;
  for (int k = 0; k < 4; ++k) v[k] = V[k][best];
}

// The similarity to ~ s R from + t that is best in least squares (Umeyama's solution; the rotation is
// found as Horn's quaternion, which is the same optimum and needs no SVD).
void fit_similarity(const std::vector<double>& from, const std::vector<double>& to, double R[9], double t[3], double* s) {
  const size_t n = from.size() / 3;
  double ma[3] = {0, 0, 0}, mb[3] = {0, 0, 0};
  for (size_t i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d) {
      ma[d] += from[3 * i + d] / n;
      mb[d] += to[3 * i + d] / n;
    }
  double S[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, var = 0.0;
  for (size_t i = 0; i < n; ++i)
    for (int r = 0; r < 3; ++r) {
      const double a = from[3 * i + r] - ma[r];
      var += a * a;
      for (int c = 0; c < 3; ++c) S[r][c] += a * (to[3 * i + c] - mb[c]);
    }
  double N[4][4] = {
      {S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]},
      {S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]},
      {S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]},
      {S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]}};
  double q[4];
  largest_eigenvector4(N, q);
  jpo::quat_to_rotmat(q, R);
  double num = 0.0;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) num += R[3 * c + r] * S[r][c];  // trace(R S)
  *s = var > 0.0 ? num / var : 1.0;
  for (int r = 0; r < 3; ++r) t[r] = mb[r] - *s * (R[3 * r] * ma[0] + R[3 * r + 1] * ma[1] + R[3 * r + 2] * ma[2]);
}

struct Metrics {
  size_t images = 0, registered = 0, points = 0, observations = 0;
  double rot_median = 0.0, rot_max = 0.0;       // degrees
  double centre_median = 0.0, centre_max = 0.0; // percent of the scene extent
  double centre_median_cam = 0.0;               // percent of the extent of the camera centres alone
  double point_median = 0.0;                    // percent of the scene extent
  double mean_error_px = 0.0;
  double extent = 0.0, cam_extent = 0.0;
  // The pair verification, against the ground truth.
  int false_pairs_kept = 0, true_pairs_lost = 0;
  uint32_t max_false_inliers = 0;  // the most inliers jpo found in a pair with no true overlap
  int false_pairs_ok = 0;          // such pairs that came back with status OK
  double mask_precision = 0.0, mask_recall = 0.0;
  double pair_rot_median = 0.0, pair_dir_median = 0.0;  // degrees, over kept pairs with status OK
  bool pass = false;
  std::string why;
};

double bbox_diagonal(const std::vector<double>& pts) {
  double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
  for (size_t i = 0; i < pts.size(); ++i) {
    lo[i % 3] = std::min(lo[i % 3], pts[i]);
    hi[i % 3] = std::max(hi[i % 3], pts[i]);
  }
  return std::sqrt((hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) + (hi[2] - lo[2]) * (hi[2] - lo[2]));
}

Metrics compare(const Scene& scene, const Run& run) {
  Metrics mt;
  const Map& m = run.map;
  const size_t n_img = scene.n_images();
  mt.images = n_img;
  mt.points = run.final.track_of.size();
  mt.observations = run.final.problem.num_observations();
  mt.mean_error_px = run.final.mean_error_px;

  // The scene extent: the box around the cameras and every point some image saw.
  std::vector<double> all, cam_centres;
  std::vector<char> seen(scene.points.size() / 3, 0);
  for (size_t i = 0; i < n_img; ++i) {
    double C[3];
    inc::camera_centre(&scene.poses[7 * i], C);
    cam_centres.insert(cam_centres.end(), C, C + 3);
    for (int j : scene.kp_point[i]) seen[j] = 1;
  }
  all = cam_centres;
  for (size_t j = 0; j < seen.size(); ++j)
    if (seen[j]) all.insert(all.end(), &scene.points[3 * j], &scene.points[3 * j] + 3);
  mt.extent = bbox_diagonal(all);
  mt.cam_extent = bbox_diagonal(cam_centres);

  // Pair verification against the truth.
  size_t kept_matches = 0, kept_correct = 0, correct_total = 0;
  std::vector<double> pair_rot, pair_dir;
  for (size_t i = 0; i < scene.pairs.size(); ++i) {
    const inc::Pair& p = scene.pairs[i];
    if (p.shared == 0 && i < run.pair_results.size()) {
      mt.max_false_inliers = std::max(mt.max_false_inliers, run.pair_results[i].n_inliers);
      mt.false_pairs_ok += run.pair_results[i].status == JPO_OK;
    }
    if (!run.pair_kept.empty() && run.pair_kept[i]) {
      if (p.shared == 0) ++mt.false_pairs_kept;
      for (size_t k = 0; k < p.ka.size(); ++k) {
        correct_total += p.correct[k];
        if (!run.pair_inliers[run.pair_offsets[i] + k]) continue;
        ++kept_matches;
        kept_correct += p.correct[k];
      }
      if (run.pair_results[i].status == JPO_OK) {
        double Ra[9], Rb[9], Rat[9], Rab[9], gt[7];
        jpo::quat_to_rotmat(&scene.poses[7 * p.a], Ra);
        jpo::quat_to_rotmat(&scene.poses[7 * p.b], Rb);
        for (int r = 0; r < 3; ++r)
          for (int c = 0; c < 3; ++c) Rat[3 * r + c] = Ra[3 * c + r];
        jpo::mat3_mul(Rb, Rat, Rab);
        jpo::rotmat_to_quat(Rab, gt);
        const double* ta = &scene.poses[7 * p.a + 4];
        const double* tb = &scene.poses[7 * p.b + 4];
        for (int r = 0; r < 3; ++r) gt[4 + r] = tb[r] - (Rab[3 * r] * ta[0] + Rab[3 * r + 1] * ta[1] + Rab[3 * r + 2] * ta[2]);
        double rot, dir;
        jposynth::pose_errors(0, run.pair_results[i].pose, gt, &rot, &dir);
        pair_rot.push_back(rot);
        pair_dir.push_back(dir);
      }
    } else if (p.shared >= static_cast<int>(2 * kMinPairInliers)) {
      ++mt.true_pairs_lost;  // a pair with plenty of true matches that verification threw away
    }
  }
  mt.mask_precision = kept_matches ? static_cast<double>(kept_correct) / kept_matches : 0.0;
  mt.mask_recall = correct_total ? static_cast<double>(kept_correct) / correct_total : 0.0;
  mt.pair_rot_median = median(pair_rot);
  mt.pair_dir_median = median(pair_dir);

  std::vector<double> est, gt;
  std::vector<int> images;
  for (size_t i = 0; i < n_img && i < m.registered.size(); ++i) {
    if (!m.registered[i]) continue;
    double C[3];
    inc::camera_centre(&m.cams.poses[7 * i], C);
    est.insert(est.end(), C, C + 3);
    gt.insert(gt.end(), &cam_centres[3 * i], &cam_centres[3 * i] + 3);
    images.push_back(static_cast<int>(i));
  }
  mt.registered = images.size();
  if (!run.error.empty()) {
    mt.why = run.error;
    return mt;
  }
  if (images.size() < 3) {
    mt.why = "fewer than three images registered";
    return mt;
  }
  // The alignment uses the camera centres only, so its own rotation is as uncertain as the centre errors
  // over the spread of the cameras (millimetres over a metre in the grid scene: a few hundredths of a
  // degree). That shared offset, not the poses, is most of the rotation error reported below.
  double Rs[9], ts[3], s;
  fit_similarity(est, gt, Rs, ts, &s);
  std::vector<double> rot_err, centre_err;
  for (size_t k = 0; k < images.size(); ++k) {
    double c[3];
    for (int r = 0; r < 3; ++r)
      c[r] = s * (Rs[3 * r] * est[3 * k] + Rs[3 * r + 1] * est[3 * k + 1] + Rs[3 * r + 2] * est[3 * k + 2]) + ts[r];
    centre_err.push_back(std::hypot(c[0] - gt[3 * k], c[1] - gt[3 * k + 1], c[2] - gt[3 * k + 2]));
    // World to camera in the true frame: R_est Rs'. Compared with the true rotation.
    double Re[9], Rg[9], Rst[9], Ra[9], Rgt[9], D[9];
    jpo::quat_to_rotmat(&m.cams.poses[7 * static_cast<size_t>(images[k])], Re);
    jpo::quat_to_rotmat(&scene.poses[7 * static_cast<size_t>(images[k])], Rg);
    for (int r = 0; r < 3; ++r)
      for (int c2 = 0; c2 < 3; ++c2) {
        Rst[3 * r + c2] = Rs[3 * c2 + r];
        Rgt[3 * r + c2] = Rg[3 * c2 + r];
      }
    jpo::mat3_mul(Re, Rst, Ra);
    jpo::mat3_mul(Ra, Rgt, D);
    rot_err.push_back(std::acos(std::min(1.0, std::max(-1.0, 0.5 * (D[0] + D[4] + D[8] - 1.0)))) * 180.0 / kPi);
  }
  mt.rot_median = median(rot_err);
  mt.rot_max = *std::max_element(rot_err.begin(), rot_err.end());
  mt.centre_median = 100.0 * median(centre_err) / mt.extent;
  mt.centre_max = 100.0 * *std::max_element(centre_err.begin(), centre_err.end()) / mt.extent;
  mt.centre_median_cam = 100.0 * median(centre_err) / mt.cam_extent;

  // Points: a track is the scene point most of its keypoints come from.
  std::vector<double> point_err;
  for (int t : run.final.track_of) {
    std::vector<int> ids;
    for (int64_t o = m.tracks.begin[t]; o < m.tracks.begin[t + 1]; ++o)
      if (m.obs_in[o]) ids.push_back(scene.kp_point[m.tracks.image[o]][m.tracks.keypoint[o]]);
    if (ids.empty()) continue;
    std::sort(ids.begin(), ids.end());
    const int id = ids[ids.size() / 2];
    const double* X = &m.X[3 * static_cast<size_t>(t)];
    double c[3];
    for (int r = 0; r < 3; ++r) c[r] = s * (Rs[3 * r] * X[0] + Rs[3 * r + 1] * X[1] + Rs[3 * r + 2] * X[2]) + ts[r];
    const double* G = &scene.points[3 * static_cast<size_t>(id)];
    point_err.push_back(std::hypot(c[0] - G[0], c[1] - G[1], c[2] - G[2]));
  }
  mt.point_median = 100.0 * median(point_err) / mt.extent;

  // The verdict.
  if (mt.registered != mt.images) mt.why = "not every image registered";
  else if (!(mt.rot_median < 0.5)) mt.why = "median rotation error";
  else if (!(mt.centre_median < 1.0)) mt.why = "median centre error";
  else if (!(mt.mean_error_px < 1.5)) mt.why = "mean reprojection error";
  else mt.pass = true;
  return mt;
}

}  // namespace

int main(int argc, char** argv) {
  uint64_t seed = 0;  // 0: the scenes of the test; others are for trying the mapper on more draws by hand
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-v") == 0) g_verbose = true;
    else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
    else if (std::strcmp(argv[i], "--first-pair") == 0 && i + 1 < argc) {
      const char* v = argv[++i];
      g_first_rule = std::strcmp(v, "inliers") == 0 ? 1 : std::strcmp(v, "degenerate") == 0 ? 2 : 0;
    } else {
      std::printf("usage: incremental_test [-v] [--seed n] [--first-pair angle|inliers|degenerate]\n");
      return 2;
    }
  }

  jpo_config config;
  jpo_config_init(&config);
  config.backend = JPO_BACKEND_CPU;
  config.seed = 7 + static_cast<uint32_t>(seed);
  jpo_ctx* ctx = jpo_create(&config);
  if (!ctx) {
    std::printf("jpo_create failed: %s\n", jpo_last_error());
    return 2;
  }

  struct Row {
    Scene scene;
    Run run;
    Metrics mt;
    double seconds = 0.0;
  };
  std::vector<Row> rows;
  int failed = 0;
  try {
    for (int k = 0; k < 3; ++k) {
      Row row;
      row.scene = k == 0 ? inc::make_orbit(seed) : k == 1 ? inc::make_wall(seed) : inc::make_grid(seed);
      if (g_verbose) std::printf("%s\n", row.scene.name.c_str());
      const auto t0 = std::chrono::steady_clock::now();
      row.run = reconstruct(row.scene, ctx);
      row.seconds = seconds_since(t0);
      row.mt = compare(row.scene, row.run);
      if (!row.mt.pass) ++failed;
      rows.push_back(std::move(row));
    }
  } catch (const std::exception& e) {
    std::printf("exception: %s\n", e.what());
    jpo_destroy(ctx);
    return 2;
  }
  jpo_destroy(ctx);

  std::printf("Pair verification (jpo_relative_pose, one call per scene; thresholds: defaults)\n");
  std::printf("%-6s %6s %5s %5s %6s %5s %8s %8s %8s %8s %6s %7s %8s %8s %8s %7s\n", "scene", "pairs", "ok", "degen", "failed",
              "kept", "kept-dg", "false-ok", "false-in", "false+", "lost", "prec", "recall", "rot med°", "dir med°", "ms");
  for (const Row& r : rows)
    std::printf("%-6s %6zu %5d %5d %6d %5d %8d %8d %8u %8d %6d %6.1f%% %7.1f%% %8.3f %8.3f %7.1f\n", r.scene.name.c_str(),
                r.scene.pairs.size(), r.run.n_ok, r.run.n_degenerate, r.run.n_failed, r.run.n_kept, r.run.n_kept_degenerate,
                r.mt.false_pairs_ok, r.mt.max_false_inliers, r.mt.false_pairs_kept, r.mt.true_pairs_lost,
                100.0 * r.mt.mask_precision, 100.0 * r.mt.mask_recall, r.mt.pair_rot_median, r.mt.pair_dir_median,
                1e3 * r.run.t_verify);
  std::printf("  kept: pairs whose inliers go on to the tracks (%u inliers and %.0f %% of the matches); kept-dg: those\n"
              "  flagged DEGENERATE. false-ok: pairs with no true overlap that came back with status OK, and false-in the\n"
              "  most inliers one of them had. false+: such pairs kept. lost: dropped pairs with 60 or more true matches.\n"
              "  prec/recall: inlier masks of the kept pairs against the true matches. rot/dir: error of the relative\n"
              "  pose of the kept OK pairs.\n\n",
              kMinPairInliers, 100.0 * kMinPairInlierRatio);

  std::printf("Mapper\n");
  std::printf("%-6s %10s %6s %8s %6s %9s %8s %7s %6s %6s %6s\n", "scene", "first pair", "inl", "angle", "floor", "pnp calls",
              "problems", "reject", "BA", "tracks", "s");
  for (const Row& r : rows) {
    char first[32];
    std::snprintf(first, sizeof(first), "%d-%d", r.run.first_a, r.run.first_b);
    std::printf("%-6s %10s %6u %7.1f° %5.0f° %9d %8d %7d %6d %6zu %6.2f\n", r.scene.name.c_str(), first, r.run.first_inliers,
                r.run.first_angle_deg, r.run.first_angle_floor, r.run.pnp_calls, r.run.pnp_problems, r.run.pnp_rejected,
                r.run.adjustments, r.run.map.tracks.size(), r.seconds);
  }
  std::printf("  time by step, s (verify / tracks / pnp / triangulate / adjust):");
  for (const Row& r : rows)
    std::printf("  %s %.2f/%.2f/%.2f/%.2f/%.2f", r.scene.name.c_str(), r.run.t_verify, r.run.t_tracks, r.run.t_pnp, r.run.t_points,
                r.run.t_adjust);
  std::printf("\n\n");

  std::printf("Against the ground truth, after a similarity alignment of the camera centres\n");
  std::printf("%-6s %7s %7s %7s %9s %9s %9s %9s %9s %9s %9s  %s\n", "scene", "images", "points", "obs", "reproj px", "rot med°",
              "rot max°", "ctr med%", "ctr max%", "ctr/cam%", "pts med%", "result");
  for (const Row& r : rows) {
    char images[32];
    std::snprintf(images, sizeof(images), "%zu/%zu", r.mt.registered, r.mt.images);
    std::printf("%-6s %7s %7zu %7zu %9.3f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f  %s%s%s\n", r.scene.name.c_str(), images,
                r.mt.points, r.mt.observations, r.mt.mean_error_px, r.mt.rot_median, r.mt.rot_max, r.mt.centre_median,
                r.mt.centre_max, r.mt.centre_median_cam, r.mt.point_median, r.mt.pass ? "PASS" : "FAIL",
                r.mt.why.empty() ? "" : ": ", r.mt.why.c_str());
  }
  std::printf("  ctr: camera-centre error over the scene extent (box around cameras and seen points); ctr/cam: the\n"
              "  median over the extent of the camera centres alone. pts: point error over the scene extent.\n"
              "  Pass: all images, median rotation < 0.5 deg, median centre < 1 %%, mean reprojection < 1.5 px.\n");
  return failed ? 1 : 0;
}
