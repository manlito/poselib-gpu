// Synthetic captures with ground truth for the incremental reconstruction test: cameras, points, noisy
// keypoints, and pairwise matches corrupted the way a matcher's output is (wrong pairings inside
// overlapping pairs, and pairs that do not overlap at all but still carry matches).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "cli/synth.h"
#include "core/geometry.h"
#include "jpo/jpo.h"

namespace inc {

using jposynth::Rng;

const double kWidth = 1600.0, kHeight = 1200.0;
const double kNoisePx = 0.5;         // standard deviation per coordinate
const double kWrongFraction = 0.25;  // of each overlapping pair's matches
const double kPi = 3.14159265358979323846;

struct Pair {
  int a = 0, b = 0;
  std::vector<int> ka, kb;      // keypoint in a, keypoint in b, per match
  std::vector<uint8_t> correct; // ground truth: both keypoints are the same scene point
  int shared = 0;               // scene points really seen by both images
};

struct Scene {
  std::string name;
  jpo_camera camera;                         // one camera for the whole capture, as a phone has
  std::vector<double> poses;                 // 7 per image, world to camera (ground truth)
  std::vector<double> points;                // 3 per point (ground truth)
  std::vector<double> normals;               // 3 per point, or empty: a point is seen from every side
  double max_view_angle_deg = 180.0;         // with normals: seen only within this angle of the normal
  std::vector<std::vector<double>> xy;       // per image: 2 per keypoint, with noise
  std::vector<std::vector<int>> kp_point;    // per image: the scene point of each keypoint
  std::vector<Pair> pairs;
  size_t n_images() const { return poses.size() / 7; }
};

inline void cross(const double a[3], const double b[3], double c[3]) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}
inline double norm3(const double a[3]) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }

inline void camera_centre(const double pose[7], double C[3]) {
  double R[9];
  jpo::quat_to_rotmat(pose, R);
  for (int c = 0; c < 3; ++c) C[c] = -(R[c] * pose[4] + R[3 + c] * pose[5] + R[6 + c] * pose[6]);
}

// A camera at C looking at `target`, image y along the world's y (down), then turned by a small random
// rotation so no two cameras are perfectly aligned.
inline void add_look_at(Scene& s, Rng& rng, const double C[3], const double target[3], double jitter_deg) {
  double z[3] = {target[0] - C[0], target[1] - C[1], target[2] - C[2]};
  const double zn = norm3(z);
  for (double& v : z) v /= zn;
  const double down[3] = {0.0, 1.0, 0.0};
  double x[3], y[3];
  cross(down, z, x);
  const double xn = norm3(x);
  for (double& v : x) v /= xn;
  cross(z, x, y);
  const double R0[9] = {x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]};
  double J[9], R[9];
  jposynth::random_rotation(rng, jitter_deg, J);
  jpo::mat3_mul(J, R0, R);
  double pose[7];
  jpo::rotmat_to_quat(R, pose);
  for (int r = 0; r < 3; ++r) pose[4 + r] = -(R[3 * r] * C[0] + R[3 * r + 1] * C[1] + R[3 * r + 2] * C[2]);
  s.poses.insert(s.poses.end(), pose, pose + 7);
}

// Keypoints: every point that projects inside the image (and faces the camera, when the scene has
// normals) is detected with probability 0.85, at its projection plus noise.
inline void observe(Scene& s, Rng& rng) {
  const size_t n_img = s.n_images(), n_pts = s.points.size() / 3;
  s.xy.assign(n_img, {});
  s.kp_point.assign(n_img, {});
  const double cos_max = std::cos(s.max_view_angle_deg * kPi / 180.0);
  for (size_t i = 0; i < n_img; ++i) {
    const double* pose = &s.poses[7 * i];
    double R[9], C[3];
    jpo::quat_to_rotmat(pose, R);
    camera_centre(pose, C);
    for (size_t j = 0; j < n_pts; ++j) {
      const double* X = &s.points[3 * j];
      double Y[3], u, v;
      jposynth::apply(R, pose + 4, X, Y);
      if (!(Y[2] > 0.1) || !jpo::camera_project(s.camera, Y, &u, &v)) continue;
      if (u < 0.0 || v < 0.0 || u > kWidth - 1.0 || v > kHeight - 1.0) continue;
      if (!s.normals.empty()) {
        const double* n = &s.normals[3 * j];
        const double d[3] = {C[0] - X[0], C[1] - X[1], C[2] - X[2]};
        if ((d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) / norm3(d) < cos_max) continue;
      }
      if (rng.uniform() > 0.85) continue;
      s.xy[i].push_back(u + kNoisePx * rng.normal());
      s.xy[i].push_back(v + kNoisePx * rng.normal());
      s.kp_point[i].push_back(static_cast<int>(j));
    }
  }
}

// Matches of every pair of images.
//   Overlapping pairs: one match per shared point, then a quarter of them get a wrong keypoint in b
//   (a keypoint of b that this pair does not otherwise use, so matches stay one to one as a mutual
//   nearest-neighbour matcher gives them).
//   Pairs that share no point: up to `max_false_pairs` of them get 60 to 140 random matches.
inline void match(Scene& s, Rng& rng, int max_false_pairs) {
  const int n_img = static_cast<int>(s.n_images());
  const size_t n_pts = s.points.size() / 3;
  std::vector<std::vector<int>> kp_of(n_img, std::vector<int>(n_pts, -1));  // point -> keypoint
  for (int i = 0; i < n_img; ++i)
    for (size_t k = 0; k < s.kp_point[i].size(); ++k) kp_of[i][s.kp_point[i][k]] = static_cast<int>(k);
  std::vector<std::pair<int, int>> disjoint;
  for (int a = 0; a < n_img; ++a)
    for (int b = a + 1; b < n_img; ++b) {
      Pair p;
      p.a = a;
      p.b = b;
      std::vector<char> used_b(s.kp_point[b].size(), 0);
      for (size_t k = 0; k < s.kp_point[a].size(); ++k) {
        const int kb = kp_of[b][s.kp_point[a][k]];
        if (kb < 0) continue;
        p.ka.push_back(static_cast<int>(k));
        p.kb.push_back(kb);
        used_b[kb] = 1;
      }
      p.shared = static_cast<int>(p.ka.size());
      if (p.shared == 0) {
        disjoint.emplace_back(a, b);
        continue;
      }
      p.correct.assign(p.ka.size(), 1);
      std::vector<int> free_b;
      for (size_t k = 0; k < used_b.size(); ++k)
        if (!used_b[k]) free_b.push_back(static_cast<int>(k));
      std::vector<int> wrong;  // the matches that get a wrong keypoint
      for (size_t m = 0; m < p.ka.size(); ++m)
        if (rng.uniform() < kWrongFraction) wrong.push_back(static_cast<int>(m));
      std::vector<int> rotated;  // no unused keypoint left: these swap among themselves
      for (int m : wrong) {
        if (!free_b.empty()) {
          const size_t pick = rng.next() % free_b.size();
          p.kb[m] = free_b[pick];
          free_b[pick] = free_b.back();
          free_b.pop_back();
          p.correct[m] = 0;
        } else {
          rotated.push_back(m);
        }
      }
      if (rotated.size() >= 2) {
        const int first = p.kb[rotated[0]];
        for (size_t r = 0; r + 1 < rotated.size(); ++r) p.kb[rotated[r]] = p.kb[rotated[r + 1]];
        p.kb[rotated.back()] = first;
        for (int m : rotated) p.correct[m] = 0;
      }
      s.pairs.push_back(std::move(p));
    }
  // Pairs without overlap, spread over the list rather than the first few.
  for (size_t i = disjoint.size(); i > 1; --i) std::swap(disjoint[i - 1], disjoint[rng.next() % i]);
  for (size_t d = 0; d < disjoint.size() && d < static_cast<size_t>(max_false_pairs); ++d) {
    Pair p;
    p.a = disjoint[d].first;
    p.b = disjoint[d].second;
    const size_t na = s.kp_point[p.a].size(), nb = s.kp_point[p.b].size();
    const size_t want = std::min<size_t>(60 + rng.next() % 81, std::min(na, nb));
    std::vector<int> ia(na), ib(nb);
    for (size_t k = 0; k < na; ++k) ia[k] = static_cast<int>(k);
    for (size_t k = 0; k < nb; ++k) ib[k] = static_cast<int>(k);
    for (size_t k = 0; k < want; ++k) {
      std::swap(ia[k], ia[k + rng.next() % (na - k)]);
      std::swap(ib[k], ib[k + rng.next() % (nb - k)]);
      p.ka.push_back(ia[k]);
      p.kb.push_back(ib[k]);
    }
    p.correct.assign(want, 0);
    s.pairs.push_back(std::move(p));
  }
  std::sort(s.pairs.begin(), s.pairs.end(),
            [](const Pair& x, const Pair& y) { return x.a != y.a ? x.a < y.a : x.b < y.b; });
}

inline jpo_camera scene_camera(int model, Rng& rng, double focal_over_width) {
  jpo_camera c = jposynth::make_camera(model, rng, kWidth, kHeight);
  // A fixed field of view, so which cameras overlap is decided by the layout below and not by the seed.
  const double f = focal_over_width * kWidth;
  if (model == JPO_PINHOLE) {
    c.params[1] *= f / c.params[0];
    c.params[0] = f;
  } else {
    c.params[0] = f;
  }
  return c;
}

// a. Orbit: 16 cameras on a 220 degree arc around a bumpy ball of 1500 surface points. A point is seen
//    only from its own side, so cameras far apart on the arc share nothing.
inline Scene make_orbit(uint64_t seed = 0) {
  Scene s;
  s.name = "orbit";
  Rng rng(101 + seed);
  s.camera = scene_camera(JPO_PINHOLE, rng, 0.9);
  s.max_view_angle_deg = 65.0;
  for (int j = 0; j < 1500; ++j) {
    double n[3] = {rng.normal(), rng.normal(), rng.normal()};
    const double nn = norm3(n);
    for (double& v : n) v /= nn;
    // A radius of 1.5 with relief of a few tenths, so the surface is not one smooth quadric.
    const double r = 1.5 * (1.0 + 0.12 * std::sin(5.0 * n[0]) * std::cos(4.0 * n[1]) + 0.05 * rng.normal());
    for (int d = 0; d < 3; ++d) {
      s.points.push_back(r * n[d]);
      s.normals.push_back(n[d]);
    }
  }
  const int n_cam = 16;
  for (int i = 0; i < n_cam; ++i) {
    const double ang = (-110.0 + 220.0 * i / (n_cam - 1)) * kPi / 180.0;
    const double radius = 5.0 + rng.uniform(-0.4, 0.4);
    const double C[3] = {radius * std::sin(ang), rng.uniform(-0.6, 0.6), -radius * std::cos(ang)};
    const double target[3] = {rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15)};
    add_look_at(s, rng, C, target, 3.0);
  }
  observe(s, rng);
  match(s, rng, 10);
  return s;
}

// b. Walk along a wall: 14 cameras, one metre apart, moving sideways and a little toward and away from a
//    wall six metres off that is flat except for shallow relief. SIMPLE_RADIAL, mild distortion.
inline Scene make_wall(uint64_t seed = 0) {
  Scene s;
  s.name = "wall";
  Rng rng(202 + seed);
  s.camera = scene_camera(JPO_SIMPLE_RADIAL, rng, 0.85);
  s.camera.params[3] = -0.06;
  for (int j = 0; j < 3200; ++j) {
    const double x = rng.uniform(-6.0, 19.0), y = rng.uniform(-3.5, 3.5);
    // Relief: a slow wave of 15 cm, roughness of 3 cm, and one point in ten on something standing in
    // front of the wall (up to a metre).
    double z = 6.0 + 0.15 * std::sin(0.9 * x) * std::cos(1.3 * y) + 0.03 * rng.normal();
    if (rng.uniform() < 0.1) z -= rng.uniform(0.2, 1.0);
    s.points.insert(s.points.end(), {x, y, z});
  }
  const int n_cam = 14;
  for (int i = 0; i < n_cam; ++i) {
    const double C[3] = {1.0 * i + rng.uniform(-0.1, 0.1), rng.uniform(-0.15, 0.15), 0.8 * std::sin(0.7 * i)};
    const double target[3] = {C[0] + rng.uniform(-1.0, 1.0), rng.uniform(-0.3, 0.3), 6.0};
    add_look_at(s, rng, C, target, 2.0);
  }
  observe(s, rng);
  match(s, rng, 10);
  return s;
}

// c. Grid: two rows of six cameras 20 cm apart (rows 30 cm apart), fanned out by 14 degrees a column,
//    in front of a deep volume of points. Neighbours have a baseline of a few percent of the depth, and
//    the two ends of a row look at different parts of the scene. A thirteenth image is taken from the
//    place of image 3 (within a centimetre) after turning by 3 degrees: with image 3 it makes a pair with
//    hundreds of good matches and no baseline, the pair a first-pair rule must not pick.
inline Scene make_grid(uint64_t seed = 0) {
  Scene s;
  s.name = "grid";
  Rng rng(303 + seed);
  s.camera = scene_camera(JPO_PINHOLE, rng, 0.85);
  for (int j = 0; j < 3000; ++j) {
    const double x = rng.uniform(-10.0, 10.0), y = rng.uniform(-3.5, 3.5), z = rng.uniform(4.5, 8.5);
    s.points.insert(s.points.end(), {x, y, z});
  }
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 6; ++c) {
      const double C[3] = {0.2 * (c - 2.5) + rng.uniform(-0.02, 0.02), 0.3 * (r - 0.5) + rng.uniform(-0.02, 0.02),
                           rng.uniform(-0.03, 0.03)};
      const double yaw = 14.0 * (c - 2.5) * kPi / 180.0;
      const double target[3] = {C[0] + 6.0 * std::sin(yaw), C[1] + rng.uniform(-0.2, 0.2), C[2] + 6.0 * std::cos(yaw)};
      add_look_at(s, rng, C, target, 1.5);
    }
  {
    double C[3];
    camera_centre(&s.poses[7 * 3], C);
    for (double& v : C) v += rng.uniform(-0.006, 0.006);
    const double yaw = (14.0 * 0.5 + 3.0) * kPi / 180.0;
    const double target[3] = {C[0] + 6.0 * std::sin(yaw), C[1] + 0.1, C[2] + 6.0 * std::cos(yaw)};
    add_look_at(s, rng, C, target, 1.0);
  }
  observe(s, rng);
  match(s, rng, 8);
  return s;
}

}  // namespace inc
