// Synthetic problems with known poses: used by `jpo gen`, the tests and the benchmark.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "cli/problem_file.h"
#include "core/geometry.h"

namespace jposynth {

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9e3779b97f4a7c15ull + 0x1234567ull) {}
  uint64_t next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
  double uniform(double a, double b) { return a + (b - a) * uniform(); }
  double normal() {
    const double u1 = std::max(uniform(), 1e-300), u2 = uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
};

struct Spec {
  uint32_t kind = 0;            // 0 relative, 1 absolute
  uint32_t count = 1;           // problems
  uint32_t points = 1000;       // matches per problem
  double inlier_ratio = 0.5;    // 0: no model at all
  double noise_px = 0.5;        // standard deviation, per coordinate
  int model = JPO_PINHOLE;
  bool planar = false;          // all points on one plane
  double baseline = 0.15;       // relative: |t| over the mean depth
  double max_rotation_deg = 25.0;
  double world_offset = 0.0;    // absolute: the world origin is this far from the scene
  uint64_t seed = 1;
};

inline jpo_camera make_camera(int model, Rng& rng, double width, double height) {
  jpo_camera c;
  std::memset(&c, 0, sizeof(c));
  c.model = model;
  const double f = rng.uniform(0.7, 1.3) * width;
  const double cx = 0.5 * (width - 1.0) + rng.uniform(-10.0, 10.0);
  const double cy = 0.5 * (height - 1.0) + rng.uniform(-10.0, 10.0);
  double* p = c.params;
  switch (model) {
    case JPO_SIMPLE_PINHOLE: p[0] = f; p[1] = cx; p[2] = cy; break;
    case JPO_PINHOLE: p[0] = f; p[1] = f * rng.uniform(0.98, 1.02); p[2] = cx; p[3] = cy; break;
    case JPO_SIMPLE_RADIAL: p[0] = f; p[1] = cx; p[2] = cy; p[3] = rng.uniform(-0.1, 0.1); break;
    case JPO_RADIAL: p[0] = f; p[1] = cx; p[2] = cy; p[3] = rng.uniform(-0.1, 0.1); p[4] = rng.uniform(-0.02, 0.02); break;
    case JPO_OPENCV:
      p[0] = f; p[1] = f * rng.uniform(0.98, 1.02); p[2] = cx; p[3] = cy;
      p[4] = rng.uniform(-0.1, 0.1); p[5] = rng.uniform(-0.02, 0.02);
      p[6] = rng.uniform(-0.002, 0.002); p[7] = rng.uniform(-0.002, 0.002);
      break;
    default:  // OPENCV_FISHEYE: a shorter focal length, as such lenses have
      p[0] = 0.45 * f; p[1] = 0.45 * f * rng.uniform(0.98, 1.02); p[2] = cx; p[3] = cy;
      p[4] = rng.uniform(-0.02, 0.02); p[5] = rng.uniform(-0.005, 0.005);
      p[6] = rng.uniform(-0.001, 0.001); p[7] = rng.uniform(-0.0005, 0.0005);
  }
  return c;
}

inline void random_rotation(Rng& rng, double max_deg, double R[9]) {
  double w[3] = {rng.normal(), rng.normal(), rng.normal()};
  const double n = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  const double ang = rng.uniform(0.0, max_deg) * 3.14159265358979323846 / 180.0;
  for (int i = 0; i < 3; ++i) w[i] *= ang / n;
  jpo::rotmat_exp(w, R);
}

inline void apply(const double R[9], const double t[3], const double X[3], double Y[3]) {
  for (int r = 0; r < 3; ++r) Y[r] = R[3 * r] * X[0] + R[3 * r + 1] * X[1] + R[3 * r + 2] * X[2] + t[r];
}

// A point of the scene seen at a random pixel of the camera, in the camera frame.
inline bool scene_point(const jpo_camera& cam, Rng& rng, double width, double height, bool planar,
                        const double plane[4], double X[3]) {
  double x, y;
  if (!jpo::camera_unproject(cam, rng.uniform(0.0, width - 1.0), rng.uniform(0.0, height - 1.0), &x, &y)) return false;
  double depth = rng.uniform(2.0, 10.0);
  if (planar) {  // plane n . X = d, ray (x, y, 1) z
    const double den = plane[0] * x + plane[1] * y + plane[2];
    if (std::abs(den) < 1e-6) return false;
    depth = plane[3] / den;
    if (!(depth > 0.5)) return false;
  }
  X[0] = x * depth;
  X[1] = y * depth;
  X[2] = depth;
  return true;
}

inline jpofile::Problem make_problem(const Spec& spec, Rng& rng) {
  const double width = 1600.0, height = 1200.0;
  jpofile::Problem p;
  p.cam_a = make_camera(spec.model, rng, width, height);
  p.cam_b = make_camera(spec.model, rng, width, height);
  double plane[4] = {rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 1.0, rng.uniform(4.0, 7.0)};

  // Camera b (relative) or the world (absolute) with respect to camera a.
  double R[9], t[3];
  random_rotation(rng, spec.max_rotation_deg, R);
  double dir[3] = {rng.normal(), rng.normal(), 0.3 * rng.normal()};
  const double dn = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  for (int i = 0; i < 3; ++i) t[i] = dir[i] / dn * spec.baseline * 6.0;

  const uint32_t n_in = static_cast<uint32_t>(std::lround(spec.inlier_ratio * spec.points));
  const uint32_t n = spec.points;
  std::vector<uint32_t> order(n);
  for (uint32_t i = 0; i < n; ++i) order[i] = i;
  for (uint32_t i = n; i > 1; --i) std::swap(order[i - 1], order[rng.next() % i]);
  p.xy_a.assign(2 * static_cast<size_t>(n), 0.0);
  p.gt_inlier.assign(n, 0);

  if (spec.kind == 0) {
    p.xy_b.assign(2 * static_cast<size_t>(n), 0.0);
    for (uint32_t k = 0; k < n; ++k) {
      const uint32_t i = order[k];
      if (k < n_in) {
        for (int tries = 0; tries < 1000; ++tries) {
          double X[3], Y[3], ua, va, ub, vb;
          if (!scene_point(p.cam_a, rng, width, height, spec.planar, plane, X)) continue;
          apply(R, t, X, Y);
          if (!jpo::camera_project(p.cam_a, X, &ua, &va) || !jpo::camera_project(p.cam_b, Y, &ub, &vb)) continue;
          if (ub < 0.0 || vb < 0.0 || ub > width - 1.0 || vb > height - 1.0) continue;
          if (p.cam_b.model != JPO_OPENCV_FISHEYE && !(Y[2] > 0.1)) continue;
          p.xy_a[2 * i] = ua + spec.noise_px * rng.normal();
          p.xy_a[2 * i + 1] = va + spec.noise_px * rng.normal();
          p.xy_b[2 * i] = ub + spec.noise_px * rng.normal();
          p.xy_b[2 * i + 1] = vb + spec.noise_px * rng.normal();
          p.gt_inlier[i] = 1;
          break;
        }
      }
      if (!p.gt_inlier[i]) {
        p.xy_a[2 * i] = rng.uniform(0.0, width - 1.0);
        p.xy_a[2 * i + 1] = rng.uniform(0.0, height - 1.0);
        p.xy_b[2 * i] = rng.uniform(0.0, width - 1.0);
        p.xy_b[2 * i + 1] = rng.uniform(0.0, height - 1.0);
      }
    }
    jpo::rotmat_to_quat(R, p.gt_pose);
    const double tn = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    for (int i = 0; i < 3; ++i) p.gt_pose[4 + i] = t[i] / tn;
  } else {
    // World frame: X_world = Rw X_cam + tw, so the pose (world to camera) is (Rw', -Rw' tw).
    double tw[3] = {spec.world_offset + rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0)};
    random_rotation(rng, 180.0, R);
    p.xyz.assign(3 * static_cast<size_t>(n), 0.0);
    for (uint32_t k = 0; k < n; ++k) {
      const uint32_t i = order[k];
      double X[3] = {0.0, 0.0, 5.0}, W[3], ua = 0.0, va = 0.0;
      for (int tries = 0; tries < 1000; ++tries) {
        if (scene_point(p.cam_a, rng, width, height, spec.planar, plane, X) && jpo::camera_project(p.cam_a, X, &ua, &va)) break;
      }
      apply(R, tw, X, W);
      for (int d = 0; d < 3; ++d) p.xyz[3 * i + d] = W[d];
      if (k < n_in) {
        p.xy_a[2 * i] = ua + spec.noise_px * rng.normal();
        p.xy_a[2 * i + 1] = va + spec.noise_px * rng.normal();
        p.gt_inlier[i] = 1;
      } else {
        p.xy_a[2 * i] = rng.uniform(0.0, width - 1.0);
        p.xy_a[2 * i + 1] = rng.uniform(0.0, height - 1.0);
      }
    }
    double Rt[9];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) Rt[3 * r + c] = R[3 * c + r];
    jpo::rotmat_to_quat(Rt, p.gt_pose);
    for (int r = 0; r < 3; ++r) p.gt_pose[4 + r] = -(Rt[3 * r] * tw[0] + Rt[3 * r + 1] * tw[1] + Rt[3 * r + 2] * tw[2]);
  }
  if (spec.inlier_ratio <= 0.0)
    for (int i = 0; i < 7; ++i) p.gt_pose[i] = std::numeric_limits<double>::quiet_NaN();
  return p;
}

inline jpofile::ProblemSet make_set(const Spec& spec) {
  jpofile::ProblemSet set;
  set.kind = spec.kind;
  Rng rng(spec.seed);
  for (uint32_t i = 0; i < spec.count; ++i) set.problems.push_back(make_problem(spec, rng));
  return set;
}

// Errors against the ground truth, in degrees. For a relative pose the translation error is the angle
// between directions; for an absolute pose it is returned as a distance over the scene depth scale.
inline void pose_errors(uint32_t kind, const double est[7], const double gt[7], double* rot_deg, double* trans) {
  double Re[9], Rg[9], D[9];
  jpo::quat_to_rotmat(est, Re);
  jpo::quat_to_rotmat(gt, Rg);
  double Rgt[9];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) Rgt[3 * r + c] = Rg[3 * c + r];
  jpo::mat3_mul(Re, Rgt, D);
  const double tr = std::min(3.0, std::max(-1.0, D[0] + D[4] + D[8]));
  *rot_deg = std::acos(std::min(1.0, std::max(-1.0, 0.5 * (tr - 1.0)))) * 180.0 / 3.14159265358979323846;
  const double* te = est + 4;
  const double* tg = gt + 4;
  if (kind == 0) {
    const double ne = std::sqrt(te[0] * te[0] + te[1] * te[1] + te[2] * te[2]);
    const double ng = std::sqrt(tg[0] * tg[0] + tg[1] * tg[1] + tg[2] * tg[2]);
    const double c = (te[0] * tg[0] + te[1] * tg[1] + te[2] * tg[2]) / (ne * ng);
    *trans = std::acos(std::min(1.0, std::max(-1.0, c))) * 180.0 / 3.14159265358979323846;
  } else {
    // Distance between camera centres -R' t.
    double ce[3], cg[3];
    for (int c = 0; c < 3; ++c) {
      ce[c] = -(Re[c] * te[0] + Re[3 + c] * te[1] + Re[6 + c] * te[2]);
      cg[c] = -(Rg[c] * tg[0] + Rg[3 + c] * tg[1] + Rg[6 + c] * tg[2]);
    }
    *trans = std::sqrt((ce[0] - cg[0]) * (ce[0] - cg[0]) + (ce[1] - cg[1]) * (ce[1] - cg[1]) + (ce[2] - cg[2]) * (ce[2] - cg[2]));
  }
}

}  // namespace jposynth
