// Problem files (.jpop) and result files: what the CLI, the tests and the PoseLib reference tool share.
//
// .jpop, little endian:
//   "JPOP" u32 version (1)  u32 kind (0 relative, 1 absolute)  u32 count
//   per problem:
//     u32 n
//     camera a: i32 model, f64 params[8]          (relative: then camera b the same)
//     f64 gt_pose[7]   qw qx qy qz tx ty tz, NaN if unknown
//     relative: f64 xy_a[2 n], f64 xy_b[2 n]       absolute: f64 xy[2 n], f64 xyz[3 n]
//     u8 gt_inlier[n]  (2 = unknown)
//
// Results, text: a header line "# jpo-results <kind> <count> <seconds> <tool>", then per problem
//   status n_inliers iterations score qw qx qy qz tx ty tz tri_angle mask
// where mask is a string of 0/1, one per point ("-" if empty).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "jpo/jpo.h"

namespace jpofile {

struct Problem {
  jpo_camera cam_a, cam_b;
  double gt_pose[7];
  std::vector<double> xy_a, xy_b;  // relative; absolute uses xy_a as xy
  std::vector<double> xyz;         // absolute
  std::vector<uint8_t> gt_inlier;
  uint32_t n() const { return static_cast<uint32_t>(xy_a.size() / 2); }
};

struct ProblemSet {
  uint32_t kind = 0;  // 0 relative, 1 absolute
  std::vector<Problem> problems;
};

namespace detail {
inline void wr(std::FILE* f, const void* p, size_t n) {
  if (n && std::fwrite(p, 1, n, f) != n) throw std::runtime_error("write failed");
}
inline void rd(std::FILE* f, void* p, size_t n) {
  if (n && std::fread(p, 1, n, f) != n) throw std::runtime_error("problem file is truncated");
}
inline void wr_cam(std::FILE* f, const jpo_camera& c) {
  wr(f, &c.model, 4);
  wr(f, c.params, 64);
}
inline void rd_cam(std::FILE* f, jpo_camera* c) {
  std::memset(c, 0, sizeof(*c));
  rd(f, &c->model, 4);
  rd(f, c->params, 64);
}
}  // namespace detail

inline void save(const std::string& path, const ProblemSet& set) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path);
  try {
    const uint32_t head[3] = {1u, set.kind, static_cast<uint32_t>(set.problems.size())};
    detail::wr(f, "JPOP", 4);
    detail::wr(f, head, 12);
    for (const Problem& p : set.problems) {
      const uint32_t n = p.n();
      detail::wr(f, &n, 4);
      detail::wr_cam(f, p.cam_a);
      if (set.kind == 0) detail::wr_cam(f, p.cam_b);
      detail::wr(f, p.gt_pose, 56);
      detail::wr(f, p.xy_a.data(), 16 * static_cast<size_t>(n));
      if (set.kind == 0)
        detail::wr(f, p.xy_b.data(), 16 * static_cast<size_t>(n));
      else
        detail::wr(f, p.xyz.data(), 24 * static_cast<size_t>(n));
      detail::wr(f, p.gt_inlier.data(), n);
    }
  } catch (...) {
    std::fclose(f);
    throw;
  }
  std::fclose(f);
}

inline ProblemSet load(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot read " + path);
  ProblemSet set;
  try {
    char magic[4];
    uint32_t head[3];
    detail::rd(f, magic, 4);
    detail::rd(f, head, 12);
    if (std::memcmp(magic, "JPOP", 4) != 0 || head[0] != 1 || head[1] > 1) throw std::runtime_error("not a .jpop file: " + path);
    set.kind = head[1];
    set.problems.resize(head[2]);
    for (Problem& p : set.problems) {
      uint32_t n;
      detail::rd(f, &n, 4);
      detail::rd_cam(f, &p.cam_a);
      if (set.kind == 0) detail::rd_cam(f, &p.cam_b);
      detail::rd(f, p.gt_pose, 56);
      p.xy_a.resize(2 * static_cast<size_t>(n));
      detail::rd(f, p.xy_a.data(), 16 * static_cast<size_t>(n));
      if (set.kind == 0) {
        p.xy_b.resize(2 * static_cast<size_t>(n));
        detail::rd(f, p.xy_b.data(), 16 * static_cast<size_t>(n));
      } else {
        p.xyz.resize(3 * static_cast<size_t>(n));
        detail::rd(f, p.xyz.data(), 24 * static_cast<size_t>(n));
      }
      p.gt_inlier.resize(n);
      detail::rd(f, p.gt_inlier.data(), n);
    }
  } catch (...) {
    std::fclose(f);
    throw;
  }
  std::fclose(f);
  return set;
}

// The set as the flat buffers of the C API.
struct Flat {
  std::vector<uint32_t> offsets, cam_a, cam_b;
  std::vector<jpo_camera> cameras;
  std::vector<double> xy_a, xy_b, xyz;
  jpo_pairs pairs() const {
    jpo_pairs p;
    p.n_pairs = static_cast<uint32_t>(offsets.size() - 1);
    p.n_cameras = static_cast<uint32_t>(cameras.size());
    p.offsets = offsets.data();
    p.xy_a = xy_a.data();
    p.xy_b = xy_b.data();
    p.cameras = cameras.data();
    p.camera_a = cam_a.data();
    p.camera_b = cam_b.data();
    return p;
  }
  jpo_resections resections() const {
    jpo_resections r;
    r.n_images = static_cast<uint32_t>(offsets.size() - 1);
    r.n_cameras = static_cast<uint32_t>(cameras.size());
    r.offsets = offsets.data();
    r.xy = xy_a.data();
    r.xyz = xyz.data();
    r.cameras = cameras.data();
    r.camera = cam_a.data();
    return r;
  }
};

inline Flat flatten(const ProblemSet& set) {
  Flat f;
  f.offsets.push_back(0);
  for (const Problem& p : set.problems) {
    f.cam_a.push_back(static_cast<uint32_t>(f.cameras.size()));
    f.cameras.push_back(p.cam_a);
    if (set.kind == 0) {
      f.cam_b.push_back(static_cast<uint32_t>(f.cameras.size()));
      f.cameras.push_back(p.cam_b);
      f.xy_b.insert(f.xy_b.end(), p.xy_b.begin(), p.xy_b.end());
    } else {
      f.xyz.insert(f.xyz.end(), p.xyz.begin(), p.xyz.end());
    }
    f.xy_a.insert(f.xy_a.end(), p.xy_a.begin(), p.xy_a.end());
    f.offsets.push_back(f.offsets.back() + p.n());
  }
  return f;
}

inline void save_results(const std::string& path, uint32_t kind, const std::vector<jpo_pose_result>& results,
                         const std::vector<uint32_t>& offsets, const std::vector<uint8_t>& inliers, double seconds,
                         const std::string& tool) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) throw std::runtime_error("cannot write " + path);
  std::fprintf(f, "# jpo-results %u %zu %.9g %s\n", kind, results.size(), seconds, tool.c_str());
  std::string mask;
  for (size_t i = 0; i < results.size(); ++i) {
    const jpo_pose_result& r = results[i];
    std::fprintf(f, "%d %u %u %.17g", r.status, r.n_inliers, r.iterations, r.score);
    for (int k = 0; k < 7; ++k) std::fprintf(f, " %.17g", r.pose[k]);
    std::fprintf(f, " %.9g ", r.median_tri_angle_deg);
    mask.clear();
    for (uint32_t k = offsets[i]; k < offsets[i + 1]; ++k) mask.push_back(inliers[k] ? '1' : '0');
    if (mask.empty()) mask = "-";
    std::fprintf(f, "%s\n", mask.c_str());
  }
  std::fclose(f);
}

inline const char* model_name(int model) {
  switch (model) {
    case JPO_SIMPLE_PINHOLE: return "SIMPLE_PINHOLE";
    case JPO_PINHOLE: return "PINHOLE";
    case JPO_SIMPLE_RADIAL: return "SIMPLE_RADIAL";
    case JPO_RADIAL: return "RADIAL";
    case JPO_OPENCV: return "OPENCV";
    case JPO_OPENCV_FISHEYE: return "OPENCV_FISHEYE";
    default: return "?";
  }
}

inline int model_from_name(const std::string& name) {
  for (int m = 0; m <= JPO_OPENCV_FISHEYE; ++m)
    if (name == model_name(m)) return m;
  return -1;
}

}  // namespace jpofile
