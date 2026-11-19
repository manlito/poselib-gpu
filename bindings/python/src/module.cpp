// Python bindings: a thin layer over the C API (include/jpo/jpo.h). Problems cross packed, as the C API
// takes them (concatenated matches and offsets); poselib_gpu/__init__.py builds them from lists of arrays,
// names the cameras and backends, and gathers the results. Arrays are copied in and out.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "jpo/jpo.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T>
using In = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;
template <typename T>
using Out = nb::ndarray<nb::numpy, T>;

template <typename T>
Out<T> to_numpy(std::vector<T>&& v, std::vector<size_t> shape) {
  auto* owned = new std::vector<T>(std::move(v));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return Out<T>(owned->data(), shape.size(), shape.data(), owner);
}

// The array's rows, checking that it has shape (rows, cols) (cols 0: one dimension).
template <typename T>
size_t rows(const In<T>& a, size_t cols, const char* name) {
  const bool ok = cols ? (a.ndim() == 2 && a.shape(1) == cols) : a.ndim() == 1;
  if (!ok) {
    const std::string want = cols ? "(n, " + std::to_string(cols) + ")" : "(n,)";
    throw nb::value_error((std::string(name) + " must have shape " + want).c_str());
  }
  return a.shape(0);
}

[[noreturn]] void fail() { throw std::runtime_error(jpo_last_error()); }

std::vector<jpo_camera> cameras(const In<int32_t>& models, const In<double>& params) {
  const size_t n = rows(models, 0, "camera models");
  if (rows(params, 8, "camera params") != n) throw nb::value_error("one row of camera params per camera");
  std::vector<jpo_camera> out(n);
  for (size_t i = 0; i < n; ++i) {
    std::memset(&out[i], 0, sizeof(jpo_camera));
    out[i].model = models.data()[i];
    std::copy(params.data() + 8 * i, params.data() + 8 * i + 8, out[i].params);
  }
  return out;
}

jpo_ransac_options options(double max_error_px, double confidence, uint32_t min_iterations,
                           uint32_t max_iterations, bool refine, double min_inlier_ratio) {
  jpo_ransac_options o;
  jpo_ransac_options_init(&o);
  o.max_error_px = max_error_px;
  o.confidence = confidence;
  o.min_iterations = min_iterations;
  o.max_iterations = max_iterations;
  o.refine = refine ? 1 : 0;
  o.min_inlier_ratio = min_inlier_ratio;
  return o;
}

// pose (n, 7), score, median triangulation angle, inlier count, iterations, status, and the inlier mask
// (one byte per match, or None).
using Results = std::tuple<Out<double>, Out<double>, Out<double>, Out<uint32_t>, Out<uint32_t>, Out<int32_t>,
                           nb::object>;

Results gather(const std::vector<jpo_pose_result>& r, std::vector<uint8_t>&& mask, bool want_inliers) {
  const size_t n = r.size();
  std::vector<double> pose(7 * n), score(n), angle(n);
  std::vector<uint32_t> inl(n), it(n);
  std::vector<int32_t> status(n);
  for (size_t i = 0; i < n; ++i) {
    std::copy(r[i].pose, r[i].pose + 7, pose.data() + 7 * i);
    score[i] = r[i].score;
    angle[i] = r[i].median_tri_angle_deg;
    inl[i] = r[i].n_inliers;
    it[i] = r[i].iterations;
    status[i] = r[i].status;
  }
  nb::object m = nb::none();
  if (want_inliers) {
    const size_t len = mask.size();
    m = nb::cast(to_numpy(std::move(mask), {len}));
  }
  return {to_numpy(std::move(pose), {n, 7}), to_numpy(std::move(score), {n}), to_numpy(std::move(angle), {n}),
          to_numpy(std::move(inl), {n}),     to_numpy(std::move(it), {n}),    to_numpy(std::move(status), {n}),
          m};
}

class Context {
 public:
  Context(int32_t backend, uint32_t seed, int32_t threads, bool cpu_float, uint32_t gpu_round) {
    jpo_config cfg;
    jpo_config_init(&cfg);
    cfg.backend = backend;
    cfg.seed = seed;
    cfg.threads = threads;
    cfg.cpu_float = cpu_float ? 1 : 0;
    cfg.gpu_round = gpu_round;
    ctx_ = jpo_create(&cfg);
    if (!ctx_) fail();
  }
  ~Context() { jpo_destroy(ctx_); }
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  std::string backend() const { return jpo_backend_name(ctx_); }

  Results relative(const In<uint32_t>& offsets, const In<double>& xy_a, const In<double>& xy_b,
                   const In<int32_t>& models, const In<double>& params, const In<uint32_t>& camera_a,
                   const In<uint32_t>& camera_b, double max_error_px, double confidence, uint32_t min_iterations,
                   uint32_t max_iterations, bool refine, double min_inlier_ratio, bool want_inliers) {
    const size_t n = check_offsets(offsets, rows(xy_a, 2, "xy_a"));
    if (rows(xy_b, 2, "xy_b") != xy_a.shape(0)) throw nb::value_error("xy_a and xy_b must have the same rows");
    if (rows(camera_a, 0, "camera_a") != n || rows(camera_b, 0, "camera_b") != n)
      throw nb::value_error("camera_a and camera_b need one index per pair");
    const std::vector<jpo_camera> cams = cameras(models, params);
    jpo_pairs p;
    p.n_pairs = static_cast<uint32_t>(n);
    p.n_cameras = static_cast<uint32_t>(cams.size());
    p.offsets = offsets.data();
    p.xy_a = xy_a.data();
    p.xy_b = xy_b.data();
    p.cameras = cams.data();
    p.camera_a = camera_a.data();
    p.camera_b = camera_b.data();
    const jpo_ransac_options o =
        options(max_error_px, confidence, min_iterations, max_iterations, refine, min_inlier_ratio);
    std::vector<jpo_pose_result> r(n);
    std::vector<uint8_t> mask(want_inliers ? xy_a.shape(0) : 0);
    int rc;
    {
      nb::gil_scoped_release release;
      rc = jpo_relative_pose(ctx_, &p, &o, r.data(), want_inliers ? mask.data() : nullptr);
    }
    if (rc != 0) fail();
    return gather(r, std::move(mask), want_inliers);
  }

  Results absolute(const In<uint32_t>& offsets, const In<double>& xy, const In<double>& xyz,
                   const In<int32_t>& models, const In<double>& params, const In<uint32_t>& camera,
                   double max_error_px, double confidence, uint32_t min_iterations, uint32_t max_iterations,
                   bool refine, double min_inlier_ratio, bool want_inliers) {
    const size_t n = check_offsets(offsets, rows(xy, 2, "xy"));
    if (rows(xyz, 3, "xyz") != xy.shape(0)) throw nb::value_error("xy and xyz must have the same rows");
    if (rows(camera, 0, "camera") != n) throw nb::value_error("camera needs one index per image");
    const std::vector<jpo_camera> cams = cameras(models, params);
    jpo_resections p;
    p.n_images = static_cast<uint32_t>(n);
    p.n_cameras = static_cast<uint32_t>(cams.size());
    p.offsets = offsets.data();
    p.xy = xy.data();
    p.xyz = xyz.data();
    p.cameras = cams.data();
    p.camera = camera.data();
    const jpo_ransac_options o =
        options(max_error_px, confidence, min_iterations, max_iterations, refine, min_inlier_ratio);
    std::vector<jpo_pose_result> r(n);
    std::vector<uint8_t> mask(want_inliers ? xy.shape(0) : 0);
    int rc;
    {
      nb::gil_scoped_release release;
      rc = jpo_absolute_pose(ctx_, &p, &o, r.data(), want_inliers ? mask.data() : nullptr);
    }
    if (rc != 0) fail();
    return gather(r, std::move(mask), want_inliers);
  }

 private:
  // The problems the offsets delimit; the last offset must be the number of matches.
  static size_t check_offsets(const In<uint32_t>& offsets, size_t matches) {
    const size_t k = rows(offsets, 0, "offsets");
    if (k == 0) throw nb::value_error("offsets must have n + 1 entries");
    if (offsets.data()[0] != 0 || offsets.data()[k - 1] != matches)
      throw nb::value_error("offsets must start at 0 and end at the number of matches");
    return k - 1;
  }

  jpo_ctx* ctx_ = nullptr;
};

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "poselib-gpu's C API (include/jpo/jpo.h); use the poselib_gpu package, which wraps it.";

  m.def("camera_param_count", &jpo_camera_param_count, "model"_a,
        "Number of parameters of a camera model id (3 .. 8), 0 for an unknown model.");
  m.def("backend_available", [](int32_t b) { return jpo_backend_available(b) != 0; }, "backend"_a,
        "Whether this build has the backend id (CPU and auto: always).");

  nb::class_<Context>(m, "Context")
      .def(nb::init<int32_t, uint32_t, int32_t, bool, uint32_t>(), "backend"_a, "seed"_a, "threads"_a,
           "cpu_float"_a, "gpu_round"_a)
      .def_prop_ro("backend", &Context::backend)
      .def("relative", &Context::relative, "offsets"_a, "xy_a"_a, "xy_b"_a, "models"_a, "params"_a,
           "camera_a"_a, "camera_b"_a, "max_error_px"_a, "confidence"_a, "min_iterations"_a, "max_iterations"_a,
           "refine"_a, "min_inlier_ratio"_a, "want_inliers"_a)
      .def("absolute", &Context::absolute, "offsets"_a, "xy"_a, "xyz"_a, "models"_a, "params"_a, "camera"_a,
           "max_error_px"_a, "confidence"_a, "min_iterations"_a, "max_iterations"_a, "refine"_a,
           "min_inlier_ratio"_a, "want_inliers"_a);
}
