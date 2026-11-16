// The WebGPU backend on a device the caller created (jpo_config.webgpu_instance / webgpu_device), as an
// app does that runs other GPU work on the same device: a small batch of relative poses and of absolute
// poses must be solved, with the same results as on the backend's own device, and the caller's device
// must still work after jpo_destroy.
//   webgpu_shared_test
//
// The problems are generated here (pinhole cameras, a known motion, 60 % inliers) with the public header
// only: this test is built without the include paths of cli/synth.h.
#include <webgpu/webgpu.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "jpo/jpo.h"

namespace {

const double kPi = 3.14159265358979323846;
const double kWidth = 1600.0, kHeight = 1200.0;

struct Rng {
  uint64_t s;
  double uniform() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return static_cast<double>((z ^ (z >> 31)) >> 11) * (1.0 / 9007199254740992.0);
  }
  double uniform(double a, double b) { return a + (b - a) * uniform(); }
  double normal() { return std::sqrt(-2.0 * std::log(uniform() + 1e-300)) * std::cos(2.0 * kPi * uniform()); }
};

void quat_to_rotmat(const double q[4], double R[9]) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  const double m[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
                       2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                       2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)};
  for (int i = 0; i < 9; ++i) R[i] = m[i];
}

// kind 0: image pairs, kind 1: images against world points. All problems share one pinhole camera.
struct Batch {
  uint32_t kind = 0;
  std::vector<uint32_t> offsets, camera;
  std::vector<double> xy_a, xy_b, xyz, gt;  // gt: 7 per problem
  jpo_camera cam;
};

Batch make_batch(uint32_t kind, uint32_t count, uint32_t points, uint64_t seed) {
  Batch b;
  b.kind = kind;
  Rng rng{seed};
  b.cam.model = JPO_PINHOLE;
  b.cam.reserved = 0;
  for (double& p : b.cam.params) p = 0.0;
  const double f = 1200.0, cx = 800.0, cy = 600.0;
  b.cam.params[0] = f, b.cam.params[1] = f, b.cam.params[2] = cx, b.cam.params[3] = cy;
  b.offsets.push_back(0);
  for (uint32_t p = 0; p < count; ++p) {
    // The motion: a rotation of up to 20 degrees about a random axis, a translation of length 0.9.
    double axis[3] = {rng.normal(), rng.normal(), rng.normal()}, t[3] = {rng.normal(), rng.normal(), 0.3 * rng.normal()};
    const double an = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    const double tn = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    const double angle = rng.uniform(2.0, 20.0) * kPi / 180.0;
    double q[4] = {std::cos(0.5 * angle), 0, 0, 0}, R[9];
    for (int d = 0; d < 3; ++d) q[1 + d] = std::sin(0.5 * angle) * axis[d] / an, t[d] *= (kind == 0 ? 0.9 : 3.0) / tn;
    quat_to_rotmat(q, R);
    for (int d = 0; d < 4; ++d) b.gt.push_back(q[d]);
    for (int d = 0; d < 3; ++d) b.gt.push_back(kind == 0 ? t[d] / 0.9 : t[d]);
    for (uint32_t i = 0; i < points; ++i) {
      const bool inlier = i % 5 < 3;
      double ua = rng.uniform(0.0, kWidth - 1.0), va = rng.uniform(0.0, kHeight - 1.0);
      double ub = rng.uniform(0.0, kWidth - 1.0), vb = rng.uniform(0.0, kHeight - 1.0);
      double W[3] = {rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0)};
      for (int tries = 0; inlier && tries < 1000; ++tries) {
        const double depth = rng.uniform(4.0, 8.0);
        const double u = rng.uniform(0.0, kWidth - 1.0), v = rng.uniform(0.0, kHeight - 1.0);
        const double X[3] = {(u - cx) / f * depth, (v - cy) / f * depth, depth};
        if (kind == 0) {
          // Camera b sees Y = R X + t.
          double Y[3];
          for (int r = 0; r < 3; ++r) Y[r] = R[3 * r] * X[0] + R[3 * r + 1] * X[1] + R[3 * r + 2] * X[2] + t[r];
          const double pu = f * Y[0] / Y[2] + cx, pv = f * Y[1] / Y[2] + cy;
          if (!(Y[2] > 0.1) || pu < 0.0 || pv < 0.0 || pu > kWidth - 1.0 || pv > kHeight - 1.0) continue;
          ub = pu + 0.5 * rng.normal(), vb = pv + 0.5 * rng.normal();
        } else {
          // The camera sees X = R W + t, so W = R' (X - t).
          for (int c = 0; c < 3; ++c) W[c] = R[c] * (X[0] - t[0]) + R[3 + c] * (X[1] - t[1]) + R[6 + c] * (X[2] - t[2]);
        }
        ua = u + 0.5 * rng.normal(), va = v + 0.5 * rng.normal();
        break;
      }
      b.xy_a.push_back(ua), b.xy_a.push_back(va);
      if (kind == 0) b.xy_b.push_back(ub), b.xy_b.push_back(vb);
      else b.xyz.insert(b.xyz.end(), W, W + 3);
    }
    b.offsets.push_back(b.offsets.back() + points);
    b.camera.push_back(0);
  }
  return b;
}

bool run(const jpo_config& config, const Batch& b, std::vector<jpo_pose_result>* res) {
  jpo_ctx* ctx = jpo_create(&config);
  if (!ctx) {
    std::fprintf(stderr, "jpo_create: %s\n", jpo_last_error());
    return false;
  }
  const uint32_t count = static_cast<uint32_t>(b.camera.size());
  res->assign(count, jpo_pose_result{});
  int rc = 0;
  for (int pass = 0; pass < 2 && rc == 0; ++pass) {  // twice: the buffers are reused
    if (b.kind == 0) {
      const jpo_pairs pairs = {count, 1, b.offsets.data(), b.xy_a.data(), b.xy_b.data(), &b.cam, b.camera.data(), b.camera.data()};
      rc = jpo_relative_pose(ctx, &pairs, nullptr, res->data(), nullptr);
    } else {
      const jpo_resections rs = {count, 1, b.offsets.data(), b.xy_a.data(), b.xyz.data(), &b.cam, b.camera.data()};
      rc = jpo_absolute_pose(ctx, &rs, nullptr, res->data(), nullptr);
    }
  }
  if (rc != 0) std::fprintf(stderr, "%s: %s\n", b.kind == 0 ? "jpo_relative_pose" : "jpo_absolute_pose", jpo_last_error());
  jpo_destroy(ctx);
  return rc == 0;
}

// Problems solved: rotation within 1 degree, and the translation direction within 5 degrees (relative)
// or the translation within 0.05 (absolute; the scene is 4 to 8 away).
int solved(const Batch& b, const std::vector<jpo_pose_result>& res) {
  int ok = 0;
  for (size_t i = 0; i < res.size(); ++i) {
    if (res[i].status != JPO_OK) continue;
    const double* e = res[i].pose;
    const double* g = &b.gt[7 * i];
    const double dot = std::fabs(e[0] * g[0] + e[1] * g[1] + e[2] * g[2] + e[3] * g[3]);
    const double rot = 2.0 * std::acos(std::fmin(1.0, dot)) * 180.0 / kPi;
    double tr;
    if (b.kind == 0) {
      const double c = e[4] * g[4] + e[5] * g[5] + e[6] * g[6];
      tr = std::acos(std::fmax(-1.0, std::fmin(1.0, c))) * 180.0 / kPi;
    } else {
      tr = std::sqrt((e[4] - g[4]) * (e[4] - g[4]) + (e[5] - g[5]) * (e[5] - g[5]) + (e[6] - g[6]) * (e[6] - g[6])) * 100.0;
    }
    if (rot < 1.0 && tr < 5.0 && res[i].n_inliers >= 5) ++ok;
  }
  return ok;
}

void wait(WGPUInstance instance, WGPUFuture f) {
  WGPUFutureWaitInfo info{f, false};
  wgpuInstanceWaitAny(instance, 1, &info, UINT64_MAX);
}

}  // namespace

int main() {
  WGPUInstanceDescriptor idesc = WGPU_INSTANCE_DESCRIPTOR_INIT;
  const WGPUInstanceFeatureName feature = WGPUInstanceFeatureName_TimedWaitAny;
  idesc.requiredFeatureCount = 1;
  idesc.requiredFeatures = &feature;
  WGPUInstance instance = wgpuCreateInstance(&idesc);
  WGPUAdapter adapter = nullptr;
  WGPURequestAdapterCallbackInfo acb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
  acb.mode = WGPUCallbackMode_WaitAnyOnly;
  acb.callback = [](WGPURequestAdapterStatus st, WGPUAdapter a, WGPUStringView, void* u, void*) {
    if (st == WGPURequestAdapterStatus_Success) *static_cast<WGPUAdapter*>(u) = a;
  };
  acb.userdata1 = &adapter;
  wait(instance, wgpuInstanceRequestAdapter(instance, nullptr, acb));
  if (!adapter) {
    std::fprintf(stderr, "no WebGPU adapter\n");
    return 1;
  }
  // The caller's device with default limits, as an app that knows nothing of this library has it. An
  // error nobody catches is printed, so a validation error of the library's fails the test.
  int device_errors = 0;
  WGPUDeviceDescriptor ddesc = WGPU_DEVICE_DESCRIPTOR_INIT;
  ddesc.uncapturedErrorCallbackInfo.callback = [](WGPUDevice const*, WGPUErrorType, WGPUStringView m, void* u, void*) {
    std::fprintf(stderr, "device error: %.*s\n", static_cast<int>(m.length), m.data);
    ++*static_cast<int*>(u);
  };
  ddesc.uncapturedErrorCallbackInfo.userdata1 = &device_errors;
  WGPUDevice device = nullptr;
  WGPURequestDeviceCallbackInfo dcb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
  dcb.mode = WGPUCallbackMode_WaitAnyOnly;
  dcb.callback = [](WGPURequestDeviceStatus st, WGPUDevice d, WGPUStringView, void* u, void*) {
    if (st == WGPURequestDeviceStatus_Success) *static_cast<WGPUDevice*>(u) = d;
  };
  dcb.userdata1 = &device;
  wait(instance, wgpuAdapterRequestDevice(adapter, &ddesc, dcb));
  if (!device) {
    std::fprintf(stderr, "no WebGPU device\n");
    return 1;
  }

  int failed = 0;
  for (uint32_t kind = 0; kind < 2; ++kind) {
    const uint32_t count = 12;
    const Batch batch = make_batch(kind, count, 500, 7 + kind);
    jpo_config own;
    jpo_config_init(&own);
    own.backend = JPO_BACKEND_WEBGPU;
    jpo_config shared = own;
    shared.webgpu_instance = instance;
    shared.webgpu_device = device;
    std::vector<jpo_pose_result> r_own, r_shared;
    const bool ran = run(own, batch, &r_own) && run(shared, batch, &r_shared);
    const int ok_own = ran ? solved(batch, r_own) : 0, ok_shared = ran ? solved(batch, r_shared) : 0;
    // The same kernels on the same GPU: the same poses, up to the last bits.
    double diff = 0.0;
    for (size_t i = 0; ran && i < r_own.size(); ++i)
      for (int d = 0; d < 7; ++d) diff = std::fmax(diff, std::fabs(r_own[i].pose[d] - r_shared[i].pose[d]));
    const bool good = ran && ok_own >= static_cast<int>(count) - 1 && ok_shared >= static_cast<int>(count) - 1 && diff < 1e-6;
    std::printf("%s: %d of %u solved on its own device, %d on the caller's, poses differ by %.3g: %s\n",
                kind == 0 ? "relpose" : "pnp", ok_own, count, ok_shared, diff, good ? "ok" : "FAILED");
    failed += !good;
  }
  // The caller's device is still usable after the library released its references.
  WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
  bd.size = 256;
  bd.usage = WGPUBufferUsage_CopyDst;
  WGPUBuffer b = wgpuDeviceCreateBuffer(device, &bd);
  const bool alive = b != nullptr;
  if (b) wgpuBufferRelease(b);
  std::printf("caller's device after jpo_destroy: %s, %d uncaught device errors\n", alive ? "usable" : "LOST", device_errors);
  failed += !alive + (device_errors != 0);
  wgpuDeviceRelease(device);
  wgpuAdapterRelease(adapter);
  wgpuInstanceRelease(instance);
  std::printf(failed ? "FAILED (%d)\n" : "shared device ok\n", failed);
  return failed ? 1 : 0;
}
