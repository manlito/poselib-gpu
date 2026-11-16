#include "core/geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace jpo {

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kPi = 3.14159265358979323846;

inline double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline void cross3(const double* a, const double* b, double* c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}
inline bool normalize3(double* a) {
  const double n = std::sqrt(dot3(a, a));
  if (!(n > 0.0)) return false;
  a[0] /= n;
  a[1] /= n;
  a[2] /= n;
  return true;
}
inline void mat3_vec(const double* R, const double* x, double* y) {
  y[0] = R[0] * x[0] + R[1] * x[1] + R[2] * x[2];
  y[1] = R[3] * x[0] + R[4] * x[1] + R[5] * x[2];
  y[2] = R[6] * x[0] + R[7] * x[1] + R[8] * x[2];
}
inline void skew(const double* t, double* S) {
  S[0] = 0.0;
  S[1] = -t[2];
  S[2] = t[1];
  S[3] = t[2];
  S[4] = 0.0;
  S[5] = -t[0];
  S[6] = -t[1];
  S[7] = t[0];
  S[8] = 0.0;
}

// Solves A x = b for a small symmetric positive system by elimination with partial pivoting.
template <int N>
bool solve_small(double* A, double* b, double* x) {
  for (int c = 0; c < N; ++c) {
    int piv = c;
    for (int r = c + 1; r < N; ++r)
      if (std::abs(A[r * N + c]) > std::abs(A[piv * N + c])) piv = r;
    if (!(std::abs(A[piv * N + c]) > 1e-300)) return false;
    if (piv != c) {
      for (int k = 0; k < N; ++k) std::swap(A[c * N + k], A[piv * N + k]);
      std::swap(b[c], b[piv]);
    }
    for (int r = c + 1; r < N; ++r) {
      const double f = A[r * N + c] / A[c * N + c];
      for (int k = c; k < N; ++k) A[r * N + k] -= f * A[c * N + k];
      b[r] -= f * b[c];
    }
  }
  for (int r = N - 1; r >= 0; --r) {
    double s = b[r];
    for (int k = r + 1; k < N; ++k) s -= A[r * N + k] * x[k];
    x[r] = s / A[r * N + r];
  }
  return true;
}

// Levenberg-Marquardt on a state with N tangent parameters. accum(state, JtJ, Jtr) returns the cost and
// the normal equations; apply(state, delta) returns the moved state.
template <int N, class State, class Accum, class Apply>
void levenberg_marquardt(State* state, Accum accum, Apply apply, int max_iterations) {
  double JtJ[N * N], Jtr[N];
  double cost = accum(*state, JtJ, Jtr);
  double lambda = 1e-3;
  for (int iter = 0; iter < max_iterations; ++iter) {
    double grad = 0.0;
    for (int i = 0; i < N; ++i) grad = std::max(grad, std::abs(Jtr[i]));
    if (grad < 1e-12) break;
    double A[N * N], b[N], delta[N];
    for (int i = 0; i < N * N; ++i) A[i] = JtJ[i];
    for (int i = 0; i < N; ++i) {
      A[i * N + i] += lambda * (JtJ[i * N + i] + 1e-12);
      b[i] = -Jtr[i];
    }
    if (!solve_small<N>(A, b, delta)) {
      lambda *= 10.0;
      if (lambda > 1e10) break;
      continue;
    }
    State cand = apply(*state, delta);
    double JtJ2[N * N], Jtr2[N];
    const double cost2 = accum(cand, JtJ2, Jtr2);
    if (cost2 < cost) {
      *state = cand;
      const double gain = cost - cost2;
      for (int i = 0; i < N * N; ++i) JtJ[i] = JtJ2[i];
      for (int i = 0; i < N; ++i) Jtr[i] = Jtr2[i];
      const double prev = cost;
      cost = cost2;
      lambda = std::max(lambda * 0.1, 1e-10);
      double step = 0.0;
      for (int i = 0; i < N; ++i) step += delta[i] * delta[i];
      if (gain < 1e-10 * prev || step < 1e-16) break;
    } else {
      lambda *= 10.0;
      if (lambda > 1e10) break;
    }
  }
}

// ---- distortion ----

// Normalized (u, v) -> distorted normalized coordinates, for the pinhole-type models.
inline void distort(const jpo_camera& cam, double u, double v, double* ud, double* vd) {
  const double* p = cam.params;
  const double r2 = u * u + v * v;
  switch (cam.model) {
    case JPO_SIMPLE_RADIAL: {
      const double s = 1.0 + p[3] * r2;
      *ud = u * s;
      *vd = v * s;
      return;
    }
    case JPO_RADIAL: {
      const double s = 1.0 + p[3] * r2 + p[4] * r2 * r2;
      *ud = u * s;
      *vd = v * s;
      return;
    }
    case JPO_OPENCV: {
      const double k1 = p[4], k2 = p[5], p1 = p[6], p2 = p[7];
      const double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
      *ud = u * radial + 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u);
      *vd = v * radial + 2.0 * p2 * u * v + p1 * (r2 + 2.0 * v * v);
      return;
    }
    default:
      *ud = u;
      *vd = v;
  }
}

inline bool has_distortion(int model) { return model == JPO_SIMPLE_RADIAL || model == JPO_RADIAL || model == JPO_OPENCV; }
inline bool two_focals(int model) { return model == JPO_PINHOLE || model == JPO_OPENCV || model == JPO_OPENCV_FISHEYE; }

inline double fisheye_theta_d(const double* k, double theta) {
  const double t2 = theta * theta;
  return theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
}

}  // namespace

int camera_param_count(int model) {
  switch (model) {
    case JPO_SIMPLE_PINHOLE: return 3;
    case JPO_PINHOLE: return 4;
    case JPO_SIMPLE_RADIAL: return 4;
    case JPO_RADIAL: return 5;
    case JPO_OPENCV: return 8;
    case JPO_OPENCV_FISHEYE: return 8;
    default: return 0;
  }
}

double camera_focal(const jpo_camera& cam) {
  return two_focals(cam.model) ? 0.5 * (cam.params[0] + cam.params[1]) : cam.params[0];
}

bool camera_unproject(const jpo_camera& cam, double px, double py, double* x, double* y) {
  const double* p = cam.params;
  double ud, vd;
  if (two_focals(cam.model)) {
    ud = (px - p[2]) / p[0];
    vd = (py - p[3]) / p[1];
  } else {
    ud = (px - p[1]) / p[0];
    vd = (py - p[2]) / p[0];
  }
  if (cam.model == JPO_OPENCV_FISHEYE) {
    const double rd = std::sqrt(ud * ud + vd * vd);
    if (rd < 1e-12) {
      *x = ud;
      *y = vd;
      return true;
    }
    double theta = rd;
    for (int i = 0; i < 20; ++i) {
      const double h = 1e-6;
      const double f = fisheye_theta_d(p + 4, theta) - rd;
      const double df = (fisheye_theta_d(p + 4, theta + h) - fisheye_theta_d(p + 4, theta - h)) / (2.0 * h);
      if (!(std::abs(df) > 1e-12)) break;
      const double step = f / df;
      theta -= step;
      if (std::abs(step) < 1e-14) break;
    }
    if (!(theta > 0.0) || !(theta < 0.5 * kPi - 1e-6)) {  // not on the image plane in front of the camera
      *x = kNaN;
      *y = kNaN;
      return false;
    }
    const double s = std::tan(theta) / rd;
    *x = ud * s;
    *y = vd * s;
    return true;
  }
  if (!has_distortion(cam.model)) {
    *x = ud;
    *y = vd;
    return true;
  }
  // Newton on distort(u, v) = (ud, vd), numeric Jacobian.
  double u = ud, v = vd;
  for (int i = 0; i < 25; ++i) {
    double fu, fv, fu_u, fv_u, fu_v, fv_v;
    const double h = 1e-7;
    distort(cam, u, v, &fu, &fv);
    distort(cam, u + h, v, &fu_u, &fv_u);
    distort(cam, u, v + h, &fu_v, &fv_v);
    const double j00 = (fu_u - fu) / h, j10 = (fv_u - fv) / h, j01 = (fu_v - fu) / h, j11 = (fv_v - fv) / h;
    const double det = j00 * j11 - j01 * j10;
    if (!(std::abs(det) > 1e-12)) break;
    const double ru = fu - ud, rv = fv - vd;
    const double du = (j11 * ru - j01 * rv) / det;
    const double dv = (-j10 * ru + j00 * rv) / det;
    u -= du;
    v -= dv;
    if (du * du + dv * dv < 1e-28) break;
  }
  *x = u;
  *y = v;
  return std::isfinite(u) && std::isfinite(v);
}

bool camera_project(const jpo_camera& cam, const double X[3], double* px, double* py) {
  const double* p = cam.params;
  double ud, vd;
  if (cam.model == JPO_OPENCV_FISHEYE) {
    const double r = std::sqrt(X[0] * X[0] + X[1] * X[1]);
    if (r < 1e-12 * std::abs(X[2]) || r == 0.0) {
      if (!(X[2] > 0.0)) return false;
      ud = X[0] / X[2];
      vd = X[1] / X[2];
    } else {
      const double theta = std::atan2(r, X[2]);
      const double s = fisheye_theta_d(p + 4, theta) / r;
      ud = X[0] * s;
      vd = X[1] * s;
    }
  } else {
    if (!(X[2] > 0.0)) return false;
    distort(cam, X[0] / X[2], X[1] / X[2], &ud, &vd);
  }
  if (two_focals(cam.model)) {
    *px = p[0] * ud + p[2];
    *py = p[1] * vd + p[3];
  } else {
    *px = p[0] * ud + p[1];
    *py = p[0] * vd + p[2];
  }
  return true;
}

// ---- rotations ----

void quat_to_rotmat(const double q[4], double R[9]) {
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const double w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n;
  R[0] = 1.0 - 2.0 * (y * y + z * z);
  R[1] = 2.0 * (x * y - w * z);
  R[2] = 2.0 * (x * z + w * y);
  R[3] = 2.0 * (x * y + w * z);
  R[4] = 1.0 - 2.0 * (x * x + z * z);
  R[5] = 2.0 * (y * z - w * x);
  R[6] = 2.0 * (x * z - w * y);
  R[7] = 2.0 * (y * z + w * x);
  R[8] = 1.0 - 2.0 * (x * x + y * y);
}

void rotmat_to_quat(const double R[9], double q[4]) {
  const double tr = R[0] + R[4] + R[8];
  if (tr > 0.0) {
    const double s = std::sqrt(tr + 1.0) * 2.0;
    q[0] = 0.25 * s;
    q[1] = (R[7] - R[5]) / s;
    q[2] = (R[2] - R[6]) / s;
    q[3] = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const double s = std::sqrt(1.0 + R[0] - R[4] - R[8]) * 2.0;
    q[0] = (R[7] - R[5]) / s;
    q[1] = 0.25 * s;
    q[2] = (R[1] + R[3]) / s;
    q[3] = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const double s = std::sqrt(1.0 + R[4] - R[0] - R[8]) * 2.0;
    q[0] = (R[2] - R[6]) / s;
    q[1] = (R[1] + R[3]) / s;
    q[2] = 0.25 * s;
    q[3] = (R[5] + R[7]) / s;
  } else {
    const double s = std::sqrt(1.0 + R[8] - R[0] - R[4]) * 2.0;
    q[0] = (R[3] - R[1]) / s;
    q[1] = (R[2] + R[6]) / s;
    q[2] = (R[5] + R[7]) / s;
    q[3] = 0.25 * s;
  }
  double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (q[0] < 0.0) n = -n;
  for (int i = 0; i < 4; ++i) q[i] /= n;
}

void rotmat_exp(const double w[3], double R[9]) {
  const double th2 = dot3(w, w);
  const double th = std::sqrt(th2);
  double a, b;  // R = I + a [w]x + b [w]x^2
  if (th < 1e-8) {
    a = 1.0 - th2 / 6.0;
    b = 0.5 - th2 / 24.0;
  } else {
    a = std::sin(th) / th;
    b = (1.0 - std::cos(th)) / th2;
  }
  double S[9], S2[9];
  skew(w, S);
  mat3_mul(S, S, S2);
  for (int i = 0; i < 9; ++i) R[i] = a * S[i] + b * S2[i];
  R[0] += 1.0;
  R[4] += 1.0;
  R[8] += 1.0;
}

void mat3_mul(const double A[9], const double B[9], double C[9]) {
  double T[9];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) T[3 * r + c] = A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c];
  for (int i = 0; i < 9; ++i) C[i] = T[i];
}

namespace {
// Re-orthonormalizes a rotation that has drifted (first two rows, then their cross product).
void orthonormalize(double R[9]) {
  double q[4];
  rotmat_to_quat(R, q);
  quat_to_rotmat(q, R);
}
}  // namespace

// ---- relative pose ----

void essential_from_pose(const RelPose& pose, double E[9]) {
  double S[9];
  skew(pose.t, S);
  mat3_mul(S, pose.R, E);
}

void poses_from_essential(const double E[9], RelPose out[4]) {
  double c0[3] = {E[0], E[3], E[6]}, c1[3] = {E[1], E[4], E[7]}, c2[3] = {E[2], E[5], E[8]};
  double u12[3], u13[3], u23[3];
  cross3(c0, c1, u12);
  cross3(c0, c2, u13);
  cross3(c1, c2, u23);
  const double n12 = dot3(u12, u12), n13 = dot3(u13, u13), n23 = dot3(u23, u23);
  double uw0[3], uw1[3], uw2[3];  // columns of U W
  const double* e = c0;
  const double* u = u12;
  if (n12 > n13) {
    if (n12 > n23) {
      e = c0;
      u = u12;
    } else {
      e = c1;
      u = u23;
    }
  } else {
    if (n13 > n23) {
      e = c0;
      u = u13;
    } else {
      e = c1;
      u = u23;
    }
  }
  for (int i = 0; i < 3; ++i) {
    uw1[i] = e[i];
    uw2[i] = u[i];
  }
  normalize3(uw1);
  normalize3(uw2);
  cross3(uw2, uw1, uw0);
  for (int i = 0; i < 3; ++i) uw0[i] = -uw0[i];

  double vt0[3], vt1[3], vt2[3];  // rows of V'
  for (int c = 0; c < 3; ++c) {
    vt0[c] = uw1[0] * E[c] + uw1[1] * E[3 + c] + uw1[2] * E[6 + c];
    vt1[c] = -(uw0[0] * E[c] + uw0[1] * E[3 + c] + uw0[2] * E[6 + c]);
  }
  normalize3(vt0);
  const double d = dot3(vt0, vt1);
  for (int c = 0; c < 3; ++c) vt1[c] -= d * vt0[c];
  normalize3(vt1);
  cross3(vt0, vt1, vt2);

  for (int k = 0; k < 2; ++k) {
    const double s = k == 0 ? 1.0 : -1.0;  // the second rotation flips the first two columns of U W
    double R[9];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) R[3 * r + c] = s * uw0[r] * vt0[c] + s * uw1[r] * vt1[c] + uw2[r] * vt2[c];
    for (int j = 0; j < 2; ++j) {
      RelPose& p = out[2 * k + j];
      for (int i = 0; i < 9; ++i) p.R[i] = R[i];
      for (int i = 0; i < 3; ++i) p.t[i] = j == 0 ? uw2[i] : -uw2[i];
    }
  }
}

bool check_cheirality(const RelPose& pose, const double b1[3], const double b2[3], double min_depth) {
  double Rx1[3];
  mat3_vec(pose.R, b1, Rx1);
  const double a = -dot3(Rx1, b2);
  const double bb1 = -dot3(Rx1, pose.t);
  const double bb2 = dot3(b2, pose.t);
  const double lambda1 = bb1 - a * bb2;
  const double lambda2 = -a * bb1 + bb2;
  min_depth = min_depth * (1.0 - a * a);
  return lambda1 > min_depth && lambda2 > min_depth;
}

namespace {
inline void bearings(const double* p, double* b1, double* b2) {
  const double i1 = 1.0 / std::sqrt(p[0] * p[0] + p[1] * p[1] + 1.0);
  const double i2 = 1.0 / std::sqrt(p[2] * p[2] + p[3] * p[3] + 1.0);
  b1[0] = p[0] * i1;
  b1[1] = p[1] * i1;
  b1[2] = i1;
  b2[0] = p[2] * i2;
  b2[1] = p[3] * i2;
  b2[2] = i2;
}
inline double sampson_sq(const double* E, const double* p) {
  const double a0 = E[0] * p[0] + E[1] * p[1] + E[2];
  const double a1 = E[3] * p[0] + E[4] * p[1] + E[5];
  const double a2 = E[6] * p[0] + E[7] * p[1] + E[8];
  const double b0 = E[0] * p[2] + E[3] * p[3] + E[6];
  const double b1 = E[1] * p[2] + E[4] * p[3] + E[7];
  const double C = p[2] * a0 + p[3] * a1 + a2;
  return C * C / (a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1);
}
}  // namespace

double score_relpose(const RelPose& pose, const double* pts, size_t n, double sq_thr, size_t* n_inliers,
                     uint8_t* inliers) {
  double E[9];
  essential_from_pose(pose, E);
  double score = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < n; ++i) {
    const double* p = pts + 4 * i;
    const double r2 = sampson_sq(E, p);
    bool in = r2 < sq_thr;
    if (in) {
      double b1[3], b2[3];
      bearings(p, b1, b2);
      in = check_cheirality(pose, b1, b2);
    }
    score += in ? r2 / sq_thr : 1.0;
    count += in ? 1 : 0;
    if (inliers) inliers[i] = in ? 1 : 0;
  }
  if (n_inliers) *n_inliers = count;
  return score;
}

void refine_relpose(RelPose* pose, const double* pts, size_t n, double sq_thr, int max_iterations) {
  auto accum = [&](const RelPose& s, double* JtJ, double* Jtr) {
    for (int i = 0; i < 25; ++i) JtJ[i] = 0.0;
    for (int i = 0; i < 5; ++i) Jtr[i] = 0.0;
    double E[9];
    essential_from_pose(s, E);
    // dE / d(parameter): rotation R (I + [w]x), translation t + B d on the sphere.
    double dE[5][9];
    double S[9];
    skew(s.t, S);
    for (int k = 0; k < 3; ++k) {
      double w[3] = {0.0, 0.0, 0.0}, W[9], RW[9];
      w[k] = 1.0;
      skew(w, W);
      mat3_mul(s.R, W, RW);
      mat3_mul(S, RW, dE[k]);
    }
    double axis[3] = {0.0, 0.0, 0.0};
    int small = 0;
    if (std::abs(s.t[1]) < std::abs(s.t[small])) small = 1;
    if (std::abs(s.t[2]) < std::abs(s.t[small])) small = 2;
    axis[small] = 1.0;
    double bt1[3], bt2[3];
    cross3(s.t, axis, bt1);
    normalize3(bt1);
    cross3(s.t, bt1, bt2);
    double B1[9], B2[9];
    skew(bt1, B1);
    skew(bt2, B2);
    mat3_mul(B1, s.R, dE[3]);
    mat3_mul(B2, s.R, dE[4]);

    double cost = 0.0;
    const double inv = 1.0 / sq_thr;
    for (size_t i = 0; i < n; ++i) {
      const double* p = pts + 4 * i;
      const double x1[3] = {p[0], p[1], 1.0}, x2[3] = {p[2], p[3], 1.0};
      const double a0 = E[0] * p[0] + E[1] * p[1] + E[2];
      const double a1 = E[3] * p[0] + E[4] * p[1] + E[5];
      const double a2 = E[6] * p[0] + E[7] * p[1] + E[8];
      const double b0 = E[0] * p[2] + E[3] * p[3] + E[6];
      const double b1 = E[1] * p[2] + E[4] * p[3] + E[7];
      const double C = p[2] * a0 + p[3] * a1 + a2;
      const double D = a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1;
      const double r2 = C * C / D;
      if (!(r2 < sq_thr)) {
        cost += 1.0;
        continue;
      }
      cost += r2 * inv;
      const double isd = 1.0 / std::sqrt(D);
      const double r = C * isd;
      const double a[2] = {a0, a1}, b[2] = {b0, b1};
      double J[5];
      for (int k = 0; k < 5; ++k) {
        double dC = 0.0, dD = 0.0;
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            const double de = dE[k][3 * row + col];
            dC += de * x2[row] * x1[col];
            if (row < 2) dD += 2.0 * a[row] * x1[col] * de;
            if (col < 2) dD += 2.0 * b[col] * x2[row] * de;
          }
        }
        J[k] = dC * isd - 0.5 * r * dD / D;
      }
      for (int k = 0; k < 5; ++k) {
        Jtr[k] += J[k] * r * inv;
        for (int l = 0; l < 5; ++l) JtJ[5 * k + l] += J[k] * J[l] * inv;
      }
    }
    return cost;
  };
  auto apply = [](const RelPose& s, const double* d) {
    RelPose o;
    double dR[9];
    rotmat_exp(d, dR);
    mat3_mul(s.R, dR, o.R);
    orthonormalize(o.R);
    double axis[3] = {0.0, 0.0, 0.0};
    int small = 0;
    if (std::abs(s.t[1]) < std::abs(s.t[small])) small = 1;
    if (std::abs(s.t[2]) < std::abs(s.t[small])) small = 2;
    axis[small] = 1.0;
    double bt1[3], bt2[3];
    cross3(s.t, axis, bt1);
    normalize3(bt1);
    cross3(s.t, bt1, bt2);
    for (int i = 0; i < 3; ++i) o.t[i] = s.t[i] + d[3] * bt1[i] + d[4] * bt2[i];
    normalize3(o.t);
    return o;
  };
  levenberg_marquardt<5>(pose, accum, apply, max_iterations);
}

double median_triangulation_angle(const RelPose& pose, const double* pts, size_t n, const uint8_t* inliers) {
  std::vector<double> angles;
  angles.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    if (inliers && !inliers[i]) continue;
    double b1[3], b2[3], r[3];
    bearings(pts + 4 * i, b1, b2);
    mat3_vec(pose.R, b1, r);
    double c[3];
    cross3(r, b2, c);
    angles.push_back(std::atan2(std::sqrt(dot3(c, c)), dot3(r, b2)));
  }
  if (angles.empty()) return 0.0;
  auto mid = angles.begin() + angles.size() / 2;
  std::nth_element(angles.begin(), mid, angles.end());
  return *mid * 180.0 / kPi;
}

// ---- absolute pose ----

double score_abspose(const AbsPose& pose, const double* pts, size_t n, double sq_thr, size_t* n_inliers,
                     uint8_t* inliers) {
  double score = 0.0;
  size_t count = 0;
  const double inv = 1.0 / sq_thr;
  for (size_t i = 0; i < n; ++i) {
    const double* p = pts + 5 * i;
    double Z[3];
    mat3_vec(pose.R, p + 2, Z);
    Z[0] += pose.t[0];
    Z[1] += pose.t[1];
    Z[2] += pose.t[2];
    bool in = false;
    double r2 = 0.0;
    if (Z[2] > 0.0) {
      const double d0 = Z[0] / Z[2] - p[0], d1 = Z[1] / Z[2] - p[1];
      r2 = d0 * d0 + d1 * d1;
      in = r2 < sq_thr;
    }
    score += in ? r2 * inv : 1.0;
    count += in ? 1 : 0;
    if (inliers) inliers[i] = in ? 1 : 0;
  }
  if (n_inliers) *n_inliers = count;
  return score;
}

namespace {
AbsPose apply_abs(const AbsPose& s, const double* d) {
  AbsPose o;
  double dR[9];
  rotmat_exp(d, dR);
  mat3_mul(dR, s.R, o.R);
  orthonormalize(o.R);
  for (int i = 0; i < 3; ++i) o.t[i] = s.t[i] + d[3 + i];
  return o;
}

// Adds one 2D residual r with Jacobian dr/dZ (2x3) at the camera-frame point Z = R X + t; the pose
// moves as R <- exp(w) R, t <- t + d.
inline void add_abs_residual(const double* Z, const double* RX, const double r[2], const double Jz[6], double w,
                             double* JtJ, double* Jtr) {
  (void)Z;
  // dZ/dw = -[R X]x, dZ/dt = I.
  double J[2][6];
  for (int k = 0; k < 2; ++k) {
    const double* j = Jz + 3 * k;
    J[k][0] = j[1] * (-RX[2]) + j[2] * RX[1];
    J[k][1] = j[0] * RX[2] + j[2] * (-RX[0]);
    J[k][2] = j[0] * (-RX[1]) + j[1] * RX[0];
    J[k][3] = j[0];
    J[k][4] = j[1];
    J[k][5] = j[2];
  }
  for (int k = 0; k < 2; ++k)
    for (int a = 0; a < 6; ++a) {
      Jtr[a] += J[k][a] * r[k] * w;
      for (int b = 0; b < 6; ++b) JtJ[6 * a + b] += J[k][a] * J[k][b] * w;
    }
}
}  // namespace

void refine_abspose(AbsPose* pose, const double* pts, size_t n, double sq_thr, int max_iterations) {
  auto accum = [&](const AbsPose& s, double* JtJ, double* Jtr) {
    for (int i = 0; i < 36; ++i) JtJ[i] = 0.0;
    for (int i = 0; i < 6; ++i) Jtr[i] = 0.0;
    double cost = 0.0;
    const double inv = 1.0 / sq_thr;
    for (size_t i = 0; i < n; ++i) {
      const double* p = pts + 5 * i;
      double RX[3];
      mat3_vec(s.R, p + 2, RX);
      const double Z[3] = {RX[0] + s.t[0], RX[1] + s.t[1], RX[2] + s.t[2]};
      if (!(Z[2] > 0.0)) {
        cost += 1.0;
        continue;
      }
      const double iz = 1.0 / Z[2];
      const double r[2] = {Z[0] * iz - p[0], Z[1] * iz - p[1]};
      const double r2 = r[0] * r[0] + r[1] * r[1];
      if (!(r2 < sq_thr)) {
        cost += 1.0;
        continue;
      }
      cost += r2 * inv;
      const double Jz[6] = {iz, 0.0, -Z[0] * iz * iz, 0.0, iz, -Z[1] * iz * iz};
      add_abs_residual(Z, RX, r, Jz, inv, JtJ, Jtr);
    }
    return cost;
  };
  levenberg_marquardt<6>(pose, accum, apply_abs, max_iterations);
}

double score_abspose_px(const AbsPose& pose, const jpo_camera& cam, const double* xy, const double* xyz, size_t n,
                        double sq_thr_px, size_t* n_inliers, uint8_t* inliers) {
  double score = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < n; ++i) {
    double Z[3], px, py;
    mat3_vec(pose.R, xyz + 3 * i, Z);
    Z[0] += pose.t[0];
    Z[1] += pose.t[1];
    Z[2] += pose.t[2];
    bool in = false;
    double r2 = 0.0;
    if (camera_project(cam, Z, &px, &py)) {
      const double d0 = px - xy[2 * i], d1 = py - xy[2 * i + 1];
      r2 = d0 * d0 + d1 * d1;
      in = r2 < sq_thr_px;
    }
    score += in ? r2 / sq_thr_px : 1.0;
    count += in ? 1 : 0;
    if (inliers) inliers[i] = in ? 1 : 0;
  }
  if (n_inliers) *n_inliers = count;
  return score;
}

void refine_abspose_px(AbsPose* pose, const jpo_camera& cam, const double* xy, const double* xyz, size_t n,
                       double sq_thr_px, int max_iterations) {
  auto accum = [&](const AbsPose& s, double* JtJ, double* Jtr) {
    for (int i = 0; i < 36; ++i) JtJ[i] = 0.0;
    for (int i = 0; i < 6; ++i) Jtr[i] = 0.0;
    double cost = 0.0;
    const double inv = 1.0 / sq_thr_px;
    for (size_t i = 0; i < n; ++i) {
      double RX[3], px, py;
      mat3_vec(s.R, xyz + 3 * i, RX);
      const double Z[3] = {RX[0] + s.t[0], RX[1] + s.t[1], RX[2] + s.t[2]};
      if (!camera_project(cam, Z, &px, &py)) {
        cost += 1.0;
        continue;
      }
      const double r[2] = {px - xy[2 * i], py - xy[2 * i + 1]};
      const double r2 = r[0] * r[0] + r[1] * r[1];
      if (!(r2 < sq_thr_px)) {
        cost += 1.0;
        continue;
      }
      cost += r2 * inv;
      // d(pixel)/dZ by central differences: six projections per point, any camera model.
      double Jz[6];
      const double h = 1e-6 * std::max(1.0, std::sqrt(dot3(Z, Z)));
      bool ok = true;
      for (int k = 0; k < 3; ++k) {
        double Zp[3] = {Z[0], Z[1], Z[2]}, Zm[3] = {Z[0], Z[1], Z[2]};
        Zp[k] += h;
        Zm[k] -= h;
        double pxp, pyp, pxm, pym;
        ok = ok && camera_project(cam, Zp, &pxp, &pyp) && camera_project(cam, Zm, &pxm, &pym);
        if (!ok) break;
        Jz[k] = (pxp - pxm) / (2.0 * h);
        Jz[3 + k] = (pyp - pym) / (2.0 * h);
      }
      if (!ok) continue;
      add_abs_residual(Z, RX, r, Jz, inv, JtJ, Jtr);
    }
    return cost;
  };
  levenberg_marquardt<6>(pose, accum, apply_abs, max_iterations);
}

}  // namespace jpo
