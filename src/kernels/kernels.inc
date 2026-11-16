// The kernels: sampler, minimal solvers and scores. One source for every backend that can compile C:
// the CPU backend (as C++, real = double and real = float), CUDA (nvcc) and Metal (MSL). The WGSL
// kernels (src/backend/webgpu/kernels.wgsl) are a port of this file, function by function.
//
// The solvers are ported from PoseLib (https://github.com/PoseLib/PoseLib), BSD-3-Clause:
//   Copyright (c) 2020, Viktor Larsson. All rights reserved.
//   relpose_5pt (Nister's five-point method), p3p (Ding et al., authors Yaqing Ding and Mark Shachkov),
//   the Sturm bracketing of misc/sturm.h and the cubic of misc/univariate.cc.
// See LICENSE.PoseLib for the full notice. Changes: no Eigen, no recursion, no heap, loops with fixed
// bounds, the trace constraints computed by polynomial products instead of generated code.
//
// Written in the subset the three compilers share. The file that includes it defines:
//   real                 float or double (a typedef)
//   jpo_u32              32-bit unsigned
//   JPO_FN               function qualifier
//   JPO_P                qualifier of a pointer to memory of the calling thread (MSL: thread)
//   JPO_PTS              type of a pointer into the point buffer (MSL: const device float*)
//   JPO_C(x)             a literal of type real
//   JPO_SQRT JPO_ABS JPO_CBRT JPO_ACOS JPO_COS
//   JPO_TOL              root tolerance (1e-10 in double)
//   JPO_STURM_DEPTH      bisection depth limit
// No include guard: the CPU backend includes it twice, in two namespaces.

#define JPO_MAX_E 10   // essential matrices per five-point sample
#define JPO_MAX_P 4    // poses per P3P sample
#define JPO_REL_STRIDE 4  // x1 y1 x2 y2 (normalized image coordinates)
#define JPO_ABS_STRIDE 5  // x y (normalized image coordinates), X Y Z
#define JPO_BIG JPO_C(3.0e38)
#define JPO_POLISH_STEPS 2

// ---------------------------------------------------------------------------------------------------
// Sampler. Counter based: the sample of (seed, problem, iteration) is the same on every backend.

JPO_FN jpo_u32 jpo_hash(jpo_u32 x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

JPO_FN jpo_u32 jpo_sample_base(jpo_u32 seed, jpo_u32 problem, jpo_u32 iter) {
  jpo_u32 h = jpo_hash(seed + 0x9e3779b9U);
  h = jpo_hash(h ^ problem);
  h = jpo_hash(h ^ iter);
  return h;
}

// m distinct indices below n (n >= m, m <= 5).
JPO_FN void jpo_sample(jpo_u32 base, jpo_u32 n, int m, JPO_P jpo_u32* idx) {
  for (int k = 0; k < m; ++k) {
    jpo_u32 v = jpo_hash(base + (jpo_u32)(k + 1) * 0x9e3779b9U) % n;
    for (int t = 0; t < m; ++t) {
      int clash = 0;
      for (int j = 0; j < k; ++j) {
        if (idx[j] == v) clash = 1;
      }
      if (clash == 0) break;
      v = (v + 1U) % n;
    }
    idx[k] = v;
  }
}

// ---------------------------------------------------------------------------------------------------
// Real roots of a polynomial of degree 10 by Sturm bracketing (PoseLib misc/sturm.h, N = 10).

// Monic polynomial of degree deg: f[deg] is taken as 1.
JPO_FN real jpo_polyval(JPO_P const real* f, int off, int deg, real x) {
  real fx = x + f[off + deg - 1];
  for (int i = deg - 2; i >= 0; --i) fx = x * fx + f[off + i];
  return fx;
}

JPO_FN int jpo_signchanges(JPO_P const real* svec, real x) {
  real f[11];
  f[10] = svec[29];
  f[9] = svec[27] + x * svec[28];
  for (int i = 8; i >= 0; --i) f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2];
  int count = 0;
  int neg1 = f[0] < JPO_C(0.0) ? 1 : 0;
  for (int i = 0; i < 10; ++i) {
    int neg2 = f[i + 1] < JPO_C(0.0) ? 1 : 0;
    if (neg1 != neg2) count += 1;
    neg1 = neg2;
  }
  return count;
}

// Ridders' method until close to the root, then Newton. Returns 1 and the root, or 0.
JPO_FN int jpo_ridders_newton(JPO_P const real* fvec, real a, real b, JPO_P real* root) {
  real fa = jpo_polyval(fvec, 0, 10, a);
  real fb = jpo_polyval(fvec, 0, 10, b);
  if ((fa < JPO_C(0.0)) == (fb < JPO_C(0.0))) return 0;
  for (int iter = 0; iter < 30; ++iter) {
    if (JPO_ABS(a - b) < JPO_C(1.0e-3)) break;
    real c = (a + b) * JPO_C(0.5);
    real fc = jpo_polyval(fvec, 0, 10, c);
    real s = JPO_SQRT(fc * fc - fa * fb);
    if (!(s > JPO_C(0.0))) break;
    real d = (fa < fb) ? c + (a - c) * fc / s : c + (c - a) * fc / s;
    real fd = jpo_polyval(fvec, 0, 10, d);
    if (fd >= JPO_C(0.0) ? (fc < JPO_C(0.0)) : (fc > JPO_C(0.0))) {
      a = c;
      fa = fc;
      b = d;
      fb = fd;
    } else if (fd >= JPO_C(0.0) ? (fa < JPO_C(0.0)) : (fa > JPO_C(0.0))) {
      b = d;
      fb = fd;
    } else {
      a = d;
      fa = fd;
    }
  }
  real x = (a + b) * JPO_C(0.5);
  for (int iter = 0; iter < 10; ++iter) {
    real fx = jpo_polyval(fvec, 0, 10, x);
    if (JPO_ABS(fx) < JPO_TOL) break;
    real fpx = JPO_C(10.0) * jpo_polyval(fvec, 11, 9, x);
    real dx = fx / fpx;
    if (!(JPO_ABS(dx) < JPO_BIG)) break;  // flat: keep the bracketed estimate
    x = x - dx;
    if (JPO_ABS(dx) < JPO_TOL) break;
  }
  root[0] = x;
  return 1;
}

// coeffs[0..10], constant term first. Up to 10 roots.
JPO_FN int jpo_sturm10(JPO_P const real* coeffs, JPO_P real* roots) {
  if (coeffs[10] == JPO_C(0.0)) return 0;
  real fvec[21];
  real svec[30];
  real c_inv = JPO_C(1.0) / coeffs[10];
  for (int i = 0; i < 10; ++i) fvec[i] = coeffs[i] * c_inv;
  fvec[10] = JPO_C(1.0);
  for (int i = 0; i < 9; ++i) fvec[11 + i] = fvec[i + 1] * ((real)(i + 1) / JPO_C(10.0));
  fvec[20] = JPO_C(1.0);

  // The Sturm sequence, as the quotients needed to evaluate it.
  {
    real f[30];
    for (int i = 0; i < 21; ++i) f[i] = fvec[i];
    for (int i = 21; i < 30; ++i) f[i] = JPO_C(0.0);
    int o1 = 0, o2 = 11, o3 = 21;
    for (int i = 0; i < 9; ++i) {
      real q1 = f[o1 + 10 - i] * f[o2 + 9 - i];
      real q0 = f[o1 + 9 - i] * f[o2 + 9 - i] - f[o1 + 10 - i] * f[o2 + 8 - i];
      f[o3] = f[o1] - q0 * f[o2];
      for (int j = 1; j < 9 - i; ++j) f[o3 + j] = f[o1 + j] - q1 * f[o2 + j - 1] - q0 * f[o2 + j];
      real c = -JPO_ABS(f[o3 + 8 - i]);
      real ci = JPO_C(1.0) / c;
      for (int j = 0; j < 9 - i; ++j) f[o3 + j] = f[o3 + j] * ci;
      int tmp = o1;
      o1 = o2;
      o2 = o3;
      o3 = tmp;
      svec[3 * i] = q0;
      svec[3 * i + 1] = q1;
      svec[3 * i + 2] = c;
    }
    svec[27] = f[o1];
    svec[28] = f[o1 + 1];
    svec[29] = f[o2];
  }

  // Cauchy bound on the real roots.
  real r0 = JPO_C(0.0);
  for (int i = 0; i < 10; ++i) {
    real v = JPO_ABS(fvec[i]);
    if (v > r0) r0 = v;
  }
  r0 = r0 + JPO_C(1.0);
  if (!(r0 < JPO_BIG)) return 0;  // also catches NaN

  int sa0 = jpo_signchanges(svec, -r0);
  int sb0 = jpo_signchanges(svec, r0);
  if (sa0 - sb0 <= 0) return 0;

  // Isolate the roots by bisection; an explicit stack instead of PoseLib's recursion.
  real sta[JPO_STURM_DEPTH + 4];
  real stb[JPO_STURM_DEPTH + 4];
  int stsa[JPO_STURM_DEPTH + 4];
  int stsb[JPO_STURM_DEPTH + 4];
  int stdp[JPO_STURM_DEPTH + 4];
  int sp = 0;
  sta[0] = -r0;
  stb[0] = r0;
  stsa[0] = sa0;
  stsb[0] = sb0;
  stdp[0] = 0;
  sp = 1;
  int n_roots = 0;
  // Each level is visited at most twice per root.
  for (int guard = 0; guard < 20 * (JPO_STURM_DEPTH + 2) && sp > 0 && n_roots < 10; ++guard) {
    sp -= 1;
    real a = sta[sp];
    real b = stb[sp];
    int sa = stsa[sp];
    int sb = stsb[sp];
    int depth = stdp[sp];
    if (depth > JPO_STURM_DEPTH) continue;
    if (b - a < JPO_TOL) {
      roots[n_roots] = b;
      n_roots += 1;
      continue;
    }
    int n_rts = sa - sb;
    if (n_rts > 1) {
      real c = (a + b) * JPO_C(0.5);
      int sc = jpo_signchanges(svec, c);
      sta[sp] = c;
      stb[sp] = b;
      stsa[sp] = sc;
      stsb[sp] = sb;
      stdp[sp] = depth + 1;
      sp += 1;
      sta[sp] = a;
      stb[sp] = c;
      stsa[sp] = sa;
      stsb[sp] = sc;
      stdp[sp] = depth + 1;
      sp += 1;
    } else if (n_rts == 1) {
      real r;
      if (jpo_ridders_newton(fvec, a, b, &r) != 0) {
        roots[n_roots] = r;
        n_roots += 1;
      }
    }
  }
  return n_roots;
}

// ---------------------------------------------------------------------------------------------------
// Five-point relative pose (Nister). b1, b2: five unit bearings each (5x3). E: up to 10 essential
// matrices, row major, each of unit Frobenius norm, with b2' E b1 = 0. Returns the count.

JPO_FN int jpo_relpose_5pt(JPO_P const real* b1, JPO_P const real* b2, JPO_P real* E) {
  // Null space of the 5x9 epipolar constraints by Householder reflections of their transpose.
  real V[45];
  for (int k = 0; k < 5; ++k) {
    real col[9];
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) col[3 * r + c] = b2[3 * k + r] * b1[3 * k + c];
    }
    for (int j = 0; j < k; ++j) {
      real d = JPO_C(0.0);
      for (int i = 0; i < 9; ++i) d += V[9 * j + i] * col[i];
      d = d * JPO_C(2.0);
      for (int i = 0; i < 9; ++i) col[i] -= d * V[9 * j + i];
    }
    real nrm = JPO_C(0.0);
    for (int i = k; i < 9; ++i) nrm += col[i] * col[i];
    nrm = JPO_SQRT(nrm);
    if (!(nrm > JPO_C(0.0))) return 0;
    for (int i = 0; i < k; ++i) V[9 * k + i] = JPO_C(0.0);
    V[9 * k + k] = col[k] + (col[k] < JPO_C(0.0) ? -nrm : nrm);
    for (int i = k + 1; i < 9; ++i) V[9 * k + i] = col[i];
    real vn = JPO_C(0.0);
    for (int i = k; i < 9; ++i) vn += V[9 * k + i] * V[9 * k + i];
    vn = JPO_SQRT(vn);
    if (!(vn > JPO_C(0.0))) return 0;
    vn = JPO_C(1.0) / vn;
    for (int i = k; i < 9; ++i) V[9 * k + i] *= vn;
  }
  // N[v][e]: the four null vectors; E(x, y, z) = x N0 + y N1 + z N2 + N3.
  real N[36];
  for (int v = 0; v < 4; ++v) {
    real q[9];
    for (int i = 0; i < 9; ++i) q[i] = JPO_C(0.0);
    q[5 + v] = JPO_C(1.0);
    for (int k = 4; k >= 0; --k) {
      real d = JPO_C(0.0);
      for (int i = 0; i < 9; ++i) d += V[9 * k + i] * q[i];
      d = d * JPO_C(2.0);
      for (int i = 0; i < 9; ++i) q[i] -= d * V[9 * k + i];
    }
    for (int i = 0; i < 9; ++i) N[9 * v + i] = q[i];
  }

  // Monomials. Degree 1: x y z 1. Degree 2: x2 xy xz x y2 yz y z2 z 1. Degree 3, Nister's order:
  // x3 y3 x2y xy2 x2z x2 y2z y2 xyz xy | xz2 xz x yz2 yz y z3 z2 z 1.
  const int M2[16] = {0, 1, 2, 3, 1, 4, 5, 6, 2, 5, 7, 8, 3, 6, 8, 9};
  const int M3[40] = {0, 2, 4, 5, 2, 3, 8,  9,  4,  8,  10, 11, 5,  9,  11, 12, 3,  1,  6,  7,
                      8, 6, 13, 14, 9, 7, 14, 15, 10, 13, 16, 17, 11, 14, 17, 18, 12, 15, 18, 19};

  // The ten cubic constraints, 10x20: det(E) = 0 and 2 E E' E - trace(E E') E = 0.
  real C[200];
  for (int i = 0; i < 200; ++i) C[i] = JPO_C(0.0);
  // L = E E' - 0.5 trace(E E') I, degree 2, symmetric: L[i][j] at 10 * (3 i + j).
  real L[90];
  for (int i = 0; i < 90; ++i) L[i] = JPO_C(0.0);
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int a = 0; a < 4; ++a) {
          real ea = N[9 * a + 3 * i + k];
          for (int b = 0; b < 4; ++b) L[10 * (3 * i + j) + M2[4 * a + b]] += ea * N[9 * b + 3 * j + k];
        }
      }
    }
  }
  for (int m = 0; m < 10; ++m) {
    real tr = JPO_C(0.5) * (L[m] + L[40 + m] + L[80 + m]);
    L[m] -= tr;
    L[40 + m] -= tr;
    L[80 + m] -= tr;
    L[30 + m] = L[10 + m];
    L[60 + m] = L[20 + m];
    L[70 + m] = L[50 + m];
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int row = 20 * (3 * i + j);
      for (int k = 0; k < 3; ++k) {
        for (int m = 0; m < 10; ++m) {
          real lm = L[10 * (3 * i + k) + m];
          for (int b = 0; b < 4; ++b) C[row + M3[4 * m + b]] += lm * N[9 * b + 3 * k + j];
        }
      }
    }
  }
  // det(E), expanded along the first row.
  for (int t = 0; t < 3; ++t) {
    int c1 = (t + 1) % 3;
    int c2 = (t + 2) % 3;
    real minor[10];
    for (int m = 0; m < 10; ++m) minor[m] = JPO_C(0.0);
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        minor[M2[4 * a + b]] += N[9 * a + 3 + c1] * N[9 * b + 6 + c2] - N[9 * a + 3 + c2] * N[9 * b + 6 + c1];
      }
    }
    for (int m = 0; m < 10; ++m) {
      for (int b = 0; b < 4; ++b) C[180 + M3[4 * m + b]] += minor[m] * N[9 * b + t];
    }
  }

  // The constraints as they are, for the polish of each solution below.
  real C0[200];
  for (int i = 0; i < 200; ++i) C0[i] = C[i];

  // Gauss-Jordan on the first ten columns, partial pivoting.
  for (int col = 0; col < 10; ++col) {
    int piv = col;
    real best = JPO_ABS(C[20 * col + col]);
    for (int r = col + 1; r < 10; ++r) {
      real v = JPO_ABS(C[20 * r + col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (!(best > JPO_C(0.0))) return 0;
    if (piv != col) {
      for (int c = col; c < 20; ++c) {
        real tmp = C[20 * col + c];
        C[20 * col + c] = C[20 * piv + c];
        C[20 * piv + c] = tmp;
      }
    }
    real inv = JPO_C(1.0) / C[20 * col + col];
    for (int c = col; c < 20; ++c) C[20 * col + c] *= inv;
    for (int r = 0; r < 10; ++r) {
      if (r != col) {
        real f = C[20 * r + col];
        if (f != JPO_C(0.0)) {
          for (int c = col; c < 20; ++c) C[20 * r + c] -= f * C[20 * col + c];
        }
      }
    }
  }

  // Rows 4..9 pair up into three equations x p(z) + y q(z) + r(z) = 0; p, q of degree 3, r of degree 4.
  // P[i][d], Q[i][d], R[i][d]: coefficient of z^d.
  real P[12], Q[12], R[15];
  for (int i = 0; i < 3; ++i) {
    int ra = 20 * (4 + 2 * i) + 10;
    int rb = 20 * (5 + 2 * i) + 10;
    P[4 * i + 0] = C[ra + 2];
    P[4 * i + 1] = C[ra + 1] - C[rb + 2];
    P[4 * i + 2] = C[ra + 0] - C[rb + 1];
    P[4 * i + 3] = -C[rb + 0];
    Q[4 * i + 0] = C[ra + 5];
    Q[4 * i + 1] = C[ra + 4] - C[rb + 5];
    Q[4 * i + 2] = C[ra + 3] - C[rb + 4];
    Q[4 * i + 3] = -C[rb + 3];
    R[5 * i + 0] = C[ra + 9];
    R[5 * i + 1] = C[ra + 8] - C[rb + 9];
    R[5 * i + 2] = C[ra + 7] - C[rb + 8];
    R[5 * i + 3] = C[ra + 6] - C[rb + 7];
    R[5 * i + 4] = -C[rb + 6];
  }
  // The determinant of [p q r], degree 10 in z.
  real c[11];
  for (int i = 0; i < 11; ++i) c[i] = JPO_C(0.0);
  for (int t = 0; t < 3; ++t) {
    int j = (t + 1) % 3;
    int k = (t + 2) % 3;
    real mnr[8];
    for (int d = 0; d < 8; ++d) mnr[d] = JPO_C(0.0);
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 5; ++b) mnr[a + b] += Q[4 * j + a] * R[5 * k + b] - Q[4 * k + a] * R[5 * j + b];
    }
    for (int a = 0; a < 4; ++a) {
      for (int d = 0; d < 8; ++d) c[a + d] += P[4 * t + a] * mnr[d];
    }
  }

  real roots[10];
  int n_roots = jpo_sturm10(c, roots);

  int n_sols = 0;
  for (int s = 0; s < n_roots; ++s) {
    real z = roots[s];
    real bx[3], by[3], bb[3];
    for (int i = 0; i < 3; ++i) {
      bx[i] = ((P[4 * i + 3] * z + P[4 * i + 2]) * z + P[4 * i + 1]) * z + P[4 * i + 0];
      by[i] = ((Q[4 * i + 3] * z + Q[4 * i + 2]) * z + Q[4 * i + 1]) * z + Q[4 * i + 0];
      bb[i] = (((R[5 * i + 4] * z + R[5 * i + 3]) * z + R[5 * i + 2]) * z + R[5 * i + 1]) * z + R[5 * i + 0];
    }
    // Two of the three rows: the pair with the largest determinant.
    int i0 = 0, i1 = 1;
    real det = bx[0] * by[1] - bx[1] * by[0];
    real d02 = bx[0] * by[2] - bx[2] * by[0];
    real d12 = bx[1] * by[2] - bx[2] * by[1];
    if (JPO_ABS(d02) > JPO_ABS(det)) {
      det = d02;
      i0 = 0;
      i1 = 2;
    }
    if (JPO_ABS(d12) > JPO_ABS(det)) {
      det = d12;
      i0 = 1;
      i1 = 2;
    }
    if (!(JPO_ABS(det) > JPO_C(0.0))) continue;
    real x = (-bb[i0] * by[i1] + bb[i1] * by[i0]) / det;
    real y = (-bx[i0] * bb[i1] + bx[i1] * bb[i0]) / det;
    // Polish (x, y, z) on the ten constraints before elimination: Gauss-Newton steps. The elimination,
    // the determinant and its roots lose digits, most of all in float; this gets them back.
    for (int it = 0; it < JPO_POLISH_STEPS; ++it) {
      real mon[20], gx[20], gy[20], gz[20];
      for (int i = 0; i < 20; ++i) {
        gx[i] = JPO_C(0.0);
        gy[i] = JPO_C(0.0);
        gz[i] = JPO_C(0.0);
      }
      real xx = x * x, yy = y * y, zz = z * z;
      mon[0] = xx * x;
      gx[0] = JPO_C(3.0) * xx;
      mon[1] = yy * y;
      gy[1] = JPO_C(3.0) * yy;
      mon[2] = xx * y;
      gx[2] = JPO_C(2.0) * x * y;
      gy[2] = xx;
      mon[3] = x * yy;
      gx[3] = yy;
      gy[3] = JPO_C(2.0) * x * y;
      mon[4] = xx * z;
      gx[4] = JPO_C(2.0) * x * z;
      gz[4] = xx;
      mon[5] = xx;
      gx[5] = JPO_C(2.0) * x;
      mon[6] = yy * z;
      gy[6] = JPO_C(2.0) * y * z;
      gz[6] = yy;
      mon[7] = yy;
      gy[7] = JPO_C(2.0) * y;
      mon[8] = x * y * z;
      gx[8] = y * z;
      gy[8] = x * z;
      gz[8] = x * y;
      mon[9] = x * y;
      gx[9] = y;
      gy[9] = x;
      mon[10] = x * zz;
      gx[10] = zz;
      gz[10] = JPO_C(2.0) * x * z;
      mon[11] = x * z;
      gx[11] = z;
      gz[11] = x;
      mon[12] = x;
      gx[12] = JPO_C(1.0);
      mon[13] = y * zz;
      gy[13] = zz;
      gz[13] = JPO_C(2.0) * y * z;
      mon[14] = y * z;
      gy[14] = z;
      gz[14] = y;
      mon[15] = y;
      gy[15] = JPO_C(1.0);
      mon[16] = zz * z;
      gz[16] = JPO_C(3.0) * zz;
      mon[17] = zz;
      gz[17] = JPO_C(2.0) * z;
      mon[18] = z;
      gz[18] = JPO_C(1.0);
      mon[19] = JPO_C(1.0);
      real a00 = JPO_C(0.0), a01 = JPO_C(0.0), a02 = JPO_C(0.0), a11 = JPO_C(0.0), a12 = JPO_C(0.0), a22 = JPO_C(0.0);
      real g0 = JPO_C(0.0), g1 = JPO_C(0.0), g2 = JPO_C(0.0);
      for (int r = 0; r < 10; ++r) {
        real f = JPO_C(0.0), fx = JPO_C(0.0), fy = JPO_C(0.0), fz = JPO_C(0.0);
        for (int m = 0; m < 20; ++m) {
          real cm = C0[20 * r + m];
          f += cm * mon[m];
          fx += cm * gx[m];
          fy += cm * gy[m];
          fz += cm * gz[m];
        }
        a00 += fx * fx;
        a01 += fx * fy;
        a02 += fx * fz;
        a11 += fy * fy;
        a12 += fy * fz;
        a22 += fz * fz;
        g0 += fx * f;
        g1 += fy * f;
        g2 += fz * f;
      }
      // Solve the 3x3 normal equations by cofactors.
      real c00 = a11 * a22 - a12 * a12;
      real c01 = a02 * a12 - a01 * a22;
      real c02 = a01 * a12 - a02 * a11;
      real dt = a00 * c00 + a01 * c01 + a02 * c02;
      if (!(JPO_ABS(dt) > JPO_C(0.0))) break;
      real c11 = a00 * a22 - a02 * a02;
      real c12 = a01 * a02 - a00 * a12;
      real c22 = a00 * a11 - a01 * a01;
      real idt = JPO_C(1.0) / dt;
      real dx = (c00 * g0 + c01 * g1 + c02 * g2) * idt;
      real dy = (c01 * g0 + c11 * g1 + c12 * g2) * idt;
      real dz = (c02 * g0 + c12 * g1 + c22 * g2) * idt;
      if (!(JPO_ABS(dx) + JPO_ABS(dy) + JPO_ABS(dz) < JPO_BIG)) break;
      x -= dx;
      y -= dy;
      z -= dz;
    }
    real nrm = JPO_SQRT(x * x + y * y + z * z + JPO_C(1.0));
    if (!(nrm < JPO_BIG)) continue;
    real inv = JPO_C(1.0) / nrm;
    for (int i = 0; i < 9; ++i) E[9 * n_sols + i] = (N[i] * x + N[9 + i] * y + N[18 + i] * z + N[27 + i]) * inv;
    n_sols += 1;
  }
  return n_sols;
}

// ---------------------------------------------------------------------------------------------------
// P3P (Ding et al., "Revisiting the P3P Problem"). xin: three unit bearings (3x3), Xin: three points
// (3x3). P: up to 4 poses, each 12 values: R row major (9), then t; a point maps to R X + t.

JPO_FN int jpo_cubic_single_real(real c2, real c1, real c0, JPO_P real* root) {
  real a = c1 - c2 * c2 / JPO_C(3.0);
  real b = (JPO_C(2.0) * c2 * c2 * c2 - JPO_C(9.0) * c2 * c1) / JPO_C(27.0) + c0;
  real c = b * b / JPO_C(4.0) + a * a * a / JPO_C(27.0);
  if (c != JPO_C(0.0)) {
    if (c > JPO_C(0.0)) {
      c = JPO_SQRT(c);
      b *= JPO_C(-0.5);
      root[0] = JPO_CBRT(b + c) + JPO_CBRT(b - c) - c2 / JPO_C(3.0);
      return 1;
    }
    c = JPO_C(3.0) * b / (JPO_C(2.0) * a) * JPO_SQRT(JPO_C(-3.0) / a);
    if (c > JPO_C(1.0)) c = JPO_C(1.0);
    if (c < JPO_C(-1.0)) c = JPO_C(-1.0);
    root[0] = JPO_C(2.0) * JPO_SQRT(-a / JPO_C(3.0)) * JPO_COS(JPO_ACOS(c) / JPO_C(3.0)) - c2 / JPO_C(3.0);
  } else {
    root[0] = -c2 / JPO_C(3.0) + (a != JPO_C(0.0) ? (JPO_C(3.0) * b / a) : JPO_C(0.0));
  }
  return 0;
}

JPO_FN int jpo_root2real(real b, real c, JPO_P real* r) {
  real v = b * b - JPO_C(4.0) * c;
  if (v < JPO_C(-1.0e-12)) {
    r[0] = JPO_C(-0.5) * b;
    r[1] = r[0];
    return 0;
  }
  if (v < JPO_C(0.0)) {
    r[0] = JPO_C(-0.5) * b;
    r[1] = JPO_C(-2.0);
    return 1;
  }
  real y = JPO_SQRT(v);
  if (b < JPO_C(0.0)) {
    r[0] = JPO_C(0.5) * (-b + y);
    r[1] = JPO_C(0.5) * (-b - y);
  } else {
    r[0] = JPO_C(2.0) * c / (-b + y);
    r[1] = JPO_C(2.0) * c / (-b - y);
  }
  return 1;
}

JPO_FN int jpo_p3p(JPO_P const real* xin, JPO_P const real* Xin, JPO_P real* P) {
  real x[9], X[9];
  for (int i = 0; i < 9; ++i) {
    x[i] = xin[i];
    X[i] = Xin[i];
  }
  real X01[3], X02[3], X12[3];
  for (int i = 0; i < 3; ++i) {
    X01[i] = X[i] - X[3 + i];
    X02[i] = X[i] - X[6 + i];
    X12[i] = X[3 + i] - X[6 + i];
  }
  real a01 = X01[0] * X01[0] + X01[1] * X01[1] + X01[2] * X01[2];
  real a02 = X02[0] * X02[0] + X02[1] * X02[1] + X02[2] * X02[2];
  real a12 = X12[0] * X12[0] + X12[1] * X12[1] + X12[2] * X12[2];

  // Reorder so that the largest distance is between points 1 and 2.
  if (a01 > a02) {
    if (a01 > a12) {
      for (int i = 0; i < 3; ++i) {
        real t = x[i];
        x[i] = x[6 + i];
        x[6 + i] = t;
        t = X[i];
        X[i] = X[6 + i];
        X[6 + i] = t;
        X01[i] = -X12[i];
        X02[i] = -X02[i];
      }
      real t = a01;
      a01 = a12;
      a12 = t;
    }
  } else if (a02 > a12) {
    for (int i = 0; i < 3; ++i) {
      real t = x[i];
      x[i] = x[3 + i];
      x[3 + i] = t;
      t = X[i];
      X[i] = X[3 + i];
      X[3 + i] = t;
      X01[i] = -X01[i];
      X02[i] = X12[i];
    }
    real t = a02;
    a02 = a12;
    a12 = t;
  }
  if (!(a12 > JPO_C(0.0))) return 0;

  real a12d = JPO_C(1.0) / a12;
  real a = a01 * a12d;
  real b = a02 * a12d;
  real m01 = x[0] * x[3] + x[1] * x[4] + x[2] * x[5];
  real m02 = x[0] * x[6] + x[1] * x[7] + x[2] * x[8];
  real m12 = x[3] * x[6] + x[4] * x[7] + x[5] * x[8];

  real m12sq = -m12 * m12 + JPO_C(1.0);
  real m02sq = JPO_C(-1.0) + m02 * m02;
  real m01sq = JPO_C(-1.0) + m01 * m01;
  real ab = a * b;
  real bsq = b * b;
  real asq = a * a;
  real m013 = JPO_C(-2.0) + JPO_C(2.0) * m01 * m02 * m12;
  real bsqm12sq = bsq * m12sq;
  real asqm12sq = asq * m12sq;
  real abm12sq = JPO_C(2.0) * ab * m12sq;

  real k3_inv = JPO_C(1.0) / (bsqm12sq + b * m02sq);
  real k2 = k3_inv * ((JPO_C(-1.0) + a) * m02sq + abm12sq + bsqm12sq + b * m013);
  real k1 = k3_inv * (asqm12sq + abm12sq + a * m013 + (JPO_C(-1.0) + b) * m01sq);
  real k0 = k3_inv * (asqm12sq + a * m01sq);
  if (!(JPO_ABS(k2) < JPO_BIG) || !(JPO_ABS(k1) < JPO_BIG) || !(JPO_ABS(k0) < JPO_BIG)) return 0;

  real s;
  int G = jpo_cubic_single_real(k2, k1, k0, &s);

  real Cm[9];
  Cm[0] = -a + s * (JPO_C(1.0) - b);
  Cm[1] = -m02 * s;
  Cm[2] = a * m12 + b * m12 * s;
  Cm[3] = Cm[1];
  Cm[4] = s + JPO_C(1.0);
  Cm[5] = -m01;
  Cm[6] = Cm[2];
  Cm[7] = Cm[5];
  Cm[8] = -a - b * s + JPO_C(1.0);

  // Factor the degenerate conic into two lines p, q.
  real pq[6];
  {
    real adj[9];
    adj[0] = -(Cm[4] * Cm[8] - Cm[5] * Cm[7]);
    adj[1] = -(Cm[2] * Cm[7] - Cm[1] * Cm[8]);
    adj[2] = -(Cm[1] * Cm[5] - Cm[2] * Cm[4]);
    adj[3] = -(Cm[5] * Cm[6] - Cm[3] * Cm[8]);
    adj[4] = -(Cm[0] * Cm[8] - Cm[2] * Cm[6]);
    adj[5] = -(Cm[2] * Cm[3] - Cm[0] * Cm[5]);
    adj[6] = -(Cm[3] * Cm[7] - Cm[4] * Cm[6]);
    adj[7] = -(Cm[1] * Cm[6] - Cm[0] * Cm[7]);
    adj[8] = -(Cm[0] * Cm[4] - Cm[1] * Cm[3]);
    int col = 2;
    if (adj[0] > adj[4]) {
      if (adj[0] > adj[8]) col = 0;
    } else if (adj[4] > adj[8]) {
      col = 1;
    }
    real dd = adj[4 * col];
    if (!(dd > JPO_C(0.0))) return 0;
    real inv = JPO_C(1.0) / JPO_SQRT(dd);
    real v0 = adj[col] * inv;
    real v1 = adj[3 + col] * inv;
    real v2 = adj[6 + col] * inv;
    Cm[1] -= v2;
    Cm[2] += v1;
    Cm[5] -= v0;
    Cm[3] += v2;
    Cm[6] -= v1;
    Cm[7] += v0;
    pq[0] = Cm[0];
    pq[1] = Cm[3];
    pq[2] = Cm[6];
    pq[3] = Cm[0];
    pq[4] = Cm[1];
    pq[5] = Cm[2];
  }

  // XX = inverse of [X01 X02 X01 x X02] (columns).
  real XX[9];
  {
    real M[9];
    real cr0 = X01[1] * X02[2] - X01[2] * X02[1];
    real cr1 = X01[2] * X02[0] - X01[0] * X02[2];
    real cr2 = X01[0] * X02[1] - X01[1] * X02[0];
    M[0] = X01[0];
    M[1] = X02[0];
    M[2] = cr0;
    M[3] = X01[1];
    M[4] = X02[1];
    M[5] = cr1;
    M[6] = X01[2];
    M[7] = X02[2];
    M[8] = cr2;
    real det = M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) +
               M[2] * (M[3] * M[7] - M[4] * M[6]);
    if (!(JPO_ABS(det) > JPO_C(0.0))) return 0;
    real id = JPO_C(1.0) / det;
    XX[0] = (M[4] * M[8] - M[5] * M[7]) * id;
    XX[1] = (M[2] * M[7] - M[1] * M[8]) * id;
    XX[2] = (M[1] * M[5] - M[2] * M[4]) * id;
    XX[3] = (M[5] * M[6] - M[3] * M[8]) * id;
    XX[4] = (M[0] * M[8] - M[2] * M[6]) * id;
    XX[5] = (M[2] * M[3] - M[0] * M[5]) * id;
    XX[6] = (M[3] * M[7] - M[4] * M[6]) * id;
    XX[7] = (M[1] * M[6] - M[0] * M[7]) * id;
    XX[8] = (M[0] * M[4] - M[1] * M[3]) * id;
  }

  int n_sols = 0;
  for (int i = 0; i < 2; ++i) {
    real p0 = pq[3 * i];
    real p1 = pq[3 * i + 1];
    real p2 = pq[3 * i + 2];
    int switch_12 = JPO_ABS(p0) <= JPO_ABS(p1) ? 1 : 0;
    real w0, w1, cb, cc;
    if (switch_12 != 0) {
      w0 = -p0 / p1;
      w1 = -p2 / p1;
      real ca = JPO_C(1.0) / (w1 * w1 - b);
      cb = JPO_C(2.0) * (b * m12 - m02 * w1 + w0 * w1) * ca;
      cc = (w0 * w0 - JPO_C(2.0) * m02 * w0 - b + JPO_C(1.0)) * ca;
    } else {
      w0 = -p1 / p0;
      w1 = -p2 / p0;
      real ca = JPO_C(1.0) / (-a * w1 * w1 + JPO_C(2.0) * a * m12 * w1 - a + JPO_C(1.0));
      cb = JPO_C(2.0) * (a * m12 * w0 - m01 - a * w0 * w1) * ca;
      cc = (JPO_C(1.0) - a * w0 * w0) * ca;
    }
    real taus[2];
    if (jpo_root2real(cb, cc, taus) == 0) continue;
    for (int ti = 0; ti < 2; ++ti) {
      real tau = taus[ti];
      if (!(tau > JPO_C(0.0))) continue;
      real d0, d1, d2;
      if (switch_12 != 0) {
        d2 = JPO_SQRT(a12 / (tau * (tau - JPO_C(2.0) * m12) + JPO_C(1.0)));
        d1 = tau * d2;
        d0 = w0 * d2 + w1 * d1;
        if (!(d0 >= JPO_C(0.0))) continue;
      } else {
        d0 = JPO_SQRT(a01 / (tau * (tau - JPO_C(2.0) * m01) + JPO_C(1.0)));
        d1 = tau * d0;
        d2 = w0 * d0 + w1 * d1;
        if (!(d2 >= JPO_C(0.0))) continue;
      }
      // A few Newton steps on the three distance equations.
      for (int iter = 0; iter < 5; ++iter) {
        real r1 = d0 * d0 - JPO_C(2.0) * d0 * d1 * m01 + d1 * d1 - a01;
        real r2 = d0 * d0 - JPO_C(2.0) * d0 * d2 * m02 + d2 * d2 - a02;
        real r3 = d1 * d1 - JPO_C(2.0) * d1 * d2 * m12 + d2 * d2 - a12;
        if (JPO_ABS(r1) + JPO_ABS(r2) + JPO_ABS(r3) < JPO_TOL) break;
        real x11 = d0 - d1 * m01;
        real x12 = d1 - d0 * m01;
        real x21 = d0 - d2 * m02;
        real x23 = d2 - d0 * m02;
        real x32 = d1 - d2 * m12;
        real x33 = d2 - d1 * m12;
        real den = x11 * x23 * x32 + x12 * x21 * x33;
        if (!(JPO_ABS(den) > JPO_C(0.0))) break;
        real detJ = JPO_C(0.5) / den;
        d0 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ;
        d1 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ;
        d2 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ;
      }
      real v1[3], v2[3], v3[3];
      for (int k = 0; k < 3; ++k) {
        v1[k] = d0 * x[k] - d1 * x[3 + k];
        v2[k] = d0 * x[k] - d2 * x[6 + k];
      }
      v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
      v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
      v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
      int o = 12 * n_sols;
      for (int r = 0; r < 3; ++r) {
        for (int cI = 0; cI < 3; ++cI) P[o + 3 * r + cI] = v1[r] * XX[cI] + v2[r] * XX[3 + cI] + v3[r] * XX[6 + cI];
      }
      int ok = 1;
      for (int r = 0; r < 3; ++r) {
        real tr = d0 * x[r] - (P[o + 3 * r] * X[0] + P[o + 3 * r + 1] * X[1] + P[o + 3 * r + 2] * X[2]);
        P[o + 9 + r] = tr;
        if (!(JPO_ABS(tr) < JPO_BIG)) ok = 0;
      }
      if (ok != 0 && n_sols < JPO_MAX_P) n_sols += 1;
    }
    if (n_sols > 0 && G != 0) break;
    if (n_sols >= JPO_MAX_P) break;
  }
  return n_sols;
}

// ---------------------------------------------------------------------------------------------------
// Scores. Truncated squared error (MSAC) in units of the squared threshold: each point adds
// min(r2 / thr2, 1). A score that passes `bound` cannot win, so the loop stops there; the value
// returned is then only known to be above the bound.

// The four motions of an essential matrix (PoseLib motion_from_essential): P holds 4 poses of 12
// values, R row major then t. No singular value decomposition: cross products of the columns of E.
JPO_FN void jpo_essential_poses(JPO_P const real* E, JPO_P real* P) {
  real c0[3], c1[3], c2[3];
  for (int i = 0; i < 3; ++i) {
    c0[i] = E[3 * i];
    c1[i] = E[3 * i + 1];
    c2[i] = E[3 * i + 2];
  }
  real u12[3], u13[3], u23[3];
  u12[0] = c0[1] * c1[2] - c0[2] * c1[1];
  u12[1] = c0[2] * c1[0] - c0[0] * c1[2];
  u12[2] = c0[0] * c1[1] - c0[1] * c1[0];
  u13[0] = c0[1] * c2[2] - c0[2] * c2[1];
  u13[1] = c0[2] * c2[0] - c0[0] * c2[2];
  u13[2] = c0[0] * c2[1] - c0[1] * c2[0];
  u23[0] = c1[1] * c2[2] - c1[2] * c2[1];
  u23[1] = c1[2] * c2[0] - c1[0] * c2[2];
  u23[2] = c1[0] * c2[1] - c1[1] * c2[0];
  real n12 = u12[0] * u12[0] + u12[1] * u12[1] + u12[2] * u12[2];
  real n13 = u13[0] * u13[0] + u13[1] * u13[1] + u13[2] * u13[2];
  real n23 = u23[0] * u23[0] + u23[1] * u23[1] + u23[2] * u23[2];
  real w0[3], w1[3], w2[3];  // columns of U W
  int pick = 0;              // 0: (c0, u12), 1: (c0, u13), 2: (c1, u23)
  if (n12 > n13) {
    pick = n12 > n23 ? 0 : 2;
  } else {
    pick = n13 > n23 ? 1 : 2;
  }
  for (int i = 0; i < 3; ++i) {
    w1[i] = pick == 2 ? c1[i] : c0[i];
    w2[i] = pick == 0 ? u12[i] : (pick == 1 ? u13[i] : u23[i]);
  }
  real i1 = JPO_C(1.0) / JPO_SQRT(w1[0] * w1[0] + w1[1] * w1[1] + w1[2] * w1[2]);
  real i2 = JPO_C(1.0) / JPO_SQRT(w2[0] * w2[0] + w2[1] * w2[1] + w2[2] * w2[2]);
  for (int i = 0; i < 3; ++i) {
    w1[i] *= i1;
    w2[i] *= i2;
  }
  w0[0] = -(w2[1] * w1[2] - w2[2] * w1[1]);
  w0[1] = -(w2[2] * w1[0] - w2[0] * w1[2]);
  w0[2] = -(w2[0] * w1[1] - w2[1] * w1[0]);
  real v0[3], v1[3], v2[3];  // rows of V'
  for (int c = 0; c < 3; ++c) {
    v0[c] = w1[0] * E[c] + w1[1] * E[3 + c] + w1[2] * E[6 + c];
    v1[c] = -(w0[0] * E[c] + w0[1] * E[3 + c] + w0[2] * E[6 + c]);
  }
  real iv = JPO_C(1.0) / JPO_SQRT(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
  for (int c = 0; c < 3; ++c) v0[c] *= iv;
  real d = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
  for (int c = 0; c < 3; ++c) v1[c] -= d * v0[c];
  iv = JPO_C(1.0) / JPO_SQRT(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
  for (int c = 0; c < 3; ++c) v1[c] *= iv;
  v2[0] = v0[1] * v1[2] - v0[2] * v1[1];
  v2[1] = v0[2] * v1[0] - v0[0] * v1[2];
  v2[2] = v0[0] * v1[1] - v0[1] * v1[0];
  for (int k = 0; k < 2; ++k) {
    real sg = k == 0 ? JPO_C(1.0) : JPO_C(-1.0);  // the second rotation flips the first two columns of U W
    for (int j = 0; j < 2; ++j) {
      int o = 12 * (2 * k + j);
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) P[o + 3 * r + c] = sg * (w0[r] * v0[c] + w1[r] * v1[c]) + w2[r] * v2[c];
        P[o + 9 + r] = j == 0 ? w2[r] : -w2[r];
      }
    }
  }
}

// PoseLib's cheirality test for unit bearings b1 (camera a) and b2 (camera b): both depths of the
// point they triangulate to are above 0.01.
JPO_FN int jpo_cheirality(JPO_P const real* P, real b10, real b11, real b12, real b20, real b21, real b22) {
  real r0 = P[0] * b10 + P[1] * b11 + P[2] * b12;
  real r1 = P[3] * b10 + P[4] * b11 + P[5] * b12;
  real r2 = P[6] * b10 + P[7] * b11 + P[8] * b12;
  real a = -(r0 * b20 + r1 * b21 + r2 * b22);
  real bb1 = -(r0 * P[9] + r1 * P[10] + r2 * P[11]);
  real bb2 = b20 * P[9] + b21 * P[10] + b22 * P[11];
  real l1 = bb1 - a * bb2;
  real l2 = -a * bb1 + bb2;
  real md = JPO_C(0.01) * (JPO_C(1.0) - a * a);
  return (l1 > md && l2 > md) ? 1 : 0;
}

// Sampson error of the pairs pts[off + 4 i ..] against the relative pose P (12); a pair within the
// threshold that fails the cheirality test counts as an outlier (as PoseLib).
JPO_FN real jpo_score_rel(JPO_P const real* P, JPO_PTS pts, jpo_u32 off, jpo_u32 n, real inv_sq_thr, real bound) {
  real score = JPO_C(0.0);
  // E = [t]x R
  real e0 = -P[11] * P[3] + P[10] * P[6];
  real e1 = -P[11] * P[4] + P[10] * P[7];
  real e2 = -P[11] * P[5] + P[10] * P[8];
  real e3 = P[11] * P[0] - P[9] * P[6];
  real e4 = P[11] * P[1] - P[9] * P[7];
  real e5 = P[11] * P[2] - P[9] * P[8];
  real e6 = -P[10] * P[0] + P[9] * P[3];
  real e7 = -P[10] * P[1] + P[9] * P[4];
  real e8 = -P[10] * P[2] + P[9] * P[5];
  for (jpo_u32 i = 0; i < n; ++i) {
    jpo_u32 o = off + JPO_REL_STRIDE * i;
    real x1 = pts[o], y1 = pts[o + 1], x2 = pts[o + 2], y2 = pts[o + 3];
    real a0 = e0 * x1 + e1 * y1 + e2;
    real a1 = e3 * x1 + e4 * y1 + e5;
    real a2 = e6 * x1 + e7 * y1 + e8;
    real b0 = e0 * x2 + e3 * y2 + e6;
    real b1 = e1 * x2 + e4 * y2 + e7;
    real Cc = x2 * a0 + y2 * a1 + a2;
    real r2 = Cc * Cc / (a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1) * inv_sq_thr;
    real add = JPO_C(1.0);  // NaN counts as an outlier
    if (r2 < JPO_C(1.0)) {
      real i1 = JPO_C(1.0) / JPO_SQRT(x1 * x1 + y1 * y1 + JPO_C(1.0));
      real i2 = JPO_C(1.0) / JPO_SQRT(x2 * x2 + y2 * y2 + JPO_C(1.0));
      if (jpo_cheirality(P, x1 * i1, y1 * i1, i1, x2 * i2, y2 * i2, i2) != 0) add = r2;
    }
    score += add;
    if ((i & 15U) == 15U && score > bound) break;
  }
  return score;
}

// Reprojection error on the normalized image plane of pts[off + 5 i ..] against the pose P (12).
JPO_FN real jpo_score_abs(JPO_P const real* P, JPO_PTS pts, jpo_u32 off, jpo_u32 n, real inv_sq_thr, real bound) {
  real score = JPO_C(0.0);
  real r0 = P[0], r1 = P[1], r2 = P[2], r3 = P[3], r4 = P[4], r5 = P[5], r6 = P[6], r7 = P[7], r8 = P[8];
  real t0 = P[9], t1 = P[10], t2 = P[11];
  for (jpo_u32 i = 0; i < n; ++i) {
    jpo_u32 o = off + JPO_ABS_STRIDE * i;
    real X = pts[o + 2], Y = pts[o + 3], Z = pts[o + 4];
    real z0 = r0 * X + r1 * Y + r2 * Z + t0;
    real z1 = r3 * X + r4 * Y + r5 * Z + t1;
    real z2 = r6 * X + r7 * Y + r8 * Z + t2;
    real e2 = JPO_C(1.0);
    if (z2 > JPO_C(0.0)) {
      real inv = JPO_C(1.0) / z2;
      real d0 = z0 * inv - pts[o];
      real d1 = z1 * inv - pts[o + 1];
      real v = (d0 * d0 + d1 * d1) * inv_sq_thr;
      if (v < JPO_C(1.0)) e2 = v;
    }
    score += e2;
    if ((i & 15U) == 15U && score > bound) break;
  }
  return score;
}

// ---------------------------------------------------------------------------------------------------
// One hypothesis: the sample of (seed, problem, iter), solved, each solution scored. Returns the best
// score and writes the index of that solution (-1: none). A model is always a pose: 12 values.

JPO_FN void jpo_rel_sample(JPO_PTS pts, jpo_u32 off, jpo_u32 n, jpo_u32 seed, jpo_u32 problem, jpo_u32 iter,
                           JPO_P real* b1, JPO_P real* b2) {
  jpo_u32 idx[5];
  jpo_sample(jpo_sample_base(seed, problem, iter), n, 5, idx);
  for (int k = 0; k < 5; ++k) {
    jpo_u32 o = off + JPO_REL_STRIDE * idx[k];
    real x1 = pts[o], y1 = pts[o + 1], x2 = pts[o + 2], y2 = pts[o + 3];
    real i1 = JPO_C(1.0) / JPO_SQRT(x1 * x1 + y1 * y1 + JPO_C(1.0));
    real i2 = JPO_C(1.0) / JPO_SQRT(x2 * x2 + y2 * y2 + JPO_C(1.0));
    b1[3 * k] = x1 * i1;
    b1[3 * k + 1] = y1 * i1;
    b1[3 * k + 2] = i1;
    b2[3 * k] = x2 * i2;
    b2[3 * k + 1] = y2 * i2;
    b2[3 * k + 2] = i2;
  }
}

// The motions of sample solution e that put all five sample points in front of both cameras, scored;
// the solution index is 4 e + k.
JPO_FN real jpo_rel_hypothesis(JPO_PTS pts, jpo_u32 off, jpo_u32 n, jpo_u32 seed, jpo_u32 problem, jpo_u32 iter,
                               real inv_sq_thr, real bound, JPO_P int* best_sol) {
  real b1[15], b2[15];
  real E[9 * JPO_MAX_E];
  jpo_rel_sample(pts, off, n, seed, problem, iter, b1, b2);
  int ns = jpo_relpose_5pt(b1, b2, E);
  real best = JPO_BIG;
  best_sol[0] = -1;
  for (int s = 0; s < ns; ++s) {
    real Es[9];
    real P4[48];
    for (int i = 0; i < 9; ++i) Es[i] = E[9 * s + i];
    jpo_essential_poses(Es, P4);
    for (int k = 0; k < 4; ++k) {
      real Ps[12];
      for (int i = 0; i < 12; ++i) Ps[i] = P4[12 * k + i];
      int front = 1;
      for (int j = 0; j < 5; ++j) {
        if (jpo_cheirality(Ps, b1[3 * j], b1[3 * j + 1], b1[3 * j + 2], b2[3 * j], b2[3 * j + 1], b2[3 * j + 2]) == 0)
          front = 0;
      }
      if (front == 0) continue;
      real sc = jpo_score_rel(Ps, pts, off, n, inv_sq_thr, best < bound ? best : bound);
      if (sc < best) {
        best = sc;
        best_sol[0] = 4 * s + k;
      }
    }
  }
  return best;
}

// The model of a hypothesis, solved again: 12 values. Returns 0 if the solution does not exist.
JPO_FN int jpo_rel_model(JPO_PTS pts, jpo_u32 off, jpo_u32 n, jpo_u32 seed, jpo_u32 problem, jpo_u32 iter, int sol,
                         JPO_P real* model) {
  real b1[15], b2[15];
  real E[9 * JPO_MAX_E];
  jpo_rel_sample(pts, off, n, seed, problem, iter, b1, b2);
  int ns = jpo_relpose_5pt(b1, b2, E);
  if (sol < 0 || sol >= 4 * ns) return 0;
  real Es[9];
  real P4[48];
  int s = sol / 4;
  int k = sol - 4 * s;
  for (int i = 0; i < 9; ++i) Es[i] = E[9 * s + i];
  jpo_essential_poses(Es, P4);
  for (int i = 0; i < 12; ++i) model[i] = P4[12 * k + i];
  return 1;
}

JPO_FN void jpo_abs_sample(JPO_PTS pts, jpo_u32 off, jpo_u32 n, jpo_u32 seed, jpo_u32 problem, jpo_u32 iter,
                           JPO_P real* xs, JPO_P real* Xs) {
  jpo_u32 idx[5];
  jpo_sample(jpo_sample_base(seed, problem, iter), n, 3, idx);
  for (int k = 0; k < 3; ++k) {
    jpo_u32 o = off + JPO_ABS_STRIDE * idx[k];
    real x = pts[o], y = pts[o + 1];
    real inv = JPO_C(1.0) / JPO_SQRT(x * x + y * y + JPO_C(1.0));
    xs[3 * k] = x * inv;
    xs[3 * k + 1] = y * inv;
    xs[3 * k + 2] = inv;
    Xs[3 * k] = pts[o + 2];
    Xs[3 * k + 1] = pts[o + 3];
    Xs[3 * k + 2] = pts[o + 4];
  }
}

JPO_FN real jpo_abs_hypothesis(JPO_PTS pts, jpo_u32 off, jpo_u32 n, jpo_u32 seed, jpo_u32 problem, jpo_u32 iter,
                               real inv_sq_thr, real bound, JPO_P int* best_sol) {
  real xs[9], Xs[9];
  real P[12 * JPO_MAX_P];
  jpo_abs_sample(pts, off, n, seed, problem, iter, xs, Xs);
  int ns = jpo_p3p(xs, Xs, P);
  real best = JPO_BIG;
  best_sol[0] = -1;
  for (int s = 0; s < ns; ++s) {
    real Ps[12];
    for (int i = 0; i < 12; ++i) Ps[i] = P[12 * s + i];
    real sc = jpo_score_abs(Ps, pts, off, n, inv_sq_thr, best < bound ? best : bound);
    if (sc < best) {
      best = sc;
      best_sol[0] = s;
    }
  }
  return best;
}

JPO_FN int jpo_abs_model(JPO_PTS pts, jpo_u32 off, jpo_u32 n, jpo_u32 seed, jpo_u32 problem, jpo_u32 iter, int sol,
                         JPO_P real* model) {
  real xs[9], Xs[9];
  real P[12 * JPO_MAX_P];
  jpo_abs_sample(pts, off, n, seed, problem, iter, xs, Xs);
  int ns = jpo_p3p(xs, Xs, P);
  if (sol < 0 || sol >= ns) return 0;
  for (int i = 0; i < 12; ++i) model[i] = P[12 * sol + i];
  return 1;
}
