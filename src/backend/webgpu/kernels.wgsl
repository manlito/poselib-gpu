// The kernels in WGSL: a port of src/kernels/kernels.inc, function by function, with real = f32, and the
// entry points of src/backend/metal/entry.metal. Same function names, same order of floating-point
// operations; when kernels.inc changes, this file changes with it.
//
// The solvers are ported from PoseLib (https://github.com/PoseLib/PoseLib), BSD-3-Clause:
//   Copyright (c) 2020, Viktor Larsson. All rights reserved.
//   relpose_5pt (Nister's five-point method), p3p (Ding et al., authors Yaqing Ding and Mark Shachkov),
//   the Sturm bracketing of misc/sturm.h and the cubic of misc/univariate.cc.
// See LICENSE.PoseLib for the full notice. Changes: no Eigen, no recursion, no heap, loops with fixed
// bounds, the trace constraints computed by polynomial products instead of generated code.
//
// What differs from the C source, because WGSL asks for it:
//   - A pointer argument is a pointer to a whole local array of one size, so every array parameter has
//     the size its callers use (E: 90, P: 48 or 12, ...).
//   - The point buffer is the global `pts`; functions take the offset into it, not a pointer.
//   - a ? b : c is written as if / else; integers that are flags stay i32 as in the C source.
//   - Guards of the form !(x > 0.0) are kept as written: they are false for NaN on purpose.
//   - `score` and `sol` are the names of the scratch buffers, so the local score is score_sum and the
//     solution index argument of jpo_*_model is sol_index.
//   - One copy of an array in jpo_relpose_5pt (Pc) that works around a shader compiler fault; see there.

// The float configuration of the Metal backend (src/backend/metal/prelude.metal).
const JPO_TOL: f32 = 1.0e-6;
const JPO_STURM_DEPTH: i32 = 36;
const JPO_STURM_STACK: i32 = 40;  // JPO_STURM_DEPTH + 4

const JPO_MAX_E: i32 = 10;  // essential matrices per five-point sample
const JPO_MAX_P: i32 = 4;   // poses per P3P sample
const JPO_REL_STRIDE: u32 = 4u;  // x1 y1 x2 y2 (normalized image coordinates)
const JPO_ABS_STRIDE: u32 = 5u;  // x y (normalized image coordinates), X Y Z
const JPO_BIG: f32 = 3.0e38;
const JPO_POLISH_STEPS: i32 = 2;  // Gauss-Newton steps on each five-point solution

struct JpoItem {
  first: u32,    // first point of the problem
  n: u32,        // points
  problem: u32,  // sample counter
  iter0: u32,
  count: u32,    // hypotheses of this item (<= round)
  inv_sq_thr: f32,
  bound: f32,
  pre_k: u32,    // early rejection (kernels.inc, "Scores")
  pre_min: u32,
  pad: u32,
}

struct JpoParams {
  seed: u32,
  round: u32,     // hypotheses per item in the grid
  n_items: u32,   // items of this dispatch: the grid is rounded up to whole workgroups
  pts_base: u32,  // index in the batch of the first value of `pts` (the bound part of the point buffer)
}

struct JpoResult {
  score: f32,
  iter: u32,
  sol: i32,
  pad: u32,
  model: array<f32, 12>,
}

@group(0) @binding(0) var<storage, read> pts: array<f32>;
@group(0) @binding(1) var<storage, read> items: array<JpoItem>;
@group(0) @binding(2) var<storage, read_write> score: array<f32>;
@group(0) @binding(3) var<storage, read_write> sol: array<i32>;
@group(0) @binding(4) var<uniform> prm: JpoParams;
@group(0) @binding(5) var<storage, read_write> results: array<JpoResult>;

fn jpo_cbrt(x: f32) -> f32 {
  let r = pow(abs(x), 1.0 / 3.0);
  if (x < 0.0) {
    return -r;
  }
  return r;
}

// ---------------------------------------------------------------------------------------------------
// Sampler. Counter based: the sample of (seed, problem, iteration) is the same on every backend.
// u32 arithmetic wraps in WGSL, as in C.

fn jpo_hash(x_in: u32) -> u32 {
  var x = x_in;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

fn jpo_sample_base(seed: u32, problem: u32, iter: u32) -> u32 {
  var h = jpo_hash(seed + 0x9e3779b9u);
  h = jpo_hash(h ^ problem);
  h = jpo_hash(h ^ iter);
  return h;
}

// m distinct indices below n (n >= m, m <= 5).
fn jpo_sample(base: u32, n: u32, m: i32, idx: ptr<function, array<u32, 5>>) {
  for (var k = 0; k < m; k++) {
    var v = jpo_hash(base + u32(k + 1) * 0x9e3779b9u) % n;
    for (var t = 0; t < m; t++) {
      var clash = 0;
      for (var j = 0; j < k; j++) {
        if ((*idx)[j] == v) {
          clash = 1;
        }
      }
      if (clash == 0) {
        break;
      }
      v = (v + 1u) % n;
    }
    (*idx)[k] = v;
  }
}

// ---------------------------------------------------------------------------------------------------
// Real roots of a polynomial of degree 10 by Sturm bracketing (PoseLib misc/sturm.h, N = 10).

// Monic polynomial of degree deg: f[deg] is taken as 1.
fn jpo_polyval(f: ptr<function, array<f32, 21>>, off: i32, deg: i32, x: f32) -> f32 {
  var fx = x + (*f)[off + deg - 1];
  for (var i = deg - 2; i >= 0; i--) {
    fx = x * fx + (*f)[off + i];
  }
  return fx;
}

fn jpo_signchanges(svec: ptr<function, array<f32, 30>>, x: f32) -> i32 {
  var f: array<f32, 11>;
  f[10] = (*svec)[29];
  f[9] = (*svec)[27] + x * (*svec)[28];
  for (var i = 8; i >= 0; i--) {
    f[i] = ((*svec)[3 * i] + x * (*svec)[3 * i + 1]) * f[i + 1] + (*svec)[3 * i + 2] * f[i + 2];
  }
  var count = 0;
  var neg1 = 0;
  if (f[0] < 0.0) {
    neg1 = 1;
  }
  for (var i = 0; i < 10; i++) {
    var neg2 = 0;
    if (f[i + 1] < 0.0) {
      neg2 = 1;
    }
    if (neg1 != neg2) {
      count += 1;
    }
    neg1 = neg2;
  }
  return count;
}

// Ridders' method until close to the root, then Newton. Returns 1 and the root, or 0.
fn jpo_ridders_newton(fvec: ptr<function, array<f32, 21>>, a_in: f32, b_in: f32, root: ptr<function, f32>) -> i32 {
  var a = a_in;
  var b = b_in;
  var fa = jpo_polyval(fvec, 0, 10, a);
  var fb = jpo_polyval(fvec, 0, 10, b);
  if ((fa < 0.0) == (fb < 0.0)) {
    return 0;
  }
  for (var iter = 0; iter < 30; iter++) {
    if (abs(a - b) < 1.0e-3) {
      break;
    }
    let c = (a + b) * 0.5;
    let fc = jpo_polyval(fvec, 0, 10, c);
    let s = sqrt(fc * fc - fa * fb);
    if (!(s > 0.0)) {
      break;
    }
    var d: f32;
    if (fa < fb) {
      d = c + (a - c) * fc / s;
    } else {
      d = c + (c - a) * fc / s;
    }
    let fd = jpo_polyval(fvec, 0, 10, d);
    var c_side: bool;  // fd >= 0 ? (fc < 0) : (fc > 0)
    var a_side: bool;  // fd >= 0 ? (fa < 0) : (fa > 0)
    if (fd >= 0.0) {
      c_side = fc < 0.0;
      a_side = fa < 0.0;
    } else {
      c_side = fc > 0.0;
      a_side = fa > 0.0;
    }
    if (c_side) {
      a = c;
      fa = fc;
      b = d;
      fb = fd;
    } else if (a_side) {
      b = d;
      fb = fd;
    } else {
      a = d;
      fa = fd;
    }
  }
  var x = (a + b) * 0.5;
  for (var iter = 0; iter < 10; iter++) {
    let fx = jpo_polyval(fvec, 0, 10, x);
    if (abs(fx) < JPO_TOL) {
      break;
    }
    let fpx = 10.0 * jpo_polyval(fvec, 11, 9, x);
    let dx = fx / fpx;
    if (!(abs(dx) < JPO_BIG)) {  // flat: keep the bracketed estimate
      break;
    }
    x = x - dx;
    if (abs(dx) < JPO_TOL) {
      break;
    }
  }
  *root = x;
  return 1;
}

// coeffs[0..10], constant term first. Up to 10 roots.
fn jpo_sturm10(coeffs: ptr<function, array<f32, 11>>, roots: ptr<function, array<f32, 10>>) -> i32 {
  if ((*coeffs)[10] == 0.0) {
    return 0;
  }
  var fvec: array<f32, 21>;
  var svec: array<f32, 30>;
  let c_inv = 1.0 / (*coeffs)[10];
  for (var i = 0; i < 10; i++) {
    fvec[i] = (*coeffs)[i] * c_inv;
  }
  fvec[10] = 1.0;
  for (var i = 0; i < 9; i++) {
    fvec[11 + i] = fvec[i + 1] * (f32(i + 1) / 10.0);
  }
  fvec[20] = 1.0;

  // The Sturm sequence, as the quotients needed to evaluate it.
  {
    var f: array<f32, 30>;
    for (var i = 0; i < 21; i++) {
      f[i] = fvec[i];
    }
    for (var i = 21; i < 30; i++) {
      f[i] = 0.0;
    }
    var o1 = 0;
    var o2 = 11;
    var o3 = 21;
    for (var i = 0; i < 9; i++) {
      let q1 = f[o1 + 10 - i] * f[o2 + 9 - i];
      let q0 = f[o1 + 9 - i] * f[o2 + 9 - i] - f[o1 + 10 - i] * f[o2 + 8 - i];
      f[o3] = f[o1] - q0 * f[o2];
      for (var j = 1; j < 9 - i; j++) {
        f[o3 + j] = f[o1 + j] - q1 * f[o2 + j - 1] - q0 * f[o2 + j];
      }
      let c = -abs(f[o3 + 8 - i]);
      let ci = 1.0 / c;
      for (var j = 0; j < 9 - i; j++) {
        f[o3 + j] = f[o3 + j] * ci;
      }
      let tmp = o1;
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
  var r0 = 0.0;
  for (var i = 0; i < 10; i++) {
    let v = abs(fvec[i]);
    if (v > r0) {
      r0 = v;
    }
  }
  r0 = r0 + 1.0;
  if (!(r0 < JPO_BIG)) {  // also catches NaN
    return 0;
  }

  let sa0 = jpo_signchanges(&svec, -r0);
  let sb0 = jpo_signchanges(&svec, r0);
  if (sa0 - sb0 <= 0) {
    return 0;
  }

  // Isolate the roots by bisection; an explicit stack instead of PoseLib's recursion.
  var sta: array<f32, JPO_STURM_STACK>;
  var stb: array<f32, JPO_STURM_STACK>;
  var stsa: array<i32, JPO_STURM_STACK>;
  var stsb: array<i32, JPO_STURM_STACK>;
  var stdp: array<i32, JPO_STURM_STACK>;
  var sp = 0;
  sta[0] = -r0;
  stb[0] = r0;
  stsa[0] = sa0;
  stsb[0] = sb0;
  stdp[0] = 0;
  sp = 1;
  var n_roots = 0;
  // Each level is visited at most twice per root.
  for (var guard = 0; guard < 20 * (JPO_STURM_DEPTH + 2) && sp > 0 && n_roots < 10; guard++) {
    sp -= 1;
    let a = sta[sp];
    let b = stb[sp];
    let sa = stsa[sp];
    let sb = stsb[sp];
    let depth = stdp[sp];
    if (depth > JPO_STURM_DEPTH) {
      continue;
    }
    if (b - a < JPO_TOL) {
      (*roots)[n_roots] = b;
      n_roots += 1;
      continue;
    }
    let n_rts = sa - sb;
    if (n_rts > 1) {
      let c = (a + b) * 0.5;
      let sc = jpo_signchanges(&svec, c);
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
      var r: f32;
      if (jpo_ridders_newton(&fvec, a, b, &r) != 0) {
        (*roots)[n_roots] = r;
        n_roots += 1;
      }
    }
  }
  return n_roots;
}

// ---------------------------------------------------------------------------------------------------
// Five-point relative pose (Nister). b1, b2: five unit bearings each (5x3). E: up to 10 essential
// matrices, row major, each of unit Frobenius norm, with b2' E b1 = 0. Returns the count.

fn jpo_relpose_5pt(b1: ptr<function, array<f32, 15>>, b2: ptr<function, array<f32, 15>>,
                   E: ptr<function, array<f32, 90>>) -> i32 {
  // Null space of the 5x9 epipolar constraints by Householder reflections of their transpose.
  var V: array<f32, 45>;
  for (var k = 0; k < 5; k++) {
    var col: array<f32, 9>;
    for (var r = 0; r < 3; r++) {
      for (var c = 0; c < 3; c++) {
        col[3 * r + c] = (*b2)[3 * k + r] * (*b1)[3 * k + c];
      }
    }
    for (var j = 0; j < k; j++) {
      var d = 0.0;
      for (var i = 0; i < 9; i++) {
        d += V[9 * j + i] * col[i];
      }
      d = d * 2.0;
      for (var i = 0; i < 9; i++) {
        col[i] -= d * V[9 * j + i];
      }
    }
    var nrm = 0.0;
    for (var i = k; i < 9; i++) {
      nrm += col[i] * col[i];
    }
    nrm = sqrt(nrm);
    if (!(nrm > 0.0)) {
      return 0;
    }
    for (var i = 0; i < k; i++) {
      V[9 * k + i] = 0.0;
    }
    if (col[k] < 0.0) {
      V[9 * k + k] = col[k] + -nrm;
    } else {
      V[9 * k + k] = col[k] + nrm;
    }
    for (var i = k + 1; i < 9; i++) {
      V[9 * k + i] = col[i];
    }
    var vn = 0.0;
    for (var i = k; i < 9; i++) {
      vn += V[9 * k + i] * V[9 * k + i];
    }
    vn = sqrt(vn);
    if (!(vn > 0.0)) {
      return 0;
    }
    vn = 1.0 / vn;
    for (var i = k; i < 9; i++) {
      V[9 * k + i] *= vn;
    }
  }
  // N[v][e]: the four null vectors; E(x, y, z) = x N0 + y N1 + z N2 + N3.
  var N: array<f32, 36>;
  for (var v = 0; v < 4; v++) {
    var q: array<f32, 9>;
    for (var i = 0; i < 9; i++) {
      q[i] = 0.0;
    }
    q[5 + v] = 1.0;
    for (var k = 4; k >= 0; k--) {
      var d = 0.0;
      for (var i = 0; i < 9; i++) {
        d += V[9 * k + i] * q[i];
      }
      d = d * 2.0;
      for (var i = 0; i < 9; i++) {
        q[i] -= d * V[9 * k + i];
      }
    }
    for (var i = 0; i < 9; i++) {
      N[9 * v + i] = q[i];
    }
  }

  // Monomials. Degree 1: x y z 1. Degree 2: x2 xy xz x y2 yz y z2 z 1. Degree 3, Nister's order:
  // x3 y3 x2y xy2 x2z x2 y2z y2 xyz xy | xz2 xz x yz2 yz y z3 z2 z 1.
  var M2 = array<i32, 16>(0, 1, 2, 3, 1, 4, 5, 6, 2, 5, 7, 8, 3, 6, 8, 9);
  var M3 = array<i32, 40>(0, 2, 4, 5, 2, 3, 8, 9, 4, 8, 10, 11, 5, 9, 11, 12, 3, 1, 6, 7,
                          8, 6, 13, 14, 9, 7, 14, 15, 10, 13, 16, 17, 11, 14, 17, 18, 12, 15, 18, 19);

  // The ten cubic constraints, 10x20: det(E) = 0 and 2 E E' E - trace(E E') E = 0.
  var C: array<f32, 200>;
  for (var i = 0; i < 200; i++) {
    C[i] = 0.0;
  }
  // L = E E' - 0.5 trace(E E') I, degree 2, symmetric: L[i][j] at 10 * (3 i + j).
  var L: array<f32, 90>;
  for (var i = 0; i < 90; i++) {
    L[i] = 0.0;
  }
  for (var i = 0; i < 3; i++) {
    for (var j = i; j < 3; j++) {
      for (var k = 0; k < 3; k++) {
        for (var a = 0; a < 4; a++) {
          let ea = N[9 * a + 3 * i + k];
          for (var b = 0; b < 4; b++) {
            L[10 * (3 * i + j) + M2[4 * a + b]] += ea * N[9 * b + 3 * j + k];
          }
        }
      }
    }
  }
  for (var m = 0; m < 10; m++) {
    let tr = 0.5 * (L[m] + L[40 + m] + L[80 + m]);
    L[m] -= tr;
    L[40 + m] -= tr;
    L[80 + m] -= tr;
    L[30 + m] = L[10 + m];
    L[60 + m] = L[20 + m];
    L[70 + m] = L[50 + m];
  }
  for (var i = 0; i < 3; i++) {
    for (var j = 0; j < 3; j++) {
      let row = 20 * (3 * i + j);
      for (var k = 0; k < 3; k++) {
        for (var m = 0; m < 10; m++) {
          let lm = L[10 * (3 * i + k) + m];
          for (var b = 0; b < 4; b++) {
            C[row + M3[4 * m + b]] += lm * N[9 * b + 3 * k + j];
          }
        }
      }
    }
  }
  // det(E), expanded along the first row.
  for (var t = 0; t < 3; t++) {
    let c1 = (t + 1) % 3;
    let c2 = (t + 2) % 3;
    var minor: array<f32, 10>;
    for (var m = 0; m < 10; m++) {
      minor[m] = 0.0;
    }
    for (var a = 0; a < 4; a++) {
      for (var b = 0; b < 4; b++) {
        minor[M2[4 * a + b]] += N[9 * a + 3 + c1] * N[9 * b + 6 + c2] - N[9 * a + 3 + c2] * N[9 * b + 6 + c1];
      }
    }
    for (var m = 0; m < 10; m++) {
      for (var b = 0; b < 4; b++) {
        C[180 + M3[4 * m + b]] += minor[m] * N[9 * b + t];
      }
    }
  }

  // Gauss-Jordan on the first ten columns, partial pivoting.
  for (var col = 0; col < 10; col++) {
    var piv = col;
    var best = abs(C[20 * col + col]);
    for (var r = col + 1; r < 10; r++) {
      let v = abs(C[20 * r + col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (!(best > 0.0)) {
      return 0;
    }
    if (piv != col) {
      for (var c = col; c < 20; c++) {
        let tmp = C[20 * col + c];
        C[20 * col + c] = C[20 * piv + c];
        C[20 * piv + c] = tmp;
      }
    }
    let inv = 1.0 / C[20 * col + col];
    for (var c = col; c < 20; c++) {
      C[20 * col + c] *= inv;
    }
    for (var r = 0; r < 10; r++) {
      if (r != col) {
        let f = C[20 * r + col];
        if (f != 0.0) {
          for (var c = col; c < 20; c++) {
            C[20 * r + c] -= f * C[20 * col + c];
          }
        }
      }
    }
  }

  // Rows 4..9 pair up into three equations x p(z) + y q(z) + r(z) = 0; p, q of degree 3, r of degree 4.
  // P[i][d], Q[i][d], R[i][d]: coefficient of z^d.
  var P: array<f32, 12>;
  var Q: array<f32, 12>;
  var R: array<f32, 15>;
  for (var i = 0; i < 3; i++) {
    let ra = 20 * (4 + 2 * i) + 10;
    let rb = 20 * (5 + 2 * i) + 10;
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
  // Not in kernels.inc: the products below read a copy of P. Without it, Apple's shader compiler (macOS 26,
  // M5 Pro, through Dawn's Metal backend) loses P: after this loop nest and the call of jpo_sturm10, the
  // loads P[4 * i + d] in the solution loop return 0 while P[0] still reads right, so no sample has a
  // solution. With one array per loop nest it compiles correctly. The values are the same.
  var Pc = P;
  var c: array<f32, 11>;
  for (var i = 0; i < 11; i++) {
    c[i] = 0.0;
  }
  for (var t = 0; t < 3; t++) {
    let j = (t + 1) % 3;
    let k = (t + 2) % 3;
    var mnr: array<f32, 8>;
    for (var d = 0; d < 8; d++) {
      mnr[d] = 0.0;
    }
    for (var a = 0; a < 4; a++) {
      for (var b = 0; b < 5; b++) {
        mnr[a + b] += Q[4 * j + a] * R[5 * k + b] - Q[4 * k + a] * R[5 * j + b];
      }
    }
    for (var a = 0; a < 4; a++) {
      for (var d = 0; d < 8; d++) {
        c[a + d] += Pc[4 * t + a] * mnr[d];
      }
    }
  }

  var roots: array<f32, 10>;
  let n_roots = jpo_sturm10(&c, &roots);

  var n_sols = 0;
  for (var s = 0; s < n_roots; s++) {
    var z = roots[s];
    var bx: array<f32, 3>;
    var by: array<f32, 3>;
    var bb: array<f32, 3>;
    for (var i = 0; i < 3; i++) {
      bx[i] = ((P[4 * i + 3] * z + P[4 * i + 2]) * z + P[4 * i + 1]) * z + P[4 * i + 0];
      by[i] = ((Q[4 * i + 3] * z + Q[4 * i + 2]) * z + Q[4 * i + 1]) * z + Q[4 * i + 0];
      bb[i] = (((R[5 * i + 4] * z + R[5 * i + 3]) * z + R[5 * i + 2]) * z + R[5 * i + 1]) * z + R[5 * i + 0];
    }
    // Two of the three rows: the pair with the largest determinant.
    var i0 = 0;
    var i1 = 1;
    var det = bx[0] * by[1] - bx[1] * by[0];
    let d02 = bx[0] * by[2] - bx[2] * by[0];
    let d12 = bx[1] * by[2] - bx[2] * by[1];
    if (abs(d02) > abs(det)) {
      det = d02;
      i0 = 0;
      i1 = 2;
    }
    if (abs(d12) > abs(det)) {
      det = d12;
      i0 = 1;
      i1 = 2;
    }
    if (!(abs(det) > 0.0)) {
      continue;
    }
    var x = (-bb[i0] * by[i1] + bb[i1] * by[i0]) / det;
    var y = (-bx[i0] * bb[i1] + bx[i1] * bb[i0]) / det;
    // Polish (x, y, z) on the ten eliminated constraints (row r: monomial r plus ten coefficients of
    // the monomials 10..19): Gauss-Newton steps. The determinant and its roots lose digits, most of all
    // in float; this gets them back.
    for (var it = 0; it < JPO_POLISH_STEPS; it++) {
      var mon: array<f32, 20>;
      var gx: array<f32, 20>;
      var gy: array<f32, 20>;
      var gz: array<f32, 20>;
      for (var i = 0; i < 20; i++) {
        gx[i] = 0.0;
        gy[i] = 0.0;
        gz[i] = 0.0;
      }
      let xx = x * x;
      let yy = y * y;
      let zz = z * z;
      mon[0] = xx * x;
      gx[0] = 3.0 * xx;
      mon[1] = yy * y;
      gy[1] = 3.0 * yy;
      mon[2] = xx * y;
      gx[2] = 2.0 * x * y;
      gy[2] = xx;
      mon[3] = x * yy;
      gx[3] = yy;
      gy[3] = 2.0 * x * y;
      mon[4] = xx * z;
      gx[4] = 2.0 * x * z;
      gz[4] = xx;
      mon[5] = xx;
      gx[5] = 2.0 * x;
      mon[6] = yy * z;
      gy[6] = 2.0 * y * z;
      gz[6] = yy;
      mon[7] = yy;
      gy[7] = 2.0 * y;
      mon[8] = x * y * z;
      gx[8] = y * z;
      gy[8] = x * z;
      gz[8] = x * y;
      mon[9] = x * y;
      gx[9] = y;
      gy[9] = x;
      mon[10] = x * zz;
      gx[10] = zz;
      gz[10] = 2.0 * x * z;
      mon[11] = x * z;
      gx[11] = z;
      gz[11] = x;
      mon[12] = x;
      gx[12] = 1.0;
      mon[13] = y * zz;
      gy[13] = zz;
      gz[13] = 2.0 * y * z;
      mon[14] = y * z;
      gy[14] = z;
      gz[14] = y;
      mon[15] = y;
      gy[15] = 1.0;
      mon[16] = zz * z;
      gz[16] = 3.0 * zz;
      mon[17] = zz;
      gz[17] = 2.0 * z;
      mon[18] = z;
      gz[18] = 1.0;
      mon[19] = 1.0;
      var a00 = 0.0;
      var a01 = 0.0;
      var a02 = 0.0;
      var a11 = 0.0;
      var a12 = 0.0;
      var a22 = 0.0;
      var g0 = 0.0;
      var g1 = 0.0;
      var g2 = 0.0;
      for (var r = 0; r < 10; r++) {
        var f = mon[r];
        var fx = gx[r];
        var fy = gy[r];
        var fz = gz[r];
        for (var m = 10; m < 20; m++) {
          let cm = C[20 * r + m];
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
      let c00 = a11 * a22 - a12 * a12;
      let c01 = a02 * a12 - a01 * a22;
      let c02 = a01 * a12 - a02 * a11;
      let dt = a00 * c00 + a01 * c01 + a02 * c02;
      if (!(abs(dt) > 0.0)) {
        break;
      }
      let c11 = a00 * a22 - a02 * a02;
      let c12 = a01 * a02 - a00 * a12;
      let c22 = a00 * a11 - a01 * a01;
      let idt = 1.0 / dt;
      let dx = (c00 * g0 + c01 * g1 + c02 * g2) * idt;
      let dy = (c01 * g0 + c11 * g1 + c12 * g2) * idt;
      let dz = (c02 * g0 + c12 * g1 + c22 * g2) * idt;
      if (!(abs(dx) + abs(dy) + abs(dz) < JPO_BIG)) {
        break;
      }
      x -= dx;
      y -= dy;
      z -= dz;
    }
    let nrm = sqrt(x * x + y * y + z * z + 1.0);
    if (!(nrm < JPO_BIG)) {
      continue;
    }
    let inv = 1.0 / nrm;
    for (var i = 0; i < 9; i++) {
      (*E)[9 * n_sols + i] = (N[i] * x + N[9 + i] * y + N[18 + i] * z + N[27 + i]) * inv;
    }
    n_sols += 1;
  }
  return n_sols;
}

// ---------------------------------------------------------------------------------------------------
// P3P (Ding et al., "Revisiting the P3P Problem"). xin: three unit bearings (3x3), Xin: three points
// (3x3). P: up to 4 poses, each 12 values: R row major (9), then t; a point maps to R X + t.

fn jpo_cubic_single_real(c2: f32, c1: f32, c0: f32, root: ptr<function, f32>) -> i32 {
  let a = c1 - c2 * c2 / 3.0;
  var b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
  var c = b * b / 4.0 + a * a * a / 27.0;
  if (c != 0.0) {
    if (c > 0.0) {
      c = sqrt(c);
      b *= -0.5;
      *root = jpo_cbrt(b + c) + jpo_cbrt(b - c) - c2 / 3.0;
      return 1;
    }
    c = 3.0 * b / (2.0 * a) * sqrt(-3.0 / a);
    if (c > 1.0) {
      c = 1.0;
    }
    if (c < -1.0) {
      c = -1.0;
    }
    *root = 2.0 * sqrt(-a / 3.0) * cos(acos(c) / 3.0) - c2 / 3.0;
  } else {
    if (a != 0.0) {
      *root = -c2 / 3.0 + (3.0 * b / a);
    } else {
      *root = -c2 / 3.0 + 0.0;
    }
  }
  return 0;
}

fn jpo_root2real(b: f32, c: f32, r: ptr<function, array<f32, 2>>) -> i32 {
  let v = b * b - 4.0 * c;
  if (v < -1.0e-12) {
    (*r)[0] = -0.5 * b;
    (*r)[1] = (*r)[0];
    return 0;
  }
  if (v < 0.0) {
    (*r)[0] = -0.5 * b;
    (*r)[1] = -2.0;
    return 1;
  }
  let y = sqrt(v);
  if (b < 0.0) {
    (*r)[0] = 0.5 * (-b + y);
    (*r)[1] = 0.5 * (-b - y);
  } else {
    (*r)[0] = 2.0 * c / (-b + y);
    (*r)[1] = 2.0 * c / (-b - y);
  }
  return 1;
}

fn jpo_p3p(xin: ptr<function, array<f32, 9>>, Xin: ptr<function, array<f32, 9>>,
           P: ptr<function, array<f32, 48>>) -> i32 {
  var x: array<f32, 9>;
  var X: array<f32, 9>;
  for (var i = 0; i < 9; i++) {
    x[i] = (*xin)[i];
    X[i] = (*Xin)[i];
  }
  var X01: array<f32, 3>;
  var X02: array<f32, 3>;
  var X12: array<f32, 3>;
  for (var i = 0; i < 3; i++) {
    X01[i] = X[i] - X[3 + i];
    X02[i] = X[i] - X[6 + i];
    X12[i] = X[3 + i] - X[6 + i];
  }
  var a01 = X01[0] * X01[0] + X01[1] * X01[1] + X01[2] * X01[2];
  var a02 = X02[0] * X02[0] + X02[1] * X02[1] + X02[2] * X02[2];
  var a12 = X12[0] * X12[0] + X12[1] * X12[1] + X12[2] * X12[2];

  // Reorder so that the largest distance is between points 1 and 2.
  if (a01 > a02) {
    if (a01 > a12) {
      for (var i = 0; i < 3; i++) {
        var t = x[i];
        x[i] = x[6 + i];
        x[6 + i] = t;
        t = X[i];
        X[i] = X[6 + i];
        X[6 + i] = t;
        X01[i] = -X12[i];
        X02[i] = -X02[i];
      }
      let t = a01;
      a01 = a12;
      a12 = t;
    }
  } else if (a02 > a12) {
    for (var i = 0; i < 3; i++) {
      var t = x[i];
      x[i] = x[3 + i];
      x[3 + i] = t;
      t = X[i];
      X[i] = X[3 + i];
      X[3 + i] = t;
      X01[i] = -X01[i];
      X02[i] = X12[i];
    }
    let t = a02;
    a02 = a12;
    a12 = t;
  }
  if (!(a12 > 0.0)) {
    return 0;
  }

  let a12d = 1.0 / a12;
  let a = a01 * a12d;
  let b = a02 * a12d;
  let m01 = x[0] * x[3] + x[1] * x[4] + x[2] * x[5];
  let m02 = x[0] * x[6] + x[1] * x[7] + x[2] * x[8];
  let m12 = x[3] * x[6] + x[4] * x[7] + x[5] * x[8];

  let m12sq = -m12 * m12 + 1.0;
  let m02sq = -1.0 + m02 * m02;
  let m01sq = -1.0 + m01 * m01;
  let ab = a * b;
  let bsq = b * b;
  let asq = a * a;
  let m013 = -2.0 + 2.0 * m01 * m02 * m12;
  let bsqm12sq = bsq * m12sq;
  let asqm12sq = asq * m12sq;
  let abm12sq = 2.0 * ab * m12sq;

  let k3_inv = 1.0 / (bsqm12sq + b * m02sq);
  let k2 = k3_inv * ((-1.0 + a) * m02sq + abm12sq + bsqm12sq + b * m013);
  let k1 = k3_inv * (asqm12sq + abm12sq + a * m013 + (-1.0 + b) * m01sq);
  let k0 = k3_inv * (asqm12sq + a * m01sq);
  if (!(abs(k2) < JPO_BIG) || !(abs(k1) < JPO_BIG) || !(abs(k0) < JPO_BIG)) {
    return 0;
  }

  var s: f32;
  let G = jpo_cubic_single_real(k2, k1, k0, &s);

  var Cm: array<f32, 9>;
  Cm[0] = -a + s * (1.0 - b);
  Cm[1] = -m02 * s;
  Cm[2] = a * m12 + b * m12 * s;
  Cm[3] = Cm[1];
  Cm[4] = s + 1.0;
  Cm[5] = -m01;
  Cm[6] = Cm[2];
  Cm[7] = Cm[5];
  Cm[8] = -a - b * s + 1.0;

  // Factor the degenerate conic into two lines p, q.
  var pq: array<f32, 6>;
  {
    var adj: array<f32, 9>;
    adj[0] = -(Cm[4] * Cm[8] - Cm[5] * Cm[7]);
    adj[1] = -(Cm[2] * Cm[7] - Cm[1] * Cm[8]);
    adj[2] = -(Cm[1] * Cm[5] - Cm[2] * Cm[4]);
    adj[3] = -(Cm[5] * Cm[6] - Cm[3] * Cm[8]);
    adj[4] = -(Cm[0] * Cm[8] - Cm[2] * Cm[6]);
    adj[5] = -(Cm[2] * Cm[3] - Cm[0] * Cm[5]);
    adj[6] = -(Cm[3] * Cm[7] - Cm[4] * Cm[6]);
    adj[7] = -(Cm[1] * Cm[6] - Cm[0] * Cm[7]);
    adj[8] = -(Cm[0] * Cm[4] - Cm[1] * Cm[3]);
    var col = 2;
    if (adj[0] > adj[4]) {
      if (adj[0] > adj[8]) {
        col = 0;
      }
    } else if (adj[4] > adj[8]) {
      col = 1;
    }
    let dd = adj[4 * col];
    if (!(dd > 0.0)) {
      return 0;
    }
    let inv = 1.0 / sqrt(dd);
    let v0 = adj[col] * inv;
    let v1 = adj[3 + col] * inv;
    let v2 = adj[6 + col] * inv;
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
  var XX: array<f32, 9>;
  {
    var M: array<f32, 9>;
    let cr0 = X01[1] * X02[2] - X01[2] * X02[1];
    let cr1 = X01[2] * X02[0] - X01[0] * X02[2];
    let cr2 = X01[0] * X02[1] - X01[1] * X02[0];
    M[0] = X01[0];
    M[1] = X02[0];
    M[2] = cr0;
    M[3] = X01[1];
    M[4] = X02[1];
    M[5] = cr1;
    M[6] = X01[2];
    M[7] = X02[2];
    M[8] = cr2;
    let det = M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) +
              M[2] * (M[3] * M[7] - M[4] * M[6]);
    if (!(abs(det) > 0.0)) {
      return 0;
    }
    let id = 1.0 / det;
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

  var n_sols = 0;
  for (var i = 0; i < 2; i++) {
    let p0 = pq[3 * i];
    let p1 = pq[3 * i + 1];
    let p2 = pq[3 * i + 2];
    var switch_12 = 0;
    if (abs(p0) <= abs(p1)) {
      switch_12 = 1;
    }
    var w0: f32;
    var w1: f32;
    var cb: f32;
    var cc: f32;
    if (switch_12 != 0) {
      w0 = -p0 / p1;
      w1 = -p2 / p1;
      let ca = 1.0 / (w1 * w1 - b);
      cb = 2.0 * (b * m12 - m02 * w1 + w0 * w1) * ca;
      cc = (w0 * w0 - 2.0 * m02 * w0 - b + 1.0) * ca;
    } else {
      w0 = -p1 / p0;
      w1 = -p2 / p0;
      let ca = 1.0 / (-a * w1 * w1 + 2.0 * a * m12 * w1 - a + 1.0);
      cb = 2.0 * (a * m12 * w0 - m01 - a * w0 * w1) * ca;
      cc = (1.0 - a * w0 * w0) * ca;
    }
    var taus: array<f32, 2>;
    if (jpo_root2real(cb, cc, &taus) == 0) {
      continue;
    }
    for (var ti = 0; ti < 2; ti++) {
      let tau = taus[ti];
      if (!(tau > 0.0)) {
        continue;
      }
      var d0: f32;
      var d1: f32;
      var d2: f32;
      if (switch_12 != 0) {
        d2 = sqrt(a12 / (tau * (tau - 2.0 * m12) + 1.0));
        d1 = tau * d2;
        d0 = w0 * d2 + w1 * d1;
        if (!(d0 >= 0.0)) {
          continue;
        }
      } else {
        d0 = sqrt(a01 / (tau * (tau - 2.0 * m01) + 1.0));
        d1 = tau * d0;
        d2 = w0 * d0 + w1 * d1;
        if (!(d2 >= 0.0)) {
          continue;
        }
      }
      // A few Newton steps on the three distance equations.
      for (var iter = 0; iter < 5; iter++) {
        let r1 = d0 * d0 - 2.0 * d0 * d1 * m01 + d1 * d1 - a01;
        let r2 = d0 * d0 - 2.0 * d0 * d2 * m02 + d2 * d2 - a02;
        let r3 = d1 * d1 - 2.0 * d1 * d2 * m12 + d2 * d2 - a12;
        if (abs(r1) + abs(r2) + abs(r3) < JPO_TOL) {
          break;
        }
        let x11 = d0 - d1 * m01;
        let x12 = d1 - d0 * m01;
        let x21 = d0 - d2 * m02;
        let x23 = d2 - d0 * m02;
        let x32 = d1 - d2 * m12;
        let x33 = d2 - d1 * m12;
        let den = x11 * x23 * x32 + x12 * x21 * x33;
        if (!(abs(den) > 0.0)) {
          break;
        }
        let detJ = 0.5 / den;
        d0 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ;
        d1 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ;
        d2 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ;
      }
      var v1: array<f32, 3>;
      var v2: array<f32, 3>;
      var v3: array<f32, 3>;
      for (var k = 0; k < 3; k++) {
        v1[k] = d0 * x[k] - d1 * x[3 + k];
        v2[k] = d0 * x[k] - d2 * x[6 + k];
      }
      v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
      v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
      v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
      let o = 12 * n_sols;
      for (var r = 0; r < 3; r++) {
        for (var cI = 0; cI < 3; cI++) {
          (*P)[o + 3 * r + cI] = v1[r] * XX[cI] + v2[r] * XX[3 + cI] + v3[r] * XX[6 + cI];
        }
      }
      var ok = 1;
      for (var r = 0; r < 3; r++) {
        let tr = d0 * x[r] - ((*P)[o + 3 * r] * X[0] + (*P)[o + 3 * r + 1] * X[1] + (*P)[o + 3 * r + 2] * X[2]);
        (*P)[o + 9 + r] = tr;
        if (!(abs(tr) < JPO_BIG)) {
          ok = 0;
        }
      }
      if (ok != 0 && n_sols < JPO_MAX_P) {
        n_sols += 1;
      }
    }
    if (n_sols > 0 && G != 0) {
      break;
    }
    if (n_sols >= JPO_MAX_P) {
      break;
    }
  }
  return n_sols;
}

// ---------------------------------------------------------------------------------------------------
// Scores. Truncated squared error (MSAC) in units of the squared threshold: each point adds
// min(r2 / thr2, 1). A score that passes `bound` cannot win, so the loop stops there; the value
// returned is then only known to be above the bound.
//
// Early rejection: points are visited from `start` (a per-hypothesis offset, so that the first pre_k
// points are a different subset for every hypothesis). A hypothesis with fewer than pre_min inliers
// among its first pre_k points is dropped (JPO_BIG). The host chooses pre_min so that a hypothesis as
// good as the best so far passes with probability 0.999 (core/estimator.cpp); pre_min = 0 turns the
// test off. pre_k is a multiple of 16.

// The four motions of an essential matrix (PoseLib motion_from_essential): P holds 4 poses of 12
// values, R row major then t. No singular value decomposition: cross products of the columns of E.
fn jpo_essential_poses(E: ptr<function, array<f32, 9>>, P: ptr<function, array<f32, 48>>) {
  var c0: array<f32, 3>;
  var c1: array<f32, 3>;
  var c2: array<f32, 3>;
  for (var i = 0; i < 3; i++) {
    c0[i] = (*E)[3 * i];
    c1[i] = (*E)[3 * i + 1];
    c2[i] = (*E)[3 * i + 2];
  }
  var u12: array<f32, 3>;
  var u13: array<f32, 3>;
  var u23: array<f32, 3>;
  u12[0] = c0[1] * c1[2] - c0[2] * c1[1];
  u12[1] = c0[2] * c1[0] - c0[0] * c1[2];
  u12[2] = c0[0] * c1[1] - c0[1] * c1[0];
  u13[0] = c0[1] * c2[2] - c0[2] * c2[1];
  u13[1] = c0[2] * c2[0] - c0[0] * c2[2];
  u13[2] = c0[0] * c2[1] - c0[1] * c2[0];
  u23[0] = c1[1] * c2[2] - c1[2] * c2[1];
  u23[1] = c1[2] * c2[0] - c1[0] * c2[2];
  u23[2] = c1[0] * c2[1] - c1[1] * c2[0];
  let n12 = u12[0] * u12[0] + u12[1] * u12[1] + u12[2] * u12[2];
  let n13 = u13[0] * u13[0] + u13[1] * u13[1] + u13[2] * u13[2];
  let n23 = u23[0] * u23[0] + u23[1] * u23[1] + u23[2] * u23[2];
  var w0: array<f32, 3>;  // columns of U W
  var w1: array<f32, 3>;
  var w2: array<f32, 3>;
  var pick = 0;  // 0: (c0, u12), 1: (c0, u13), 2: (c1, u23)
  if (n12 > n13) {
    if (n12 > n23) {
      pick = 0;
    } else {
      pick = 2;
    }
  } else {
    if (n13 > n23) {
      pick = 1;
    } else {
      pick = 2;
    }
  }
  for (var i = 0; i < 3; i++) {
    if (pick == 2) {
      w1[i] = c1[i];
    } else {
      w1[i] = c0[i];
    }
    if (pick == 0) {
      w2[i] = u12[i];
    } else if (pick == 1) {
      w2[i] = u13[i];
    } else {
      w2[i] = u23[i];
    }
  }
  let i1 = 1.0 / sqrt(w1[0] * w1[0] + w1[1] * w1[1] + w1[2] * w1[2]);
  let i2 = 1.0 / sqrt(w2[0] * w2[0] + w2[1] * w2[1] + w2[2] * w2[2]);
  for (var i = 0; i < 3; i++) {
    w1[i] *= i1;
    w2[i] *= i2;
  }
  w0[0] = -(w2[1] * w1[2] - w2[2] * w1[1]);
  w0[1] = -(w2[2] * w1[0] - w2[0] * w1[2]);
  w0[2] = -(w2[0] * w1[1] - w2[1] * w1[0]);
  var v0: array<f32, 3>;  // rows of V'
  var v1: array<f32, 3>;
  var v2: array<f32, 3>;
  for (var c = 0; c < 3; c++) {
    v0[c] = w1[0] * (*E)[c] + w1[1] * (*E)[3 + c] + w1[2] * (*E)[6 + c];
    v1[c] = -(w0[0] * (*E)[c] + w0[1] * (*E)[3 + c] + w0[2] * (*E)[6 + c]);
  }
  var iv = 1.0 / sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
  for (var c = 0; c < 3; c++) {
    v0[c] *= iv;
  }
  let d = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
  for (var c = 0; c < 3; c++) {
    v1[c] -= d * v0[c];
  }
  iv = 1.0 / sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
  for (var c = 0; c < 3; c++) {
    v1[c] *= iv;
  }
  v2[0] = v0[1] * v1[2] - v0[2] * v1[1];
  v2[1] = v0[2] * v1[0] - v0[0] * v1[2];
  v2[2] = v0[0] * v1[1] - v0[1] * v1[0];
  for (var k = 0; k < 2; k++) {
    var sg = 1.0;  // the second rotation flips the first two columns of U W
    if (k != 0) {
      sg = -1.0;
    }
    for (var j = 0; j < 2; j++) {
      let o = 12 * (2 * k + j);
      for (var r = 0; r < 3; r++) {
        for (var c = 0; c < 3; c++) {
          (*P)[o + 3 * r + c] = sg * (w0[r] * v0[c] + w1[r] * v1[c]) + w2[r] * v2[c];
        }
        if (j == 0) {
          (*P)[o + 9 + r] = w2[r];
        } else {
          (*P)[o + 9 + r] = -w2[r];
        }
      }
    }
  }
}

// PoseLib's cheirality test for unit bearings b1 (camera a) and b2 (camera b): both depths of the
// point they triangulate to are above 0.01.
fn jpo_cheirality(P: ptr<function, array<f32, 12>>, b10: f32, b11: f32, b12: f32, b20: f32, b21: f32,
                  b22: f32) -> i32 {
  let r0 = (*P)[0] * b10 + (*P)[1] * b11 + (*P)[2] * b12;
  let r1 = (*P)[3] * b10 + (*P)[4] * b11 + (*P)[5] * b12;
  let r2 = (*P)[6] * b10 + (*P)[7] * b11 + (*P)[8] * b12;
  let a = -(r0 * b20 + r1 * b21 + r2 * b22);
  let bb1 = -(r0 * (*P)[9] + r1 * (*P)[10] + r2 * (*P)[11]);
  let bb2 = b20 * (*P)[9] + b21 * (*P)[10] + b22 * (*P)[11];
  let l1 = bb1 - a * bb2;
  let l2 = -a * bb1 + bb2;
  let md = 0.01 * (1.0 - a * a);
  if (l1 > md && l2 > md) {
    return 1;
  }
  return 0;
}

// Sampson error of the pairs pts[off + 4 i ..] against the relative pose P (12); a pair within the
// threshold that fails the cheirality test counts as an outlier (as PoseLib).
fn jpo_score_rel(P: ptr<function, array<f32, 12>>, off: u32, n: u32, inv_sq_thr: f32, bound: f32, start: u32,
                 pre_k: u32, pre_min: u32) -> f32 {
  var score_sum = 0.0;
  var inl = 0u;
  // E = [t]x R
  let e0 = -(*P)[11] * (*P)[3] + (*P)[10] * (*P)[6];
  let e1 = -(*P)[11] * (*P)[4] + (*P)[10] * (*P)[7];
  let e2 = -(*P)[11] * (*P)[5] + (*P)[10] * (*P)[8];
  let e3 = (*P)[11] * (*P)[0] - (*P)[9] * (*P)[6];
  let e4 = (*P)[11] * (*P)[1] - (*P)[9] * (*P)[7];
  let e5 = (*P)[11] * (*P)[2] - (*P)[9] * (*P)[8];
  let e6 = -(*P)[10] * (*P)[0] + (*P)[9] * (*P)[3];
  let e7 = -(*P)[10] * (*P)[1] + (*P)[9] * (*P)[4];
  let e8 = -(*P)[10] * (*P)[2] + (*P)[9] * (*P)[5];
  for (var i = 0u; i < n; i++) {
    var pi = start + i;
    if (pi >= n) {
      pi -= n;
    }
    let o = off + JPO_REL_STRIDE * pi;
    let x1 = pts[o];
    let y1 = pts[o + 1u];
    let x2 = pts[o + 2u];
    let y2 = pts[o + 3u];
    let a0 = e0 * x1 + e1 * y1 + e2;
    let a1 = e3 * x1 + e4 * y1 + e5;
    let a2 = e6 * x1 + e7 * y1 + e8;
    let b0 = e0 * x2 + e3 * y2 + e6;
    let b1 = e1 * x2 + e4 * y2 + e7;
    let Cc = x2 * a0 + y2 * a1 + a2;
    let r2 = Cc * Cc / (a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1) * inv_sq_thr;
    var add = 1.0;  // NaN counts as an outlier
    if (r2 < 1.0) {
      let i1 = 1.0 / sqrt(x1 * x1 + y1 * y1 + 1.0);
      let i2 = 1.0 / sqrt(x2 * x2 + y2 * y2 + 1.0);
      if (jpo_cheirality(P, x1 * i1, y1 * i1, i1, x2 * i2, y2 * i2, i2) != 0) {
        add = r2;
        inl += 1u;
      }
    }
    score_sum += add;
    if ((i & 15u) == 15u) {
      if (score_sum > bound) {
        break;
      }
      if (i + 1u == pre_k && inl < pre_min) {
        return JPO_BIG;
      }
    }
  }
  return score_sum;
}

// Reprojection error on the normalized image plane of pts[off + 5 i ..] against the pose P (12).
fn jpo_score_abs(P: ptr<function, array<f32, 12>>, off: u32, n: u32, inv_sq_thr: f32, bound: f32, start: u32,
                 pre_k: u32, pre_min: u32) -> f32 {
  var score_sum = 0.0;
  var inl = 0u;
  let r0 = (*P)[0];
  let r1 = (*P)[1];
  let r2 = (*P)[2];
  let r3 = (*P)[3];
  let r4 = (*P)[4];
  let r5 = (*P)[5];
  let r6 = (*P)[6];
  let r7 = (*P)[7];
  let r8 = (*P)[8];
  let t0 = (*P)[9];
  let t1 = (*P)[10];
  let t2 = (*P)[11];
  for (var i = 0u; i < n; i++) {
    var pi = start + i;
    if (pi >= n) {
      pi -= n;
    }
    let o = off + JPO_ABS_STRIDE * pi;
    let X = pts[o + 2u];
    let Y = pts[o + 3u];
    let Z = pts[o + 4u];
    let z0 = r0 * X + r1 * Y + r2 * Z + t0;
    let z1 = r3 * X + r4 * Y + r5 * Z + t1;
    let z2 = r6 * X + r7 * Y + r8 * Z + t2;
    var e2 = 1.0;
    if (z2 > 0.0) {
      let inv = 1.0 / z2;
      let d0 = z0 * inv - pts[o];
      let d1 = z1 * inv - pts[o + 1u];
      let v = (d0 * d0 + d1 * d1) * inv_sq_thr;
      if (v < 1.0) {
        e2 = v;
        inl += 1u;
      }
    }
    score_sum += e2;
    if ((i & 15u) == 15u) {
      if (score_sum > bound) {
        break;
      }
      if (i + 1u == pre_k && inl < pre_min) {
        return JPO_BIG;
      }
    }
  }
  return score_sum;
}

// ---------------------------------------------------------------------------------------------------
// One hypothesis: the sample of (seed, problem, iter), solved, each solution scored. Returns the best
// score and writes the index of that solution (-1: none). A model is always a pose: 12 values.

fn jpo_rel_sample(off: u32, n: u32, seed: u32, problem: u32, iter: u32, b1: ptr<function, array<f32, 15>>,
                  b2: ptr<function, array<f32, 15>>) {
  var idx: array<u32, 5>;
  jpo_sample(jpo_sample_base(seed, problem, iter), n, 5, &idx);
  for (var k = 0; k < 5; k++) {
    let o = off + JPO_REL_STRIDE * idx[k];
    let x1 = pts[o];
    let y1 = pts[o + 1u];
    let x2 = pts[o + 2u];
    let y2 = pts[o + 3u];
    let i1 = 1.0 / sqrt(x1 * x1 + y1 * y1 + 1.0);
    let i2 = 1.0 / sqrt(x2 * x2 + y2 * y2 + 1.0);
    (*b1)[3 * k] = x1 * i1;
    (*b1)[3 * k + 1] = y1 * i1;
    (*b1)[3 * k + 2] = i1;
    (*b2)[3 * k] = x2 * i2;
    (*b2)[3 * k + 1] = y2 * i2;
    (*b2)[3 * k + 2] = i2;
  }
}

// The motions of sample solution e that put all five sample points in front of both cameras, scored;
// the solution index is 4 e + k.
fn jpo_rel_hypothesis(off: u32, n: u32, seed: u32, problem: u32, iter: u32, inv_sq_thr: f32, bound: f32,
                      pre_k: u32, pre_min: u32, best_sol: ptr<function, i32>) -> f32 {
  let start = jpo_hash(jpo_sample_base(seed, problem, iter) ^ 0x5bd1e995u) % n;
  var b1: array<f32, 15>;
  var b2: array<f32, 15>;
  var E: array<f32, 90>;  // 9 * JPO_MAX_E
  jpo_rel_sample(off, n, seed, problem, iter, &b1, &b2);
  let ns = jpo_relpose_5pt(&b1, &b2, &E);
  var best = JPO_BIG;
  *best_sol = -1;
  for (var s = 0; s < ns; s++) {
    var Es: array<f32, 9>;
    var P4: array<f32, 48>;
    for (var i = 0; i < 9; i++) {
      Es[i] = E[9 * s + i];
    }
    jpo_essential_poses(&Es, &P4);
    for (var k = 0; k < 4; k++) {
      var Ps: array<f32, 12>;
      for (var i = 0; i < 12; i++) {
        Ps[i] = P4[12 * k + i];
      }
      var front = 1;
      for (var j = 0; j < 5; j++) {
        if (jpo_cheirality(&Ps, b1[3 * j], b1[3 * j + 1], b1[3 * j + 2], b2[3 * j], b2[3 * j + 1], b2[3 * j + 2]) == 0) {
          front = 0;
        }
      }
      if (front == 0) {
        continue;
      }
      var bnd = bound;
      if (best < bound) {
        bnd = best;
      }
      let sc = jpo_score_rel(&Ps, off, n, inv_sq_thr, bnd, start, pre_k, pre_min);
      if (sc < best) {
        best = sc;
        *best_sol = 4 * s + k;
      }
    }
  }
  return best;
}

// The model of a hypothesis, solved again: 12 values. Returns 0 if the solution does not exist.
fn jpo_rel_model(off: u32, n: u32, seed: u32, problem: u32, iter: u32, sol_index: i32,
                 model: ptr<function, array<f32, 12>>) -> i32 {
  var b1: array<f32, 15>;
  var b2: array<f32, 15>;
  var E: array<f32, 90>;
  jpo_rel_sample(off, n, seed, problem, iter, &b1, &b2);
  let ns = jpo_relpose_5pt(&b1, &b2, &E);
  if (sol_index < 0 || sol_index >= 4 * ns) {
    return 0;
  }
  var Es: array<f32, 9>;
  var P4: array<f32, 48>;
  let s = sol_index / 4;
  let k = sol_index - 4 * s;
  for (var i = 0; i < 9; i++) {
    Es[i] = E[9 * s + i];
  }
  jpo_essential_poses(&Es, &P4);
  for (var i = 0; i < 12; i++) {
    (*model)[i] = P4[12 * k + i];
  }
  return 1;
}

fn jpo_abs_sample(off: u32, n: u32, seed: u32, problem: u32, iter: u32, xs: ptr<function, array<f32, 9>>,
                  Xs: ptr<function, array<f32, 9>>) {
  var idx: array<u32, 5>;
  jpo_sample(jpo_sample_base(seed, problem, iter), n, 3, &idx);
  for (var k = 0; k < 3; k++) {
    let o = off + JPO_ABS_STRIDE * idx[k];
    let x = pts[o];
    let y = pts[o + 1u];
    let inv = 1.0 / sqrt(x * x + y * y + 1.0);
    (*xs)[3 * k] = x * inv;
    (*xs)[3 * k + 1] = y * inv;
    (*xs)[3 * k + 2] = inv;
    (*Xs)[3 * k] = pts[o + 2u];
    (*Xs)[3 * k + 1] = pts[o + 3u];
    (*Xs)[3 * k + 2] = pts[o + 4u];
  }
}

fn jpo_abs_hypothesis(off: u32, n: u32, seed: u32, problem: u32, iter: u32, inv_sq_thr: f32, bound: f32,
                      pre_k: u32, pre_min: u32, best_sol: ptr<function, i32>) -> f32 {
  let start = jpo_hash(jpo_sample_base(seed, problem, iter) ^ 0x5bd1e995u) % n;
  var xs: array<f32, 9>;
  var Xs: array<f32, 9>;
  var P: array<f32, 48>;  // 12 * JPO_MAX_P
  jpo_abs_sample(off, n, seed, problem, iter, &xs, &Xs);
  let ns = jpo_p3p(&xs, &Xs, &P);
  var best = JPO_BIG;
  *best_sol = -1;
  for (var s = 0; s < ns; s++) {
    var Ps: array<f32, 12>;
    for (var i = 0; i < 12; i++) {
      Ps[i] = P[12 * s + i];
    }
    var bnd = bound;
    if (best < bound) {
      bnd = best;
    }
    let sc = jpo_score_abs(&Ps, off, n, inv_sq_thr, bnd, start, pre_k, pre_min);
    if (sc < best) {
      best = sc;
      *best_sol = s;
    }
  }
  return best;
}

fn jpo_abs_model(off: u32, n: u32, seed: u32, problem: u32, iter: u32, sol_index: i32,
                 model: ptr<function, array<f32, 12>>) -> i32 {
  var xs: array<f32, 9>;
  var Xs: array<f32, 9>;
  var P: array<f32, 48>;
  jpo_abs_sample(off, n, seed, problem, iter, &xs, &Xs);
  let ns = jpo_p3p(&xs, &Xs, &P);
  if (sol_index < 0 || sol_index >= ns) {
    return 0;
  }
  for (var i = 0; i < 12; i++) {
    (*model)[i] = P[12 * sol_index + i];
  }
  return 1;
}

// ---------------------------------------------------------------------------------------------------
// Entry points, as src/backend/metal/entry.metal. A round is two dispatches: round_* runs one thread per
// hypothesis (x) and item (y) and writes its score and solution index; select_* runs one thread per
// item, finds the best hypothesis and solves it once more for the model. Grids are whole workgroups, so
// every entry point first drops the threads past the end.

@compute @workgroup_size(64, 1, 1)
fn round_rel(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.y >= prm.n_items) {
    return;
  }
  let it = items[gid.y];
  if (gid.x >= it.count) {
    return;
  }
  var s = -1;
  let sc = jpo_rel_hypothesis(it.first * JPO_REL_STRIDE - prm.pts_base, it.n, prm.seed, it.problem, it.iter0 + gid.x,
                              it.inv_sq_thr, it.bound, it.pre_k, it.pre_min, &s);
  let o = gid.y * prm.round + gid.x;
  score[o] = sc;
  sol[o] = s;
}

@compute @workgroup_size(64, 1, 1)
fn round_abs(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.y >= prm.n_items) {
    return;
  }
  let it = items[gid.y];
  if (gid.x >= it.count) {
    return;
  }
  var s = -1;
  let sc = jpo_abs_hypothesis(it.first * JPO_ABS_STRIDE - prm.pts_base, it.n, prm.seed, it.problem, it.iter0 + gid.x,
                              it.inv_sq_thr, it.bound, it.pre_k, it.pre_min, &s);
  let o = gid.y * prm.round + gid.x;
  score[o] = sc;
  sol[o] = s;
}

fn jpo_best(o: u32, count: u32, best: ptr<function, f32>, best_h: ptr<function, u32>, best_sol: ptr<function, i32>) {
  *best = JPO_BIG;
  *best_h = 0u;
  *best_sol = -1;
  for (var h = 0u; h < count; h++) {  // in iteration order: a tie keeps the earlier hypothesis
    let s = sol[o + h];
    let sc = score[o + h];
    if (s >= 0 && sc < *best) {
      *best = sc;
      *best_h = h;
      *best_sol = s;
    }
  }
}

fn jpo_write_result(i: u32, best: f32, iter: u32, s: i32, m: ptr<function, array<f32, 12>>) {
  if (s >= 0) {
    results[i].score = best;
  } else {
    results[i].score = JPO_BIG;
  }
  results[i].iter = iter;
  results[i].sol = s;
  results[i].pad = 0u;
  results[i].model = *m;
}

@compute @workgroup_size(64, 1, 1)
fn select_rel(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= prm.n_items) {
    return;
  }
  let it = items[gid.x];
  var best: f32;
  var h: u32;
  var s: i32;
  jpo_best(gid.x * prm.round, it.count, &best, &h, &s);
  var m: array<f32, 12>;
  if (s >= 0) {
    if (jpo_rel_model(it.first * JPO_REL_STRIDE - prm.pts_base, it.n, prm.seed, it.problem, it.iter0 + h, s, &m) == 0) {
      s = -1;
    }
  }
  jpo_write_result(gid.x, best, it.iter0 + h, s, &m);
}

@compute @workgroup_size(64, 1, 1)
fn select_abs(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= prm.n_items) {
    return;
  }
  let it = items[gid.x];
  var best: f32;
  var h: u32;
  var s: i32;
  jpo_best(gid.x * prm.round, it.count, &best, &h, &s);
  var m: array<f32, 12>;
  if (s >= 0) {
    if (jpo_abs_model(it.first * JPO_ABS_STRIDE - prm.pts_base, it.n, prm.seed, it.problem, it.iter0 + h, s, &m) == 0) {
      s = -1;
    }
  }
  jpo_write_result(gid.x, best, it.iter0 + h, s, &m);
}
