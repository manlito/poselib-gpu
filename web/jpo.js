// JavaScript wrapper of the WebAssembly module (jpo_wasm.js, built by CMake under Emscripten).
//
//   import { createJpoModule, Jpo, BACKEND } from "./jpo.js";
//   const module = await createJpoModule();
//   const jpo = await Jpo.create(module, { backend: BACKEND.AUTO });
//   const out = await jpo.relativePose({ offsets, xyA, xyB, cameras, cameraA, cameraB }, { maxErrorPx: 1 });
//   // out.results[i] = { pose (Float64Array 7), score, medianTriAngleDeg, nInliers, iterations, status }
//   // out.inliers: Uint8Array, one byte per match
//
// cameras: array of { model, params } with COLMAP's ids and parameter order. Poses: qw qx qy qz tx ty tz.
import createJpo from "./jpo_wasm.js";

export const BACKEND = { AUTO: 0, CPU: 1, WEBGPU: 4 };
export const STATUS = { OK: 0, NO_MODEL: 1, TOO_FEW_POINTS: 2, DEGENERATE: 3 };

export function createJpoModule(options) {
  return createJpo(options);
}

export class Jpo {
  static async create(module, { backend = BACKEND.AUTO, seed = 0, device = null } = {}) {
    let handle = 0;
    if (device) {
      if (!module.WebGPU) throw new Error("this build has no WebGPU backend");
      handle = module.WebGPU.importJsDevice(device);
    }
    const ctx = await module._jpow_create(backend, seed, 0, handle);
    if (!ctx) throw new Error(module.UTF8ToString(module._jpow_error()));
    return new Jpo(module, ctx);
  }

  constructor(module, ctx) {
    this.m = module;
    this.ctx = ctx;
  }

  destroy() {
    if (this.ctx) this.m._jpow_destroy(this.ctx);
    this.ctx = 0;
  }

  relativePose(problem, options) {
    return this.#run(0, problem, options);
  }

  // problem: { offsets, xy, xyz, cameras, camera }
  absolutePose(problem, options) {
    return this.#run(1, { offsets: problem.offsets, xyA: problem.xy, xyB: problem.xyz, cameras: problem.cameras, cameraA: problem.camera }, options);
  }

  async #run(kind, p, { maxErrorPx = kind === 0 ? 1 : 4, confidence = 0.9999, minIterations = 100, maxIterations = 10000, refine = true } = {}) {
    const m = this.m;
    const count = p.offsets.length - 1;
    const total = p.offsets[count];
    const cams = new Float64Array(9 * p.cameras.length);
    p.cameras.forEach((c, i) => {
      cams[9 * i] = c.model;
      cams.set(c.params, 9 * i + 1);
    });
    const allocs = [];
    const put = (array, Type) => {
      const typed = array instanceof Type ? array : Type.from(array);
      const ptr = m._malloc(Math.max(8, typed.byteLength));
      allocs.push(ptr);
      new Type(m.HEAPU8.buffer, ptr, typed.length).set(typed);
      return ptr;
    };
    try {
      const offsets = put(p.offsets, Uint32Array);
      const a = put(p.xyA, Float64Array);
      const b = put(p.xyB, Float64Array);
      const cameras = put(cams, Float64Array);
      const cameraA = put(p.cameraA, Uint32Array);
      const opts = put([maxErrorPx, confidence, minIterations, maxIterations, refine ? 1 : 0], Float64Array);
      const results = m._malloc(Math.max(8, 96 * count));
      const inliers = m._malloc(Math.max(8, total));
      allocs.push(results, inliers);
      let rc;
      if (kind === 0) {
        const cameraB = put(p.cameraB, Uint32Array);
        rc = await m._jpow_relative(this.ctx, count, offsets, a, b, p.cameras.length, cameras, cameraA, cameraB, opts, results, inliers);
      } else {
        rc = await m._jpow_absolute(this.ctx, count, offsets, a, b, p.cameras.length, cameras, cameraA, opts, results, inliers);
      }
      if (rc !== 0) throw new Error(m.UTF8ToString(m._jpow_error()));
      const r = new Float64Array(m.HEAPU8.buffer, results, 12 * count).slice();
      const out = [];
      for (let i = 0; i < count; ++i) {
        out.push({
          pose: r.slice(12 * i, 12 * i + 7), score: r[12 * i + 7], medianTriAngleDeg: r[12 * i + 8],
          nInliers: r[12 * i + 9], iterations: r[12 * i + 10], status: r[12 * i + 11],
        });
      }
      return { results: out, inliers: new Uint8Array(m.HEAPU8.buffer, inliers, total).slice() };
    } finally {
      allocs.forEach((ptr) => m._free(ptr));
    }
  }
}

// The synthetic problems of the native tools (`jpo gen` with the same arguments), for tests and demos.
// Returns a problem for relativePose (kind 0) or absolutePose (kind 1), with gt: Float64Array (7 per problem).
export function synth(module, { kind = 0, count = 1, points = 1000, inliers = 0.5, noise = 0.5, model = 1, seed = 1 } = {}) {
  const m = module;
  const total = m._jpow_synth(kind, count, points, inliers, noise, model, seed, 0, 0, 0, 0, 0, 0, 0);
  const nCams = kind === 0 ? 2 * count : count;
  const sizes = [4 * (count + 1), 16 * total, (kind === 0 ? 16 : 24) * total, 72 * nCams, 4 * count, 4 * count, 56 * count];
  const ptrs = sizes.map((s) => m._malloc(Math.max(8, s)));
  m._jpow_synth(kind, count, points, inliers, noise, model, seed, ...ptrs);
  const f64 = (ptr, n) => new Float64Array(m.HEAPU8.buffer, ptr, n).slice();
  const u32 = (ptr, n) => new Uint32Array(m.HEAPU8.buffer, ptr, n).slice();
  const cams = f64(ptrs[3], 9 * nCams);
  const cameras = [];
  for (let i = 0; i < nCams; ++i) cameras.push({ model: cams[9 * i], params: cams.slice(9 * i + 1, 9 * i + 9) });
  const out = { offsets: u32(ptrs[0], count + 1), cameras, gt: f64(ptrs[6], 7 * count) };
  if (kind === 0) {
    Object.assign(out, { xyA: f64(ptrs[1], 2 * total), xyB: f64(ptrs[2], 2 * total), cameraA: u32(ptrs[4], count), cameraB: u32(ptrs[5], count) });
  } else {
    Object.assign(out, { xy: f64(ptrs[1], 2 * total), xyz: f64(ptrs[2], 3 * total), camera: u32(ptrs[4], count) });
  }
  ptrs.forEach((ptr) => m._free(ptr));
  return out;
}

// Rotation error in degrees and translation error (angle between directions for a relative pose,
// distance between camera centres for an absolute pose), as tools/jpofile.py.
export function poseErrors(kind, est, gt) {
  const rot = (q) => {
    const n = Math.hypot(q[0], q[1], q[2], q[3]);
    const [w, x, y, z] = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
    return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
      2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
  };
  const Re = rot(est), Rg = rot(gt);
  let trace = 0;
  for (let i = 0; i < 9; ++i) trace += Re[i] * Rg[i];
  const deg = (c) => (Math.acos(Math.max(-1, Math.min(1, c))) * 180) / Math.PI;
  const rotErr = deg(0.5 * (trace - 1));
  const te = est.slice(4, 7), tg = gt.slice(4, 7);
  if (kind === 0) {
    const c = (te[0] * tg[0] + te[1] * tg[1] + te[2] * tg[2]) / (Math.hypot(...te) * Math.hypot(...tg));
    return [rotErr, deg(c)];
  }
  const centre = (R, t) => [0, 1, 2].map((c) => -(R[c] * t[0] + R[3 + c] * t[1] + R[6 + c] * t[2]));
  const ce = centre(Re, te), cg = centre(Rg, tg);
  return [rotErr, Math.hypot(ce[0] - cg[0], ce[1] - cg[1], ce[2] - cg[2])];
}
