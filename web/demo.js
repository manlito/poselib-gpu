// The live demo (demo.html). Loads the module and the sample data, builds a batch of problems from the real
// matches (wrong matches and noise added per copy, from a fixed seed), runs it on each chosen backend, and draws
// the first problem of the batch with the result of the backend picked in the table.
const config = (() => {
  const el = document.getElementById("demo-config");
  return { lib: "./", samples: "samples/", ...(el ? JSON.parse(el.textContent) : {}) };
})();

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const statusEl = $("status");
const setStatus = (text, bad = false) => {
  statusEl.textContent = text;
  statusEl.classList.toggle("bad", bad);
};
const notice = (html, bad = false) => {
  const el = $("notice");
  el.innerHTML = html;
  el.classList.toggle("bad", bad);
  el.hidden = false;
};

// ---------------------------------------------------------------- small maths

function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rand) {
  const u = Math.max(rand(), 1e-12), v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function quatToRot(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  const [w, x, y, z] = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
  return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
}
const deg = (r) => (r * 180) / Math.PI;
const clampAcos = (c) => Math.acos(Math.max(-1, Math.min(1, c)));
// Rotation angle between two quaternions, degrees.
function rotDiff(a, b) {
  const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]) / (Math.hypot(...a.slice(0, 4)) * Math.hypot(...b.slice(0, 4)));
  return deg(2 * clampAcos(d));
}
// Centre of the camera of a pose (world to camera): -R^T t.
function centre(pose) {
  const R = quatToRot(pose), t = pose.slice(4, 7);
  return [0, 1, 2].map((c) => -(R[c] * t[0] + R[3 + c] * t[1] + R[6 + c] * t[2]));
}
function angleBetween(u, v) {
  return deg(clampAcos((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (Math.hypot(...u) * Math.hypot(...v))));
}
// Fundamental matrix of a relative pose (camera a to camera b) for one pinhole camera: K^-T [t]x R K^-1.
function fundamental(pose, cam) {
  const [fx, fy, cx, cy] = cam.params;
  const R = quatToRot(pose), [tx, ty, tz] = pose.slice(4, 7);
  const tX = [0, -tz, ty, tz, 0, -tx, -ty, tx, 0];
  const E = new Array(9).fill(0);
  for (let r = 0; r < 3; ++r) for (let c = 0; c < 3; ++c) for (let k = 0; k < 3; ++k) E[3 * r + c] += tX[3 * r + k] * R[3 * k + c];
  const Ki = [1 / fx, 0, -cx / fx, 0, 1 / fy, -cy / fy, 0, 0, 1];
  const KiT = [Ki[0], Ki[3], Ki[6], Ki[1], Ki[4], Ki[7], Ki[2], Ki[5], Ki[8]];
  const mul = (A, B) => {
    const C = new Array(9).fill(0);
    for (let r = 0; r < 3; ++r) for (let c = 0; c < 3; ++c) for (let k = 0; k < 3; ++k) C[3 * r + c] += A[3 * r + k] * B[3 * k + c];
    return C;
  };
  return mul(mul(KiT, E), Ki);
}
const lineOf = (F, x, y, transpose) => transpose
  ? [F[0] * x + F[3] * y + F[6], F[1] * x + F[4] * y + F[7], F[2] * x + F[5] * y + F[8]]
  : [F[0] * x + F[1] * y + F[2], F[3] * x + F[4] * y + F[5], F[6] * x + F[7] * y + F[8]];

// ---------------------------------------------------------------- state

const state = {
  tab: "rel", pairs: null, cloud: null, images: {}, jpo: {}, module: null, lib: null,
  results: null, shown: null, reference: {}, click: null, running: false,
};
const NAMES = { cpu: "CPU", webgpu: "WebGPU", auto: "auto" };

function settings() {
  return {
    wrong: Number($("wrong").value) / 100,
    noise: Number($("noise").value),
    batch: Number(document.querySelector("input[name=batch]:checked").value),
    backends: [...document.querySelectorAll("input[name=backend]:checked")].map((e) => e.value).filter((b) => state.jpo[b]),
  };
}
const currentPair = () => state.pairs.pairs.find((p) => p.id === $("pair").value);

// A batch of copies of the problem: real matches with noise, then random wrong matches, from seed 1000 + copy.
function buildProblem(kind, { wrong, noise, batch }) {
  const W = state.pairs.size[0], H = state.pairs.size[1];
  const src = kind === 0 ? currentPair() : state.cloud;
  const real = kind === 0 ? src.xyA.length / 2 : src.xy.length / 2;
  const added = wrong > 0 ? Math.round((real * wrong) / (1 - wrong)) : 0;
  const n = real + added;
  const offsets = new Uint32Array(batch + 1);
  const a = new Float64Array(2 * n * batch);
  const b = new Float64Array((kind === 0 ? 2 : 3) * n * batch);
  for (let c = 0; c < batch; ++c) {
    offsets[c + 1] = (c + 1) * n;
    const rand = rng(1000 + c);
    const oa = 2 * n * c, ob = (kind === 0 ? 2 : 3) * n * c;
    for (let i = 0; i < real; ++i) {
      if (kind === 0) {
        a[oa + 2 * i] = src.xyA[2 * i];
        a[oa + 2 * i + 1] = src.xyA[2 * i + 1];
        b[ob + 2 * i] = src.xyB[2 * i] + noise * gauss(rand);
        b[ob + 2 * i + 1] = src.xyB[2 * i + 1] + noise * gauss(rand);
      } else {
        a[oa + 2 * i] = src.xy[2 * i] + noise * gauss(rand);
        a[oa + 2 * i + 1] = src.xy[2 * i + 1] + noise * gauss(rand);
        b.set(src.xyz.slice(3 * i, 3 * i + 3), ob + 3 * i);
      }
    }
    for (let i = real; i < n; ++i) {
      a[oa + 2 * i] = rand() * W;
      a[oa + 2 * i + 1] = rand() * H;
      if (kind === 0) {
        b[ob + 2 * i] = rand() * W;
        b[ob + 2 * i + 1] = rand() * H;
      } else {
        const j = Math.floor(rand() * real);
        b.set(src.xyz.slice(3 * j, 3 * j + 3), ob + 3 * i);
      }
    }
  }
  const zeros = new Uint32Array(batch);
  const cameras = [state.pairs.camera];
  const problem = kind === 0
    ? { offsets, xyA: a, xyB: b, cameras, cameraA: zeros, cameraB: zeros }
    : { offsets, xy: a, xyz: b, cameras, camera: zeros };
  return { problem, n, real, batch, kind, pairId: kind === 0 ? src.id : null };
}

async function timed(fn) {
  let t0 = performance.now();
  let out = await fn();
  let best = performance.now() - t0;
  if (best < 300) {
    for (let r = 0; r < 3; ++r) {
      t0 = performance.now();
      out = await fn();
      best = Math.min(best, performance.now() - t0);
    }
  }
  return [best, out];
}

// The pose found on the matches as they are (CPU, no wrong matches, no noise): what the other runs are compared to.
async function reference(kind) {
  const key = kind === 0 ? currentPair().id : "cloud";
  if (!state.reference[key]) {
    const { problem } = buildProblem(kind, { wrong: 0, noise: 0, batch: 1 });
    const jpo = state.jpo.cpu;
    const out = kind === 0 ? await jpo.relativePose(problem) : await jpo.absolutePose(problem);
    state.reference[key] = out.results[0];
  }
  return state.reference[key];
}

function sameAsReference(kind, r, ref) {
  if (r.status !== 0 || ref.status !== 0) return false;
  if (kind === 0) return rotDiff(r.pose, ref.pose) < 1 && angleBetween(r.pose.slice(4, 7), ref.pose.slice(4, 7)) < 5;
  const c = centre(r.pose), cr = centre(ref.pose);
  return rotDiff(r.pose, ref.pose) < 1 && Math.hypot(c[0] - cr[0], c[1] - cr[1], c[2] - cr[2]) < 0.05;
}

async function run() {
  if (state.running) return;
  const s = settings();
  if (!s.backends.length) {
    setStatus("Pick at least one backend.", true);
    return;
  }
  state.running = true;
  $("run").disabled = true;
  const kind = state.tab === "rel" ? 0 : 1;
  try {
    const ref = await reference(kind);
    const built = buildProblem(kind, s);
    const rows = [];
    for (const name of s.backends) {
      setStatus(`Running on ${NAMES[name]}…`);
      await new Promise((r) => setTimeout(r, 0));
      const jpo = state.jpo[name];
      const [ms, out] = await timed(() => (kind === 0 ? jpo.relativePose(built.problem) : jpo.absolutePose(built.problem)));
      const stats = jpo.lastStats();
      const same = out.results.filter((r) => sameAsReference(kind, r, ref)).length;
      rows.push({ name, ms, out, stats, same });
    }
    state.results = { ...built, rows, ref, settings: s };
    state.shown = rows[0].name;
    const n = built.n, total = n * built.batch;
    setStatus(`${built.batch} ${built.batch === 1 ? "problem" : "problems"} of ${n} matches (${built.real} real, ${n - built.real} added): ${total.toLocaleString()} in all.`);
    renderTable();
    draw();
  } catch (e) {
    setStatus(`Failed: ${e.message}`, true);
    console.error(e);
  } finally {
    state.running = false;
    $("run").disabled = false;
  }
}

// ---------------------------------------------------------------- table

const STATUS_TEXT = ["ok", "no model", "too few points", "degenerate"];

function renderTable() {
  const R = state.results;
  const thead = $("results").tHead, tbody = $("results").tBodies[0];
  const cols = ["Backend", "Time (ms)", "Status", "Inliers", "Iterations", "On the GPU",
    R.kind === 0 ? "Pose (first problem)" : "Camera (first problem)"];
  if (R.batch > 1) cols.splice(6, 0, "Same pose");
  thead.innerHTML = "<tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>";
  tbody.innerHTML = "";
  for (const row of R.rows) {
    const r = row.out.results[0];
    const tr = tbody.insertRow();
    tr.classList.toggle("shown", row.name === state.shown);
    tr.title = "Draw this backend's result";
    tr.addEventListener("click", () => {
      state.shown = row.name;
      renderTable();
      draw();
    });
    const gpuShare = row.stats.cpuHypotheses + row.stats.gpuHypotheses > 0
      ? row.stats.gpuHypotheses / (row.stats.cpuHypotheses + row.stats.gpuHypotheses) : 0;
    const cells = [NAMES[row.name], row.ms.toFixed(row.ms < 10 ? 2 : 1), STATUS_TEXT[r.status] ?? r.status,
      `${r.nInliers} of ${R.n}`, r.iterations.toLocaleString(), `${Math.round(100 * gpuShare)} %`];
    if (R.batch > 1) cells.push(`${row.same} of ${R.batch}`);
    cells.push(describePose(R.kind, r, R.ref));
    cells.forEach((v, i) => {
      const td = tr.insertCell();
      td.textContent = v;
      if (i === cells.length - 1) td.className = "l";
    });
  }
  const best = R.rows.reduce((a, b) => (a.ms <= b.ms ? a : b));
  const ref = R.ref;
  $("results-note").textContent = (R.rows.length > 1 ? `Fastest here: ${NAMES[best.name]}. ` : "")
    + `Times are the best of four runs (one run if it takes over 300 ms), after loading. "On the GPU" is the share of hypotheses drawn there. `
    + (R.batch > 1 ? `"Same pose" counts the problems whose pose is within 1° (and ${R.kind === 0 ? "5° in direction" : "5 cm"}) of the one found on the matches as they are. ` : "")
    + (ref.status === 0 ? "" : "On the matches as they are, the solver finds no pose. ");
}

function describePose(kind, r, ref) {
  if (r.status !== 0) return "—";
  if (kind === 0) {
    const c = centre(r.pose);
    const n = Math.hypot(...c);
    const rot = deg(2 * clampAcos(Math.abs(r.pose[0]) / Math.hypot(...r.pose.slice(0, 4))));
    const d = ref.status === 0 ? `; ${rotDiff(r.pose, ref.pose).toFixed(2)}° and ${angleBetween(r.pose.slice(4, 7), ref.pose.slice(4, 7)).toFixed(1)}° from the clean run` : "";
    return `turn ${rot.toFixed(2)}°, moved toward (${c.map((v) => (v / n).toFixed(2)).join(", ")})${d}`;
  }
  const c = centre(r.pose);
  const [, c106] = state.cloud.cameras.map((x) => x.centre);
  const along = (c[0] * c106[0] + c[1] * c106[1] + c[2] * c106[2]) / (c106[0] ** 2 + c106[1] ** 2 + c106[2] ** 2);
  return `${along.toFixed(3)} of the way from frame 100 to 106`;
}

// ---------------------------------------------------------------- drawing

const GAP = 8;
const EPI_COLOURS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"];

function shownRow() {
  return state.results?.rows.find((r) => r.name === state.shown) ?? null;
}

function draw() {
  if (state.tab === "rel") drawRel();
  else drawAbs();
  drawLegend();
}

function drawLegend() {
  const items = [["var(--inlier)", "inlier"], ["var(--outlier)", "outlier"]];
  if (state.tab === "rel") items.push(["#4363d8", "epipolar line and its point (lines: up to 150 matches of each kind)"]);
  else items.push(["var(--accent)", "reprojected point"]);
  $("legend").innerHTML = items.map(([c, t]) => `<span><span class="sw" style="background:${c}"></span>${t}</span>`).join("")
    + (state.results ? `<span>result drawn: ${NAMES[state.shown]}, first problem of the batch</span>` : "");
}

function drawRel() {
  const pair = currentPair();
  const W = state.pairs.size[0], H = state.pairs.size[1];
  const cv = $("rel-canvas"), g = cv.getContext("2d");
  cv.width = W;
  cv.height = 2 * H + GAP;
  g.fillStyle = css("--panel");
  g.fillRect(0, 0, cv.width, cv.height);
  const ia = state.images[pair.a], ib = state.images[pair.b];
  if (ia) g.drawImage(ia, 0, 0, W, H);
  if (ib) g.drawImage(ib, 0, H + GAP, W, H);
  const R = state.results, row = shownRow();
  const valid = R && R.kind === 0 && R.pairId === pair.id;
  const xyA = valid ? R.problem.xyA : pair.xyA, xyB = valid ? R.problem.xyB : pair.xyB;
  const n = valid ? R.n : pair.xyA.length / 2;
  const mask = valid ? row.out.inliers : null;
  // Every match as a dot in both images; lines for at most LINES of each kind, spread over the matches, so the
  // images stay visible.
  const LINES = 150;
  for (const want of [0, 1]) {
    if (want === 1 && !$("show-in").checked) continue;
    if (want === 0 && !$("show-out").checked) continue;
    const idx = [];
    for (let i = 0; i < n; ++i) if ((mask ? (mask[i] ? 1 : 0) : 1) === want) idx.push(i);
    const colour = css(want ? "--inlier" : "--outlier");
    g.fillStyle = colour;
    g.globalAlpha = want ? 0.95 : 0.6;
    for (const i of idx) {
      g.fillRect(xyA[2 * i] - 2, xyA[2 * i + 1] - 2, 4, 4);
      g.fillRect(xyB[2 * i] - 2, H + GAP + xyB[2 * i + 1] - 2, 4, 4);
    }
    g.strokeStyle = colour;
    g.lineWidth = 1;
    g.globalAlpha = want ? 0.55 : 0.3;
    g.beginPath();
    const step = Math.max(1, idx.length / LINES);
    for (let k = 0; k < idx.length; k += step) {
      const i = idx[Math.floor(k)];
      g.moveTo(xyA[2 * i], xyA[2 * i + 1]);
      g.lineTo(xyB[2 * i], H + GAP + xyB[2 * i + 1]);
    }
    g.stroke();
  }
  g.globalAlpha = 1;
  const r = valid ? row.out.results[0] : null;
  if (!r || r.status !== 0) return;
  const F = fundamental(r.pose, state.pairs.camera);
  const epi = (l, top, colour, dashed) => {
    // a x + b y + c = 0 across the image at vertical offset top.
    const [a, b, c] = l;
    g.save();
    g.beginPath();
    g.rect(0, top, W, H);
    g.clip();
    g.strokeStyle = colour;
    g.lineWidth = 2;
    g.setLineDash(dashed ? [8, 6] : []);
    g.beginPath();
    if (Math.abs(b) > 1e-12) {
      g.moveTo(0, top - c / b);
      g.lineTo(W, top - (a * W + c) / b);
    } else {
      g.moveTo(-c / a, top);
      g.lineTo(-c / a, top + H);
    }
    g.stroke();
    g.restore();
  };
  const dot = (x, y, colour) => {
    g.fillStyle = colour;
    g.strokeStyle = "rgba(0,0,0,0.6)";
    g.lineWidth = 1.5;
    g.beginPath();
    g.arc(x, y, 5, 0, 2 * Math.PI);
    g.fill();
    g.stroke();
  };
  if ($("show-epi").checked) {
    const inl = [];
    for (let i = 0; i < n; ++i) if (mask[i]) inl.push(i);
    inl.sort((i, j) => xyA[2 * i] - xyA[2 * j]);
    const pick = [];
    const k = Math.min(EPI_COLOURS.length, inl.length);
    for (let s = 0; s < k; ++s) pick.push(inl[Math.floor(((s + 0.5) * inl.length) / k)]);
    pick.forEach((i, s) => {
      const col = EPI_COLOURS[s];
      epi(lineOf(F, xyA[2 * i], xyA[2 * i + 1], false), H + GAP, col);
      epi(lineOf(F, xyB[2 * i], xyB[2 * i + 1], true), 0, col);
      dot(xyA[2 * i], xyA[2 * i + 1], col);
      dot(xyB[2 * i], H + GAP + xyB[2 * i + 1], col);
    });
  }
  if (state.click) {
    const { x, y, inA } = state.click;
    const accent = css("--accent");
    if (inA) {
      dot(x, y, accent);
      epi(lineOf(F, x, y, false), H + GAP, accent, true);
    } else {
      dot(x, H + GAP + y, accent);
      epi(lineOf(F, x, y, true), 0, accent, true);
    }
  }
}

function drawAbs() {
  const cloud = state.cloud;
  const W = state.pairs.size[0], H = state.pairs.size[1];
  const cv = $("abs-canvas"), g = cv.getContext("2d");
  g.fillStyle = css("--panel");
  g.fillRect(0, 0, W, H);
  const img = state.images[cloud.image];
  if (img) g.drawImage(img, 0, 0, W, H);
  const R = state.results, row = shownRow();
  const valid = R && R.kind === 1;
  const xy = valid ? R.problem.xy : cloud.xy, xyz = valid ? R.problem.xyz : cloud.xyz;
  const n = valid ? R.n : cloud.xy.length / 2;
  const mask = valid ? row.out.inliers : null;
  const r = valid ? row.out.results[0] : null;
  const [fx, fy, cx, cy] = state.pairs.camera.params;
  const P = r && r.status === 0 ? { R: quatToRot(r.pose), t: r.pose.slice(4, 7) } : null;
  for (let i = 0; i < n; ++i) {
    const isIn = mask ? mask[i] === 1 : true;
    if (isIn ? !$("show-in").checked : !$("show-out").checked) continue;
    g.fillStyle = css(isIn ? "--inlier" : "--outlier");
    g.globalAlpha = isIn ? 0.9 : 0.6;
    g.beginPath();
    g.arc(xy[2 * i], xy[2 * i + 1], 3, 0, 2 * Math.PI);
    g.fill();
    if (P && isIn && mask) {
      const X = xyz.slice(3 * i, 3 * i + 3);
      const Xc = [0, 1, 2].map((k) => P.R[3 * k] * X[0] + P.R[3 * k + 1] * X[1] + P.R[3 * k + 2] * X[2] + P.t[k]);
      if (Xc[2] > 0) {
        g.strokeStyle = css("--accent");
        g.globalAlpha = 1;
        g.lineWidth = 1.5;
        const u = fx * Xc[0] / Xc[2] + cx, v = fy * Xc[1] / Xc[2] + cy;
        g.beginPath();
        g.moveTo(u - 3, v - 3); g.lineTo(u + 3, v + 3);
        g.moveTo(u - 3, v + 3); g.lineTo(u + 3, v - 3);
        g.stroke();
      }
    }
  }
  g.globalAlpha = 1;
  drawTop();
}

// The cloud from above: x across, z (the direction of travel) up, fitted to the canvas.
function drawTop() {
  const cloud = state.cloud;
  const cv = $("top-canvas"), g = cv.getContext("2d");
  const S = cv.width;
  g.fillStyle = css("--bg");
  g.fillRect(0, 0, S, S);
  const pts = cloud.cloud;
  const xs = [], zs = [];
  for (let i = 0; i < pts.length; i += 3) {
    xs.push(pts[i]);
    zs.push(pts[i + 2]);
  }
  const q = (a, f) => [...a].sort((u, v) => u - v)[Math.floor(f * (a.length - 1))];
  const x0 = Math.min(q(xs, 0.03), -1), x1 = Math.max(q(xs, 0.97), 1);
  const z0 = Math.min(q(zs, 0.0), -0.5), z1 = q(zs, 0.95);
  const scale = (0.9 * S) / Math.max(x1 - x0, z1 - z0);
  const ox = S / 2 - scale * (x0 + x1) / 2, oz = S / 2 + scale * (z0 + z1) / 2;
  const to = (x, z) => [ox + scale * x, oz - scale * z];
  g.fillStyle = css("--cloud");
  g.globalAlpha = 0.7;
  for (let i = 0; i < pts.length; i += 3) {
    const [u, v] = to(pts[i], pts[i + 2]);
    g.fillRect(u - 1, v - 1, 2, 2);
  }
  g.globalAlpha = 1;
  const [c100, c106] = cloud.cameras.map((c) => c.centre);
  const found = [];
  if (state.results?.kind === 1) {
    for (const row of state.results.rows) {
      const r = row.out.results[0];
      if (r.status === 0) found.push({ c: centre(r.pose), shown: row.name === state.shown, name: row.name });
    }
  }
  // The cameras, in a view with its own scale: `map` takes x, z to the canvas.
  const cams = (map, labels) => {
    const cam = (c, colour, label, open) => {
      const [u, v] = map(c[0], c[2]);
      g.strokeStyle = colour;
      g.fillStyle = colour;
      g.lineWidth = 2;
      g.beginPath();
      g.arc(u, v, labels ? 6 : 3.5, 0, 2 * Math.PI);
      if (open) g.stroke();
      else g.fill();
      if (labels && label) {
        g.font = "17px system-ui, sans-serif";
        g.fillText(label, u + 11, v + 6);
      }
    };
    g.strokeStyle = css("--faint");
    g.lineWidth = 1;
    g.setLineDash([4, 4]);
    g.beginPath();
    g.moveTo(...map(c100[0], c100[2]));
    g.lineTo(...map(c106[0], c106[2]));
    g.stroke();
    g.setLineDash([]);
    cam(c100, css("--fg"), "frame 100");
    cam(c106, css("--fg"), "frame 106");
    for (const f of found.filter((x) => !x.shown)) cam(f.c, css("--muted"), "", true);
    for (const f of found.filter((x) => x.shown)) cam(f.c, css("--accent"), `frame 103 (${NAMES[f.name]})`, true);
  };
  cams(to, false);
  // Inset: the three cameras enlarged, the path from frame 100 to frame 106 across most of it.
  const I = Math.round(S * 0.42), pad = 10;
  g.fillStyle = css("--panel");
  g.strokeStyle = css("--line");
  g.lineWidth = 1;
  g.fillRect(pad, pad, I, I);
  g.strokeRect(pad + 0.5, pad + 0.5, I, I);
  const mx = (c100[0] + c106[0]) / 2, mz = (c100[2] + c106[2]) / 2;
  const k = (0.55 * I) / Math.max(Math.hypot(c106[0] - c100[0], c106[2] - c100[2]), 1e-9);
  g.save();
  g.beginPath();
  g.rect(pad, pad, I, I);
  g.clip();
  cams((x, z) => [pad + I * 0.3 + k * (x - mx), pad + I / 2 - k * (z - mz)], true);
  g.restore();
  g.fillStyle = css("--muted");
  g.font = "15px system-ui, sans-serif";
  g.fillText("the cameras, enlarged", pad + 8, pad + I - 8);
}

// ---------------------------------------------------------------- page

function setTab(tab) {
  state.tab = tab;
  const rel = tab === "rel";
  $("rel-view").hidden = !rel;
  $("abs-view").hidden = rel;
  $("pair-field").hidden = !rel;
  $("cloud-field").hidden = rel;
  $("show-epi-row").hidden = !rel;
  $("show-hint").textContent = rel ? "Click a point in either image to draw its epipolar line in the other."
    : "Crosses: the inliers' 3D points projected with the pose found.";
  $("results-title").textContent = rel ? "Relative pose" : "Camera registration (PnP)";
  clearResults();
}

function clearResults() {
  state.results = null;
  state.click = null;
  $("results").tHead.innerHTML = "";
  $("results").tBodies[0].innerHTML = '<tr><td class="l" colspan="7">Press Run.</td></tr>';
  $("results-note").textContent = "";
  if (state.pairs) draw();
}

async function loadImage(name) {
  if (state.images[name]) return;
  // The load event, not img.decode(): Chrome can hold decode() back while the page is not drawn.
  const img = new Image();
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error(`cannot load ${name}`));
    img.src = config.samples + name;
  });
  state.images[name] = img;
}

function wire() {
  for (const el of document.querySelectorAll("input[name=tab]")) el.addEventListener("change", () => setTab(el.value));
  const out = () => {
    $("wrong-out").textContent = `${$("wrong").value} %`;
    $("noise-out").textContent = `${Number($("noise").value).toFixed(2)} px`;
  };
  out();
  for (const id of ["wrong", "noise"]) $(id).addEventListener("input", out);
  for (const id of ["wrong", "noise"]) $(id).addEventListener("change", () => state.results && run());
  for (const el of document.querySelectorAll("input[name=batch]")) el.addEventListener("change", () => state.results && run());
  for (const id of ["show-in", "show-out", "show-epi"]) $(id).addEventListener("change", draw);
  $("pair").addEventListener("change", async () => {
    const p = currentPair();
    $("pair-note").textContent = p.note;
    await Promise.all([loadImage(p.a), loadImage(p.b)]);
    clearResults();
    run();
  });
  $("run").addEventListener("click", run);
  $("rel-canvas").addEventListener("click", (ev) => {
    const cv = ev.currentTarget, rect = cv.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) * cv.width) / rect.width, y = ((ev.clientY - rect.top) * cv.height) / rect.height;
    const H = state.pairs.size[1];
    state.click = y < H ? { x, y, inA: true } : y > H + GAP ? { x, y: y - H - GAP, inA: false } : null;
    draw();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
}

async function main() {
  wire();
  if (typeof WebAssembly.Suspending !== "function") {
    notice("This page needs a browser with JavaScript promise integration for WebAssembly: Chrome or Edge 137 or later. "
      + "The library itself runs anywhere; only this browser build waits for the GPU that way.", true);
    setStatus("Not supported in this browser.", true);
    return;
  }
  try {
    const [pairs, cloud, lib] = await Promise.all([
      fetch(config.samples + "pairs.json").then((r) => r.json()),
      fetch(config.samples + "cloud.json").then((r) => r.json()),
      import(config.lib + "jpo.js"),
    ]);
    state.pairs = pairs;
    state.cloud = cloud;
    state.lib = lib;
    for (const p of pairs.pairs) $("pair").add(new Option(`${p.title} (${p.xyA.length / 2} matches)`, p.id));
    $("pair-note").textContent = pairs.pairs[0].note;
    $("cloud-note").textContent = `${cloud.note} ${cloud.xy.length / 2} of its matches have a point; the cloud has ${cloud.cloud.length / 3}.`;
    await Promise.all([loadImage(pairs.pairs[0].a), loadImage(pairs.pairs[0].b), loadImage(cloud.image)]);
    setStatus("Loading the module…");
    state.module = await lib.createJpoModule({ locateFile: (f) => config.lib + f });
    state.jpo.cpu = await lib.Jpo.create(state.module, { backend: lib.BACKEND.CPU });
    let gpuMsg = "";
    if (!navigator.gpu) gpuMsg = "This browser has no WebGPU, so only the CPU backend runs.";
    else if (!state.module._jpow_backend_available(lib.BACKEND.WEBGPU)) gpuMsg = "This build has no WebGPU backend.";
    else {
      try {
        state.jpo.webgpu = await lib.Jpo.create(state.module, { backend: lib.BACKEND.WEBGPU });
        state.jpo.auto = await lib.Jpo.create(state.module, { backend: lib.BACKEND.AUTO });
      } catch (e) {
        gpuMsg = `WebGPU is not available here (${e.message}), so only the CPU backend runs.`;
      }
    }
    if (gpuMsg) {
      notice(gpuMsg);
      for (const el of document.querySelectorAll("input[name=backend]")) {
        if (el.value !== "cpu") {
          el.checked = false;
          el.disabled = true;
        }
      }
    }
    $("run").disabled = false;
    draw();
    await run();
  } catch (e) {
    setStatus(`Failed to load: ${e.message}`, true);
    console.error(e);
  }
}

main();
