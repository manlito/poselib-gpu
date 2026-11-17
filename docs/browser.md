# In the browser

The browser build is the same C++ library compiled to WebAssembly: the CPU backend on one thread, and the
WebGPU backend on the visitor's GPU. A small JavaScript wrapper, `jpo.js`, loads it.

```js
import { createJpoModule, Jpo, BACKEND, STATUS } from './jpo.js';

const module = await createJpoModule();
const jpo = await Jpo.create(module, { backend: BACKEND.AUTO });   // or BACKEND.CPU, BACKEND.WEBGPU

// Two pairs: 100 and 80 matches, concatenated. One pinhole camera (COLMAP's PINHOLE: fx fy cx cy).
const out = await jpo.relativePose({
  offsets: [0, 100, 180],
  xyA, xyB,                         // Float64Array(2 * 180): pixels in image a and in image b
  cameras: [{ model: 1, params: [718.856, 718.856, 607.19, 185.22] }],
  cameraA: [0, 0], cameraB: [0, 0], // index into cameras, per pair
}, { maxErrorPx: 1 });
// out.results[i] = { pose (Float64Array 7), score, medianTriAngleDeg, nInliers, iterations, status }
// out.inliers: Uint8Array, one byte per match

const reg = await jpo.absolutePose({ offsets, xy, xyz, cameras, camera }, { maxErrorPx: 4 });
jpo.destroy();
```

Options: `maxErrorPx`, `confidence`, `minIterations`, `maxIterations`, `refine`, `minInlierRatio`, as in the
[C API](api.md). `jpo.lastStats()` tells how many hypotheses the last call drew on the CPU and on the GPU,
which shows where `auto` sent the work.

Serve `jpo.js`, `jpo_wasm.js` and `jpo_wasm.wasm` as static files; nothing runs on a server. The module is
about 260 KB.

## Requirements

The module waits for the GPU through JavaScript promise integration (JSPI), so it needs a browser that has
it: Chrome or Edge 137 or later. WebGPU is optional: without it, `BACKEND.CPU` works and creating a WebGPU or
`auto` context fails with a message. Check before creating one:

```js
const hasJspi = typeof WebAssembly.Suspending === 'function';
const hasGpu = !!navigator.gpu && module._jpow_backend_available(BACKEND.WEBGPU);
```

Only Chrome has been measured; Safari and Firefox were not tried.

## Sharing the GPU with other libraries

A page that already has a `GPUDevice`, for example for keypoints and matching, can hand it over, so that all
the work runs on one device:

```js
const jpo = await Jpo.create(module, { backend: BACKEND.WEBGPU, device });
```

## Speed in the browser

The CPU backend has one thread here and runs about 1.35 times slower than one native thread, so WebGPU wins
from about four problems up, easy ones included. `auto` has its own numbers in the browser for that reason (a
32-hypothesis CPU phase), and moves batches to the GPU much sooner than natively. Measured in Chrome on an
M5 Pro, 20 problems per set, ms:

| set | CPU (WebAssembly) | WebGPU |
|---|---|---|
| relative pose, 1000 matches, 70 % inliers | 25.7 | 18.8 |
| relative pose, 1000 matches, 30 % inliers | 678.2 | 30.0 |
| relative pose, 300 matches, no true pose | 703.8 | 22.2 |
| PnP, 1000 matches, 70 % inliers | 9.3 | 7.6 |
| PnP, 1000 matches, 10 % inliers | 465.2 | 11.9 |
| PnP, 300 matches, no true pose | 214.6 | 5.0 |

Every set is solved 20 of 20 on both. <a href="benchmark/index.html">The benchmark page</a> runs the same sets
in your browser.
