# Build

poselib-gpu is built from source with CMake 3.16 or later and a C++17 compiler. The CPU backend needs nothing
else. Each GPU backend is built when its toolkit is found, and uses no library beyond it.

```bash
cmake -S . -B build/release -G Ninja
cmake --build build/release
ctest --test-dir build/release
```

This gives the static library `libjpo.a`, the single public header `jpo/jpo.h` and the `jpo` command-line tool.
To use the library from another CMake project, add its folder and link the target:

```cmake
add_subdirectory(poselib-gpu)
target_link_libraries(my_app PRIVATE jpo)
```

## Backends

| Backend | Built when | Notes |
|---|---|---|
| CPU | always | Plain C++, all threads, double precision. The reference the others are checked against. |
| Metal | on Apple platforms | Needs only Metal.framework; the kernels are compiled at run time. |
| CUDA | the CUDA toolkit (`nvcc`) is found | Needs only the CUDA runtime library. |
| WebGPU | Dawn's prebuilt release is found (natively), or emdawnwebgpu (under Emscripten) | WGSL kernels, compiled at run time. The caller may hand over its own device. |

`-DJPO_METAL=OFF`, `-DJPO_CUDA=OFF` and `-DJPO_WEBGPU=OFF` leave a backend out; `ON` makes a missing toolkit an
error. `-DJPO_BUILD_CLI=OFF` and `-DJPO_BUILD_TESTS=OFF` skip the tool and the tests. The script
`tools/fetch_webgpu.sh` downloads Dawn and emdawnwebgpu into `third_party/`.

Which backends a build has, and which one `auto` picks on this machine:

```bash
build/release/jpo backends
```

## WebAssembly

With Emscripten 7.0.0 (and emdawnwebgpu for the WebGPU backend):

```bash
tools/fetch_webgpu.sh browser
emcmake cmake -S . -B build/wasm -G Ninja
cmake --build build/wasm
```

`build/wasm/web` then holds the module (`jpo_wasm.js`, `jpo_wasm.wasm`), the JavaScript wrapper `jpo.js` and the
demo and benchmark pages, ready to serve as static files. See [In the browser](browser.md).

## Command line

The `jpo` tool generates problem sets, solves them and times the backends:

```bash
jpo gen relpose -o pairs.jpop --count 200 --points 1000 --inliers 0.3     # generated problems
jpo run pairs.jpop --backend auto -o results.txt                          # poses, masks, accuracy
jpo bench --backends cpu,metal,auto                                       # the timing grid
```

## Tests

| Test | What it checks |
|---|---|
| `core_test` | Camera models, both solvers on exact data, both estimators on every camera model, hard cases. |
| `precision_test` | The float32 build of the kernels against the double build. |
| `backend_test <gpu>` | A GPU backend against the float32 CPU build: the same best hypothesis per round, the same problems solved. |
| `parity_test` | Each solver against PoseLib's on the same samples (built when PoseLib is fetched with `tools/fetch_reference.sh`). |

The GPU tests run on a machine with that GPU.
