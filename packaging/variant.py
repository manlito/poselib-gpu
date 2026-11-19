"""Turns pyproject.toml into one of the CUDA packages, in place.

    python packaging/variant.py cu12 --root /opt/jpo-cuda --platform linux
    python packaging/variant.py cu13 --root C:/jpo-cuda --platform windows

Run by the release workflow only, before cibuildwheel; never commit the result. ``--root`` is where
packaging/fetch_cuda.py put the CUDA toolkit (``cuda/``), as the build sees it (inside the manylinux container
on Linux). The CUDA runtime is linked statically (CMakeLists.txt, CUDA::cudart_static), so a CUDA package needs
only NVIDIA's driver at run time: no NVIDIA pip package.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# GPU architectures per package, as just-ba's: real code for each, plus PTX for the newest so later GPUs can
# still run it. CUDA 12 keeps Pascal and Volta, which CUDA 13 dropped.
ARCHITECTURES = {
    "cu12": "60-real;70-real;75-real;80-real;86-real;89-real;90-real;100-real;120",
    "cu13": "75-real;80-real;86-real;89-real;90-real;100-real;120",
}


def sub_once(text: str, pattern: str, repl: str) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"variant.py: pattern not found in pyproject.toml: {pattern}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(ARCHITECTURES))
    ap.add_argument("--root", required=True)
    ap.add_argument("--platform", choices=["linux", "windows"], required=True)
    args = ap.parse_args()
    v, root = args.variant, args.root.rstrip("/\\")
    nvcc = f"{root}/cuda/bin/nvcc" + (".exe" if args.platform == "windows" else "")

    text = PYPROJECT.read_text()
    text = sub_once(text, r'^name = "poselib-gpu"$', f'name = "poselib-gpu-{v}"')
    text = sub_once(text, r'^(description = ".*)"$', rf'\1 (CUDA {v[2:]} package)"')
    text = sub_once(text, r'^JPO_CUDA = "OFF"$', 'JPO_CUDA = "ON"')
    defines = "\n".join([
        f'JPO_PACKAGE_VARIANT = "{v}"',
        f'CMAKE_CUDA_ARCHITECTURES = "{ARCHITECTURES[v]}"',
        f'CMAKE_CUDA_COMPILER = "{nvcc}"',
        f'CUDAToolkit_ROOT = "{root}/cuda"',
    ])
    text = sub_once(text, r'^JPO_PACKAGE_VARIANT = "cpu"$', defines)
    with open(PYPROJECT, "w", newline="") as f:  # keep the checkout's line endings
        f.write(text)
    print(f"pyproject.toml is now poselib-gpu-{v} ({args.platform}, CUDA under {root})")


if __name__ == "__main__":
    main()
