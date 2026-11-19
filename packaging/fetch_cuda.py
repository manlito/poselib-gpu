"""Fetches the parts of the CUDA toolkit the CUDA backend builds with from NVIDIA's
redistributable archives (no installer), as just-ba's script does:

    python packaging/fetch_cuda.py --cuda 12.9.1 --dest /opt/jpo-cuda

gives ``<dest>/cuda``: nvcc, the runtime (linked statically) and the headers. The
same archives exist for Linux and Windows, so the release workflow builds both the
same way. Each archive's SHA-256 is checked against NVIDIA's index. Standard
library only (it runs in the manylinux container).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

CUDA_REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/"
# What nvcc and the build need. The names changed in CUDA 13 (cccl, and the compiler's
# crt and libnvvm split out of cuda_nvcc); each is taken when the index has it.
COMPONENTS = ["cuda_nvcc", "cuda_crt", "libnvvm", "cuda_cudart", "cuda_cccl", "cccl"]


def platform_name(override: str | None) -> str:
    if override:
        return override
    if sys.platform.startswith("linux"):
        return "linux-x86_64"
    if sys.platform == "win32":
        return "windows-x86_64"
    sys.exit("fetch_cuda.py: Linux or Windows x86_64 only")


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def download(url: str, sha256: str, into: Path) -> Path:
    path = into / url.rsplit("/", 1)[-1]
    h = hashlib.sha256()
    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != sha256:
        sys.exit(f"fetch_cuda.py: SHA-256 mismatch for {url}")
    return path


def unpack_into(archive: Path, dest: Path, scratch: Path) -> None:
    out = scratch / (archive.name + ".d")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    else:
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data")
    (top,) = list(out.iterdir())  # each archive holds one folder, <name>-archive
    shutil.copytree(top, dest, symlinks=True, dirs_exist_ok=True)
    shutil.rmtree(out)
    archive.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", required=True, help="CUDA release, e.g. 12.9.1 or 13.4.3")
    ap.add_argument("--dest", required=True)
    # For checking this script on another machine: another platform, fewer parts.
    ap.add_argument("--platform", choices=["linux-x86_64", "windows-x86_64"])
    ap.add_argument("--components", help="comma-separated subset of " + ",".join(COMPONENTS))
    args = ap.parse_args()
    plat, dest = platform_name(args.platform), Path(args.dest)
    wanted = args.components.split(",") if args.components else COMPONENTS
    cuda_dir = dest / "cuda"

    index = get_json(f"{CUDA_REDIST}redistrib_{args.cuda}.json")
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        for name in wanted:
            entry = index.get(name, {}).get(plat)
            if not entry:
                continue
            print(f"{name} {index[name]['version']}", flush=True)
            unpack_into(download(CUDA_REDIST + entry["relative_path"], entry["sha256"], scratch), cuda_dir, scratch)

    # The redistributable archives use lib/; some tools look in lib64/.
    if plat.startswith("linux") and (cuda_dir / "lib").is_dir() and not (cuda_dir / "lib64").exists():
        (cuda_dir / "lib64").symlink_to("lib")
    print("nvcc:", next(cuda_dir.glob("bin/nvcc*"), "not fetched"))


if __name__ == "__main__":
    main()
