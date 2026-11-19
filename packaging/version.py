"""The version a release publishes (README.md, "Releases"), and writing it into CMakeLists.txt.

    python packaging/version.py next poselib-gpu poselib-gpu-cu12 poselib-gpu-cu13
    python packaging/version.py set 0.1.3

``next`` prints the version the release workflow publishes: the highest of the git
tags ``vX.Y.Z`` and the versions of the named packages on PyPI, with its patch
bumped; or ``project(VERSION)`` in CMakeLists.txt when that is higher, which is
how a minor or major release is made (edit it in the change that warrants it).
Nothing released yet: ``project(VERSION)`` itself. Reading PyPI as well as the
tags means a release whose tag was not pushed is still never published twice.

``set`` writes a version into ``project(VERSION)``, the one source the package,
``poselib_gpu._build_info`` and the C++ library read. Run by the release workflow
only, before building; never commit the result.

Standard library only (it runs on every runner before anything is installed).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CMAKELISTS = ROOT / "CMakeLists.txt"
PROJECT = re.compile(r"^(project\(\w+ VERSION )([0-9]+\.[0-9]+\.[0-9]+)", re.M)
SEMVER = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)$")

Version = tuple[int, int, int]


def parse(text: str) -> Version | None:
    m = SEMVER.match(text)
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def cmake_version() -> Version:
    m = PROJECT.search(CMAKELISTS.read_text())
    if not m:
        sys.exit("version.py: no project(<name> VERSION X.Y.Z) in CMakeLists.txt")
    return parse(m[2])


def tag_versions() -> list[Version]:
    out = subprocess.run(["git", "tag", "--list", "v*"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [v for t in out.stdout.split() if (v := parse(t[1:]))]


def pypi_versions(package: str) -> list[Version]:
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=30) as r:
            releases = json.load(r)["releases"]
    except urllib.error.HTTPError as e:
        if e.code == 404:  # not on PyPI yet
            return []
        raise
    return [v for t in releases if (v := parse(t))]


def next_version(packages: list[str]) -> Version:
    base = cmake_version()
    released = tag_versions() + [v for p in packages for v in pypi_versions(p)]
    if not released:
        return base
    latest = max(released)
    bumped = (latest[0], latest[1], latest[2] + 1)
    return max(base, bumped)


def set_version(text: str) -> None:
    if not parse(text):
        sys.exit(f"version.py: not a version X.Y.Z: {text}")
    source = CMAKELISTS.read_text()
    out, n = PROJECT.subn(rf"\g<1>{text}", source, count=1)
    if n != 1:
        sys.exit("version.py: no project(<name> VERSION X.Y.Z) in CMakeLists.txt")
    with open(CMAKELISTS, "w", newline="") as f:  # keep the checkout's line endings
        f.write(out)
    print(f"CMakeLists.txt: project VERSION {text}")


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "next":
        print("%d.%d.%d" % next_version(sys.argv[2:]))
    elif len(sys.argv) == 3 and sys.argv[1] == "set":
        set_version(sys.argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
