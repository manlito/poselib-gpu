#!/usr/bin/env bash
# Fetch the WebGPU dependencies from the pinned Dawn release into third_party/webgpu (not committed):
#
#   tools/fetch_webgpu.sh            # emdawnwebgpu (the browser build) and Dawn for this machine (native)
#   tools/fetch_webgpu.sh browser    # only emdawnwebgpu, as the docs workflow does
#   tools/fetch_webgpu.sh native     # only Dawn
#
# The layout is the one CMakeLists.txt looks for by default.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/webgpu"
TAG=v20261113.214538
COMMIT=7daf3b0bee2c73621001ab15b256b5a84fb0b45f
BASE="https://github.com/google/dawn/releases/download/$TAG"
EMDAWN_SHA256=dde49956462cede6b2dfae74daf22479056e05035ca8283b03f9b725c1c41982
WHAT="${1:-all}"

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

browser() {
  local dir="$DEST/emdawnwebgpu" zip="emdawnwebgpu_pkg-$TAG.zip"
  if [ -f "$dir/emdawnwebgpu_pkg/emdawnwebgpu.port.py" ]; then
    echo "emdawnwebgpu: $dir/emdawnwebgpu_pkg"
    return
  fi
  mkdir -p "$dir"
  curl -fL -o "$dir/$zip.part" "$BASE/$zip"
  local got
  got="$(sha256 "$dir/$zip.part")"
  if [ "$got" != "$EMDAWN_SHA256" ]; then
    echo "emdawnwebgpu: sha256 $got, expected $EMDAWN_SHA256" >&2
    exit 1
  fi
  (cd "$dir" && unzip -q "$zip.part" && rm "$zip.part")
  echo "emdawnwebgpu: $dir/emdawnwebgpu_pkg"
}

native() {
  local platform
  case "$(uname -s)" in
    Darwin) platform=macos-latest ;;
    Linux) platform=ubuntu-latest ;;
    *) echo "native Dawn: unpack Dawn-$COMMIT-<platform>-Release from $BASE into $DEST/dawn by hand" >&2; exit 1 ;;
  esac
  local name="Dawn-$COMMIT-$platform-Release"
  if [ -d "$DEST/dawn/$name/lib/cmake/Dawn" ]; then
    echo "Dawn: $DEST/dawn/$name"
    return
  fi
  mkdir -p "$DEST/dawn"
  curl -fL "$BASE/$name.tar.gz" | tar -xz -C "$DEST/dawn"
  echo "Dawn: $DEST/dawn/$name"
}

case "$WHAT" in
  all) browser; native ;;
  browser) browser ;;
  native) native ;;
  *) echo "usage: $0 [all|browser|native]" >&2; exit 2 ;;
esac
