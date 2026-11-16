#!/usr/bin/env bash
# Fetch the test-only reference into third_party/ (not committed): PoseLib at the pinned commit and
# the Eigen headers it needs. The library itself uses neither.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party"
POSELIB_COMMIT=6a636b1d84273d5e3ba97676463476569611572a   # after v2.0.5
EIGEN_TAG=3.4.0
EIGEN_SHA256=b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626

mkdir -p "$DEST"
if [ ! -d "$DEST/poselib/.git" ]; then
  git clone -q https://github.com/PoseLib/PoseLib.git "$DEST/poselib"
fi
git -C "$DEST/poselib" fetch -q origin "$POSELIB_COMMIT" 2>/dev/null || true
git -C "$DEST/poselib" checkout -q "$POSELIB_COMMIT"
echo "PoseLib: $DEST/poselib @ $POSELIB_COMMIT"

if [ ! -f "$DEST/eigen/Eigen/Dense" ]; then
  tmp="$DEST/eigen.tar.bz2.part"
  curl -fL -o "$tmp" "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_TAG/eigen-$EIGEN_TAG.tar.bz2"
  if command -v sha256sum >/dev/null; then got="$(sha256sum "$tmp" | cut -d' ' -f1)"; else got="$(shasum -a 256 "$tmp" | cut -d' ' -f1)"; fi
  if [ "$got" != "$EIGEN_SHA256" ]; then
    echo "eigen: sha256 $got, expected $EIGEN_SHA256" >&2
    exit 1
  fi
  mkdir -p "$DEST/eigen"
  tar -xjf "$tmp" -C "$DEST/eigen" --strip-components 1
  rm "$tmp"
fi
echo "Eigen: $DEST/eigen"
