#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  assets/bunny \
  assets/buddha \
  assets/dragon \
  assets/sponza \
  assets/rungholt \
  .cache/assets

# Bunny
if [ ! -f assets/bunny/bunny.obj ]; then
  curl -L --fail \
    -o assets/bunny/bunny.obj \
    https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj
fi
# Happy Buddha
if [ ! -f assets/buddha/buddha.obj ]; then
  curl -L --fail \
    -o assets/buddha/buddha.obj \
    https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/happy.obj
fi

# XYZ Dragon
if [ ! -f assets/dragon/dragon.obj ]; then
  curl -L --fail \
    -o assets/dragon/dragon.obj \
    https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/xyzrgb_dragon.obj
fi

# Sponza
if [ ! -f assets/sponza/sponza.obj ]; then
  curl -L --fail \
    -o .cache/assets/sponza.zip \
    https://github.com/jimmiebergmann/Sponza/archive/refs/heads/master.zip

  rm -rf .cache/assets/Sponza-master
  unzip -q .cache/assets/sponza.zip -d .cache/assets
  cp -r .cache/assets/Sponza-master/* assets/sponza/
fi

# Rungholt
if [ ! -f assets/rungholt/rungholt.obj ]; then
  curl -L --fail \
    -o .cache/assets/rungholt.zip \
    https://casual-effects.com/g3d/data10/research/model/rungholt/rungholt.zip

  rm -rf .cache/assets/rungholt
  mkdir -p .cache/assets/rungholt
  unzip -q .cache/assets/rungholt.zip -d .cache/assets/rungholt

  found_obj="$(find .cache/assets/rungholt -name 'rungholt.obj' | head -n 1)"
  found_mtl="$(find .cache/assets/rungholt -name 'rungholt.mtl' | head -n 1 || true)"

  if [ -z "$found_obj" ]; then
    echo "Could not find rungholt.obj after extracting archive"
    exit 1
  fi

  cp "$found_obj" assets/rungholt/rungholt.obj

  if [ -n "$found_mtl" ]; then
    cp "$found_mtl" assets/rungholt/rungholt.mtl
  fi
fi

echo "Assets downloaded."