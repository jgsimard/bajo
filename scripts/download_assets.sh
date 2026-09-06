#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  assets/bunny \
  assets/buddha \
  assets/dragon \
  assets/sponza \
  assets/rungholt \
  assets/lucy \
  assets/igea \
  assets/nefertiti \
  assets/armadillo \
  assets/crown \
  assets/pbrt/killeroos/geometry \
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

# Lucy statue
if [ ! -f assets/lucy/lucy.obj ]; then
  curl -L --fail \
    -o assets/lucy/lucy.obj \
    https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/lucy.obj
fi

# Igea bust/head
if [ ! -f assets/igea/igea.obj ]; then
  curl -L --fail \
    -o assets/igea/igea.obj \
    https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/igea.obj
fi

# Nefertiti bust
if [ ! -f assets/nefertiti/nefertiti.obj ]; then
  curl -L --fail \
    -o assets/nefertiti/nefertiti.obj \
    https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/nefertiti.obj
fi

# Stanford Armadillo
if [ ! -f assets/armadillo/armadillo.obj ]; then
  curl -L --fail \
    -o assets/armadillo/armadillo.obj \
    https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj
fi

# Complete PBRT-v4 Crown scene: scene description, 794 PLY meshes, and 40
# textures. Use a sparse checkout so the other large scene assets in the
# repository are not downloaded. The pinned revision keeps benchmarks stable.
if [ ! -f assets/crown/crown.pbrt ]; then
  crown_checkout=.cache/assets/pbrt-v4-scenes-crown
  crown_revision=30cf4a0346ae5a80a2d7a530a3ef7d0fa4f70572

  rm -rf "$crown_checkout"
  git init -q "$crown_checkout"
  git -C "$crown_checkout" remote add origin https://github.com/mmp/pbrt-v4-scenes.git
  git -C "$crown_checkout" sparse-checkout init --cone
  git -C "$crown_checkout" sparse-checkout set crown
  git -C "$crown_checkout" fetch -q --depth 1 --filter=blob:none origin "$crown_revision"
  git -C "$crown_checkout" checkout -q --detach FETCH_HEAD
  cp -r "$crown_checkout/crown/." assets/crown/
fi

# PBRT-v4 Killeroo gallery scene. Model courtesy of headus; scene maintained
# by the official pbrt-v4-scenes repository.
if [ ! -f assets/pbrt/killeroos/killeroo-simple.pbrt ]; then
  curl -L --fail \
    -o assets/pbrt/killeroos/killeroo-simple.pbrt \
    https://raw.githubusercontent.com/mmp/pbrt-v4-scenes/master/killeroos/killeroo-simple.pbrt
fi
if [ ! -f assets/pbrt/killeroos/geometry/killeroo.pbrt ]; then
  curl -L --fail \
    -o assets/pbrt/killeroos/geometry/killeroo.pbrt \
    https://raw.githubusercontent.com/mmp/pbrt-v4-scenes/master/killeroos/geometry/killeroo.pbrt
fi
echo "Assets downloaded."
