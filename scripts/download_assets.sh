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

# Fetch one complete directory from pbrt-v4-scenes without downloading the
# other multi-gigabyte scenes. Pinning the revision keeps tests and benchmarks
# reproducible.
download_pbrt_scene() {
  local scene_name=$1
  local destination=$2
  local sentinel=$3
  local revision=30cf4a0346ae5a80a2d7a530a3ef7d0fa4f70572
  local checkout=.cache/assets/pbrt-v4-scenes-$scene_name

  if [ -f "$destination/$sentinel" ]; then
    return
  fi

  rm -rf "$checkout"
  git init -q "$checkout"
  git -C "$checkout" remote add origin https://github.com/mmp/pbrt-v4-scenes.git
  git -C "$checkout" sparse-checkout init --cone
  git -C "$checkout" sparse-checkout set "$scene_name"
  git -C "$checkout" fetch -q --depth 1 --filter=blob:none origin "$revision"
  git -C "$checkout" checkout -q --detach FETCH_HEAD
  mkdir -p "$destination"
  cp -r "$checkout/$scene_name/." "$destination/"
}

# Progression scenes, from the nearly-supported Killeroos scene through PLY,
# textures, large mesh sets, environment lighting, media, and full stress cases.
download_pbrt_scene killeroos assets/pbrt/killeroos killeroo-gold.pbrt
download_pbrt_scene pbrt-book assets/pbrt/pbrt-book book.pbrt
download_pbrt_scene \
  contemporary-bathroom \
  assets/pbrt/contemporary-bathroom \
  contemporary-bathroom.pbrt
download_pbrt_scene bmw-m6 assets/pbrt/bmw-m6 bmw-m6.pbrt
download_pbrt_scene crown assets/crown crown.pbrt
download_pbrt_scene bistro assets/bistro bistro_cafe.pbrt

echo "Assets downloaded."
