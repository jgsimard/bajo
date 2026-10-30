"""PBRT-v4 text scene loading for Bajo.

The loader currently maps perspective cameras, film/sampler settings, static
affine transforms, attribute/transform scopes, diffuse/conductor/dielectric
materials, diffuse area lights, spheres, and triangle meshes. Unsupported
directives fail explicitly instead of being silently dropped. Loop subdivision
control meshes are accepted as their unsmoothed triangle control cages, and
binary little-endian PLY meshes are loaded relative to their declaring file.
PBRT constant/scale texture graphs and PNG spectrum image maps are loaded.
PLY vertex UVs and normals are retained for CPU and GPU shading.
Homogeneous absorption media are tracked across geometry boundaries on both
backends. Scalar image maps support material mixing, conductor roughness and
bump mapping. Optical spectra and Canon sensor response use RGB approximations;
dielectric IOR curves use their 550 nm value, without spectral dispersion.
"""

from bajo.parser.asset_loader import (
    MemoryAssetLoader,
    PathAssetLoader,
    AssetLoader,
)
from .parser import _parse_pbrt, _parse_pbrt_camera
from bajo.core.camera import Camera
from bajo.rt.scene_description import SceneDescription


def read_pbrt(path: String) raises -> SceneDescription:
    var loader = PathAssetLoader()
    return _parse_pbrt(loader.read_text(path), path, loader)


def read_pbrt_camera(path: String) raises -> Camera:
    """Read the camera without parsing geometry, materials, or image assets."""
    var loader = PathAssetLoader()
    return _parse_pbrt_camera(loader.read_text(path), path, loader)


def read_pbrt_camera[
    Loader: AssetLoader
](path: String, loader: Loader) raises -> Camera:
    return _parse_pbrt_camera(loader.read_text(path), path, loader)


def read_pbrt[
    Loader: AssetLoader
](path: String, loader: Loader) raises -> SceneDescription:
    return _parse_pbrt(loader.read_text(path), path, loader)


def parse_pbrt(text: String, path: String = "") raises -> SceneDescription:
    var loader = MemoryAssetLoader()
    return _parse_pbrt(text, path, loader)
