"""PBRT-v4 text scene loading for Bajo.

The loader currently maps perspective cameras, film/sampler settings, static
affine transforms, attribute/transform scopes, diffuse/conductor/dielectric
materials, diffuse area lights, spheres, and triangle meshes. Unsupported
directives fail explicitly instead of being silently dropped. Loop subdivision
control meshes are accepted as their unsmoothed triangle control cages.
"""

from .loaders import MemoryPbrtTextLoader, PathPbrtTextLoader, PbrtTextLoader
from .parser import _parse_pbrt
from .types import PbrtScene


def read_pbrt(path: String) raises -> PbrtScene:
    var loader = PathPbrtTextLoader()
    return _parse_pbrt(loader.read_text(path), path, loader)


def read_pbrt[
    Loader: PbrtTextLoader
](path: String, loader: Loader) raises -> PbrtScene:
    return _parse_pbrt(loader.read_text(path), path, loader)


def parse_pbrt(text: String, path: String = "") raises -> PbrtScene:
    var loader = MemoryPbrtTextLoader()
    return _parse_pbrt(text, path, loader)
