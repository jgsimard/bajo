"""PBRT-v4 text scene loading for Bajo.

The loader currently maps perspective cameras, film/sampler settings, static
affine transforms, attribute/transform scopes, diffuse/conductor/dielectric
materials, diffuse area lights, spheres, and triangle meshes. Unsupported
directives fail explicitly instead of being silently dropped. Loop subdivision
control meshes are accepted as their unsmoothed triangle control cages.
"""

from bajo.parser.text_loader import MemoryTextLoader, PathTextLoader, TextLoader
from .parser import _parse_pbrt
from bajo.rt.scene_description import SceneDescription


def read_pbrt(path: String) raises -> SceneDescription:
    var loader = PathTextLoader()
    return _parse_pbrt(loader.read_text(path), path, loader)


def read_pbrt[
    Loader: TextLoader
](path: String, loader: Loader) raises -> SceneDescription:
    return _parse_pbrt(loader.read_text(path), path, loader)


def parse_pbrt(text: String, path: String = "") raises -> SceneDescription:
    var loader = MemoryTextLoader()
    return _parse_pbrt(text, path, loader)
