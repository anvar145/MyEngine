"""Встроенные загрузчики ресурсов и их набор по умолчанию."""

from __future__ import annotations

from engine.core.resources.loaders.mesh_loader import ObjMeshLoader, parse_obj
from engine.core.resources.loaders.shader_loader import GlslShaderLoader
from engine.core.resources.loaders.texture_loader import ImageTextureLoader, probe_image

__all__ = [
    "GlslShaderLoader",
    "ImageTextureLoader",
    "ObjMeshLoader",
    "default_loaders",
    "parse_obj",
    "probe_image",
]


def default_loaders() -> tuple[ObjMeshLoader, ImageTextureLoader, GlslShaderLoader]:
    """Собрать набор загрузчиков, который использует движок по умолчанию."""
    return (ObjMeshLoader(), ImageTextureLoader(), GlslShaderLoader())
