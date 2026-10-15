"""Cooker'ы: офлайн-подготовка данных ассетов (без GL, только байты и numpy).

Имена пакета загружаются лениво (PEP 562): ``from ...cookers import cook_mesh``
импортирует только ``mesh_cooker``, а не все cooker'ы сразу. Плееру нужны лишь
форматы готовых артефактов, и он не платит за импорт тяжёлых cooker'ов на старте.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources.cookers.material_cooker import (
        MaterialSpec,
        pack_orm,
        parse_material,
    )
    from engine.core.resources.cookers.mesh_cooker import (
        MeshCookSettings,
        cook_mesh,
        reorder_vertices,
        simplify_clustering,
    )
    from engine.core.resources.cookers.mesh_format import (
        CookedLod,
        CookedMesh,
        read_cooked_mesh,
    )
    from engine.core.resources.cookers.png_codec import decode_png, encode_png
    from engine.core.resources.cookers.shader_cooker import (
        CookedShader,
        ShaderCookSettings,
        ShaderVariant,
        cook_shader,
        scan_features,
        source_hash,
        variant_key,
    )
    from engine.core.resources.cookers.sprite_atlas_cooker import (
        AtlasSpec,
        BuiltAtlas,
        Sprite,
        SpriteFrame,
        build_atlas,
        pack_sprites,
        parse_atlas_spec,
        slice_alpha,
        slice_grid,
        to_rgba,
    )
    from engine.core.resources.cookers.texture_cooker import (
        TexconvCompressor,
        TextureCompressor,
        TextureCookSettings,
        cook_texture,
        decode_image,
        find_texconv,
        mip_count,
    )
    from engine.core.resources.cookers.texture_format import (
        CookedMip,
        CookedTexture,
        read_cooked_texture,
    )

# Имя → подмодуль, в котором оно определено.
_LAZY: dict[str, str] = {
    **dict.fromkeys(
        ("MaterialSpec", "pack_orm", "parse_material"), "material_cooker"
    ),
    **dict.fromkeys(
        ("MeshCookSettings", "cook_mesh", "reorder_vertices", "simplify_clustering"),
        "mesh_cooker",
    ),
    **dict.fromkeys(("CookedLod", "CookedMesh", "read_cooked_mesh"), "mesh_format"),
    **dict.fromkeys(("decode_png", "encode_png"), "png_codec"),
    **dict.fromkeys(
        (
            "CookedShader",
            "ShaderCookSettings",
            "ShaderVariant",
            "cook_shader",
            "scan_features",
            "source_hash",
            "variant_key",
        ),
        "shader_cooker",
    ),
    **dict.fromkeys(
        (
            "AtlasSpec",
            "BuiltAtlas",
            "Sprite",
            "SpriteFrame",
            "build_atlas",
            "pack_sprites",
            "parse_atlas_spec",
            "slice_alpha",
            "slice_grid",
            "to_rgba",
        ),
        "sprite_atlas_cooker",
    ),
    **dict.fromkeys(
        (
            "TexconvCompressor",
            "TextureCompressor",
            "TextureCookSettings",
            "cook_texture",
            "decode_image",
            "find_texconv",
            "mip_count",
        ),
        "texture_cooker",
    ),
    **dict.fromkeys(
        ("CookedMip", "CookedTexture", "read_cooked_texture"), "texture_format"
    ),
}

__all__ = [
    "AtlasSpec",
    "BuiltAtlas",
    "CookedLod",
    "CookedMesh",
    "CookedMip",
    "CookedShader",
    "CookedTexture",
    "MaterialSpec",
    "MeshCookSettings",
    "ShaderCookSettings",
    "ShaderVariant",
    "Sprite",
    "SpriteFrame",
    "TexconvCompressor",
    "TextureCompressor",
    "TextureCookSettings",
    "build_atlas",
    "cook_mesh",
    "cook_shader",
    "cook_texture",
    "decode_image",
    "decode_png",
    "encode_png",
    "find_texconv",
    "mip_count",
    "pack_orm",
    "pack_sprites",
    "parse_atlas_spec",
    "parse_material",
    "read_cooked_mesh",
    "read_cooked_texture",
    "reorder_vertices",
    "scan_features",
    "simplify_clustering",
    "slice_alpha",
    "slice_grid",
    "source_hash",
    "to_rgba",
    "variant_key",
]


def __getattr__(name: str) -> Any:
    """Загрузить подмодуль при первом обращении к реэкспортированному имени."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
