"""Сборка :class:`PbrMaterial` из импортированных ассетов.

Мост между пайплайном ассетов (``engine.core.resources``: ``material.json``,
``texture.bin``) и GPU: читает кукованные файлы, создаёт ``GpuTexture`` и
заполняет материал. Сами текстуры в видеопамять уходят при первой отрисовке.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from engine.core.math3d import Vector3
from engine.core.resources import (
    ResourceError,
    load_material,
    load_material_artifact,
)
from engine.core.resources.cookers import (
    CookedMip,
    CookedTexture,
    read_cooked_texture,
)
from engine.core.resources.cookers.texture_format import mip_byte_size
from engine.core.resources.vfs import as_assets
from engine.rendering.pbr_material import PbrMaterial
from engine.rendering.texture import GpuTexture

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources import AssetDatabase
    from engine.rendering.renderer import Renderer

__all__ = ["load_pbr_material"]

_log = logging.getLogger("engine.rendering.material_loader")


def load_pbr_material(
    renderer: Renderer,
    db: AssetDatabase,
    guid: str,
    *,
    artifact_name: str | None = None,
    textures: dict[str, GpuTexture] | None = None,
) -> PbrMaterial:
    """Собрать PBR-материал из ассета ``.mat.json`` (или материала glTF).

    Текстуры, которых нет в базе или которые не прошли cooker (``cooked=False``),
    пропускаются с предупреждением: материал остаётся рабочим, но без них.
    Карта нормалей всегда читается как линейная.

    Args:
        renderer: Рендерер (даёт программу ``pbr``).
        db: База ассетов или VFS.
        guid: GUID ассета материала (или glTF-модели при ``artifact_name``).
        artifact_name: Имя ``material_NNN.json`` для материала из glTF.
        textures: Общий кэш ``GUID текстуры → GpuTexture``: одна текстура на
            все материалы, которые на неё ссылаются.

    Returns:
        Материал, готовый к ``Renderer.submit``.

    Raises:
        ResourceError: Если ассета или его артефактов нет.
    """
    db = as_assets(db)
    cache = textures if textures is not None else {}
    if artifact_name is None:
        resolved = load_material(db, guid)
    else:
        resolved = load_material_artifact(db, guid, artifact_name)
    params = resolved.params

    def texture_of(slot: str, *, srgb: bool) -> GpuTexture | None:
        texture_guid = resolved.texture_guids.get(slot)
        if texture_guid is None:
            return None
        key = f"{texture_guid}:{int(srgb)}"
        if key not in cache:
            loaded = _load_texture(db, texture_guid, srgb)
            if loaded is None:
                _log.warning(
                    "Материал '%s': текстура слота %s недоступна", resolved.name, slot
                )
                return None
            cache[key] = loaded
        return cache[key]

    orm = None
    if resolved.orm_path is not None:
        key = f"{resolved.orm_path}"
        if key not in cache:
            cooked = read_cooked_texture(resolved.orm_path.read_bytes())
            cache[key] = GpuTexture.from_cooked(
                f"{resolved.name}:orm", cooked, srgb=False
            )
        orm = cache[key]

    base = params.get("base_color", [1.0, 1.0, 1.0, 1.0])
    return PbrMaterial(
        name=resolved.name or guid,
        program=renderer.program("pbr"),
        base_color=Vector3(base[0], base[1], base[2]),
        alpha=float(base[3]),
        roughness=float(params.get("roughness", 1.0)),
        metallic=float(params.get("metallic", 0.0)),
        double_sided=bool(params.get("double_sided", False)),
        texture=texture_of("base_color", srgb=True),
        normal_texture=texture_of("normal", srgb=False),
        orm_texture=orm,
        emissive_texture=texture_of("emissive", srgb=True),
        emissive=Vector3(*params.get("emissive", [0.0, 0.0, 0.0])),
        normal_scale=float(params.get("normal_scale", 1.0)),
        occlusion_strength=float(params.get("occlusion_strength", 1.0)),
        alpha_mode=str(params.get("alpha_mode", "opaque")),
        alpha_cutoff=float(params.get("alpha_cutoff", 0.5)),
        sorting_layer=int(params.get("sorting_layer", 0)),
        uv_scale=tuple(params.get("uv_scale", (1.0, 1.0))),
        uv_offset=tuple(params.get("uv_offset", (0.0, 0.0))),
    )


def _load_texture(db: AssetDatabase, guid: str, srgb: bool) -> GpuTexture | None:
    """Прочитать ``texture.bin`` текстурного ассета; ``None`` — не получилось."""
    try:
        handle = db.resolve(guid)
        artifact = handle.artifact("texture")
        if artifact is None:
            return None
        data = handle.artifact_path(artifact).read_bytes()
        if artifact.info.get("cooked", False):
            cooked = read_cooked_texture(data)
        elif artifact.info.get("compressed", False):
            cooked = _cooked_from_dds(data, artifact.info, srgb)
        else:
            return None
    except (ResourceError, OSError, KeyError, ValueError):
        return None
    return GpuTexture.from_cooked(handle.path.name, cooked, srgb=srgb)


def _cooked_from_dds(data: bytes, info: dict, srgb: bool) -> CookedTexture:
    """Собрать :class:`CookedTexture` из сжатого ``.dds`` (BC-блоки как есть)."""
    view = memoryview(data)
    pixel_format = str(info["pixel_format"])
    width, height = int(info["width"]), int(info["height"])
    offset = int(info["data_offset"])
    mips = []
    for _ in range(int(info.get("mip_levels", 1))):
        size = mip_byte_size(pixel_format, width, height)
        if offset + size > view.nbytes:
            raise ValueError("dds: мип выходит за файл")
        mips.append(CookedMip(width, height, view[offset : offset + size]))
        offset += size
        width, height = max(1, width // 2), max(1, height // 2)
    first = mips[0]
    return CookedTexture(first.width, first.height, pixel_format, srgb, tuple(mips))
