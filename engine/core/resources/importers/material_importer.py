"""Импортёр PBR-материалов ``.mat.json`` и общая сборка материала для glTF."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.core.resources.errors import AssetImportError
from engine.core.resources.importer import Artifact, AssetPreview, ImportContext

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources.cookers.material_cooker import MaterialSpec

__all__ = ["MaterialImporter", "build_material_artifacts"]

LINKED = ("base_color", "normal", "emissive")


class MaterialImporter:
    """``.mat.json`` → артефакты ``material`` (``material.json``) и ``orm_texture``.

    Карты ``ao``/``roughness``/``metallic`` (или ``metallic_roughness``)
    упаковываются в ``material_orm.bin`` (R = occlusion, G = roughness,
    B = metallic, линейная, с мипами). ``base_color``/``normal``/``emissive``
    остаются ссылками на отдельные текстурные ассеты: их ``srgb``/``normal_map``
    задаются в ``.meta`` самих текстур.

    Настройки:
        generate_mips (bool), max_texture_size (int), compression (str):
        параметры упакованной текстуры (см. ``ImageImporter``).
    """

    name = "material"
    extensions = (".mat.json",)
    version = 1

    def default_settings(self) -> dict[str, Any]:
        """Настройки по умолчанию."""
        return {"generate_mips": True, "max_texture_size": 0, "compression": "none"}

    def inspect(self, path: Path) -> AssetPreview:
        """Разобрать материал и вернуть краткую сводку."""
        spec = _read_spec(path)
        return AssetPreview(
            "material",
            {
                "name": spec.name or path.name.removesuffix(".mat.json"),
                "shader": spec.shader,
                "alpha_mode": spec.alpha_mode,
                "textures": sorted(spec.textures),
            },
        )

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        """Проверить материал, упаковать карты и записать ``material.json``."""
        spec = _read_spec(ctx.source)
        name = spec.name or ctx.source.name.removesuffix(".mat.json")
        return build_material_artifacts(ctx, spec, ctx.source.parent, "material", name)


def _read_spec(path: Path) -> MaterialSpec:
    from engine.core.resources.cookers.material_cooker import parse_material

    try:
        return parse_material(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise AssetImportError(f"Не удалось прочитать '{path}': {exc}") from exc


def build_material_artifacts(
    ctx: ImportContext, spec: MaterialSpec, base_dir: Path, stem: str, name: str
) -> list[Artifact]:
    """Записать ``<stem>.json`` и (при наличии PBR-карт) ``<stem>_orm.bin``.

    Args:
        ctx: Контекст импорта (зависимости регистрируются в нём).
        spec: Проверенное описание материала.
        base_dir: Каталог, от которого отсчитаны пути текстур.
        stem: Префикс имён артефактов.
        name: Имя материала для ``info``.

    Raises:
        AssetImportError: Если файл текстуры не найден или не декодируется.
    """
    from engine.core.resources.cookers import (
        TexconvCompressor,
        TextureCookSettings,
        cook_texture,
        decode_image,
        find_texconv,
        pack_orm,
    )
    from engine.core.resources.cookers.material_cooker import MATERIAL_VERSION

    files = {slot: _existing(base_dir, rel) for slot, rel in spec.textures.items()}
    textures: dict[str, dict[str, str]] = {}
    for slot in LINKED:
        if slot in files:
            ctx.add_dependency(files[slot])
            textures[slot] = {"source": _portable(files[slot], base_dir)}

    artifacts: list[Artifact] = []
    packed = {slot: files[slot] for slot in spec.packed_sources}
    orm_name = f"{stem}_orm.bin"
    if packed:
        for path in dict.fromkeys(packed.values()):
            ctx.add_baked_input(path)
        pixels = pack_orm(
            **{slot: decode_image(path) for slot, path in packed.items()}  # type: ignore[arg-type]
        )
        settings = TextureCookSettings(
            srgb=False,
            generate_mips=bool(ctx.setting("generate_mips", True)),
            max_size=int(ctx.setting("max_texture_size", 0)),
            compression=str(ctx.setting("compression", "none")),
        )
        texconv = find_texconv() if settings.compression != "none" else None
        compressor = (
            TexconvCompressor(texconv, work_dir=ctx.output_dir) if texconv else None
        )
        (ctx.output_dir / orm_name).write_bytes(
            cook_texture(pixels, settings, compressor, ctx.log)
        )
        textures["orm"] = {"artifact": orm_name}

    document = {
        "format": "mymat",
        "version": MATERIAL_VERSION,
        **spec.to_dict(),
        "textures": textures,
    }
    document["name"] = name
    json_name = f"{stem}.json"
    (ctx.output_dir / json_name).write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    artifacts.append(
        ctx.artifact(
            "material",
            json_name,
            material_name=name,
            shader=spec.shader,
            alpha_mode=spec.alpha_mode,
            textures=sorted(textures),
        )
    )
    if packed:
        artifacts.append(
            ctx.artifact("orm_texture", orm_name, sources=sorted(packed))
        )
    return artifacts


def _existing(base_dir: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise AssetImportError(f"Путь текстуры должен быть относительным: '{relative}'")
    full = Path(os.path.abspath(base_dir / path))
    if not full.is_file():
        raise AssetImportError(f"Текстура '{relative}' не найдена")
    return full


def _portable(path: Path, base_dir: Path) -> str:
    """Путь относительно каталога материала с ``/`` в качестве разделителя."""
    return Path(os.path.relpath(path, base_dir)).as_posix()
