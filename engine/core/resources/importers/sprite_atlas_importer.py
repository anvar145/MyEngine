"""Импортёр спрайт-атласов ``*.atlas.json``: нарезка листа или упаковка файлов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.core.resources.errors import AssetImportError
from engine.core.resources.importer import Artifact, AssetPreview, ImportContext

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources.cookers import AtlasSpec

__all__ = ["SpriteAtlasImporter"]

TEXTURE_FILE = "texture.bin"
FRAMES_FILE = "atlas.json"


class SpriteAtlasImporter:
    """``*.atlas.json`` → артефакты ``texture`` и ``sprite_atlas``.

    Описание атласа лежит в самом файле (``mode``: ``grid`` | ``alpha`` |
    ``pack``, см. ``cookers/sprite_atlas_cooker.py``). Исходные картинки —
    «запечённые» зависимости: их правка переимпортирует атлас.

    Настройки (параметры итоговой текстуры):
        srgb (bool): цветовая текстура;
        generate_mips (bool): мипы (по умолчанию выключены: на краях спрайтов
            они подмешивают соседей);
        compression (str): ``none`` | ``auto`` | ``bc1`` | ``bc3`` внешним ``texconv``.
    """

    name = "sprite_atlas"
    extensions = (".atlas.json",)
    version = 1

    def default_settings(self) -> dict[str, Any]:
        """Настройки по умолчанию."""
        return {"srgb": True, "generate_mips": False, "compression": "none"}

    def inspect(self, path: Path) -> AssetPreview:
        """Разобрать описание и вернуть сводку (без чтения картинок)."""
        spec = _read_spec(path)
        return AssetPreview(
            "sprite_atlas",
            {
                "mode": spec.mode,
                "image": spec.image,
                "sources": len(spec.sources),
                "folder": spec.folder,
            },
        )

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        """Построить атлас, записать ``texture.bin`` и ``atlas.json``."""
        from engine.core.resources.cookers import (
            TexconvCompressor,
            TextureCookSettings,
            build_atlas,
            cook_texture,
            find_texconv,
        )

        spec = _read_spec(ctx.source)
        built = build_atlas(spec, ctx.source.parent)
        for path in built.inputs:
            ctx.add_baked_input(path)

        settings = TextureCookSettings(
            srgb=bool(ctx.setting("srgb", True)),
            generate_mips=bool(ctx.setting("generate_mips", False)),
            compression=str(ctx.setting("compression", "none")),
        )
        texconv = find_texconv() if settings.compression != "none" else None
        compressor = (
            TexconvCompressor(texconv, work_dir=ctx.output_dir) if texconv else None
        )
        (ctx.output_dir / TEXTURE_FILE).write_bytes(
            cook_texture(built.pixels, settings, compressor, ctx.log)
        )
        (ctx.output_dir / FRAMES_FILE).write_text(
            json.dumps(built.document(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return [
            ctx.artifact(
                "texture",
                TEXTURE_FILE,
                cooked=True,
                width=built.width,
                height=built.height,
                srgb=settings.srgb,
                generate_mips=settings.generate_mips,
            ),
            ctx.artifact(
                "sprite_atlas",
                FRAMES_FILE,
                mode=spec.mode,
                width=built.width,
                height=built.height,
                count=len(built.frames),
                names=[frame.name for frame in built.frames],
            ),
        ]


def _read_spec(path: Path) -> AtlasSpec:
    from engine.core.resources.cookers import parse_atlas_spec

    try:
        return parse_atlas_spec(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise AssetImportError(f"Не удалось прочитать '{path}': {exc}") from exc
