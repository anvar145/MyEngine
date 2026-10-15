"""Импортёр растровых изображений: PNG, JPEG, BMP, GIF."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from engine.core.resources.errors import AssetImportError, ResourceFormatError
from engine.core.resources.importer import Artifact, AssetPreview, ImportContext
from engine.core.resources.loaders.texture_loader import probe_image

__all__ = ["ImageImporter"]

TEXTURE_FILE = "texture.bin"


class ImageImporter:
    """Изображение → артефакт ``texture`` (``texture.bin``: мипы, опц. BC).

    Настройки:
        srgb (bool): цветовая текстура (иначе — линейные данные: нормали, маски);
        generate_mips (bool): строить цепочку мипов;
        max_size (int): ограничить большую сторону (0 — без ограничения);
        compression (str): ``none`` | ``auto`` | ``bc1`` | ``bc3`` | ``bc5`` —
            сжатие внешним ``texconv``; если его нет, текстура остаётся без сжатия;
        normal_map (bool): карта нормалей (перенормировка мипов, BC5 в ``auto``).

    Версия 2: вместо копии исходника кукуется ``texture.bin``. Если формат
    нельзя декодировать (JPEG/BMP/GIF без Pillow), исходник копируется как есть
    (``cooked=False``), и мипы строит графический слой.
    """

    name = "image"
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    version = 2

    def default_settings(self) -> dict[str, Any]:
        """Настройки по умолчанию."""
        return {
            "srgb": True,
            "generate_mips": True,
            "max_size": 0,
            "compression": "none",
            "normal_map": False,
        }

    def inspect(self, path: Path) -> AssetPreview:
        """Прочитать размеры и формат из заголовка."""
        try:
            info = probe_image(path)
        except ResourceFormatError as exc:
            raise AssetImportError(str(exc)) from exc
        return AssetPreview("texture", info)

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        """Декодировать, приготовить мипы/сжатие и записать ``texture.bin``."""
        from engine.core.resources.cookers import (
            TexconvCompressor,
            TextureCookSettings,
            cook_texture,
            decode_image,
            find_texconv,
            read_cooked_texture,
        )

        info = self.inspect(ctx.source).info
        settings = TextureCookSettings.from_mapping(dict(ctx.settings))
        try:
            pixels = decode_image(ctx.source)
        except AssetImportError as exc:
            if "Pillow" not in str(exc):
                raise
            ctx.log.warning("%s; исходник сохранён без обработки", exc)
            return [self._raw_copy(ctx, info)]

        texconv = find_texconv() if settings.compression != "none" else None
        compressor = (
            TexconvCompressor(texconv, work_dir=ctx.output_dir) if texconv else None
        )
        blob = cook_texture(pixels, settings, compressor, ctx.log)
        (ctx.output_dir / TEXTURE_FILE).write_bytes(blob)

        cooked = read_cooked_texture(blob)
        return [
            ctx.artifact(
                "texture",
                TEXTURE_FILE,
                cooked=True,
                width=cooked.width,
                height=cooked.height,
                source_width=info["width"],
                source_height=info["height"],
                channels=pixels.shape[2],
                pixel_format=cooked.pixel_format,
                mip_levels=len(cooked.mips),
                compressed=cooked.compressed,
                srgb=cooked.srgb,
                generate_mips=settings.generate_mips,
            )
        ]

    def _raw_copy(self, ctx: ImportContext, info: Any) -> Artifact:
        """Запасной путь: скопировать исходник и описать его заголовок."""
        name = "texture" + ctx.source.suffix.lower()
        shutil.copyfile(ctx.source, ctx.output_dir / name)
        return ctx.artifact(
            "texture",
            name,
            cooked=False,
            width=info["width"],
            height=info["height"],
            channels=info["channels"],
            pixel_format=info["pixel_format"],
            srgb=bool(ctx.setting("srgb", True)),
            generate_mips=bool(ctx.setting("generate_mips", True)),
        )
