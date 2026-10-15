"""Импортёр DDS: проверка заголовка и блоков BC1/BC3/BC5 без перекодирования."""

from __future__ import annotations

import shutil
import struct
from pathlib import Path
from typing import Any

from engine.core.resources.errors import AssetImportError
from engine.core.resources.importer import Artifact, AssetPreview, ImportContext

__all__ = ["DdsImporter", "parse_dds_header"]

_MAGIC = b"DDS "
_HEADER_SIZE = 128
_DX10_HEADER_SIZE = 20
_CAPS2_CUBEMAP = 0x200
_CAPS2_VOLUME = 0x200000

# FourCC → (формат, байт на блок 4x4)
_FOURCC = {
    b"DXT1": ("bc1", 8),
    b"DXT5": ("bc3", 16),
    b"ATI2": ("bc5", 16),
    b"BC5U": ("bc5", 16),
}
# DXGI_FORMAT_BCn_UNORM из DX10-расширения заголовка.
_DXGI = {71: ("bc1", 8), 77: ("bc3", 16), 83: ("bc5", 16)}


def parse_dds_header(path: Path) -> dict[str, Any]:
    """Разобрать и проверить DDS-файл.

    Args:
        path: Путь к ``.dds``.

    Returns:
        Словарь ``width``, ``height``, ``mip_levels``, ``pixel_format``,
        ``block_bytes``, ``data_offset``, ``data_bytes``.

    Raises:
        AssetImportError: Если заголовок повреждён, формат не BC1/BC3/BC5,
            это кубемап/объём или блоки данных обрезаны.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            head = stream.read(_HEADER_SIZE + _DX10_HEADER_SIZE)
    except OSError as exc:
        raise AssetImportError(f"Не удалось прочитать '{path}': {exc}") from exc
    if len(head) < _HEADER_SIZE or not head.startswith(_MAGIC):
        raise AssetImportError(f"'{path.name}': повреждён заголовок DDS")

    height, width = struct.unpack_from("<II", head, 12)
    mip_levels = max(1, struct.unpack_from("<I", head, 28)[0])
    four_cc = head[84:88]
    caps2 = struct.unpack_from("<I", head, 112)[0]
    if caps2 & (_CAPS2_CUBEMAP | _CAPS2_VOLUME):
        raise AssetImportError(f"'{path.name}': кубемапы и объёмные DDS не поддержаны")

    data_offset = _HEADER_SIZE
    if four_cc == b"DX10":
        if len(head) < _HEADER_SIZE + _DX10_HEADER_SIZE:
            raise AssetImportError(f"'{path.name}': обрезан заголовок DX10")
        dxgi, _dimension, _misc, array_size = struct.unpack_from("<IIII", head, 128)
        if array_size > 1:
            raise AssetImportError(f"'{path.name}': массивы текстур не поддержаны")
        data_offset += _DX10_HEADER_SIZE
        fmt = _DXGI.get(dxgi)
    else:
        fmt = _FOURCC.get(four_cc)
    if fmt is None:
        raise AssetImportError(
            f"'{path.name}': поддерживаются только BC1/BC3/BC5 (DXT1, DXT5, ATI2)"
        )
    if width == 0 or height == 0:
        raise AssetImportError(f"'{path.name}': нулевой размер текстуры")

    pixel_format, block_bytes = fmt
    expected = 0
    for level in range(mip_levels):
        w = max(1, width >> level)
        h = max(1, height >> level)
        expected += ((w + 3) // 4) * ((h + 3) // 4) * block_bytes
    if size < data_offset + expected:
        raise AssetImportError(
            f"'{path.name}': данных {size - data_offset} байт, "
            f"для {mip_levels} мип-уровней нужно {expected}"
        )
    return {
        "width": int(width),
        "height": int(height),
        "mip_levels": int(mip_levels),
        "pixel_format": pixel_format,
        "block_bytes": block_bytes,
        "data_offset": data_offset,
        "data_bytes": expected,
    }


class DdsImporter:
    """``.dds`` → артефакт ``texture`` с готовыми BC-блоками (копия файла)."""

    name = "dds"
    extensions = (".dds",)
    version = 1

    def default_settings(self) -> dict[str, Any]:
        """Настройки по умолчанию (нет)."""
        return {}

    def inspect(self, path: Path) -> AssetPreview:
        """Проверить заголовок и вернуть его параметры."""
        return AssetPreview("texture", parse_dds_header(path))

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        """Проверить файл целиком и сложить его в ``Library`` как есть."""
        info = parse_dds_header(ctx.source)
        shutil.copyfile(ctx.source, ctx.output_dir / "texture.dds")
        return [
            ctx.artifact(
                "texture",
                "texture.dds",
                width=info["width"],
                height=info["height"],
                mip_levels=info["mip_levels"],
                pixel_format=info["pixel_format"],
                compressed=True,
                data_offset=info["data_offset"],
            )
        ]
