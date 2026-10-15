"""Загрузчик изображений: PNG, JPEG, BMP, GIF и DDS.

Задача загрузчика — прочитать **заголовок** и отдать метаданные текстуры
(размер, каналы, формат), не декодируя пиксели. Это осознанное решение для
слабого железа: DDS/BC-данные уходят на GPU как есть, а PNG/JPEG декодируются
только если доступна библиотека декодирования (Pillow) — иначе текстура
остаётся «неразобранной» и её разберёт графический слой.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np

from engine.core.resources.asset import TextureAsset
from engine.core.resources.errors import ResourceFormatError
from engine.core.resources.loader import AssetLoader, LoadContext

__all__ = ["ImageTextureLoader", "probe_image"]

_log = logging.getLogger("engine.core.resources.image")

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8"
BMP_SIGNATURE = b"BM"
DDS_SIGNATURE = b"DDS "
GIF_SIGNATURES = (b"GIF87a", b"GIF89a")

_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_DDS_BC_FORMATS = {
    b"DXT1": ("bc1", 8),
    b"DXT2": ("bc2", 16),
    b"DXT3": ("bc2", 16),
    b"DXT4": ("bc3", 16),
    b"DXT5": ("bc3", 16),
    b"ATI1": ("bc4", 8),
    b"BC4U": ("bc4", 8),
    b"ATI2": ("bc5", 16),
    b"BC5U": ("bc5", 16),
}


class ImageTextureLoader(AssetLoader):
    """Загрузчик растровых изображений и DDS-текстур."""

    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".dds")

    def load(self, context: LoadContext) -> TextureAsset:
        """Прочитать метаданные изображения.

        Поддерживаемые параметры импорта:
            decode (bool): Попытаться декодировать пиксели (по умолчанию
                ``True`` — если доступна библиотека декодирования).
        """
        info = probe_image(context.path)
        pixels = None
        if bool(context.option("decode", True)):
            pixels = _try_decode(context.path)
        asset = TextureAsset(
            context.path,
            info["width"],
            info["height"],
            channels=info.get("channels", 4),
            pixel_format=info.get("pixel_format", "rgba8"),
            bit_depth=info.get("bit_depth", 8),
            mip_levels=info.get("mip_levels", 1),
            compressed=info.get("compressed", False),
            pixels=pixels,
        )
        return asset


def probe_image(path: Path) -> dict[str, object]:
    """Определить параметры изображения по сигнатуре и заголовку.

    Args:
        path: Путь к файлу изображения.

    Returns:
        Словарь с ключами ``width``, ``height``, ``channels``,
        ``pixel_format``, ``bit_depth``, ``mip_levels``, ``compressed``.

    Raises:
        ResourceFormatError: Если формат не распознан или заголовок повреждён.
    """
    try:
        with path.open("rb") as stream:
            head = stream.read(32)
            stream.seek(0)
            if head.startswith(PNG_SIGNATURE):
                return _probe_png(path)
            if head.startswith(JPEG_SIGNATURE):
                return _probe_jpeg(path)
            if head.startswith(BMP_SIGNATURE):
                return _probe_bmp(path)
            if head.startswith(DDS_SIGNATURE):
                return _probe_dds(path)
            if head.startswith(GIF_SIGNATURES):
                return _probe_gif(path)
    except OSError as exc:
        raise ResourceFormatError(f"Не удалось прочитать '{path}': {exc}") from exc
    raise ResourceFormatError(f"'{path}': неизвестный формат изображения")


def _probe_png(path: Path) -> dict[str, object]:
    """Разобрать IHDR-заголовок PNG."""
    with path.open("rb") as stream:
        header = stream.read(8)
        if header != PNG_SIGNATURE:
            raise ResourceFormatError(f"'{path}': повреждён заголовок PNG")
        length, chunk_type = struct.unpack(">I4s", stream.read(8))
        if chunk_type != b"IHDR" or length < 13:
            raise ResourceFormatError(f"'{path}': отсутствует блок IHDR")
        data = stream.read(13)
    width, height, bit_depth, color_type = struct.unpack(">IIBB", data[:10])
    channels = _PNG_CHANNELS.get(color_type)
    if channels is None:
        raise ResourceFormatError(
            f"'{path}': неизвестный тип цвета PNG ({color_type})"
        )
    interlace = data[12]
    return {
        "width": int(width),
        "height": int(height),
        "channels": channels,
        "bit_depth": int(bit_depth),
        "pixel_format": f"png{bit_depth * channels}",
        "mip_levels": 1,
        "compressed": False,
        "interlaced": bool(interlace),
    }


def _probe_jpeg(path: Path) -> dict[str, object]:
    """Пройти по маркерам JPEG до кадра SOFn и прочитать его заголовок."""
    with path.open("rb") as stream:
        if stream.read(2) != JPEG_SIGNATURE:
            raise ResourceFormatError(f"'{path}': повреждён заголовок JPEG")
        while True:
            marker_prefix = stream.read(1)
            if not marker_prefix:
                break
            if marker_prefix != b"\xff":
                continue
            marker = stream.read(1)
            while marker == b"\xff":
                marker = stream.read(1)
            if marker in (b"\xd8", b"\xd9") or marker.startswith(b"\xd0"):
                continue
            size_bytes = stream.read(2)
            if len(size_bytes) < 2:
                break
            segment_size = struct.unpack(">H", size_bytes)[0]
            if marker[0] in range(0xC0, 0xD0) and marker not in (
                b"\xc4",
                b"\xc8",
                b"\xcc",
            ):
                data = stream.read(5)
                bit_depth = data[0]
                height, width = struct.unpack(">HH", data[1:5])
                channels = 3 if marker[0] in (0xC0, 0xC1, 0xC2) else 4
                return {
                    "width": int(width),
                    "height": int(height),
                    "channels": channels,
                    "bit_depth": int(bit_depth),
                    "pixel_format": f"jpeg{bit_depth * channels}",
                    "mip_levels": 1,
                    "compressed": False,
                }
            stream.seek(segment_size - 2, 1)
    raise ResourceFormatError(f"'{path}': не найден заголовок кадра JPEG")


def _probe_bmp(path: Path) -> dict[str, object]:
    """Прочитать BITMAPINFOHEADER."""
    with path.open("rb") as stream:
        header = stream.read(30)
    if len(header) < 30 or not header.startswith(BMP_SIGNATURE):
        raise ResourceFormatError(f"'{path}': повреждён заголовок BMP")
    width, height = struct.unpack("<ii", header[18:26])
    bits_per_pixel = struct.unpack("<H", header[28:30])[0]
    channels = max(1, bits_per_pixel // 8)
    return {
        "width": abs(int(width)),
        "height": abs(int(height)),
        "channels": channels,
        "bit_depth": bits_per_pixel // channels if channels else bits_per_pixel,
        "pixel_format": f"bmp{bits_per_pixel}",
        "mip_levels": 1,
        "compressed": False,
    }


def _probe_dds(path: Path) -> dict[str, object]:
    """Прочитать заголовок DDS: размеры, мипы и блочный формат."""
    with path.open("rb") as stream:
        header = stream.read(128)
    if len(header) < 128 or not header.startswith(DDS_SIGNATURE):
        raise ResourceFormatError(f"'{path}': повреждён заголовок DDS")
    height, width = struct.unpack("<II", header[12:20])
    mip_levels = struct.unpack("<I", header[28:32])[0]
    four_cc = header[84:88]
    if four_cc in _DDS_BC_FORMATS:
        pixel_format, _block_bytes = _DDS_BC_FORMATS[four_cc]
        compressed = True
        channels = 3
    else:
        pixel_format = "rgba8"
        compressed = False
        channels = 4
    return {
        "width": int(width),
        "height": int(height),
        "channels": channels,
        "bit_depth": 8,
        "pixel_format": pixel_format,
        "mip_levels": max(1, int(mip_levels)),
        "compressed": compressed,
    }


def _probe_gif(path: Path) -> dict[str, object]:
    """Прочитать логический дескриптор экрана GIF."""
    with path.open("rb") as stream:
        header = stream.read(10)
    if len(header) < 10:
        raise ResourceFormatError(f"'{path}': повреждён заголовок GIF")
    width, height = struct.unpack("<HH", header[6:10])
    return {
        "width": int(width),
        "height": int(height),
        "channels": 4,
        "bit_depth": 8,
        "pixel_format": "gif8",
        "mip_levels": 1,
        "compressed": False,
    }


def _try_decode(path: Path) -> np.ndarray | None:
    """Попытаться декодировать пиксели, если доступна библиотека.

    Returns:
        Массив ``(H, W, C)`` или ``None``, если декодирование недоступно.
    """
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            data = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001 - декодирование не обязательно
        _log.warning("Не удалось декодировать '%s': %s", path.name, exc)
        return None
    return data
