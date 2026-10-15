"""Скриншоты: чтение кадрового буфера и запись PNG без внешних зависимостей.

Функции нужны для превью ассетов, golden-тестов рендера и отчётов: они
сохраняют результат кадра в файл, который можно сравнить или показать.
PNG пишется своим минимальным кодировщиком (``zlib`` + ``struct``), поэтому
модуль работает и в плеере, где нет ни Qt, ни Pillow.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from engine.rendering.errors import RenderError

__all__ = ["capture_framebuffer", "read_framebuffer", "save_png"]

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_framebuffer(width: int, height: int) -> np.ndarray:
    """Прочитать текущий кадровый буфер.

    Args:
        width: Ширина области чтения.
        height: Высота области чтения.

    Returns:
        Массив ``(height, width, 4)`` uint8 в порядке OpenGL (снизу вверх).

    Raises:
        RenderError: Если чтение не удалось.
    """
    from OpenGL import GL

    if width <= 0 or height <= 0:
        raise RenderError(f"Некорректный размер кадра: {width}x{height}")
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    raw = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    if raw is None:
        raise RenderError("glReadPixels вернул пустой результат")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)


def capture_framebuffer(
    width: int, height: int, *, flip_vertical: bool = True
) -> np.ndarray:
    """Прочитать кадровый буфер и перевернуть его в привычный порядок строк.

    Args:
        width: Ширина.
        height: Высота.
        flip_vertical: Переворачивать ли по вертикали (OpenGL отдаёт снизу вверх).

    Returns:
        Массив ``(height, width, 4)`` uint8 (сверху вниз при ``flip_vertical``).
    """
    pixels = read_framebuffer(width, height)
    return pixels[::-1].copy() if flip_vertical else pixels


def save_png(
    path: Path | str, pixels: np.ndarray, *, flip_vertical: bool = False
) -> Path:
    """Сохранить массив пикселей в PNG-файл.

    Args:
        path: Путь к файлу.
        pixels: Массив ``(H, W, 3)`` или ``(H, W, 4)`` uint8.
        flip_vertical: Перевернуть изображение перед записью.

    Returns:
        Путь к записанному файлу.

    Raises:
        RenderError: Если форма массива не поддерживается.
    """
    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise RenderError(f"Ожидался массив (H, W, 3|4), получено {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if flip_vertical:
        array = array[::-1]

    height, width, channels = array.shape
    color_type = 6 if channels == 4 else 2
    # Каждая строка PNG предваряется байтом фильтра (0 — без фильтрации).
    rows = b"".join(b"\x00" + array[y].tobytes() for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)

    payload = (
        _PNG_SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(rows, level=6))
        + _chunk(b"IEND", b"")
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def _chunk(tag: bytes, data: bytes) -> bytes:
    """Собрать PNG-чанк: длина, тип, данные, CRC."""
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )
