"""Минимальный PNG-кодек на ``zlib`` + numpy (без Pillow).

Декодер понимает все цветовые типы и глубины 1–16 бит без чересстрочной
развёртки; результат всегда 8-битный. Кодер пишет 8-битный RGBA/RGB/серый PNG
(нужен для передачи пикселей внешнему BC-компрессору и в тестах).
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from engine.core.resources.errors import AssetImportError

__all__ = ["decode_png", "encode_png"]

SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SAMPLES = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_VALID_DEPTHS = {
    0: (1, 2, 4, 8, 16),
    2: (8, 16),
    3: (1, 2, 4, 8),
    4: (8, 16),
    6: (8, 16),
}


def decode_png(source: bytes | Path) -> np.ndarray:
    """Декодировать PNG в массив ``(H, W, C)`` типа ``uint8``.

    Число каналов ``C``: 1 (серый), 2 (серый + альфа), 3 (RGB), 4 (RGBA).
    Палитровые изображения разворачиваются в RGB/RGBA, прозрачность ``tRNS``
    превращается в альфа-канал, 16 бит усекаются до старшего байта.

    Args:
        source: Путь к файлу или его содержимое.

    Raises:
        AssetImportError: Если файл повреждён, обрезан или чересстрочный.
    """
    if isinstance(source, Path):
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise AssetImportError(f"Не удалось прочитать '{source}': {exc}") from exc
    else:
        data = bytes(source)
    if not data.startswith(SIGNATURE):
        raise AssetImportError("Это не PNG: неверная сигнатура")

    header = None
    palette = trns = None
    idat = bytearray()
    position = len(SIGNATURE)
    while position + 8 <= len(data):
        length, kind = struct.unpack_from(">I4s", data, position)
        body = data[position + 8 : position + 8 + length]
        if len(body) < length:
            raise AssetImportError("PNG обрезан внутри блока")
        position += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body[:13])
        elif kind == b"PLTE":
            palette = np.frombuffer(body, dtype=np.uint8)[: len(body) // 3 * 3]
            palette = palette.reshape(-1, 3)
        elif kind == b"tRNS":
            trns = bytes(body)
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
    if header is None:
        raise AssetImportError("В PNG нет блока IHDR")
    width, height, depth, color_type, _, _, interlace = header
    if color_type not in _SAMPLES or depth not in _VALID_DEPTHS[color_type]:
        raise AssetImportError(f"Недопустимый режим PNG: тип {color_type}, {depth} бит")
    if width == 0 or height == 0:
        raise AssetImportError("PNG нулевого размера")
    if interlace:
        raise AssetImportError("Чересстрочные (Adam7) PNG не поддержаны")
    if color_type == 3 and palette is None:
        raise AssetImportError("У палитрового PNG нет блока PLTE")
    if not idat:
        raise AssetImportError("В PNG нет данных изображения")

    samples = _SAMPLES[color_type]
    bits_per_pixel = samples * depth
    stride = (width * bits_per_pixel + 7) // 8
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise AssetImportError(f"Повреждены данные PNG: {exc}") from exc
    if len(raw) < height * (stride + 1):
        raise AssetImportError("PNG обрезан: данных меньше, чем нужно")

    rows = _unfilter(raw, height, stride, max(1, bits_per_pixel // 8))
    values = _unpack(rows, width, height, samples, depth)
    return _to_rgba8(values, color_type, depth, palette, trns)


def encode_png(pixels: np.ndarray) -> bytes:
    """Закодировать массив ``(H, W, C)`` (C = 1, 3 или 4) в 8-битный PNG."""
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    height, width, channels = array.shape
    color_type = {1: 0, 3: 2, 4: 6}.get(channels)
    if color_type is None:
        raise AssetImportError(f"PNG: неподдерживаемое число каналов {channels}")
    rows = np.zeros((height, width * channels + 1), dtype=np.uint8)
    rows[:, 1:] = array.reshape(height, -1)

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    return (
        SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + chunk(b"IEND", b"")
    )


def _unfilter(raw: bytes, height: int, stride: int, bpp: int) -> np.ndarray:
    """Снять построчные фильтры PNG; возвращает ``(height, stride)``."""
    table = np.frombuffer(raw, dtype=np.uint8, count=height * (stride + 1))
    table = table.reshape(height, stride + 1)
    out = np.zeros((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    for y in range(height):
        kind = int(table[y, 0])
        line = table[y, 1:]
        if kind == 0:
            current = line
        elif kind == 1:
            current = line.reshape(-1, bpp).cumsum(axis=0, dtype=np.uint8).reshape(-1)
        elif kind == 2:
            current = line + previous
        elif kind == 3:
            current = _unfilter_average(line, previous, bpp)
        elif kind == 4:
            current = _unfilter_paeth(line, previous, bpp)
        else:
            raise AssetImportError(f"PNG: неизвестный фильтр строки {kind}")
        out[y] = current
        previous = out[y]
    return out


def _unfilter_average(line: np.ndarray, previous: np.ndarray, bpp: int) -> np.ndarray:
    """Фильтр Average: зависит от уже восстановленного левого пикселя."""
    src, up = line.tolist(), previous.tolist()
    cur = [0] * len(src)
    for i, value in enumerate(src):
        left = cur[i - bpp] if i >= bpp else 0
        cur[i] = (value + ((left + up[i]) >> 1)) & 0xFF
    return np.array(cur, dtype=np.uint8)


def _unfilter_paeth(line: np.ndarray, previous: np.ndarray, bpp: int) -> np.ndarray:
    """Фильтр Paeth."""
    src, up = line.tolist(), previous.tolist()
    cur = [0] * len(src)
    for i, value in enumerate(src):
        if i >= bpp:
            a, c = cur[i - bpp], up[i - bpp]
        else:
            a = c = 0
        b = up[i]
        p = a + b - c
        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
        if pa <= pb and pa <= pc:
            predictor = a
        elif pb <= pc:
            predictor = b
        else:
            predictor = c
        cur[i] = (value + predictor) & 0xFF
    return np.array(cur, dtype=np.uint8)


def _unpack(
    rows: np.ndarray, width: int, height: int, samples: int, depth: int
) -> np.ndarray:
    """Развернуть байты строк в массив отсчётов ``(H, W, samples)``."""
    if depth == 8:
        return rows[:, : width * samples].reshape(height, width, samples)
    if depth == 16:
        wide = rows[:, : width * samples * 2].reshape(height, -1, 2).astype(np.uint16)
        return ((wide[:, :, 0] << 8) | wide[:, :, 1]).reshape(height, width, samples)
    bits = np.unpackbits(rows, axis=1).reshape(height, -1, depth)
    weights = (1 << np.arange(depth - 1, -1, -1)).astype(np.uint8)
    packed = (bits * weights).sum(axis=2, dtype=np.uint16).astype(np.uint8)
    return packed[:, : width * samples].reshape(height, width, samples)


def _to_rgba8(
    values: np.ndarray,
    color_type: int,
    depth: int,
    palette: np.ndarray | None,
    trns: bytes | None,
) -> np.ndarray:
    """Привести отсчёты к 8 битам и развернуть палитру/прозрачность."""
    if color_type == 3:
        assert palette is not None
        indices = values[:, :, 0].astype(np.intp)
        if indices.max() >= len(palette):
            raise AssetImportError("PNG: индекс палитры вне диапазона")
        rgb = palette[indices]
        if trns:
            alpha = np.full(len(palette), 255, dtype=np.uint8)
            alpha[: len(trns)] = np.frombuffer(trns, dtype=np.uint8)[: len(palette)]
            return np.concatenate([rgb, alpha[indices][:, :, None]], axis=2)
        return rgb

    colour_key = None
    if color_type in (0, 2) and trns:
        need = 2 * _SAMPLES[color_type]
        if len(trns) >= need:
            colour_key = np.array(struct.unpack(f">{need // 2}H", trns[:need]))
    transparent = None
    if colour_key is not None:
        transparent = (values.astype(np.int32) == colour_key).all(axis=2)

    if depth == 16:
        values = (values >> 8).astype(np.uint8)
    elif depth < 8 and color_type == 0:
        values = (values.astype(np.uint16) * 255 // ((1 << depth) - 1)).astype(np.uint8)
    if transparent is not None:
        alpha = np.where(transparent, 0, 255).astype(np.uint8)[:, :, None]
        values = np.concatenate([values, alpha], axis=2)
    return np.ascontiguousarray(values, dtype=np.uint8)
