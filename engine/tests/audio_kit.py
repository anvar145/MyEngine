"""Генерация WAV для тестов аудио (без звуковых файлов в репозитории)."""

from __future__ import annotations

import struct

import numpy as np


def make_wav(
    seconds: float = 0.1,
    *,
    rate: int = 44100,
    channels: int = 1,
    bits: int = 16,
    value: float = 0.5,
    float_format: bool = False,
) -> bytes:
    """WAV с постоянным значением ``value`` (−1…1) во всех кадрах."""
    frames = round(seconds * rate)
    count = frames * channels
    if float_format:
        tag = 3
        raw = np.full(count, value, dtype="<f4" if bits == 32 else "<f8").tobytes()
    elif bits == 8:
        tag = 1
        raw = np.full(count, int(value * 127) + 128, dtype=np.uint8).tobytes()
    elif bits == 16:
        tag = 1
        raw = np.full(count, int(value * 32767), dtype="<i2").tobytes()
    elif bits == 24:
        tag = 1
        v = int(value * 8388607) & 0xFFFFFF
        raw = (v.to_bytes(3, "little")) * count
    elif bits == 32:
        tag = 1
        raw = np.full(count, int(value * 2147483647), dtype="<i4").tobytes()
    else:
        raise ValueError(bits)
    width = bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * channels * width,
                      channels * width, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
    body += b"data" + struct.pack("<I", len(raw)) + raw
    return b"RIFF" + struct.pack("<I", len(body)) + body
