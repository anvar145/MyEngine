"""Цветовые LUT 16x16x16 в виде полосы 256x16 (A10b).

Раскладка совпадает с ``apply_lut`` в ``post/tonemap.frag``: 16 срезов по
синему каналу идут вдоль X, внутри среза красный растёт по X, зелёный — по
строкам массива (строка 0 соответствует ``g = 0``). Только PNG; ``.cube`` не
поддерживается.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

__all__ = ["LUT_SIZE", "identity_lut", "make_lut", "save_lut_png"]

LUT_SIZE = 16


def make_lut(
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Построить LUT-полосу ``(16, 256, 3)`` uint8.

    Args:
        transform: Функция ``(N, 3) float32 в [0, 1] -> (N, 3)`` для цветов
            узлов решётки (``None`` — тождественная).

    Returns:
        Массив пикселей полосы.
    """
    axis = np.arange(LUT_SIZE, dtype=np.float32) / (LUT_SIZE - 1)
    blue, green, red = np.meshgrid(axis, axis, axis, indexing="ij")
    # Индексы: [срез b, строка g, столбец r].
    colors = np.stack([red, green, blue], axis=-1).reshape(-1, 3)
    if transform is not None:
        colors = np.clip(np.asarray(transform(colors), dtype=np.float32), 0.0, 1.0)
    grid = colors.reshape(LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)
    strip = grid.transpose(1, 0, 2, 3).reshape(LUT_SIZE, LUT_SIZE * LUT_SIZE, 3)
    return np.round(strip * 255.0).astype(np.uint8)


def identity_lut() -> np.ndarray:
    """Тождественная LUT (цвет не меняется)."""
    return make_lut()


def save_lut_png(path: str | Path, strip: np.ndarray) -> Path:
    """Записать полосу в PNG (RGB, без сжатия цвета)."""
    from PIL import Image

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip, "RGB").save(target)
    return target
