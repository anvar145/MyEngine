"""Раскладка текста по глифам: переносы, выравнивание, межстрочный интервал.

Чистая функция без GL. Координаты — в пикселях атласа, ось Y вверх; точка
привязки ``(0, 0)`` задаётся выравниванием (``left``/``center``/``right`` по X,
``top``/``middle``/``bottom`` по Y). Кернинга и шейпинга нет: сдвиг пера —
сумма ``advance_x``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from engine.rendering.text.atlas import TextAtlas

__all__ = ["TextLayout", "layout_text"]

#: Столбцы квада: x0, y0 (низ), x1, y1 (верх), u0, v0 (верх), u1, v1 (низ).
QUAD_COLUMNS = 8


@dataclass(frozen=True)
class TextLayout:
    """Результат раскладки.

    Attributes:
        quads: Массив ``float32`` формы ``(N, 8)`` — по строке на видимый глиф.
        width: Ширина блока в пикселях.
        height: Высота блока в пикселях.
        lines: Число строк.
    """

    quads: np.ndarray
    width: float
    height: float
    lines: int


def _line_width(atlas: TextAtlas, line: str) -> float:
    return sum(atlas.get_glyph(ord(char)).advance_x for char in line.rstrip(" "))


def _split_lines(atlas: TextAtlas, text: str, wrap_px: float) -> list[str]:
    """Разбить текст на строки по ``\n`` и по ширине переноса."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if wrap_px <= 0:
            lines.append(paragraph)
            continue
        current = ""
        for word in paragraph.split(" "):
            candidate = word if not current else f"{current} {word}"
            if _line_width(atlas, candidate) <= wrap_px:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            # Слово длиннее строки режется по символам.
            while word and _line_width(atlas, word) > wrap_px:
                cut = 1
                while (
                    cut < len(word)
                    and _line_width(atlas, word[: cut + 1]) <= wrap_px
                ):
                    cut += 1
                lines.append(word[:cut])
                word = word[cut:]
            current = word
        lines.append(current)
    return lines


@lru_cache(maxsize=1024)
def _layout_cached(
    atlas: TextAtlas,
    text: str,
    align: str,
    vertical_align: str,
    wrap_px: float,
    line_spacing: float,
) -> TextLayout:
    lines = _split_lines(atlas, text, wrap_px)
    pitch = atlas.line_height * line_spacing
    block_height = atlas.line_height + (len(lines) - 1) * pitch
    block_width = max((_line_width(atlas, line) for line in lines), default=0.0)
    top = {"top": 0.0, "middle": block_height / 2.0, "bottom": block_height}[
        vertical_align
    ]
    rows: list[tuple[float, ...]] = []
    for index, line in enumerate(lines):
        width = _line_width(atlas, line)
        pen = {"left": 0.0, "center": -width / 2.0, "right": -width}[align]
        baseline = top - atlas.ascent - index * pitch
        for char in line:
            glyph = atlas.get_glyph(ord(char))
            if glyph.width_px > 0:
                x0 = pen + glyph.bearing_x
                y1 = baseline + glyph.bearing_y
                u0, v0, u1, v1 = glyph.uv_rect
                rows.append(
                    (x0, y1 - glyph.height_px, x0 + glyph.width_px, y1, u0, v0, u1, v1)
                )
            pen += glyph.advance_x
    quads = (
        np.asarray(rows, dtype=np.float32)
        if rows
        else np.zeros((0, QUAD_COLUMNS), dtype=np.float32)
    )
    quads.setflags(write=False)
    return TextLayout(quads, float(block_width), float(block_height), len(lines))


def layout_text(
    atlas: TextAtlas,
    text: str,
    *,
    align: str = "left",
    vertical_align: str = "top",
    wrap_width: float = 0.0,
    line_spacing: float = 1.0,
) -> TextLayout:
    """Разложить текст по глифам.

    Результат кэшируется по всем аргументам: статичный текст не пересчитывается
    каждый кадр.

    Args:
        atlas: Атлас шрифта.
        text: Текст (``\n`` — новая строка).
        align: ``left``/``center``/``right``.
        vertical_align: ``top``/``middle``/``bottom``.
        wrap_width: Ширина переноса в пикселях (0 — без переноса).
        line_spacing: Множитель интервала между строками.

    Returns:
        :class:`TextLayout` (массив ``quads`` только для чтения).
    """
    return _layout_cached(
        atlas, text, align, vertical_align, float(wrap_width), float(line_spacing)
    )
