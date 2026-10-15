"""Рендеринг текста: атлас глифов (Pillow), раскладка, пакет и проход кадра."""

from __future__ import annotations

from engine.rendering.text.atlas import (
    DEFAULT_CHARSET,
    DEFAULT_FONT,
    GlyphMetrics,
    TextAtlas,
    clear_cache,
    get_atlas,
    resolve_font_path,
)
from engine.rendering.text.batch import TextBatch
from engine.rendering.text.layout import TextLayout, layout_text
from engine.rendering.text.pass_ import TextPass
from engine.rendering.text.scene import fill_text_batch

__all__ = [
    "DEFAULT_CHARSET",
    "DEFAULT_FONT",
    "GlyphMetrics",
    "TextAtlas",
    "TextBatch",
    "TextLayout",
    "TextPass",
    "clear_cache",
    "fill_text_batch",
    "get_atlas",
    "layout_text",
    "resolve_font_path",
]
