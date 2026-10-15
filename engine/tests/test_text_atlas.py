"""Атлас глифов: метрики, упаковка, кэш (без OpenGL)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.rendering.text import (
    DEFAULT_FONT,
    TextAtlas,
    clear_cache,
    get_atlas,
    layout_text,
)

pytestmark = pytest.mark.skipif(not DEFAULT_FONT.is_file(), reason="нет шрифта")


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_cache()
    yield
    clear_cache()


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def test_abc_glyphs_do_not_overlap() -> None:
    atlas = TextAtlas(DEFAULT_FONT, 32, charset="ABC")
    glyphs = [atlas.get_glyph(ord(c)) for c in "ABC"]
    for glyph in glyphs:
        assert glyph.advance_x > 0
        assert glyph.width_px > 0 and glyph.height_px > 0
    for i in range(3):
        for j in range(i + 1, 3):
            assert not _overlap(glyphs[i].uv_rect, glyphs[j].uv_rect)


def test_atlas_pixels_have_coverage() -> None:
    atlas = TextAtlas(DEFAULT_FONT, 32, charset="A")
    glyph = atlas.get_glyph(ord("A"))
    u0, v0, u1, v1 = glyph.uv_rect
    patch = atlas.pixels[
        round(v0 * atlas.height) : round(v1 * atlas.height),
        round(u0 * atlas.width) : round(u1 * atlas.width),
        3,
    ]
    assert patch.max() == 255
    assert atlas.pixels.shape == (atlas.height, atlas.width, 4)


def test_empty_charset_does_not_crash() -> None:
    atlas = TextAtlas(DEFAULT_FONT, 32, charset="")
    assert len(atlas) == 0
    glyph = atlas.get_glyph(ord("A"))
    assert glyph.width_px == 0 and glyph.advance_x > 0


def test_default_charset_has_latin_and_cyrillic() -> None:
    atlas = TextAtlas(DEFAULT_FONT, 32)
    for char in "Aa Яя Ёё 09~":
        assert ord(char) in atlas
    assert atlas.width == 512


def test_space_has_advance_but_no_raster() -> None:
    glyph = TextAtlas(DEFAULT_FONT, 32).get_glyph(ord(" "))
    assert glyph.advance_x > 0 and glyph.width_px == 0


def test_cache_returns_same_object_and_distinguishes_params() -> None:
    first = get_atlas("", 24)
    assert get_atlas("", 24) is first
    assert get_atlas(str(DEFAULT_FONT), 24) is first
    assert get_atlas("", 25) is not first
    assert get_atlas("", 24, bold=True) is not first


def test_cache_is_lru_limited() -> None:
    first = get_atlas("", 10, charset="A")
    for size in range(11, 11 + 16):
        get_atlas("", size, charset="A")
    assert get_atlas("", 10, charset="A") is not first


def test_oversized_charset_is_truncated_with_warning(caplog) -> None:
    with caplog.at_level("WARNING"):
        atlas = TextAtlas(DEFAULT_FONT, 200)
    assert atlas.width == 1024
    assert 0 < len(atlas) < 100
    assert any("обрезан" in r.message for r in caplog.records)


def test_bold_and_italic_change_pixels() -> None:
    plain = TextAtlas(DEFAULT_FONT, 32, charset="H")
    bold = TextAtlas(DEFAULT_FONT, 32, bold=True, charset="H")
    italic = TextAtlas(DEFAULT_FONT, 32, italic=True, charset="H")
    assert bold.pixels[:, :, 3].sum() > plain.pixels[:, :, 3].sum()
    assert italic.get_glyph(ord("H")).width_px > plain.get_glyph(ord("H")).width_px


def test_layout_alignment_and_wrap() -> None:
    atlas = get_atlas("", 32)
    left = layout_text(atlas, "Hello", align="left")
    center = layout_text(atlas, "Hello", align="center")
    right = layout_text(atlas, "Hello", align="right")
    assert left.quads[:, 0].min() >= 0
    assert right.quads[:, 2].max() <= 1e-3
    assert center.quads[:, 0].min() < 0 < center.quads[:, 2].max()
    wrapped = layout_text(atlas, "one two three four", wrap_width=100.0)
    assert wrapped.lines > 1
    assert wrapped.quads[:, 2].max() <= 101.0
    assert layout_text(atlas, "a\nb\nc").lines == 3
    assert layout_text(atlas, "").lines == 1
    assert len(layout_text(atlas, "").quads) == 0


def test_layout_vertical_anchor() -> None:
    atlas = get_atlas("", 32)
    top = layout_text(atlas, "A", vertical_align="top")
    middle = layout_text(atlas, "A", vertical_align="middle")
    bottom = layout_text(atlas, "A", vertical_align="bottom")
    assert top.quads[0, 3] <= 0 < bottom.quads[0, 3]
    assert np.isclose(
        middle.quads[0, 3] - top.quads[0, 3], atlas.line_height / 2, atol=1e-3
    )
