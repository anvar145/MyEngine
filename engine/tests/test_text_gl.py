"""Рисование текста на настоящем GL-контексте (offscreen)."""

# ruff: noqa: F811 - фикстуры pytest подставляются по имени аргумента

from __future__ import annotations

import numpy as np
import pytest
from test_rendering_gl import (  # noqa: F401 - фикстуры pytest
    camera,
    gl_context,
    render_to_target,
    renderer,
)

from engine.rendering.text import DEFAULT_FONT, TextBatch, TextPass, get_atlas

pytestmark = pytest.mark.skipif(not DEFAULT_FONT.is_file(), reason="нет шрифта")


def _render(renderer, camera, **kwargs):
    atlas = get_atlas("", 32, charset="A")
    batch = TextBatch()
    kwargs.setdefault("billboard", True)
    batch.add_text(atlas, "A", (0.0, 0.0, 0.0), 0.05, **kwargs)
    renderer.add_pass(TextPass(batch))
    target = render_to_target(renderer, camera, 64, 64)
    renderer.end_frame(target=target)
    pixels = target.read_pixels()
    calls = batch.last_draw_calls
    batch.dispose()
    atlas.dispose()
    return pixels, calls


def _lit(pixels) -> int:
    return int((pixels[:, :, :3].astype(int).sum(axis=2) > 120).sum())


def test_glyph_is_drawn_with_one_draw_call(renderer, camera) -> None:
    pixels, calls = _render(renderer, camera)
    assert _lit(pixels) > 20
    assert calls == 1


def test_zero_alpha_draws_nothing(renderer, camera) -> None:
    pixels, _ = _render(renderer, camera, alpha=0.0)
    assert _lit(pixels) == 0


def test_red_text_is_red(renderer, camera) -> None:
    pixels, _ = _render(renderer, camera, color=(1.0, 0.0, 0.0))
    data = pixels[:, :, :3].astype(int)
    mask = data[:, :, 0] > 150
    assert mask.sum() > 20
    assert np.all(data[mask][:, 0] > data[mask][:, 1] + 100)
    assert np.all(data[mask][:, 0] > data[mask][:, 2] + 100)
