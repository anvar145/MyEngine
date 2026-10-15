"""UIPass на настоящем GL-контексте (offscreen): квады, alpha, clip."""

# ruff: noqa: F811 - фикстуры pytest подставляются по имени аргумента

from __future__ import annotations

import numpy as np
from test_rendering_gl import (  # noqa: F401 - фикстуры pytest
    camera,
    gl_context,
    renderer,
)

from engine.rendering import RenderTarget
from engine.rendering.ui import UIPass, UIScreen

BACKGROUND = (0.2, 0.0, 0.0, 1.0)


def _render(renderer, camera, fill, size=(200, 200)):
    """Рисует UI, заполненный ``fill(batch)``; возвращает пиксели сверху вниз."""
    screen = UIScreen(size)
    ui = UIPass(screen)
    fill(ui)
    renderer.add_pass(ui)
    target = RenderTarget("ui_test", *size)
    renderer.begin_frame(camera, size, clear_color=BACKGROUND, target=target)
    renderer.end_frame(target=target)
    pixels = target.read_pixels()[::-1]
    calls = ui.batch.last_draw_calls
    ui.batch.dispose()
    target.dispose()
    return pixels, calls


def _rgb(pixels, x, y):
    return tuple(int(v) for v in pixels[y, x, :3])


def test_white_quad_in_center(renderer, camera) -> None:
    def fill(ui):
        ui.batch.add_quad(50, 50, 100, 100, (1.0, 1.0, 1.0, 1.0))

    pixels, calls = _render(renderer, camera, fill)
    assert _rgb(pixels, 100, 100) == (255, 255, 255)
    assert _rgb(pixels, 5, 5) == (51, 0, 0)
    assert calls == 1


def test_second_quad_overlaps_first(renderer, camera) -> None:
    def fill(ui):
        ui.batch.add_quad(50, 50, 100, 100, (1.0, 1.0, 1.0, 1.0))
        ui.batch.add_quad(80, 80, 100, 100, (0.0, 1.0, 0.0, 1.0))

    pixels, _ = _render(renderer, camera, fill)
    assert _rgb(pixels, 100, 100) == (0, 255, 0)
    assert _rgb(pixels, 60, 60) == (255, 255, 255)


def test_alpha_blends_with_underlying(renderer, camera) -> None:
    def fill(ui):
        ui.batch.add_quad(50, 50, 100, 100, (1.0, 1.0, 1.0, 1.0))
        ui.batch.add_quad(80, 80, 100, 100, (0.0, 0.0, 1.0, 0.5))

    pixels, _ = _render(renderer, camera, fill)
    r, g, b = _rgb(pixels, 100, 100)
    assert abs(r - 128) <= 3 and abs(g - 128) <= 3 and b == 255


def test_clip_restricts_quad(renderer, camera) -> None:
    def fill(ui):
        clip = (50.0, 50.0, 40.0, 40.0)
        ui.batch.add_quad(50, 50, 100, 100, (1.0, 1.0, 1.0, 1.0), clip=clip)

    pixels, _ = _render(renderer, camera, fill)
    assert _rgb(pixels, 60, 60) == (255, 255, 255)
    assert _rgb(pixels, 120, 120) == (51, 0, 0)


def test_top_left_origin(renderer, camera) -> None:
    def fill(ui):
        ui.batch.add_quad(0, 0, 20, 20, (1.0, 1.0, 1.0, 1.0))

    pixels, _ = _render(renderer, camera, fill)
    assert _rgb(pixels, 5, 5) == (255, 255, 255)
    assert _rgb(pixels, 5, 195) == (51, 0, 0)
    assert np.all(pixels[:, :, 3] == 255)
