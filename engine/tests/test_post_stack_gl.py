"""Tonemap, FXAA и fade/flash в пост-пути рендерера (A10a, offscreen GL)."""

# ruff: noqa: F811 - фикстуры pytest подставляются по имени аргумента

from __future__ import annotations

import numpy as np
import pytest
from test_rendering_gl import (  # noqa: F401 - фикстуры pytest
    camera,
    gl_context,
    renderer,
)

from engine.core.screen_fx import ScreenFx
from engine.rendering import RenderTarget
from engine.rendering.post import PostSettings

SIZE = (64, 48)


@pytest.fixture
def hdr_renderer(renderer):
    if not renderer.capabilities.supports_hdr_target:
        pytest.skip("RGBA16F недоступен")
    renderer.hdr_enabled = True
    return renderer


def _frame(renderer, camera, clear, size=SIZE):
    target = RenderTarget("final", *size)
    renderer.begin_frame(camera, size, clear_color=clear, target=target, post=True)
    renderer.end_frame(target=target)
    pixels = target.read_pixels()
    target.dispose()
    return pixels


def _pixel(pixels, x=10, y=10):
    return tuple(int(v) for v in pixels[y, x, :3])


def test_aces_compresses_values_above_one(hdr_renderer, camera) -> None:
    clear = (3.0, 3.0, 3.0, 1.0)
    linear = _pixel(_frame(hdr_renderer, camera, clear))
    hdr_renderer.post_settings = PostSettings(tonemap="aces")
    aces = _pixel(_frame(hdr_renderer, camera, clear))
    assert linear == (255, 255, 255)
    assert aces[0] >= 240  # ACES сводит яркий белый к ~1, но не выше
    # Яркость 0.5 в ACES заметно темнее, чем без tonemap.
    mid = (0.5, 0.5, 0.5, 1.0)
    hdr_renderer.post_settings = PostSettings(tonemap="linear")
    plain = _pixel(_frame(hdr_renderer, camera, mid))
    hdr_renderer.post_settings = PostSettings(tonemap="aces")
    mapped = _pixel(_frame(hdr_renderer, camera, mid))
    assert mapped != plain


def test_reinhard_differs_from_linear(hdr_renderer, camera) -> None:
    clear = (0.8, 0.8, 0.8, 1.0)
    linear = _pixel(_frame(hdr_renderer, camera, clear))
    hdr_renderer.post_settings = PostSettings(tonemap="reinhard")
    reinhard = _pixel(_frame(hdr_renderer, camera, clear))
    assert reinhard[0] < linear[0]


def test_exposure_scales_linear_output(hdr_renderer, camera) -> None:
    clear = (0.2, 0.2, 0.2, 1.0)
    base = _pixel(_frame(hdr_renderer, camera, clear))
    hdr_renderer.post_settings = PostSettings(exposure=0.5)
    dark = _pixel(_frame(hdr_renderer, camera, clear))
    assert dark[0] < base[0]


def test_fxaa_keeps_flat_color_and_softens_edges(hdr_renderer, camera) -> None:
    from engine.core.math3d import Vector3, compose_matrix
    from engine.core.resources import cube_mesh
    from engine.rendering import GpuMesh

    mesh = GpuMesh.from_mesh_asset(cube_mesh(1.5))
    material = hdr_renderer.create_material("m", base_color=Vector3(1.0, 1.0, 1.0))
    size = (96, 72)

    def draw():
        target = RenderTarget("final", *size)
        hdr_renderer.begin_frame(
            camera, size, clear_color=(0.0, 0.0, 0.0, 1.0), target=target, post=True
        )
        hdr_renderer.submit(
            mesh, material, compose_matrix(rotation=None, position=Vector3(0, 0, 0))
        )
        hdr_renderer.end_frame(target=target)
        pixels = target.read_pixels()
        target.dispose()
        return pixels

    off = draw().astype(int)
    hdr_renderer.post_settings = PostSettings(fxaa=True)
    on = draw().astype(int)
    assert np.abs(on - off).max() > 0  # края изменены
    assert on[0, 0, :3].tolist() == off[0, 0, :3].tolist()  # плоский фон цел


def test_fade_covers_frame_and_fxaa_does_not_touch_it(hdr_renderer, camera) -> None:
    fx = ScreenFx()
    fx.fade_to((1.0, 0.0, 0.0), 1.0, 0.0)
    hdr_renderer.screen_fx = fx
    hdr_renderer.post_settings = PostSettings(fxaa=True)
    pixels = _frame(hdr_renderer, camera, (0.0, 0.0, 1.0, 1.0))
    assert _pixel(pixels) == (255, 0, 0)
    assert _pixel(pixels, 40, 30) == (255, 0, 0)


def test_half_fade_blends_with_frame(hdr_renderer, camera) -> None:
    fx = ScreenFx()
    fx.fade_to((0.0, 0.0, 0.0), 0.5, 0.0)
    hdr_renderer.screen_fx = fx
    pixels = _frame(hdr_renderer, camera, (1.0, 1.0, 1.0, 1.0))
    assert abs(_pixel(pixels)[0] - 128) <= 2


def test_flash_draws_over_fade(hdr_renderer, camera) -> None:
    fx = ScreenFx()
    fx.flash((0.0, 1.0, 0.0), 1.0, 1.0)
    hdr_renderer.screen_fx = fx
    pixels = _frame(hdr_renderer, camera, (0.0, 0.0, 0.0, 1.0))
    assert _pixel(pixels) == (0, 255, 0)


def test_inactive_screen_fx_changes_nothing(hdr_renderer, camera) -> None:
    clear = (0.3, 0.4, 0.5, 1.0)
    without = _frame(hdr_renderer, camera, clear)
    hdr_renderer.screen_fx = ScreenFx()
    with_fx = _frame(hdr_renderer, camera, clear)
    assert np.array_equal(without, with_fx)
