"""Bloom, виньетка и LUT в fused-проходе пост-обработки (A10b, offscreen GL)."""

# ruff: noqa: F811 - фикстуры pytest подставляются по имени аргумента

from __future__ import annotations

import numpy as np
import pytest
from test_rendering_gl import (  # noqa: F401 - фикстуры pytest
    camera,
    gl_context,
    renderer,
)

from engine.core.resources.asset_ref import AssetRef
from engine.rendering import RenderTarget
from engine.rendering.lut import identity_lut, make_lut
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


def _px(pixels, x, y):
    return np.array(pixels[y, x, :3], dtype=int)


def test_bloom_adds_light_to_bright_frame(hdr_renderer, camera) -> None:
    clear = (0.8, 0.8, 0.8, 1.0)
    hdr_renderer.post_settings = PostSettings(tonemap="aces")
    plain = _frame(hdr_renderer, camera, clear)
    hdr_renderer.post_settings = PostSettings(
        tonemap="aces", bloom=True, bloom_threshold=0.1, bloom_levels=3
    )
    glow = _frame(hdr_renderer, camera, clear)
    assert _px(glow, 32, 24).sum() > _px(plain, 32, 24).sum()


def test_bloom_below_threshold_changes_nothing(hdr_renderer, camera) -> None:
    clear = (0.2, 0.2, 0.2, 1.0)
    hdr_renderer.post_settings = PostSettings(tonemap="aces")
    plain = _frame(hdr_renderer, camera, clear)
    hdr_renderer.post_settings = PostSettings(
        tonemap="aces", bloom=True, bloom_threshold=1.0
    )
    glow = _frame(hdr_renderer, camera, clear)
    assert np.abs(_px(glow, 32, 24) - _px(plain, 32, 24)).max() <= 1


def test_bloom_cascade_is_reused_on_resize(hdr_renderer, camera) -> None:
    hdr_renderer.post_settings = PostSettings(bloom=True, bloom_threshold=0.1)
    _frame(hdr_renderer, camera, (0.9, 0.9, 0.9, 1.0), (64, 48))
    chain = hdr_renderer.post._bloom
    first = [t for pair in chain._targets for t in pair]
    _frame(hdr_renderer, camera, (0.9, 0.9, 0.9, 1.0), (96, 64))
    assert chain.level_count == 3
    assert chain._targets[0][0].width == 48
    assert [t for pair in chain._targets for t in pair] == first
    hdr_renderer.post_settings = PostSettings(bloom=False)
    _frame(hdr_renderer, camera, (0.9, 0.9, 0.9, 1.0), (96, 64))
    assert chain.level_count == 0


def test_vignette_darkens_corners_not_center(hdr_renderer, camera) -> None:
    clear = (0.6, 0.6, 0.6, 1.0)
    hdr_renderer.post_settings = PostSettings()
    plain = _frame(hdr_renderer, camera, clear)
    hdr_renderer.post_settings = PostSettings(vignette=True, vignette_intensity=0.5)
    shaded = _frame(hdr_renderer, camera, clear)
    assert _px(shaded, 0, 0).sum() < _px(plain, 0, 0).sum() * 0.7
    assert np.abs(_px(shaded, 32, 24) - _px(plain, 32, 24)).max() <= 1


def test_identity_lut_keeps_colors(hdr_renderer, camera) -> None:
    clear = (0.45, 0.3, 0.6, 1.0)
    hdr_renderer.post_settings = PostSettings()
    plain = _frame(hdr_renderer, camera, clear)
    hdr_renderer.post.set_lut_pixels(identity_lut())
    mapped = _frame(hdr_renderer, camera, clear)
    assert np.abs(_px(mapped, 32, 24) - _px(plain, 32, 24)).max() <= 3


def test_lut_intensity_blends_original_and_graded(hdr_renderer, camera) -> None:
    clear = (0.5, 0.3, 0.2, 1.0)
    hdr_renderer.post_settings = PostSettings()
    plain = _px(_frame(hdr_renderer, camera, clear), 32, 24)
    hdr_renderer.post.set_lut_pixels(make_lut(lambda c: 1.0 - c))
    hdr_renderer.post_settings = PostSettings(lut_intensity=1.0)
    full = _px(_frame(hdr_renderer, camera, clear), 32, 24)
    hdr_renderer.post_settings = PostSettings(lut_intensity=0.5)
    half = _px(_frame(hdr_renderer, camera, clear), 32, 24)
    hdr_renderer.post_settings = PostSettings(lut_intensity=0.0)
    none = _px(_frame(hdr_renderer, camera, clear), 32, 24)
    assert np.abs(full - (255 - plain)).max() <= 4
    assert np.abs(none - plain).max() <= 1
    assert np.abs(half - (plain + full) / 2).max() <= 3


def test_post_settings_roundtrip_with_lut() -> None:
    ref = AssetRef(guid="ab" * 16, path="Luts/warm.png")
    settings = PostSettings(
        tonemap="aces",
        bloom=True,
        bloom_levels=4,
        vignette=True,
        lut=ref,
        lut_intensity=0.6,
    )
    data = settings.to_dict(full=True)
    assert data["lut"] == ref.to_dict()
    restored = PostSettings.from_dict(data)
    assert restored == settings


def test_old_post_section_keeps_effects_off() -> None:
    restored = PostSettings.from_dict({"tonemap": "aces", "exposure": 1.0, "fxaa": True})
    assert not restored.bloom
    assert not restored.vignette
    assert restored.lut is None
