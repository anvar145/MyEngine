"""ScreenFx и PostSettings: чистая логика без GL (A10a)."""

from __future__ import annotations

import pytest

from engine.core.math3d import Vector3
from engine.core.screen_fx import ScreenFx
from engine.rendering.post import PostSettings, tier_render_scale
from engine.rendering.settings import QualityTier


def test_fade_to_interpolates_linearly() -> None:
    fx = ScreenFx()
    fx.fade_to(Vector3(0.0, 0.0, 0.0), 1.0, 2.0)
    assert fx.fade_alpha == 0.0 and fx.is_fading
    fx.update(1.0)
    assert fx.fade_alpha == pytest.approx(0.5)
    fx.update(5.0)
    assert fx.fade_alpha == 1.0 and not fx.is_fading


def test_fade_instant_and_back() -> None:
    fx = ScreenFx()
    fx.fade_to((1, 1, 1), 1.0, 0.0)
    assert fx.fade_alpha == 1.0 and fx.active
    fx.fade_to((1, 1, 1), 0.0, 1.0)
    fx.update(0.5)
    assert fx.fade_alpha == pytest.approx(0.5)
    fx.update(0.5)
    assert fx.fade_alpha == 0.0 and not fx.active


def test_fade_color_is_clamped_and_accepts_vectors() -> None:
    fx = ScreenFx()
    fx.fade_to(Vector3(2.0, -1.0, 0.5), 1.0, 0.0)
    assert fx.fade_color == (1.0, 0.0, 0.5)


def test_flash_decays_and_ends() -> None:
    fx = ScreenFx()
    fx.flash((1.0, 1.0, 1.0), 0.8, 1.0)
    assert fx.flash_alpha == pytest.approx(0.8)
    fx.update(0.5)
    assert fx.flash_alpha == pytest.approx(0.2)
    fx.update(0.6)
    assert fx.flash_alpha == 0.0 and not fx.active


def test_clear_resets_everything() -> None:
    fx = ScreenFx()
    fx.fade_to((0, 0, 0), 1.0, 0.0)
    fx.flash()
    fx.clear()
    assert not fx.active


def test_post_settings_clamp_and_diff() -> None:
    settings = PostSettings(tonemap="bogus", exposure=100.0, fxaa=1)
    assert settings.tonemap == "linear" and settings.exposure == 8.0 and settings.fxaa is True
    assert PostSettings().to_dict() == {}
    assert PostSettings(tonemap="aces").to_dict() == {"tonemap": "aces"}


def test_post_settings_round_trip_over_tier_base() -> None:
    base = PostSettings.for_quality(QualityTier.MEDIUM)
    assert (base.tonemap, base.fxaa) == ("aces", True)
    custom = PostSettings.from_dict({"tonemap": "reinhard", "fxaa": False}, base)
    again = PostSettings.from_dict(custom.to_dict(base), base)
    assert again == custom
    assert PostSettings.for_quality(QualityTier.LOW) == PostSettings()


def test_set_value_parses_strings() -> None:
    settings = PostSettings()
    settings.set_value("fxaa", "true")
    settings.set_value("exposure", "1.5")
    assert settings.fxaa and settings.exposure == 1.5
    with pytest.raises(KeyError):
        settings.set_value("motion_blur", 1)


def test_tier_render_scale_table_and_hints() -> None:
    low = tier_render_scale(QualityTier.LOW)
    assert (low.scale, low.floor, low.ceiling, low.dynamic) == (0.5, 0.5, 0.75, True)
    medium = tier_render_scale(QualityTier.MEDIUM)
    assert (medium.scale, medium.floor, medium.ceiling) == (0.75, 0.5, 1.0)
    high = tier_render_scale(QualityTier.HIGH)
    assert (high.scale, high.dynamic) == (1.0, False)
    hinted = tier_render_scale(QualityTier.HIGH, hint=0.75, dynamic=True)
    assert hinted.scale == 0.75 and hinted.dynamic and hinted.floor <= 0.75
