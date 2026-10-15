"""Тесты темы: токены QSS, иконки, анимации."""

from __future__ import annotations

import re

from editor import anim
from editor.theme import PALETTE, icon, load_stylesheet


def test_stylesheet_has_no_unresolved_tokens() -> None:
    assert not re.search(r"@[a-z_]+", load_stylesheet())


def test_palette_has_no_blue_dominant_colors() -> None:
    for name, value in PALETTE.items():
        r, g, b = (int(value[i : i + 2], 16) for i in (1, 3, 5))
        assert not (b > r + 12 and b >= g), name


def test_motion_disabled_offscreen(monkeypatch) -> None:
    # Платформа подменяется: набор тестов может идти и под «windows»
    # (она нужна GL-тестам движка), и под offscreen.
    monkeypatch.delenv("MYENGINE_REDUCE_MOTION", raising=False)
    monkeypatch.setattr(
        anim.QGuiApplication, "platformName", staticmethod(lambda: "offscreen")
    )
    assert anim.motion_enabled() is False


def test_motion_disabled_by_environment(monkeypatch) -> None:
    monkeypatch.setenv("MYENGINE_REDUCE_MOTION", "1")
    assert anim.motion_enabled() is False


def test_icon_cache_by_name_and_color(qapp) -> None:
    assert not icon("folder").isNull()
    assert not icon("folder", "accent").isNull()
