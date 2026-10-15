"""Контроллер динамического разрешения и подсказки проекта (A10a)."""

from __future__ import annotations

import json

from engine.rendering.dynamic_resolution import (
    TARGET_FRAME_MS,
    DynamicResolution,
    read_render_hints,
)
from engine.rendering.post import tier_render_scale
from engine.rendering.settings import QualityTier


def _run(controller: DynamicResolution, ms: float, frames: int) -> list[float]:
    changes = []
    for _ in range(frames):
        new = controller.update(ms)
        if new is not None:
            changes.append(new)
    return changes


def test_scale_drops_quickly_under_load() -> None:
    controller = DynamicResolution(tier_render_scale(QualityTier.MEDIUM))
    changes = _run(controller, 30.0, 8)
    assert changes and changes[0] == 0.7
    assert controller.scale < 0.75


def test_scale_never_leaves_floor_and_ceiling() -> None:
    controller = DynamicResolution(tier_render_scale(QualityTier.MEDIUM))
    _run(controller, 60.0, 400)
    assert controller.scale == 0.5
    _run(controller, 3.0, 4000)
    assert controller.scale == 1.0


def test_scale_rises_slowly() -> None:
    controller = DynamicResolution(tier_render_scale(QualityTier.MEDIUM))
    assert _run(controller, TARGET_FRAME_MS * 0.5, 80) == []
    assert _run(controller, TARGET_FRAME_MS * 0.5, 20)


def test_stable_load_inside_dead_zone_keeps_scale() -> None:
    controller = DynamicResolution(tier_render_scale(QualityTier.MEDIUM))
    assert _run(controller, TARGET_FRAME_MS * 0.9, 1000) == []


def test_disabled_controller_does_not_move() -> None:
    controller = DynamicResolution(tier_render_scale(QualityTier.HIGH))
    assert not controller.enabled
    assert _run(controller, 100.0, 100) == []
    assert controller.scale == 1.0


def test_low_tier_ceiling_is_three_quarters() -> None:
    controller = DynamicResolution(tier_render_scale(QualityTier.LOW))
    _run(controller, 2.0, 4000)
    assert controller.scale == 0.75


def test_set_scale_widens_bounds() -> None:
    controller = DynamicResolution(tier_render_scale(QualityTier.LOW))
    controller.set_scale(0.9)
    assert controller.scale == 0.9
    assert controller.hints.ceiling >= 0.9


def test_project_hints_are_read(tmp_path) -> None:
    (tmp_path / "Game.project.json").write_text(
        json.dumps({"name": "Game", "render_scale_hint": 0.75, "dynamic_resolution": True, "hdr": True}),
        encoding="utf-8",
    )
    assert read_render_hints(tmp_path) == (0.75, True, True)


def test_project_without_hints_uses_defaults(tmp_path) -> None:
    assert read_render_hints(tmp_path) == (None, None, False)
    (tmp_path / "G.project.json").write_text("{bad", encoding="utf-8")
    assert read_render_hints(tmp_path) == (None, None, False)
    (tmp_path / "G.project.json").write_text(json.dumps({"name": "G"}), encoding="utf-8")
    assert read_render_hints(tmp_path) == (None, None, False)
