"""Динамическое разрешение: просадка под нагрузкой и возврат масштаба (A10b)."""

from __future__ import annotations

from engine.rendering.dynamic_resolution import DynamicResolution
from engine.rendering.post import tier_render_scale
from engine.rendering.settings import QualityTier


def _feed(controller: DynamicResolution, ms: float, frames: int) -> None:
    for _ in range(frames):
        controller.update(ms)


def test_scale_drops_under_load_then_recovers() -> None:
    controller = DynamicResolution(tier_render_scale(QualityTier.MEDIUM))
    start = controller.scale
    _feed(controller, 30.0, 60)
    lowered = controller.scale
    assert lowered < start
    assert lowered >= controller.hints.floor

    _feed(controller, 6.0, 2000)
    assert controller.scale > lowered
    assert controller.scale == controller.hints.ceiling
