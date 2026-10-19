"""C1: буфер ``engine.debug`` и реестр оверлеев (без GL)."""

from __future__ import annotations

import numpy as np
import pytest

from engine import debug
from engine.debug import DebugDraw, OverlayRegistry
from engine.rendering.debug_pass import DebugPass


def test_primitives_produce_segment_pairs() -> None:
    draw = DebugDraw()
    draw.line((0, 0, 0), (1, 0, 0))
    draw.box((0, 0, 0), (1, 1, 1))
    draw.sphere((0, 0, 0), 1.0)
    draw.arrow((0, 0, 0), (0, 2, 0))
    draw.path([(0, 0, 0), (1, 0, 0), (1, 0, 1)], closed=True)
    draw.cone((0, 0, 0), (0, 0, 1), 30.0, 5.0)
    draw.circle((0, 0, 0), 2.0)
    positions, colors, depth, labels = draw.collect()
    assert positions.shape == colors.shape
    assert len(positions) % 2 == 0
    assert len(depth) == len(positions)
    assert labels == []
    # 1 + 12 + 3*24 + 5 + 3 + (24 + 4) + 24 отрезков
    assert draw.line_count == 1 + 12 + 72 + 5 + 3 + 28 + 24


def test_box_rotation_and_cone_radius() -> None:
    draw = DebugDraw()
    quarter_turn_y = (0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4))
    draw.box((0, 0, 0), (2, 1, 1), rotation=quarter_turn_y)
    positions, *_ = draw.collect()
    assert np.abs(positions[:, 2]).max() == pytest.approx(2.0, abs=1e-5)
    assert np.abs(positions[:, 0]).max() == pytest.approx(1.0, abs=1e-5)
    draw.clear()
    draw.cone((0, 0, 0), (0, 0, 1), 45.0, 3.0)
    positions, *_ = draw.collect(["sight"])
    assert positions[:, 2].max() == pytest.approx(3.0, abs=1e-5)
    assert np.abs(positions[:, 0]).max() == pytest.approx(3.0, abs=1e-4)


def test_lifetime_one_frame_and_duration() -> None:
    draw = DebugDraw()
    draw.line((0, 0, 0), (1, 0, 0))
    draw.line((0, 0, 0), (0, 1, 0), duration=0.05)
    draw.text((0, 1, 0), "hp 10")
    assert draw.line_count == 2 and draw.text_count == 1
    draw.advance(1 / 60)
    assert draw.line_count == 1 and draw.text_count == 0
    draw.advance(1 / 60)
    assert draw.line_count == 1
    draw.advance(0.05)
    assert draw.line_count == 0


def test_channels_and_depth_flags() -> None:
    draw = DebugDraw()
    draw.line((0, 0, 0), (1, 0, 0), channel="nav", depth_test=True)
    draw.line((0, 0, 0), (1, 0, 0))
    draw.text((0, 0, 0), "x", channel="nav")
    positions, _, depth, labels = draw.collect(["nav"])
    assert len(positions) == 2 and depth.all() and len(labels) == 1
    _, _, depth, labels = draw.collect(["script"])
    assert not depth.any() and labels == []


def test_disabled_buffer_ignores_calls() -> None:
    draw = DebugDraw()
    draw.enabled = False
    draw.line((0, 0, 0), (1, 0, 0))
    draw.text((0, 0, 0), "x")
    assert draw.line_count == 0 and draw.text_count == 0


def test_registry_master_switch_and_external_overlay() -> None:
    registry = OverlayRegistry()
    assert {"colliders", "nav", "bones", "sight", "paths", "script"} <= set(registry.names())
    assert registry.active_channels() == []
    registry.set_visible(True)
    assert "colliders" not in registry.active_channels()
    assert "nav" in registry.active_channels()
    registry.set_enabled("nav", False)
    assert "nav" not in registry.active_channels()
    # Коллайдеры рисует свой проход (F2): общий выключатель на них не влияет.
    registry.set_enabled("colliders", True)
    registry.set_visible(False)
    assert registry.is_shown("colliders")
    assert not registry.is_shown("paths")


def test_providers_run_only_when_visible_and_errors_are_contained() -> None:
    registry = OverlayRegistry()
    calls: list[object] = []

    def nav(draw: DebugDraw, context: object) -> None:
        calls.append(context)
        draw.line((0, 0, 0), (0, 0, 1), channel="nav")

    def broken(_draw: DebugDraw, _context: object) -> None:
        raise ValueError("сломан")

    registry.register("nav", provider=nav)
    registry.register("bones", provider=broken)
    draw = DebugDraw()
    assert registry.run_providers(draw, "scene") == 0
    registry.set_visible(True)
    assert registry.run_providers(draw, "scene") == 2
    assert calls == ["scene"]
    assert draw.line_count == 1


def test_reregister_keeps_user_flag() -> None:
    registry = OverlayRegistry()
    registry.set_enabled("bones", False)
    registry.register("bones", provider=lambda d, c: None)
    assert registry.get("bones").enabled is False


def test_debug_pass_gathers_visible_channels_and_gates_buffer() -> None:
    draw = DebugDraw()
    registry = OverlayRegistry()
    debug_pass = DebugPass(draw, registry, context=lambda: "scene")
    assert draw.enabled is False  # оверлеи скрыты — вызовы не копятся
    draw.line((0, 0, 0), (1, 0, 0))
    assert draw.line_count == 0
    registry.set_visible(True)
    assert draw.enabled is True
    draw.line((0, 0, 0), (1, 0, 0))
    draw.line((0, 0, 0), (1, 0, 0), channel="paths")
    registry.set_enabled("paths", False)
    positions, _, _, _ = debug_pass.gather()
    assert len(positions) == 2
    debug_pass.dispose()
    registry.set_visible(False)
    assert draw.enabled is True  # отписан после dispose


def test_module_level_api_uses_shared_buffer() -> None:
    debug.draw.clear()
    enabled = debug.draw.enabled
    debug.draw.enabled = True
    try:
        debug.line((0, 0, 0), (1, 1, 1))
        debug.sphere((0, 0, 0), 0.5)
        debug.text((0, 0, 0), "цель")
        debug.path([(0, 0, 0), (1, 0, 0)])
        assert debug.draw.line_count == 1 + 72 + 1
        assert debug.draw.text_count == 1
    finally:
        debug.draw.clear()
        debug.draw.enabled = enabled


def test_key_replaces_previous_shape() -> None:
    draw = DebugDraw()
    draw.line((0, 0, 0), (1, 0, 0), key="enemy", duration=0.1)
    draw.line((0, 0, 0), (2, 0, 0), key="enemy", duration=0.1)
    positions, *_ = draw.collect()
    assert draw.line_count == 1 and positions[1, 0] == 2.0
    draw.advance(0.2)
    assert draw.line_count == 0
