"""C1: ``DebugPass`` рисует линии и подписи поверх кадра (настоящий GL)."""

# ruff: noqa: F811 - фикстуры pytest подставляются по имени аргумента

from __future__ import annotations

import numpy as np
from test_rendering_gl import (  # noqa: F401 - фикстуры pytest
    _top_camera,
    count_visible,
    gl_context,
    render_to_target,
    renderer,
)

from engine.debug import DebugDraw, OverlayRegistry
from engine.rendering.debug_pass import DebugPass


def _frame(renderer, debug_pass: DebugPass) -> np.ndarray:
    renderer.add_pass(debug_pass)
    try:
        target = render_to_target(renderer, _top_camera(), 128, 96)
        renderer.end_frame(target=target)
        return target
    finally:
        renderer.graph.remove_pass(debug_pass)


def test_lines_and_labels_drawn_only_when_visible(renderer) -> None:
    draw, registry = DebugDraw(), OverlayRegistry()
    debug_pass = DebugPass(draw, registry)
    registry.set_visible(True)
    draw.line((-2, 0, 0), (2, 0, 0), (1, 0, 0))
    draw.sphere((0, 0, 0), 1.0, (0, 1, 0))
    target = _frame(renderer, debug_pass)
    assert debug_pass.line_count == 1 + 72
    pixels = target.read_pixels()
    reds = (pixels[:, :, 0] > 150) & (pixels[:, :, 1] < 80)
    assert reds.sum() > 20
    assert count_visible(target) > 100

    draw.line((-2, 0, 0), (2, 0, 0), (1, 0, 0))
    registry.set_visible(False)
    target = _frame(renderer, debug_pass)
    assert debug_pass.line_count == 0
    assert count_visible(target) == 0
    debug_pass.dispose()


def test_label_renders_text(renderer) -> None:
    draw, registry = DebugDraw(), OverlayRegistry()
    debug_pass = DebugPass(draw, registry)
    registry.set_visible(True)
    draw.text((0, 0, 0), "HP 10", (1, 1, 1), size=0.03)
    target = _frame(renderer, debug_pass)
    assert debug_pass.text_count == 1
    assert count_visible(target) > 10
    debug_pass.dispose()
