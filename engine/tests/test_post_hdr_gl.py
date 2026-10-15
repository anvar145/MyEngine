"""HDR-цель и вывод кадра через пост-путь рендерера (A10 п.1, offscreen GL)."""

# ruff: noqa: F811 - фикстуры pytest подставляются по имени аргумента

from __future__ import annotations

import numpy as np
import pytest
from test_rendering_gl import (  # noqa: F401 - фикстуры pytest
    camera,
    gl_context,
    renderer,
)

from engine.rendering import RenderTarget
from engine.rendering.render_graph import FrameContext, RenderGraph
from engine.rendering.ui import UIPass, UIScreen

SIZE = (160, 120)


def _read_float(target: RenderTarget) -> np.ndarray:
    """Прочитать цвет HDR-цели как float32 ``(H, W, 4)``."""
    from OpenGL import GL

    target.resolve()
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.read_framebuffer)
    raw = GL.glReadPixels(0, 0, target.width, target.height, GL.GL_RGBA, GL.GL_FLOAT)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    return np.frombuffer(raw, dtype=np.float32).reshape(target.height, target.width, 4)


def _frame(renderer, camera, *, post, size=SIZE, clear=None, scale=1.0):
    renderer.hdr_enabled = True
    renderer.render_scale = scale
    target = RenderTarget("final", *size)
    renderer.begin_frame(camera, size, clear_color=clear, target=target, post=post)
    renderer.end_frame(target=target)
    pixels = target.read_pixels()
    target.dispose()
    return pixels


def test_capabilities_report_hdr_target(renderer) -> None:
    assert isinstance(renderer.capabilities.supports_hdr_target, bool)
    assert "HDR target" in renderer.capabilities.describe() or (
        not renderer.capabilities.supports_hdr_target
    )


def test_hdr_render_target_keeps_values_above_one(renderer) -> None:
    if not renderer.capabilities.supports_hdr_target:
        pytest.skip("RGBA16F недоступен")
    from OpenGL import GL

    target = RenderTarget("hdr", 8, 8, hdr=True)
    target.create()
    assert target.is_hdr
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.framebuffer)
    GL.glClearColor(3.0, 2.0, 0.5, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    pixel = _read_float(target)[4, 4]
    target.dispose()
    assert pixel[0] == pytest.approx(3.0, rel=1e-2)
    assert pixel[1] == pytest.approx(2.0, rel=1e-2)
    assert pixel[2] == pytest.approx(0.5, rel=1e-2)


def test_ldr_target_is_not_hdr(renderer) -> None:
    target = RenderTarget("ldr", 8, 8)
    target.create()
    assert not target.is_hdr
    target.dispose()


def test_hdr_target_with_msaa_resolves(renderer) -> None:
    if not renderer.capabilities.supports_hdr_target:
        pytest.skip("RGBA16F недоступен")
    target = RenderTarget("hdr_msaa", 16, 16, samples=4, hdr=True)
    target.create()
    # MSAA может не собраться (откат на цель без сглаживания) — оба случая валидны.
    assert target.is_hdr
    target.dispose()


def test_post_frame_matches_plain_frame(renderer, camera) -> None:
    """HDR-путь без эффектов даёт тот же кадр, что и обычный (±1 LSB)."""
    if not renderer.capabilities.supports_hdr_target:
        pytest.skip("RGBA16F недоступен")
    from engine.core.math3d import Vector3, compose_matrix
    from engine.core.resources import cube_mesh
    from engine.rendering import GpuMesh

    mesh = GpuMesh.from_mesh_asset(cube_mesh(2.0))
    material = renderer.create_material("red", base_color=Vector3(1.0, 0.2, 0.2))

    def draw(post):
        renderer.hdr_enabled = True
        target = RenderTarget("final", *SIZE)
        renderer.begin_frame(camera, SIZE, target=target, post=post)
        renderer.submit(mesh, material, compose_matrix())
        renderer.end_frame(target=target)
        pixels = target.read_pixels()
        target.dispose()
        return pixels

    plain = draw(False)
    hdr = draw(True)
    difference = np.abs(plain.astype(int) - hdr.astype(int))
    assert difference.max() <= 1
    assert (plain[:, :, :3] > 60).any()


def test_hdr_buffer_is_not_clamped_but_output_is(renderer, camera) -> None:
    if not renderer.capabilities.supports_hdr_target:
        pytest.skip("RGBA16F недоступен")
    renderer.hdr_enabled = True
    target = RenderTarget("final", *SIZE)
    renderer.begin_frame(
        camera, SIZE, clear_color=(2.0, 0.5, 0.1, 1.0), target=target, post=True
    )
    hdr = renderer.hdr_target
    assert hdr is not None and hdr.is_hdr
    renderer.end_frame(target=target)
    inner = _read_float(hdr)[10, 10]
    outer = target.read_pixels()[10, 10]
    target.dispose()
    assert inner[0] == pytest.approx(2.0, rel=1e-2)
    assert outer[0] == 255
    assert abs(int(outer[1]) - 128) <= 1


def test_post_disabled_without_hdr_flag(renderer, camera) -> None:
    renderer.hdr_enabled = False
    target = RenderTarget("final", *SIZE)
    renderer.begin_frame(camera, SIZE, target=target, post=True)
    assert not renderer.post_frame_active
    renderer.end_frame(target=target)
    target.dispose()
    assert renderer.hdr_target is None


def test_hdr_target_follows_viewport_and_scale(renderer, camera) -> None:
    if not renderer.capabilities.supports_hdr_target:
        pytest.skip("RGBA16F недоступен")
    _frame(renderer, camera, post=True, size=(160, 120))
    hdr = renderer.hdr_target
    assert (hdr.width, hdr.height) == (160, 120)
    _frame(renderer, camera, post=True, size=(200, 100))
    assert renderer.hdr_target is hdr
    assert (hdr.width, hdr.height) == (200, 100)
    _frame(renderer, camera, post=True, size=(200, 100), scale=0.5)
    assert (hdr.width, hdr.height) == (100, 50)


def test_scaled_frame_is_stretched_to_full_size(renderer, camera) -> None:
    if not renderer.capabilities.supports_hdr_target:
        pytest.skip("RGBA16F недоступен")
    pixels = _frame(
        renderer, camera, post=True, scale=0.5, clear=(0.0, 0.0, 1.0, 1.0)
    )
    assert pixels.shape == (SIZE[1], SIZE[0], 4)
    assert tuple(pixels[60, 80, :3]) == (0, 0, 255)
    assert tuple(pixels[0, 0, :3]) == (0, 0, 255)


def test_ui_is_drawn_after_post_at_full_resolution(renderer, camera) -> None:
    if not renderer.capabilities.supports_hdr_target:
        pytest.skip("RGBA16F недоступен")
    ui = UIPass(UIScreen(SIZE))
    ui.batch.add_quad(0, 0, 40, 40, (1.0, 1.0, 1.0, 1.0))
    renderer.add_pass(ui)
    renderer.hdr_enabled = True
    renderer.render_scale = 0.5
    target = RenderTarget("final", *SIZE)
    renderer.begin_frame(
        camera, SIZE, clear_color=(0.0, 0.0, 0.0, 1.0), target=target, post=True
    )
    renderer.end_frame(target=target)
    pixels = target.read_pixels()[::-1]
    hdr_pixels = _read_float(renderer.hdr_target)
    ui.batch.dispose()
    target.dispose()
    assert tuple(pixels[20, 20, :3]) == (255, 255, 255)
    assert tuple(pixels[100, 140, :3]) == (0, 0, 0)
    # В HDR-цели UI нет: он рисуется только в итоговую цель.
    assert hdr_pixels[:, :, :3].max() == 0.0


def test_ui_without_post_still_drawn_in_main_pass(renderer, camera) -> None:
    ui = UIPass(UIScreen(SIZE))
    ui.batch.add_quad(0, 0, 40, 40, (1.0, 1.0, 1.0, 1.0))
    renderer.add_pass(ui)
    target = RenderTarget("final", *SIZE)
    renderer.begin_frame(camera, SIZE, clear_color=(0, 0, 0, 1), target=target)
    renderer.end_frame(target=target)
    pixels = target.read_pixels()[::-1]
    ui.batch.dispose()
    target.dispose()
    assert tuple(pixels[20, 20, :3]) == (255, 255, 255)


class _Recorder:
    def __init__(self, name: str, late: bool) -> None:
        self.name = name
        self.after_post = late
        self.calls = 0

    def execute(self, context: FrameContext) -> None:
        self.calls += 1


def test_graph_phases_split_passes() -> None:
    graph = RenderGraph("t")
    early, late = _Recorder("a", False), _Recorder("b", True)
    graph.add_pass(early)
    graph.add_pass(late)
    assert graph.execute(None, "main") == 1  # type: ignore[arg-type]
    assert (early.calls, late.calls) == (1, 0)
    assert graph.frame_index == 0
    assert graph.execute(None, "late") == 1  # type: ignore[arg-type]
    assert (early.calls, late.calls) == (1, 1)
    assert graph.frame_index == 1
    assert graph.execute(None) == 2  # type: ignore[arg-type]
