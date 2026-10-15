"""Виджеты UI (панель, кнопка, картинка, текст) на настоящем GL-контексте."""

# ruff: noqa: F811 - фикстуры pytest подставляются по имени аргумента

from __future__ import annotations

import numpy as np
from PIL import Image
from test_rendering_gl import (  # noqa: F401 - фикстуры pytest
    camera,
    gl_context,
    renderer,
)

from engine.core.scene import Scene
from engine.core.ui_components import UIButton, UIImage, UIPanel, UIText
from engine.rendering import RenderTarget
from engine.rendering.ui import UIFiller, UIPass, UIScreen

BACKGROUND = (0.2, 0.0, 0.0, 1.0)
BG = (51, 0, 0)


def _render(renderer, camera, scene, size=(200, 200), roots=()):
    screen = UIScreen(size)
    ui = UIPass(screen)
    filler = UIFiller()
    filler.fill(scene, screen, ui.batch, ui.text_batch, roots)
    renderer.add_pass(ui)
    target = RenderTarget("ui_widgets", *size)
    renderer.begin_frame(camera, size, clear_color=BACKGROUND, target=target)
    renderer.end_frame(target=target)
    pixels = target.read_pixels()[::-1]
    calls = ui.batch.last_draw_calls + ui.text_batch.last_draw_calls
    ui.batch.dispose()
    ui.text_batch.dispose()
    filler.dispose()
    target.dispose()
    return pixels, calls


def _rgb(pixels, x, y):
    return tuple(int(v) for v in pixels[y, x, :3])


def _lit(pixels, rect):
    x0, y0, x1, y1 = rect
    area = pixels[y0:y1, x0:x1, :3].astype(int)
    return int(np.count_nonzero(np.abs(area - np.array(BG)).sum(axis=2) > 60))


def test_panel_with_border(renderer, camera) -> None:
    scene = Scene()
    panel = scene.create_game_object("P").add_component(
        UIPanel,
        size=(100.0, 100.0),
        position=(50.0, 50.0),
        color=(0.0, 1.0, 0.0, 1.0),
        border_color=(1.0, 0.0, 0.0, 1.0),
        border_width=10.0,
    )
    assert panel is not None
    pixels, _ = _render(renderer, camera, scene)
    assert _rgb(pixels, 100, 100) == (0, 255, 0)
    assert _rgb(pixels, 53, 100) == (255, 0, 0)
    assert _rgb(pixels, 20, 20) == BG


def test_button_state_changes_color(renderer, camera) -> None:
    scene = Scene()
    button = scene.create_game_object("B").add_component(
        UIButton,
        text="",
        size=(100.0, 50.0),
        position=(10.0, 10.0),
        normal_color=(1.0, 0.0, 0.0, 1.0),
        hover_color=(0.0, 1.0, 0.0, 1.0),
    )
    pixels, _ = _render(renderer, camera, scene)
    assert _rgb(pixels, 50, 30) == (255, 0, 0)
    button.state = "hover"
    pixels, _ = _render(renderer, camera, scene)
    assert _rgb(pixels, 50, 30) == (0, 255, 0)


def test_button_label_is_drawn(renderer, camera) -> None:
    scene = Scene()
    scene.create_game_object("B").add_component(
        UIButton,
        text="OK",
        size_px=32,
        size=(120.0, 60.0),
        position=(40.0, 70.0),
        normal_color=(0.0, 0.0, 0.0, 1.0),
    )
    pixels, _ = _render(renderer, camera, scene)
    assert _lit(pixels, (40, 70, 160, 130)) > 30


def test_image_from_file_with_tint(renderer, camera, tmp_path) -> None:
    Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(tmp_path / "w.png")
    scene = Scene()
    scene.create_game_object("I").add_component(
        UIImage,
        texture_path="w.png",
        color=(0.0, 0.0, 1.0, 1.0),
        size=(50.0, 50.0),
        position=(10.0, 10.0),
    )
    pixels, _ = _render(renderer, camera, scene, roots=(tmp_path,))
    assert _rgb(pixels, 30, 30) == (0, 0, 255)


def test_missing_image_is_skipped(renderer, camera) -> None:
    scene = Scene()
    scene.create_game_object("I").add_component(UIImage, texture_path="nope.png")
    pixels, _ = _render(renderer, camera, scene)
    assert _rgb(pixels, 30, 30) == BG


def test_text_at_position(renderer, camera) -> None:
    scene = Scene()
    scene.create_game_object("T").add_component(
        UIText, text="Hello", size=(200.0, 100.0), position=(10.0, 10.0)
    ).text.size_px = 32
    pixels, _ = _render(renderer, camera, scene)
    assert _lit(pixels, (10, 10, 110, 50)) > 50
    assert _lit(pixels, (0, 120, 200, 200)) == 0


def test_text_center_aligned(renderer, camera) -> None:
    scene = Scene()
    label = scene.create_game_object("T").add_component(
        UIText, text="HI", size=(200.0, 50.0)
    )
    label.text.align = "center"
    label.text.size_px = 32
    pixels, _ = _render(renderer, camera, scene)
    _, xs = np.nonzero(np.abs(pixels[:60, :, :3].astype(int) - BG).sum(axis=2) > 60)
    assert len(xs) > 0
    assert abs((xs.min() + xs.max()) / 2.0 - 100.0) < 12


def test_draw_calls_few_for_many_widgets(renderer, camera) -> None:
    scene = Scene()
    for index in range(20):
        scene.create_game_object(f"P{index}").add_component(
            UIPanel, size=(5.0, 5.0), position=(index * 8.0, 10.0)
        )
    _, calls = _render(renderer, camera, scene)
    assert calls == 1
