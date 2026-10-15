"""Секция сцены ``post``: round-trip без потерь и GpuTimer (A10a, offscreen GL)."""

# ruff: noqa: F811 - фикстуры pytest подставляются по имени аргумента

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_rendering_gl import gl_context  # noqa: F401 - фикстура pytest

from engine.core.scene import Scene
from engine.rendering.gpu_timer import GpuTimer
from engine.rendering.post import PostSettings
from engine.scene_io import load_scene_full, read_post, save_scene, scene_to_dict


def test_post_round_trip(tmp_path: Path) -> None:
    scene = Scene("s")
    scene.create_game_object("A")
    post = PostSettings(tonemap="reinhard", exposure=1.7, fxaa=True)
    path = save_scene(scene, tmp_path / "a.scene.json", post=post)
    _, _, loaded = load_scene_full(path)
    assert loaded == post
    # повторное сохранение даёт тот же словарь секции
    again = save_scene(scene, tmp_path / "b.scene.json", post=loaded)
    assert (
        json.loads(again.read_text("utf-8"))["post"]
        == json.loads(path.read_text("utf-8"))["post"]
    )


def test_no_post_section_means_none(tmp_path: Path) -> None:
    scene = Scene("s")
    data = scene_to_dict(scene)
    assert "post" not in data and read_post(data) is None
    path = save_scene(scene, tmp_path / "c.scene.json")
    assert load_scene_full(path)[2] is None


def test_bad_post_values_are_sanitized() -> None:
    post = read_post({"post": {"tonemap": "bogus", "exposure": 999, "fxaa": 1}})
    assert post is not None
    assert post.tonemap == "linear" and post.exposure <= 16.0 and post.fxaa is True


def test_gpu_timer_returns_time(gl_context) -> None:  # noqa: F811
    from OpenGL import GL

    timer = GpuTimer()
    if not timer._ensure():
        pytest.skip("GL_TIME_ELAPSED недоступен")
    for _ in range(3):
        timer.begin()
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        timer.end()
    GL.glFinish()
    value = None
    for _ in range(50):
        value = timer.poll() or value
        if value is not None:
            break
    assert value is not None and value >= 0.0
    assert timer.last_ms == value
    timer.dispose()
