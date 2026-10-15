"""Тесты процедурного неба: настройки, порядок проходов, шейдер, отрисовка."""

# ruff: noqa: F811 - фикстуры pytest подставляются по имени аргумента

from __future__ import annotations

import json

import numpy as np
import pytest
from test_rendering_gl import (  # noqa: F401 - фикстуры pytest
    _color,
    gl_context,
    render_to_target,
    renderer,
)

from engine.rendering.sky import SkyPass, SkySettings


def test_sky_settings_serialize() -> None:
    settings = SkySettings(
        mode="solid",
        zenith_color=(0.1, 0.2, 0.3),
        sun_size=2.5,
        texture_guid="abc",
    )
    data = json.loads(json.dumps(settings.to_dict()))
    restored = SkySettings.from_dict(data)
    assert restored == settings
    assert SkySettings.from_dict(None) == SkySettings()
    # Неизвестные ключи игнорируются, неизвестный режим — ошибка.
    assert SkySettings.from_dict({"nope": 1}) == SkySettings()
    with pytest.raises(ValueError):
        SkySettings().set_value("mode", "cloudy")
    with pytest.raises(KeyError):
        SkySettings().set_value("nope", 1)


def test_sky_pass_registered(renderer) -> None:
    renderer.set_sky(SkyPass(SkySettings()))
    names = [p.name for p in renderer.graph.passes]
    assert names.index("geometry") < names.index("sky") < names.index("transparent")
    # Повторная установка заменяет проход, а не дублирует.
    renderer.set_sky(SkyPass(SkySettings()))
    assert [p.name for p in renderer.graph.passes].count("sky") == 1
    renderer.set_sky(None)
    assert "sky" not in [p.name for p in renderer.graph.passes]


def test_sky_shader_compiles(renderer) -> None:
    program = renderer.program("sky")
    assert program.handle != 0
    assert program.uniform_location("u_inv_view_projection") >= 0
    assert program.uniform_location("u_zenith_color") >= 0


def _camera(look_up: bool):
    from engine.core.math3d import Vector3
    from engine.rendering import Camera

    camera = Camera(fov=60.0, far=100.0)
    camera.position = Vector3(0.0, 1.0, 5.0)
    camera.look_at(Vector3(0.0, 6.0 if look_up else 1.0, 0.0))
    return camera


def test_sky_draws_behind_geometry(renderer) -> None:
    from engine.core.math3d import Vector3, compose_matrix
    from engine.core.resources import cube_mesh
    from engine.rendering import GpuMesh

    settings = SkySettings(zenith_color=(0.2, 0.4, 1.0), horizon_color=(0.2, 0.4, 1.0))
    renderer.set_sky(SkyPass(settings))
    camera = _camera(look_up=False)
    mesh = GpuMesh.from_mesh_asset(cube_mesh(2.0))
    material = renderer.create_material("red", base_color=_color(1.0, 0.0, 0.0))

    target = render_to_target(renderer, camera)
    renderer.submit(mesh, material, compose_matrix(Vector3(0, 1, 0)))
    renderer.end_frame(target=target)
    pixels = target.read_pixels().astype(int)

    h, w = pixels.shape[:2]
    corner = pixels[h - 3, 3]  # верхний левый угол кадра: только небо
    centre = pixels[h // 2, w // 2]  # куб перекрывает небо
    assert corner[2] > corner[0] + 40, corner  # синее небо
    assert centre[0] > centre[2], centre  # красный куб

    # Без неба тот же угол — цвет очистки.
    settings.enabled = False
    target = render_to_target(renderer, camera)
    renderer.end_frame(target=target)
    off = target.read_pixels().astype(int)[h - 3, 3]
    assert not np.array_equal(off, corner)


def test_sky_does_not_break_transparent_pass(renderer) -> None:
    """После неба прозрачный материал той же программы, что и непрозрачный.

    Регрессия: кэш активной программы рендерера не сбрасывался после неба, и
    uniform-ы прозрачного материала уходили в программу неба
    (``GL_INVALID_OPERATION``).
    """
    from engine.core.math3d import Vector3, compose_matrix
    from engine.core.resources import cube_mesh
    from engine.rendering import GpuMesh

    renderer.set_sky(SkyPass(SkySettings()))
    camera = _camera(look_up=False)
    mesh = GpuMesh.from_mesh_asset(cube_mesh(2.0))
    solid = renderer.create_material("solid", base_color=_color(1.0, 0.0, 0.0))
    glass = renderer.create_material(
        "glass", base_color=_color(0.0, 1.0, 0.0), transparent=True, alpha=0.5
    )

    target = render_to_target(renderer, camera)
    renderer.submit(mesh, solid, compose_matrix(Vector3(-1.5, 1, 0)))
    renderer.submit(mesh, glass, compose_matrix(Vector3(1.5, 1, 0)))
    renderer.end_frame(target=target)  # не должен бросать GLError
    pixels = target.read_pixels().astype(int)
    h, w = pixels.shape[:2]
    assert pixels[h // 2, w // 2 + w // 6, 1] > 60  # зелёное стекло видно
