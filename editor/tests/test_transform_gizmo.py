"""Математика гизмо трансформа: пикинг осей и перетаскивание (без OpenGL)."""

from __future__ import annotations

import math

import pytest

from editor.gizmos.transform_gizmo import (
    GizmoMode,
    TransformGizmo,
    arrow_mesh,
    closest_axis_parameter,
    ring_mesh,
    scale_handle_mesh,
)
from engine.core.event_bus import EventBus
from engine.core.math3d import Vector3
from engine.core.scene import Scene
from engine.rendering.camera import Camera

W, H = 800, 600


@pytest.fixture()
def camera() -> Camera:
    view = Camera(position=Vector3(0.0, 0.0, 10.0), fov=60.0)
    view.look_at(Vector3.zero())
    view.set_aspect(W, H)
    return view


@pytest.fixture()
def cube():
    scene = Scene("t", event_bus=EventBus())
    return scene.create_game_object("Cube")


def _screen(camera: Camera, point: Vector3) -> tuple[float, float]:
    x, y, visible = camera.world_to_screen(point, W, H)
    assert visible
    return x, y


def _axis_tip_screen(gizmo, cube, camera, index) -> tuple[float, float]:
    origin = gizmo.origin_of(cube)
    length = gizmo.screen_scale(origin, camera)
    direction = gizmo.axis_direction(cube, index, gizmo.mode)
    return _screen(camera, origin + direction * length * 0.7)


def test_closest_axis_parameter_simple() -> None:
    t = closest_axis_parameter(
        Vector3(3, 0, 5), Vector3(0, 0, -1), Vector3.zero(), Vector3(1, 0, 0)
    )
    assert t == pytest.approx(3.0)


def test_parallel_ray_has_no_parameter() -> None:
    assert (
        closest_axis_parameter(
            Vector3(0, 1, 0), Vector3(1, 0, 0), Vector3.zero(), Vector3(1, 0, 0)
        )
        is None
    )


def test_gizmo_keeps_screen_size(camera) -> None:
    gizmo = TransformGizmo()
    near = gizmo.screen_scale(Vector3.zero(), camera)
    camera.position = Vector3(0, 0, 20)
    camera.look_at(Vector3.zero())
    far = gizmo.screen_scale(Vector3.zero(), camera)
    assert far == pytest.approx(near * 2.0, rel=1e-3)


@pytest.mark.parametrize("index", [0, 1])
def test_pick_finds_each_axis(camera, cube, index) -> None:
    gizmo = TransformGizmo(GizmoMode.MOVE)
    x, y = _axis_tip_screen(gizmo, cube, camera, index)
    assert gizmo.pick(cube, camera, x, y, W, H) == index


def test_pick_misses_empty_space(camera, cube) -> None:
    gizmo = TransformGizmo(GizmoMode.MOVE)
    assert gizmo.pick(cube, camera, 20, 20, W, H) is None


def test_move_along_x_changes_only_x(camera, cube) -> None:
    gizmo = TransformGizmo(GizmoMode.MOVE)
    x, y = _axis_tip_screen(gizmo, cube, camera, 0)
    assert gizmo.begin(cube, camera, x, y, W, H)
    assert gizmo.active_axis == 0
    assert gizmo.drag(cube, camera, x + 80, y + 35, W, H)
    position = cube.transform.position
    assert position.x > 0.1
    assert position.y == pytest.approx(0.0, abs=1e-6)
    assert position.z == pytest.approx(0.0, abs=1e-6)
    assert gizmo.end()
    assert not gizmo.is_dragging


def test_move_follows_cursor_exactly(camera, cube) -> None:
    """Точка оси под курсором остаётся под курсором при перетаскивании."""
    gizmo = TransformGizmo(GizmoMode.MOVE)
    x, y = _axis_tip_screen(gizmo, cube, camera, 0)
    gizmo.begin(cube, camera, x, y, W, H)
    gizmo.drag(cube, camera, x + 60, y, W, H)
    moved_x, _ = _screen(camera, cube.transform.position + Vector3(0.0, 0.0, 0.0))
    origin_x, _ = _screen(camera, Vector3.zero())
    assert moved_x > origin_x


def test_scale_changes_only_dragged_axis(camera, cube) -> None:
    gizmo = TransformGizmo(GizmoMode.SCALE)
    x, y = _axis_tip_screen(gizmo, cube, camera, 0)
    assert gizmo.begin(cube, camera, x, y, W, H)
    assert gizmo.drag(cube, camera, x + 60, y, W, H)
    scale = cube.transform.scale
    assert scale.x > 1.0
    assert scale.y == pytest.approx(1.0)
    assert scale.z == pytest.approx(1.0)


def test_scale_never_collapses(camera, cube) -> None:
    gizmo = TransformGizmo(GizmoMode.SCALE)
    x, y = _axis_tip_screen(gizmo, cube, camera, 0)
    gizmo.begin(cube, camera, x, y, W, H)
    gizmo.drag(cube, camera, x - 5000, y, W, H)
    assert cube.transform.scale.x >= 0.0099


def test_rotate_around_y_turns_object(camera, cube) -> None:
    gizmo = TransformGizmo(GizmoMode.ROTATE)
    camera.position = Vector3(0.0, 6.0, 9.0)  # кольцо XZ не должно быть ребром
    camera.look_at(Vector3.zero())
    origin = Vector3.zero()
    radius = gizmo.screen_scale(origin, camera)
    # Кольцо оси Y лежит в плоскости XZ: берём точку спереди-справа на кольце.
    start = Vector3(radius * math.cos(0.9), 0.0, radius * math.sin(0.9))
    end = Vector3(radius * math.cos(0.4), 0.0, radius * math.sin(0.4))
    sx, sy = _screen(camera, start)
    ex, ey = _screen(camera, end)
    index = gizmo.pick(cube, camera, sx, sy, W, H)
    assert index == 1
    assert gizmo.begin(cube, camera, sx, sy, W, H)
    assert gizmo.drag(cube, camera, ex, ey, W, H)
    forward = cube.transform.rotation.rotate_vector(Vector3(1, 0, 0))
    assert abs(forward.y) < 1e-6
    assert forward.z != pytest.approx(0.0, abs=1e-3)


def test_mode_switch_cancels_drag(camera, cube) -> None:
    gizmo = TransformGizmo(GizmoMode.MOVE)
    x, y = _axis_tip_screen(gizmo, cube, camera, 0)
    gizmo.begin(cube, camera, x, y, W, H)
    gizmo.set_mode(GizmoMode.SCALE)
    assert not gizmo.is_dragging


def test_drag_without_begin_does_nothing(camera, cube) -> None:
    gizmo = TransformGizmo()
    assert not gizmo.drag(cube, camera, 10, 10, W, H)
    assert not gizmo.end()


@pytest.mark.parametrize("factory", [arrow_mesh, scale_handle_mesh, ring_mesh])
def test_meshes_are_valid(factory) -> None:
    mesh = factory()
    assert mesh.positions.shape[1] == 3
    assert mesh.indices is not None and mesh.indices.max() < len(mesh.positions)
    assert len(mesh.indices) % 3 == 0


def test_axis_matrices_follow_object(camera, cube) -> None:
    cube.transform.position = Vector3(2, 3, 4)
    gizmo = TransformGizmo(GizmoMode.MOVE)
    matrices = gizmo.axis_matrices(cube, camera)
    assert [index for index, _ in matrices] == [0, 1, 2]
    for _index, matrix in matrices:
        assert tuple(matrix[:3, 3]) == pytest.approx((2, 3, 4)) or tuple(
            matrix[3, :3]
        ) == pytest.approx((2, 3, 4))
