"""Тесты гизмо камеры: геометрия иконки, фрустума и выбор мышью."""

from __future__ import annotations

import numpy as np
import pytest

from editor.gizmos import CameraGizmo, CameraGizmoStyle
from editor.gizmos.camera_gizmo import camera_icon_lines, frustum_lines
from engine.core.math3d import Vector3
from engine.rendering.camera import Camera, ProjectionMode


def _camera(position: Vector3 = Vector3(0, 0, 5)) -> Camera:
    """Камера, смотрящая в начало координат."""
    camera = Camera()
    camera.position = position
    camera.look_at(Vector3.zero())
    return camera


class TestIconGeometry:
    def test_icon_is_line_pairs(self) -> None:
        points = camera_icon_lines(1.0)
        assert points.ndim == 2 and points.shape[1] == 3
        # Вершины идут парами: начало и конец каждого отрезка.
        assert points.shape[0] % 2 == 0
        assert points.dtype == np.float32

    def test_icon_scales_with_size(self) -> None:
        small = camera_icon_lines(1.0)
        large = camera_icon_lines(2.0)
        assert large.shape == small.shape
        assert np.abs(large).max() == pytest.approx(2.0 * np.abs(small).max())

    def test_icon_without_lens_is_shorter(self) -> None:
        full = camera_icon_lines(1.0, lens=True)
        box = camera_icon_lines(1.0, lens=False)
        assert full.shape[0] > box.shape[0]

    def test_icon_bounds_are_centered(self) -> None:
        points = camera_icon_lines(2.0)
        assert points.min(axis=0)[0] == pytest.approx(-1.0)
        assert points.max(axis=0)[0] == pytest.approx(1.0)


class TestFrustumGeometry:
    def test_frustum_has_twelve_segments(self) -> None:
        points = frustum_lines(_camera())
        assert points.shape == (24, 3)

    def test_frustum_scales_with_far_plane(self) -> None:
        camera = _camera()
        short = frustum_lines(camera, far_scale=0.25)
        full = frustum_lines(camera)
        # Дальние углы подтягиваются к камере при уменьшении масштаба.
        camera_position = np.array(camera.position.to_tuple(), dtype=np.float32)
        short_reach = np.linalg.norm(short - camera_position, axis=1).max()
        full_reach = np.linalg.norm(full - camera_position, axis=1).max()
        assert short_reach < full_reach

    def test_orthographic_frustum_is_finite(self) -> None:
        camera = Camera(mode=ProjectionMode.ORTHOGRAPHIC, ortho_height=4.0)
        camera.lock_aspect()
        points = frustum_lines(camera)
        assert points.shape == (24, 3)
        assert np.isfinite(points).all()


class TestGizmoPlacement:
    def test_icon_keeps_screen_size(self) -> None:
        gizmo = CameraGizmo(_camera())
        near = gizmo.icon_scale(Vector3(0, 0, 6))
        far = gizmo.icon_scale(Vector3(0, 0, 60))
        assert far > near  # дальше — крупнее в мире, значит того же размера на экране

    def test_icon_scale_has_minimum(self) -> None:
        gizmo = CameraGizmo(_camera(), style=CameraGizmoStyle(min_icon_size=0.5))
        assert gizmo.icon_scale(Vector3(0, 0, 5.0001)) >= 0.5

    def test_icon_matrix_places_camera(self) -> None:
        camera = _camera(Vector3(2, 3, 4))
        gizmo = CameraGizmo(camera)
        matrix = gizmo.icon_matrix(Vector3(0, 0, 20))
        assert matrix.shape == (4, 4)
        assert matrix[0, 3] == pytest.approx(2.0)
        assert matrix[1, 3] == pytest.approx(3.0)
        assert matrix[2, 3] == pytest.approx(4.0)

    def test_position_and_title(self) -> None:
        camera = _camera(Vector3(1, 2, 3))
        gizmo = CameraGizmo(camera, name="Player Camera")
        assert gizmo.title == "Player Camera"
        assert gizmo.position.almost_equals(Vector3(1, 2, 3), 1e-5)

    def test_color_depends_on_selection(self) -> None:
        camera = _camera()
        plain = CameraGizmo(camera, selected=False)
        selected = CameraGizmo(camera, selected=True)
        assert not plain.color.almost_equals(selected.color, 1e-6)


class TestGizmoMeshes:
    def test_icon_mesh_is_cached(self) -> None:
        gizmo = CameraGizmo(_camera())
        assert gizmo.icon_mesh() is gizmo.icon_mesh()

    def test_frustum_mesh_rebuilt_only_on_change(self) -> None:
        camera = _camera()
        gizmo = CameraGizmo(camera)
        first = gizmo.frustum_mesh()
        assert gizmo.frustum_mesh() is first
        camera.fov = 30.0
        assert gizmo.frustum_mesh() is not first

    def test_meshes_have_expected_vertex_counts(self) -> None:
        gizmo = CameraGizmo(_camera())
        assert gizmo.icon_mesh().vertex_count == camera_icon_lines(1.0).shape[0]
        assert gizmo.frustum_mesh().vertex_count == 24

    def test_sensitive_gizmo_does_not_upload(self) -> None:
        # Меш создаётся на CPU: GPU-загрузка произойдёт при первом кадре.
        gizmo = CameraGizmo(_camera())
        assert not gizmo.icon_mesh().is_uploaded


class TestGizmoDrawing:
    def test_draw_items_include_icon(self) -> None:
        gizmo = CameraGizmo(_camera())
        items = gizmo.draw_items(object(), None)
        assert len(items) == 1
        mesh, _material, matrix, _mode = items[0]
        assert mesh is gizmo.icon_mesh()
        assert matrix.shape == (4, 4)

    def test_frustum_drawn_only_when_selected(self) -> None:
        geometry = object()
        plain = CameraGizmo(_camera(), selected=False)
        assert len(plain.draw_items(geometry, geometry)) == 1

        selected = CameraGizmo(_camera(), selected=True)
        assert len(selected.draw_items(geometry, geometry)) == 2

    def test_always_show_frustum_style(self) -> None:
        style = CameraGizmoStyle(always_show_frustum=True)
        gizmo = CameraGizmo(_camera(), style=style)
        assert len(gizmo.draw_items(object(), object())) == 2

    def test_show_frustum_override(self) -> None:
        gizmo = CameraGizmo(_camera(), selected=False)
        assert len(gizmo.draw_items(object(), object(), show_frustum=True)) == 2

    def test_no_frustum_material_skips_frustum(self) -> None:
        gizmo = CameraGizmo(_camera(), selected=True)
        assert len(gizmo.draw_items(object(), None)) == 1


class TestGizmoPicking:
    def test_pick_hits_icon_center(self) -> None:
        camera = _camera(Vector3(0, 0, 5))
        gizmo = CameraGizmo(camera)
        origin = Vector3(0, 0, 20)
        direction = (camera.position - origin).normalized
        assert gizmo.pick(origin, direction, origin) is True

    def test_pick_misses_away_from_icon(self) -> None:
        camera = _camera(Vector3(0, 0, 5))
        gizmo = CameraGizmo(camera)
        origin = Vector3(10, 10, 20)
        direction = (Vector3(10, 10, 0) - origin).normalized
        assert gizmo.pick(origin, direction, origin) is False

    def test_pick_ignores_camera_behind_viewer(self) -> None:
        camera = _camera(Vector3(0, 0, 5))
        gizmo = CameraGizmo(camera)
        origin = Vector3(0, 0, 20)
        # Луч смотрит от камеры: иконка остаётся позади наблюдателя.
        assert gizmo.pick(origin, Vector3(0, 0, 1), origin) is False

    def test_pick_tolerance_controls_radius(self) -> None:
        camera = _camera(Vector3(0, 0, 5))
        gizmo = CameraGizmo(camera)
        # Луч проходит от точки на оси Y мимо камеры: расстояние до центра
        # иконки 0.35, что меньше радиуса при обычном допуске и больше — при
        # уменьшенном.
        origin = Vector3(0.0, 0.35, 20.0)
        direction = Vector3(0.0, 0.0, -1.0)
        assert gizmo.pick(origin, direction, origin) is True
        assert gizmo.pick(origin, direction, origin, tolerance=0.01) is False

    def test_pick_with_zero_direction_is_safe(self) -> None:
        gizmo = CameraGizmo(_camera())
        assert gizmo.pick(Vector3.zero(), Vector3.zero()) is False


class TestGizmoRepresentation:
    def test_repr_mentions_state(self) -> None:
        text = repr(CameraGizmo(_camera(), selected=True, name="Main"))
        assert "Main" in text
        assert "выделено" in text
