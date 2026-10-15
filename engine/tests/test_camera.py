"""Тесты системы камер: проекции, экранные преобразования, стек и компонент.

Модуль проверяет как чистую математику (матрицы, фрустум, взаимные
преобразования экран ↔ мир), так и объектную часть: камеру-компонент
игрового объекта, стек камер с активной камерой и сохранение вида
редакторской камеры между сессиями.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from engine.core.game_object import GameObject
from engine.core.math3d import Quaternion, Vector3
from engine.core.scene import Scene
from engine.rendering.camera import (
    ASPECT_FIXED,
    ASPECT_VIEWPORT,
    PROJECTION_MODES,
    Camera,
    CameraComponent,
    CameraStack,
    EditorCameraController,
    EditorCameraSession,
    EditorCameraState,
    Frustum,
    ProjectionMode,
    quaternion_from_forward,
)

VIEWPORT = (800, 600)


def _look_at(eye: Vector3, target: Vector3) -> Camera:
    """Собрать камеру, смотрящую из точки в цель."""
    camera = Camera(fov=60.0, near=0.1, far=1000.0)
    camera.position = eye
    camera.look_at(target)
    return camera


class TestCameraProperties:
    def test_defaults_match_engine_conventions(self) -> None:
        camera = Camera()
        assert camera.mode == ProjectionMode.PERSPECTIVE
        assert camera.fov == pytest.approx(60.0)
        assert camera.near == pytest.approx(0.1)
        assert camera.far == pytest.approx(500.0)
        assert camera.aspect_mode == ASPECT_VIEWPORT
        assert camera.viewport_rect == (0.0, 0.0, 1.0, 1.0)
        assert camera.culling_mask == (1 << 32) - 1
        assert camera.enabled is True

    def test_all_projection_modes_are_supported(self) -> None:
        for mode in PROJECTION_MODES:
            camera = Camera(mode=mode)
            assert camera.get_projection_matrix().shape == (4, 4)
            assert camera.mode == mode

    def test_projection_mode_switches_matrix(self) -> None:
        camera = Camera(aspect=1.0)
        perspective = camera.get_projection_matrix().copy()
        camera.mode = ProjectionMode.ORTHOGRAPHIC
        camera.invalidate()
        orthographic = camera.get_projection_matrix()
        assert not np.allclose(perspective, orthographic)
        assert orthographic[3, 2] == pytest.approx(0.0, abs=1e-6)


class TestCameraMatrices:
    def test_get_view_matrix_matches_look_at(self) -> None:
        camera = _look_at(Vector3(0, 0, 5), Vector3.zero())
        matrix = camera.get_view_matrix()
        assert matrix.shape == (4, 4)
        point = camera.world_to_view(Vector3.zero())
        assert point.z == pytest.approx(-5.0, abs=1e-4)

    def test_get_projection_matrix_uses_fov_and_aspect(self) -> None:
        camera = Camera(fov=90.0, aspect=1.0, near=0.1, far=100.0)
        matrix = camera.get_projection_matrix()
        assert matrix[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert matrix[1, 1] == pytest.approx(1.0, abs=1e-5)

    def test_matrices_are_cached_between_calls(self) -> None:
        camera = _look_at(Vector3(1, 2, 3), Vector3.zero())
        first_view = camera.get_view_matrix()
        first_projection = camera.get_projection_matrix()
        assert camera.get_view_matrix() is first_view
        assert camera.get_projection_matrix() is first_projection

    def test_moving_camera_invalidates_cache(self) -> None:
        camera = _look_at(Vector3(0, 0, 5), Vector3.zero())
        before = camera.get_view_matrix().copy()
        camera.position = Vector3(0, 0, 10)
        after = camera.get_view_matrix()
        assert not np.allclose(before, after)
        assert camera.version > 0

    def test_version_grows_only_on_changes(self) -> None:
        camera = Camera()
        version = camera.version
        _ = camera.get_view_matrix()
        _ = camera.get_projection_matrix()
        assert camera.version == version
        camera.position = Vector3(1, 1, 1)
        assert camera.version > version

    def test_view_projection_is_product(self) -> None:
        camera = _look_at(Vector3(3, 3, 3), Vector3.zero())
        expected = camera.get_projection_matrix() @ camera.get_view_matrix()
        assert np.allclose(camera.get_view_projection_matrix(), expected)

    def test_effective_aspect_follows_viewport_by_default(self) -> None:
        camera = Camera()
        camera.set_aspect(1600, 900)
        assert camera.aspect == pytest.approx(1600 / 900)
        assert camera.effective_aspect == pytest.approx(1600 / 900)

    def test_fixed_aspect_ignores_viewport(self) -> None:
        camera = Camera(aspect=1.5, aspect_mode=ASPECT_FIXED)
        camera.set_aspect(1600, 900)
        assert camera.aspect == pytest.approx(1.5)
        assert camera.effective_aspect == pytest.approx(1.5)

    def test_lock_aspect(self) -> None:
        camera = Camera()
        camera.set_aspect(640, 480)
        camera.lock_aspect()
        assert camera.aspect_mode == ASPECT_FIXED
        camera.set_aspect(1280, 720)
        assert camera.aspect == pytest.approx(640 / 480)

    def test_degenerate_camera_does_not_crash(self) -> None:
        # Камера, смотрящая строго вниз: look_at с мировым up вырожден, но
        # видовая матрица обязана остаться корректной.
        camera = Camera()
        camera.position = Vector3(0, 5, 0)
        camera.rotation = quaternion_from_forward(Vector3(0, -1, 0))
        matrix = camera.get_view_matrix()
        assert np.isfinite(matrix).all()


class TestScreenWorldConversions:
    def test_center_of_screen_maps_to_view_target(self) -> None:
        camera = _look_at(Vector3(0, 0, 5), Vector3.zero())
        x, y, visible = camera.world_to_screen(Vector3.zero(), *VIEWPORT)
        assert visible
        assert x == pytest.approx(VIEWPORT[0] / 2, abs=1e-4)
        assert y == pytest.approx(VIEWPORT[1] / 2, abs=1e-4)

    def test_screen_to_world_inverts_world_to_screen(self) -> None:
        camera = _look_at(Vector3(2, 3, 8), Vector3.zero())
        target = Vector3(0.25, -0.5, 0.75)
        x, y, visible = camera.world_to_screen(target, *VIEWPORT)
        assert visible

        # Глубина исходной точки в NDC: развёртка с этой глубиной обязана
        # вернуть исходную точку — это и есть проверка обратимости.
        _cx, _cy, clip_z, clip_w = camera.world_to_clip(target)
        restored = camera.screen_to_world(
            x, y, *VIEWPORT, depth=clip_z / clip_w
        )
        assert restored.almost_equals(target, 5e-3)

    def test_screen_ray_passes_through_target(self) -> None:
        camera = _look_at(Vector3(2, 3, 8), Vector3.zero())
        target = Vector3(0.25, -0.5, 0.75)
        x, y, _visible = camera.world_to_screen(target, *VIEWPORT)
        origin, direction = camera.ray_from_screen(x, y, *VIEWPORT)
        assert origin.almost_equals(camera.position, 1e-4)
        # Точка лежит на луче: расстояние от неё до луча пренебрежимо мало.
        to_target = target - origin
        distance = direction.cross(to_target).length
        assert distance < 1e-3

    def test_orthographic_screen_to_world_is_exact(self) -> None:
        camera = Camera(mode=ProjectionMode.ORTHOGRAPHIC, ortho_height=10.0, aspect=1.0)
        camera.aspect_mode = ASPECT_FIXED
        camera.position = Vector3(2, 3, 8)
        camera.look_at(Vector3.zero())
        target = Vector3(0.5, -0.5, 0.0)
        x, y, visible = camera.world_to_screen(target, *VIEWPORT)
        assert visible
        _cx, _cy, clip_z, clip_w = camera.world_to_clip(target)
        restored = camera.screen_to_world(x, y, *VIEWPORT, depth=clip_z / clip_w)
        assert restored.almost_equals(target, 5e-3)

    def test_point_behind_camera_is_not_visible(self) -> None:
        camera = _look_at(Vector3(0, 0, 5), Vector3.zero())
        _x, _y, visible = camera.world_to_screen(Vector3(0, 0, 20), *VIEWPORT)
        assert visible is False

    def test_viewport_rect_shifts_projection(self) -> None:
        full = _look_at(Vector3(0, 0, 5), Vector3.zero())
        half = _look_at(Vector3(0, 0, 5), Vector3.zero())
        half.viewport_rect = (0.5, 0.0, 0.5, 1.0)
        full_x, _full_y, _ = full.world_to_screen(Vector3.zero(), *VIEWPORT)
        half_x, _half_y, _ = half.world_to_screen(Vector3.zero(), *VIEWPORT)
        # Центр второй половины экрана: 800 * 0.5 + 800 * 0.5 * 0.5 = 600.
        assert half_x == pytest.approx(full_x + VIEWPORT[0] * 0.25, abs=1e-4)

    def test_pixel_rect_in_pixels_when_values_exceed_one(self) -> None:
        camera = Camera(viewport_rect=(10, 20, 640, 480))
        assert camera.pixel_rect(1920, 1080) == (10, 20, 640, 480)

    def test_zero_viewport_returns_camera_position(self) -> None:
        camera = _look_at(Vector3(1, 1, 1), Vector3.zero())
        assert camera.screen_to_world(10, 10, 0, 0).almost_equals(camera.position)


class TestFrustum:
    def test_planes_are_normalized(self) -> None:
        camera = _look_at(Vector3(0, 0, 10), Vector3.zero())
        planes = camera.frustum_planes()
        assert planes.shape == (6, 4)
        lengths = np.linalg.norm(planes[:, :3], axis=1)
        assert np.allclose(lengths, 1.0, atol=1e-6)

    def test_frustum_is_reused_not_reallocated(self) -> None:
        camera = _look_at(Vector3(0, 0, 10), Vector3.zero())
        first = camera.frustum_planes()
        second = camera.frustum_planes()
        assert first is second  # тот же массив: горячий путь без аллокаций

    def test_frustum_recomputes_after_move(self) -> None:
        camera = _look_at(Vector3(0, 0, 10), Vector3.zero())
        version = camera.update_frustum().version
        camera.position = Vector3(0, 0, 20)
        assert camera.update_frustum().version > version

    def test_sphere_visibility(self) -> None:
        camera = _look_at(Vector3(0, 0, 10), Vector3.zero())
        assert camera.is_sphere_visible(Vector3.zero(), 0.5)
        assert not camera.is_sphere_visible(Vector3(0, 0, 50), 0.5)
        assert not camera.is_sphere_visible(Vector3(500, 0, 0), 0.5)

    def test_large_sphere_is_visible_from_inside(self) -> None:
        camera = _look_at(Vector3(0, 0, 10), Vector3.zero())
        assert camera.is_sphere_visible(Vector3(0, 0, 30), 100.0)

    def test_aabb_visibility(self) -> None:
        camera = _look_at(Vector3(0, 0, 10), Vector3.zero())
        assert camera.is_aabb_visible(Vector3(-1, -1, -1), Vector3(1, 1, 1))
        assert not camera.is_aabb_visible(
            Vector3(100, 100, 100), Vector3(101, 101, 101)
        )

    def test_frustum_corners_count_and_order(self) -> None:
        camera = _look_at(Vector3(0, 0, 10), Vector3.zero())
        corners = camera.frustum_corners()
        assert corners.shape == (8, 3)
        # Ближние углы ближе к камере, чем дальние.
        near_distance = np.linalg.norm(corners[:4] - np.array([0, 0, 10]), axis=1)
        far_distance = np.linalg.norm(corners[4:] - np.array([0, 0, 10]), axis=1)
        assert near_distance.max() < far_distance.min()

    def test_frustum_reports_dirty_state(self) -> None:
        frustum = Frustum()
        assert frustum.is_dirty
        frustum.update(np.identity(4), 0.1, 10.0)
        assert not frustum.is_dirty
        assert frustum.version == 1

    def test_orthographic_frustum_covers_box(self) -> None:
        camera = Camera(mode=ProjectionMode.ORTHOGRAPHIC, ortho_height=10.0)
        camera.aspect = 1.0
        camera.aspect_mode = ASPECT_FIXED
        camera.position = Vector3(0, 0, 10)
        camera.look_at(Vector3.zero())
        assert camera.is_sphere_visible(Vector3.zero(), 0.5)
        assert not camera.is_sphere_visible(Vector3(20, 0, 0), 0.5)


class TestCullingMask:
    def test_default_mask_accepts_every_layer(self) -> None:
        camera = Camera()
        assert camera.culling_mask == (1 << 32) - 1
        for layer in range(32):
            assert camera.layer_visible(layer)

    def test_mask_filters_layers_by_bit(self) -> None:
        # Слой N соответствует биту N: маска 0b1010 пропускает слои 1 и 3.
        camera = Camera(culling_mask=0b1010)
        assert camera.layer_visible(1)
        assert camera.layer_visible(3)
        assert not camera.layer_visible(0)
        assert not camera.layer_visible(2)

    def test_minus_one_mask_means_all_layers(self) -> None:
        camera = Camera(culling_mask=-1)
        for layer in range(32):
            assert camera.layer_visible(layer)

    def test_unknown_layer_is_visible(self) -> None:
        camera = Camera(culling_mask=0b1)
        # Слои вне диапазона 0..31 не отсекаются: это защита от опечаток в
        # данных пользователя, а не повод спрятать объект.
        assert camera.layer_visible(99)

    def test_select_objects_by_layer(self) -> None:
        class Item:
            def __init__(self, layer: int) -> None:
                self.layer = layer

        items = [Item(1), Item(2), Item(3)]
        camera = Camera(culling_mask=0b110)
        assert camera.select_objects(items) == [items[0], items[1]]


class TestCameraStack:
    def test_active_defaults_to_first(self) -> None:
        stack = CameraStack([Camera(name="A"), Camera(name="B")])
        assert stack.active is stack.cameras[0]
        assert stack.active_index == 0

    def test_set_active_by_index_and_object(self) -> None:
        first, second = Camera(name="A"), Camera(name="B")
        stack = CameraStack([first, second])
        assert stack.set_active(1) is second
        assert stack.set_active(first) is first
        assert stack.set_active(None) is first

    def test_set_active_by_name(self) -> None:
        stack = CameraStack([Camera(name="A"), Camera(name="B")])
        assert stack.set_active_by_name("B") is stack.cameras[1]
        assert stack.set_active_by_name("missing") is None

    def test_add_and_remove(self) -> None:
        stack = CameraStack()
        first = stack.add(Camera(name="A"))
        second = stack.add(Camera(name="B"), make_active=True)
        assert stack.active is second
        assert stack.remove(second) is True
        assert stack.active is first
        assert stack.remove(second) is False
        assert len(stack) == 1

    def test_out_of_range_index_raises(self) -> None:
        stack = CameraStack([Camera()])
        with pytest.raises(IndexError):
            stack.set_active(5)

    def test_ordered_by_depth_for_ui_overlay(self) -> None:
        game = Camera(name="Game", depth=0)
        overlay = Camera(name="UI", depth=10, clear_flags=("depth",))
        stack = CameraStack([overlay, game])
        assert [camera.name for camera in stack.ordered()] == ["Game", "UI"]
        assert [camera.name for camera in stack.overlay_cameras()] == ["UI"]

    def test_disabled_camera_is_skipped(self) -> None:
        game = Camera(name="Game")
        ui = Camera(name="UI", depth=5, enabled=False)
        stack = CameraStack([game, ui])
        assert [camera.name for camera in stack.ordered()] == ["Game"]

    def test_begin_frame_captures_matrices(self) -> None:
        camera = _look_at(Vector3(0, 0, 5), Vector3.zero())
        stack = CameraStack([camera])
        stack.begin_frame()
        assert np.allclose(stack.view_matrix, camera.get_view_matrix())
        assert np.allclose(
            stack.view_projection_matrix, camera.get_view_projection_matrix()
        )

    def test_captured_matrices_refresh_after_camera_move(self) -> None:
        camera = _look_at(Vector3(0, 0, 5), Vector3.zero())
        stack = CameraStack([camera])
        stack.begin_frame()
        before = stack.view_matrix.copy()
        camera.position = Vector3(0, 0, 20)
        assert not np.allclose(before, stack.view_matrix)

    def test_empty_stack_reports_none(self) -> None:
        stack = CameraStack()
        assert stack.active is None
        assert stack.begin_frame() is None
        with pytest.raises(ValueError):
            _ = stack.view_matrix

    def test_serialization_round_trip(self) -> None:
        stack = CameraStack(
            [Camera(name="A", fov=45.0), Camera(name="B", depth=3)], active=1
        )
        restored = CameraStack.from_dict(stack.to_dict())
        assert [camera.name for camera in restored] == ["A", "B"]
        assert restored.active.name == "B"
        assert restored.cameras[0].fov == pytest.approx(45.0)


class TestCameraComponent:
    def test_component_follows_owner_transform(self) -> None:
        scene = Scene("Scene")
        game_object = scene.create_game_object(
            "Camera", position=Vector3(3, 4, 5)
        )
        component = game_object.add_component(CameraComponent())
        component.sync_to_camera()
        assert component.camera.position.almost_equals(Vector3(3, 4, 5), 1e-5)

    def test_local_offset_is_applied_in_owner_axes(self) -> None:
        scene = Scene("Scene")
        game_object = scene.create_game_object("Camera")
        component = game_object.add_component(
            CameraComponent,
            local_offset=Vector3(0.0, 0.0, 2.0),
        )
        component.sync_to_camera()
        # Объект без поворота: смещение уходит вдоль мировой оси +Z.
        assert component.camera.position.almost_equals(Vector3(0, 0, 2), 1e-5)

    def test_rotation_is_taken_from_owner(self) -> None:
        scene = Scene("Scene")
        game_object = scene.create_game_object("Camera")
        game_object.transform.rotation = Quaternion.from_euler(0, 90, 0)
        component = game_object.add_component(CameraComponent())
        component.sync_to_camera()
        forward = component.camera.forward
        assert forward.x == pytest.approx(1.0, abs=1e-4)

    def test_display_name_falls_back_to_owner(self) -> None:
        scene = Scene("Scene")
        game_object = scene.create_game_object("PlayerCamera")
        component = game_object.add_component(CameraComponent())
        assert component.display_name == "PlayerCamera"
        component.camera.name = "Custom"
        assert component.display_name == "Custom"

    def test_component_can_be_detached_from_pairing(self) -> None:
        scene = Scene("Scene")
        game_object = scene.create_game_object("Camera", position=Vector3(1, 0, 0))
        component = game_object.add_component(CameraComponent())
        component.follow_owner = False
        game_object.transform.position = Vector3(9, 9, 9)
        component.sync_to_camera()
        assert component.camera.position.almost_equals(Vector3(1, 0, 0), 1e-5)

    def test_orphan_object_component_keeps_defaults(self) -> None:
        game_object = GameObject("Loose")
        component = game_object.add_component(CameraComponent())
        component.sync_to_camera()
        assert component.camera.position.almost_equals(Vector3.zero())


class TestEditorCameraController:
    def test_orbit_changes_angles(self) -> None:
        controller = EditorCameraController()
        yaw_before, pitch_before = controller.yaw, controller.pitch
        controller.orbit(10.0, 5.0)
        assert controller.yaw != yaw_before
        assert controller.pitch > pitch_before

    def test_orbit_pitch_is_clamped(self) -> None:
        controller = EditorCameraController()
        controller.orbit(0.0, 10_000.0)
        assert controller.pitch <= controller.max_pitch
        controller.orbit(0.0, -20_000.0)
        assert controller.pitch >= controller.min_pitch

    def test_zoom_is_clamped_and_directional(self) -> None:
        controller = EditorCameraController(min_distance=1.0, max_distance=10.0)
        controller.zoom(100.0)
        assert controller.distance == pytest.approx(1.0)
        controller.zoom(-100.0)
        assert controller.distance == pytest.approx(10.0)

    def test_zoom_passes_through_target_instead_of_stalling(self) -> None:
        controller = EditorCameraController(
            target=Vector3.zero(), distance=0.3, yaw=0.0, pitch=0.0
        )
        camera = Camera()
        controller.apply(camera)
        start = camera.position
        for _ in range(20):
            controller.zoom(1.0)
        controller.apply(camera)
        # Камера продолжает двигаться вперёд с постоянной скоростью.
        assert (camera.position - start).length > 20 * 0.1
        assert controller.distance == pytest.approx(controller.min_distance)

    def test_zoom_step_is_not_vanishing_near_target(self) -> None:
        controller = EditorCameraController(distance=1.0, min_distance=0.25)
        controller.zoom(1.0)
        first = 1.0 - controller.distance
        before = controller.distance
        controller.zoom(1.0)
        assert before - controller.distance >= min(first, controller.min_zoom_step) * 0.99

    def test_zoom_anchor_pulls_target_towards_cursor(self) -> None:
        controller = EditorCameraController(target=Vector3.zero(), distance=10.0)
        controller.zoom(1.0, anchor=Vector3(5, 0, 0))
        assert controller.target.x > 0.0

    def test_apply_places_camera_around_target(self) -> None:
        controller = EditorCameraController(
            target=Vector3(1, 2, 3), distance=5.0, yaw=0.0, pitch=0.0
        )
        camera = Camera()
        controller.apply(camera)
        assert (camera.position - Vector3(1, 2, 3)).length == pytest.approx(5.0)

    def test_apply_looks_at_target(self) -> None:
        controller = EditorCameraController(target=Vector3(1, 1, 1), distance=4.0)
        camera = Camera()
        controller.apply(camera)
        direction = (Vector3(1, 1, 1) - camera.position).normalized
        assert camera.forward.almost_equals(direction, 1e-4)

    def test_pan_moves_target(self) -> None:
        controller = EditorCameraController(target=Vector3.zero(), distance=10.0)
        camera = Camera()
        controller.apply(camera)
        before = controller.target
        controller.pan(50.0, 0.0, camera)
        assert not controller.target.almost_equals(before, 1e-6)

    def test_pan_without_camera_uses_angles(self) -> None:
        controller = EditorCameraController(target=Vector3.zero(), distance=10.0)
        before = controller.target
        controller.pan(25.0, 25.0)
        assert not controller.target.almost_equals(before, 1e-6)

    def test_focus_on_sets_distance_from_radius(self) -> None:
        controller = EditorCameraController()
        controller.focus_on(Vector3(5, 0, 5), radius=2.0)
        assert controller.target.almost_equals(Vector3(5, 0, 5), 1e-6)
        assert controller.distance == pytest.approx(6.0)

    def test_frame_bounds_fits_object(self) -> None:
        controller = EditorCameraController()
        camera = Camera(fov=60.0)
        controller.frame_bounds(Vector3(-1, -1, -1), Vector3(1, 1, 1), camera)
        assert controller.target.almost_equals(Vector3.zero(), 1e-6)
        assert controller.distance > 1.0

    def test_reset_restores_home_view(self) -> None:
        controller = EditorCameraController()
        controller.orbit(120.0, 40.0)
        controller.zoom(3.0)
        controller.reset()
        assert controller.distance == pytest.approx(6.0)
        assert controller.yaw == pytest.approx(45.0)

    def test_capture_and_restore_round_trip(self) -> None:
        source = EditorCameraController(
            target=Vector3(2, 3, 4), distance=12.0, yaw=30.0, pitch=15.0
        )
        camera = Camera(fov=45.0, near=0.5, far=250.0, ortho_height=8.0)
        source.apply(camera)
        state = source.capture(camera)

        other = EditorCameraController()
        other.restore(state, camera)
        assert other.target.almost_equals(Vector3(2, 3, 4), 1e-6)
        assert other.distance == pytest.approx(12.0)
        assert camera.fov == pytest.approx(45.0)
        assert camera.far == pytest.approx(250.0)

    def test_alt_orbit_flag_is_configurable(self) -> None:
        controller = EditorCameraController()
        assert controller.alt_orbit is True
        controller.alt_orbit = False
        assert controller.alt_orbit is False


class TestEditorCameraSession:
    def test_state_serialization(self) -> None:
        state = EditorCameraState(
            target=Vector3(1, 2, 3),
            distance=7.5,
            yaw=10.0,
            pitch=-5.0,
            fov=55.0,
            mode=ProjectionMode.ORTHOGRAPHIC,
        )
        restored = EditorCameraState.from_dict(state.to_dict())
        assert restored.target.almost_equals(Vector3(1, 2, 3), 1e-6)
        assert restored.distance == pytest.approx(7.5)
        assert restored.mode == ProjectionMode.ORTHOGRAPHIC

    def test_session_saves_and_loads_file(self, tmp_path) -> None:
        path = tmp_path / "camera.json"
        session = EditorCameraSession(path)
        session.set(
            EditorCameraState(target=Vector3(4, 5, 6), distance=9.0),
            "Main Camera",
        )
        assert session.save() is True

        restored = EditorCameraSession(path)
        assert restored.load() is True
        state = restored.get("Main Camera")
        assert state is not None
        assert state.target.almost_equals(Vector3(4, 5, 6), 1e-6)
        assert state.distance == pytest.approx(9.0)

    def test_load_without_file_returns_false(self, tmp_path) -> None:
        session = EditorCameraSession(tmp_path / "missing.json")
        assert session.load() is False
        assert session.get() is None

    def test_broken_file_is_ignored(self, tmp_path) -> None:
        path = tmp_path / "camera.json"
        path.write_text("{ не json", encoding="utf-8")
        session = EditorCameraSession(path)
        assert session.load() is False

    def test_unknown_entries_are_skipped(self, tmp_path) -> None:
        path = tmp_path / "camera.json"
        path.write_text(
            json.dumps({"good": EditorCameraState().to_dict(), "bad": 42}),
            encoding="utf-8",
        )
        session = EditorCameraSession(path)
        assert session.load() is True
        assert session.keys == ("good",)

    def test_default_path_is_inside_project(self, tmp_path) -> None:
        path = EditorCameraSession.default_path(tmp_path)
        assert path.parent.parent == tmp_path
        assert path.name == "camera.json"

    def test_session_without_path_does_not_write(self) -> None:
        session = EditorCameraSession(None)
        session.set(EditorCameraState())
        assert session.save() is False
        assert len(session) == 1


class TestCameraSerialization:
    def test_round_trip_preserves_all_properties(self) -> None:
        camera = Camera(
            fov=42.0,
            near=0.25,
            far=123.0,
            aspect=1.25,
            aspect_mode=ASPECT_FIXED,
            mode=ProjectionMode.ORTHOGRAPHIC,
            ortho_height=7.0,
            viewport_rect=(0.0, 0.0, 0.5, 1.0),
            clear_color=(0.1, 0.2, 0.3, 1.0),
            clear_flags=("depth",),
            culling_mask=0b110,
            depth=4,
            name="Player",
            enabled=False,
        )
        camera.position = Vector3(1.5, -2.5, 3.5)
        camera.rotation = Quaternion.from_euler(15, 30, 45)

        restored = Camera.from_dict(camera.to_dict())
        assert restored.name == "Player"
        assert restored.mode == ProjectionMode.ORTHOGRAPHIC
        assert restored.fov == pytest.approx(42.0)
        assert restored.viewport_rect == (0.0, 0.0, 0.5, 1.0)
        assert restored.clear_flags == ("depth",)
        assert restored.culling_mask == 0b110
        assert restored.enabled is False
        assert restored.position.almost_equals(Vector3(1.5, -2.5, 3.5), 1e-6)
        assert restored.rotation.almost_equals(
            camera.rotation, 1e-5
        )

    def test_clears_flags_helpers(self) -> None:
        camera = Camera(clear_flags=("color", "depth"))
        assert camera.clears_color and camera.clears_depth
        camera.clear_flags = ("depth",)
        assert not camera.clears_color
        assert camera.clears_depth

    def test_clear_color_is_clamped(self) -> None:
        camera = Camera(clear_color=(-1.0, 2.0, 0.5, 3.0))
        red, green, blue, alpha = camera.clear_color_rgba
        assert red == pytest.approx(0.0)
        assert green == pytest.approx(1.0)
        assert blue == pytest.approx(0.5)
        assert alpha == pytest.approx(1.0)

    def test_copy_is_independent(self) -> None:
        camera = Camera(name="Source", fov=30.0)
        camera.position = Vector3(1, 2, 3)
        clone = camera.copy(name="Clone")
        clone.fov = 90.0
        clone.position = Vector3(0, 0, 0)
        assert camera.fov == pytest.approx(30.0)
        assert camera.position.almost_equals(Vector3(1, 2, 3), 1e-6)
        assert clone.name == "Clone"

    def test_repr_contains_key_fields(self) -> None:
        text = repr(Camera(name="Player", fov=45.0))
        assert "Camera" in text
        assert "Player" in text
        assert "45" in text


class TestScreenWorldInvariant:
    """Свойство, которое обязано держаться для любых камер и вьюпортов.

    ``world_to_screen`` — ``screen_to_world`` — ``ray_from_screen`` живут в
    трёх разных формулах, и расхождение между ними (например, тангенс вместо
    котангенса половины угла обзора) проявляется только на реальном выборе
    объектов. Здесь оно проверяется на случайных камерах.
    """

    def test_random_cameras_keep_picking_consistent(self) -> None:
        rng = np.random.default_rng(7)
        checked = 0
        worst_ray = 0.0
        worst_round_trip = 0.0
        for trial in range(60):
            mode = (
                ProjectionMode.PERSPECTIVE
                if trial % 2 == 0
                else ProjectionMode.ORTHOGRAPHIC
            )
            camera = Camera(
                mode=mode,
                fov=float(rng.uniform(15.0, 120.0)),
                near=0.1,
                far=float(rng.uniform(50.0, 5000.0)),
                ortho_height=float(rng.uniform(2.0, 40.0)),
            )
            camera.position = Vector3(*rng.uniform(-8.0, 8.0, 3))
            camera.look_at(Vector3(*rng.uniform(-3.0, 3.0, 3)))
            width = int(rng.integers(200, 1600))
            height = int(rng.integers(200, 1000))
            camera.set_aspect(width, height)

            point = (
                camera.position
                + camera.forward * float(rng.uniform(1.0, 20.0))
                + camera.right * float(rng.uniform(-5.0, 5.0))
                + camera.up * float(rng.uniform(-5.0, 5.0))
            )
            x, y, visible = camera.world_to_screen(point, width, height)
            if not visible:
                continue
            checked += 1

            origin, direction = camera.ray_from_screen(x, y, width, height)
            worst_ray = max(worst_ray, direction.cross(point - origin).length)

            _cx, _cy, clip_z, clip_w = camera.world_to_clip(point)
            restored = camera.screen_to_world(
                x, y, width, height, depth=clip_z / clip_w
            )
            worst_round_trip = max(worst_round_trip, (restored - point).length)

        assert checked > 30, "слишком мало проверенных камер"
        assert worst_ray < 1e-2, f"луч не проходит через точку: {worst_ray}"
        assert worst_round_trip < 1e-2, f"развёртка неточна: {worst_round_trip}"


class TestFrustumMathematics:
    def test_orthographic_projection_has_unit_w(self) -> None:
        camera = Camera(mode=ProjectionMode.ORTHOGRAPHIC, ortho_height=10.0, aspect=1.0)
        _, _, _, w = camera.world_to_clip(Vector3(0, 0, -5))
        assert w == pytest.approx(1.0, abs=1e-6)

    def test_perspective_w_equals_distance(self) -> None:
        camera = Camera(fov=60.0, aspect=1.0, near=1.0, far=100.0)
        camera.position = Vector3(0, 0, 4)
        camera.look_at(Vector3.zero())
        _, _, _, w = camera.world_to_clip(Vector3.zero())
        assert w == pytest.approx(4.0, abs=1e-4)

    def test_fov_affects_visibility_of_side_point(self) -> None:
        narrow = _look_at(Vector3(0, 0, 10), Vector3.zero())
        narrow.fov = 20.0
        wide = _look_at(Vector3(0, 0, 10), Vector3.zero())
        wide.fov = 100.0
        point = Vector3(6, 0, 0)
        assert not narrow.is_sphere_visible(point, 0.1)
        assert wide.is_sphere_visible(point, 0.1)

    def test_near_far_clip_plane_distance(self) -> None:
        camera = Camera(fov=60.0, aspect=1.0, near=2.0, far=20.0)
        camera.position = Vector3(0, 0, 0)
        camera.look_at(Vector3(0, 0, -1))
        _, _, near_z, near_w = camera.world_to_clip(Vector3(0, 0, -2.0))
        _, _, far_z, far_w = camera.world_to_clip(Vector3(0, 0, -20.0))
        assert near_z / near_w == pytest.approx(-1.0, abs=1e-4)
        assert far_z / far_w == pytest.approx(1.0, abs=1e-4)

    def test_invalid_projection_parameters_raise(self) -> None:
        with pytest.raises(ValueError):
            _ = Camera(fov=0.0).projection_matrix
        with pytest.raises(ValueError):
            _ = Camera(near=10.0, far=1.0).projection_matrix

    def test_quaternion_from_forward_handles_opposite(self) -> None:
        rotation = quaternion_from_forward(Vector3(0, 0, -1))
        rotated = rotation.rotate_vector(Vector3.forward())
        assert rotated.almost_equals(Vector3(0, 0, -1), 1e-5)

    def test_ray_from_screen_center(self) -> None:
        camera = _look_at(Vector3(0, 0, 5), Vector3.zero())
        origin, direction = camera.ray_from_screen(400, 300, 800, 600)
        assert origin.almost_equals(Vector3(0, 0, 5), 1e-5)
        assert direction.almost_equals(Vector3(0, 0, -1), 1e-4)

    def test_orthographic_ray_starts_on_near_plane(self) -> None:
        camera = Camera(mode=ProjectionMode.ORTHOGRAPHIC, ortho_height=10.0, aspect=1.0)
        camera.aspect_mode = ASPECT_FIXED
        camera.position = Vector3(0, 0, 10)
        camera.look_at(Vector3.zero())
        origin, direction = camera.ray_from_screen(800, 300, 800, 600)
        # Правый край экрана: половина ширины ортографического объёма; начало
        # луча лежит на ближней плоскости (отсюда небольшой сдвиг по Z).
        assert origin.x == pytest.approx(5.0, abs=1e-2)
        assert direction.almost_equals(Vector3(0, 0, -1), 1e-4)

    def test_projection_aspect_uses_effective_value(self) -> None:
        camera = Camera(fov=90.0, near=0.1, far=100.0)
        camera.set_aspect(1000, 500)
        matrix = camera.get_projection_matrix()
        # При fov=90 focal=1, поэтому m00 = 1 / aspect = 0.5.
        assert matrix[0, 0] == pytest.approx(0.5, abs=1e-5)
        assert math.isclose(camera.effective_aspect, 2.0, rel_tol=1e-6)


def test_editor_camera_orbit_has_no_roll():
    """Орбитальная камера не кренится: right горизонтален, up смотрит вверх."""
    from engine.rendering.camera import Camera, EditorCameraController

    camera = Camera()
    controller = EditorCameraController()
    for yaw in range(0, 360, 15):
        for pitch in range(-89, 90, 8):
            controller.yaw = float(yaw)
            controller.pitch = float(pitch)
            controller.apply(camera)
            assert camera.up.y > 0.0, (yaw, pitch)
            assert abs(camera.right.y) < 1e-6, (yaw, pitch)
            direction = (controller.target - camera.position).normalized
            assert camera.forward.dot(direction) > 1.0 - 1e-6, (yaw, pitch)
