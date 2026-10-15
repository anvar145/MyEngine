"""Тесты гизмо источников света: геометрия, зоны влияния и выбор мышью."""

from __future__ import annotations

import math

import numpy as np
import pytest

from editor.gizmos import LightGizmo, LightGizmoStyle
from editor.gizmos.light_gizmo import (
    direction_lines,
    light_icon_lines,
    range_circle_lines,
    spot_cone_lines,
)
from engine.core.math3d import Vector3
from engine.rendering.lighting import (
    AmbientLight,
    DirectionalLight,
    PointLight,
    SpotLight,
)


class TestIconGeometry:
    def test_icon_is_line_pairs(self) -> None:
        points = light_icon_lines(1.0)
        assert points.ndim == 2 and points.shape[1] == 3
        assert points.shape[0] % 2 == 0
        assert points.dtype == np.float32

    def test_icon_scales_with_size(self) -> None:
        small = light_icon_lines(1.0)
        large = light_icon_lines(4.0)
        assert large.shape == small.shape
        assert np.abs(large).max() == pytest.approx(4.0 * np.abs(small).max())

    def test_icon_is_centered(self) -> None:
        points = light_icon_lines(2.0)
        assert points.min(axis=0)[1] == pytest.approx(-1.0)
        assert points.max(axis=0)[1] == pytest.approx(1.0)


class TestRangeGeometry:
    def test_circle_has_three_rings(self) -> None:
        points = range_circle_lines(5.0)
        # Три окружности по 32 сегмента, каждый — пара вершин.
        assert points.shape[0] % 2 == 0
        assert points.shape[0] >= 3 * 32 * 2

    def test_circle_radius_matches(self) -> None:
        points = range_circle_lines(7.0)
        # Все точки лежат на сфере радиуса 7: кольца в трёх плоскостях.
        lengths = np.linalg.norm(points, axis=1)
        assert lengths.max() == pytest.approx(7.0, rel=1e-4)

    def test_circle_scales(self) -> None:
        small = range_circle_lines(1.0)
        large = range_circle_lines(10.0)
        assert np.linalg.norm(large, axis=1).max() == pytest.approx(
            10.0 * np.linalg.norm(small, axis=1).max(), rel=1e-4
        )

    def test_tiny_radius_is_clamped(self) -> None:
        points = range_circle_lines(0.0)
        assert np.isfinite(points).all()


class TestSpotCone:
    def test_cone_starts_at_origin(self) -> None:
        points = spot_cone_lines(10.0, 30.0)
        # Конус растёт из начала координат: там вершина.
        assert np.isclose(points, 0.0).all(axis=1).any()

    def test_cone_radius_follows_angle(self) -> None:
        narrow = spot_cone_lines(10.0, 10.0)
        wide = spot_cone_lines(10.0, 45.0)
        # По XY основание шире у большего угла.
        narrow_reach = np.linalg.norm(narrow[:, :2], axis=1).max()
        wide_reach = np.linalg.norm(wide[:, :2], axis=1).max()
        assert wide_reach > narrow_reach

    def test_cone_matches_expected_radius(self) -> None:
        points = spot_cone_lines(10.0, 30.0)
        expected = 10.0 * math.tan(math.radians(30.0))
        reach = np.linalg.norm(points[:, :2], axis=1).max()
        assert reach == pytest.approx(expected, rel=1e-3)

    def test_extreme_angle_is_clamped(self) -> None:
        points = spot_cone_lines(5.0, 179.0)
        assert np.isfinite(points).all()


class TestDirectionLines:
    def test_lines_follow_direction(self) -> None:
        points = direction_lines(Vector3(0, -1, 0), length=2.0)
        assert points.shape[1] == 3
        # Самая дальняя точка — по направлению света (вниз).
        assert points[:, 1].min() < -1.5

    def test_zero_direction_is_empty(self) -> None:
        assert direction_lines(Vector3.zero()).shape[0] == 0

    def test_offset_moves_lines(self) -> None:
        origin = direction_lines(Vector3(0, -1, 0), offset=Vector3(5, 5, 5))
        assert origin[:, 0].max() >= 5.0

    def test_ray_count_is_configurable(self) -> None:
        few = direction_lines(Vector3(0, -1, 0), rays=1)
        many = direction_lines(Vector3(0, -1, 0), rays=8)
        assert many.shape[0] > few.shape[0]


class TestGizmoPlacement:
    def test_icon_keeps_screen_size(self) -> None:
        gizmo = LightGizmo(PointLight(position=Vector3(0, 0, 5)))
        near = gizmo.icon_scale(Vector3(0, 0, 6))
        far = gizmo.icon_scale(Vector3(0, 0, 60))
        assert far > near

    def test_icon_scale_has_minimum(self) -> None:
        gizmo = LightGizmo(
            PointLight(position=Vector3(0, 0, 5)),
            style=LightGizmoStyle(min_icon_size=0.5),
        )
        assert gizmo.icon_scale(Vector3(0, 0, 5.001)) >= 0.5

    def test_icon_matrix_places_light(self) -> None:
        light = PointLight(position=Vector3(2, 3, 4))
        gizmo = LightGizmo(light)
        matrix = gizmo.icon_matrix(Vector3(0, 0, 20))
        assert matrix[0, 3] == pytest.approx(2.0)
        assert matrix[1, 3] == pytest.approx(3.0)
        assert matrix[2, 3] == pytest.approx(4.0)

    def test_title_and_kind(self) -> None:
        gizmo = LightGizmo(SpotLight(), name="Torch")
        assert gizmo.title == "Torch"
        assert gizmo.kind == "spot"

    def test_color_depends_on_selection(self) -> None:
        light = PointLight()
        plain = LightGizmo(light, selected=False)
        selected = LightGizmo(light, selected=True)
        assert not plain.color.almost_equals(selected.color, 1e-6)

    def test_ambient_has_own_color(self) -> None:
        ambient = LightGizmo(AmbientLight(), selected=True)
        point = LightGizmo(PointLight(), selected=True)
        # У фонового света нет позы: цвет отличается, чтобы не путать.
        assert not ambient.color.almost_equals(point.color, 1e-6)


class TestGizmoDrawing:
    def test_point_light_draws_icon_and_range(self) -> None:
        gizmo = LightGizmo(PointLight(range=5.0), selected=True)
        items = gizmo.draw_items(object(), object(), viewer_position=Vector3(0, 0, 9))
        # Иконка и зона влияния. Лучей направления у точечного света нет:
        # он светит во все стороны, и «направление» у него бессмысленно.
        assert len(items) == 2

    def test_directional_draws_icon_and_rays(self) -> None:
        gizmo = LightGizmo(DirectionalLight(), selected=True)
        items = gizmo.draw_items(object(), object())
        # У солнца нет радиуса: иконка и лучи направления.
        assert len(items) == 2

    def test_spot_draws_icon_cone_and_rays(self) -> None:
        gizmo = LightGizmo(SpotLight(range=8.0), selected=True)
        items = gizmo.draw_items(object(), object())
        # Иконка, конус влияния и лучи направления.
        assert len(items) == 3

    def test_ambient_draws_only_icon(self) -> None:
        gizmo = LightGizmo(AmbientLight(), selected=True)
        items = gizmo.draw_items(object(), object())
        assert len(items) == 1

    def test_range_hidden_for_unselected_point_light(self) -> None:
        gizmo = LightGizmo(PointLight(range=5.0), selected=False)
        items = gizmo.draw_items(object(), object())
        # Без выделения остаётся только иконка: зона влияния не рисуется.
        assert len(items) == 1

    def test_range_can_be_forced(self) -> None:
        gizmo = LightGizmo(PointLight(range=5.0), selected=False)
        items = gizmo.draw_items(object(), object(), show_range=True)
        assert len(items) == 2

    def test_no_wire_material_draws_icon_only(self) -> None:
        gizmo = LightGizmo(PointLight(range=5.0), selected=True)
        assert len(gizmo.draw_items(object(), None)) == 1

    def test_style_can_disable_range(self) -> None:
        style = LightGizmoStyle(show_range=False)
        gizmo = LightGizmo(SpotLight(range=5.0), selected=True, style=style)
        items = gizmo.draw_items(object(), object())
        # Конус выключен, остаются иконка и лучи направления.
        assert len(items) == 2

    def test_style_can_disable_direction(self) -> None:
        style = LightGizmoStyle(show_range=False, show_direction=False)
        gizmo = LightGizmo(DirectionalLight(), selected=True, style=style)
        assert len(gizmo.draw_items(object(), object())) == 1

    def test_empty_direction_draws_no_rays(self) -> None:
        # Источник с нулевым направлением: лучей нет, пустой меш не рисуется.
        light = DirectionalLight(direction=Vector3.zero())
        gizmo = LightGizmo(light, selected=True)
        assert len(gizmo.draw_items(object(), object())) == 1


class TestGizmoMeshes:
    def test_icon_mesh_is_cached(self) -> None:
        gizmo = LightGizmo(PointLight())
        assert gizmo.icon_mesh() is gizmo.icon_mesh()

    def test_wire_mesh_is_cached_per_parameters(self) -> None:
        gizmo = LightGizmo(PointLight(range=5.0))
        first = gizmo.wire_mesh("range", gizmo.range_key())
        assert gizmo.wire_mesh("range", gizmo.range_key()) is first

    def test_wire_mesh_rebuilt_for_new_range(self) -> None:
        gizmo = LightGizmo(PointLight(range=5.0))
        first = gizmo.wire_mesh("range", gizmo.range_key())
        gizmo.light.range = 20.0
        assert gizmo.wire_mesh("range", gizmo.range_key()) is not first

    def test_range_key_is_empty_for_directional(self) -> None:
        assert LightGizmo(DirectionalLight()).range_key() == ()

    def test_meshes_are_cpu_side_until_uploaded(self) -> None:
        gizmo = LightGizmo(PointLight())
        assert not gizmo.icon_mesh().is_uploaded


class TestGizmoPicking:
    def test_pick_hits_icon(self) -> None:
        light = PointLight(position=Vector3(0, 0, 5))
        gizmo = LightGizmo(light)
        origin = Vector3(0, 0, 20)
        direction = (light.position - origin).normalized
        assert gizmo.pick(origin, direction, origin) is True

    def test_pick_misses_far_away(self) -> None:
        gizmo = LightGizmo(PointLight(position=Vector3(0, 0, 5)))
        origin = Vector3(10, 10, 20)
        direction = (Vector3(10, 10, 0) - origin).normalized
        assert gizmo.pick(origin, direction, origin) is False

    def test_pick_ignores_light_behind(self) -> None:
        gizmo = LightGizmo(PointLight(position=Vector3(0, 0, 5)))
        assert gizmo.pick(Vector3(0, 0, 20), Vector3(0, 0, 1), Vector3(0, 0, 20)) is False

    def test_pick_with_zero_direction_is_safe(self) -> None:
        assert LightGizmo(PointLight()).pick(Vector3.zero(), Vector3.zero()) is False

    def test_repr_mentions_kind(self) -> None:
        text = repr(LightGizmo(SpotLight(), name="Torch", selected=True))
        assert "Torch" in text
        assert "spot" in text


class TestGizmosForScene:
    def test_collects_lights_from_scene(self) -> None:
        from editor.gizmos.light_gizmo import gizmos_for
        from engine.core.scene import Scene
        from engine.rendering.lighting import LightComponent

        scene = Scene("Scene")
        scene.create_game_object("Sun").add_component(
            LightComponent, light=DirectionalLight()
        )
        scene.create_game_object("Lamp").add_component(
            LightComponent, light=PointLight()
        )
        scene.create_game_object("Empty")
        gizmos = gizmos_for(scene)
        assert len(gizmos) == 2
        assert {gizmo.kind for gizmo in gizmos} == {"directional", "point"}

    def test_empty_scene_gives_no_gizmos(self) -> None:
        from editor.gizmos.light_gizmo import gizmos_for
        from engine.core.scene import Scene

        assert gizmos_for(Scene("Empty")) == []
