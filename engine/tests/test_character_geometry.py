"""Верификация CharacterController: геометрия капсулы (radius, height).

Центр капсулы в покое стоит на ``height/2 + skin_width`` над полом; нижний полюс
(``position[1] - height/2``) — на ``skin_width``. Позиция трансформа = центр капсулы.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from character_kit import DT, foot_y, make_char, make_world, settle

from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
)
from engine.core.scene import Scene
from engine.physics.character import CharacterController
from engine.physics.shape import CapsuleShape

SKIN = 0.02


# --- 1a/1b: размеры капсулы -------------------------------------------------------

def test_default_radius_height_and_skin() -> None:
    char = CharacterController(make_world())
    assert (char.radius, char.height, char.skin_width) == (0.4, 1.8, 0.02)


def test_shape_is_capsule_of_requested_total_height() -> None:
    char = make_char(make_world())
    assert isinstance(char.shape, CapsuleShape)
    assert char.shape.radius == 0.4
    assert char.shape.half_height == pytest.approx(1.8 / 2 - 0.4)
    lo, hi = char.shape.aabb(np.zeros(3), np.eye(3))
    assert hi[1] - lo[1] == pytest.approx(1.8)
    assert hi[0] - lo[0] == pytest.approx(0.8)


@pytest.mark.parametrize(("radius", "height"),
                         [(0.4, 1.8), (0.5, 2.0), (0.3, 1.8), (0.5, 1.01), (0.2, 1.2)])
def test_resting_center_is_half_height_plus_skin(radius: float, height: float) -> None:
    char = make_char(make_world(), y=height / 2 + 0.05, radius=radius, height=height)
    settle(char)
    assert char.grounded
    assert char.position[1] == pytest.approx(height / 2 + SKIN, abs=1e-3)
    assert foot_y(char) == pytest.approx(SKIN, abs=1e-3)


def test_foot_is_below_center_by_half_height() -> None:
    char = make_char(make_world(), y=3.0)
    assert foot_y(char) == pytest.approx(3.0 - 0.9)


def test_larger_radius_stops_earlier_at_wall() -> None:
    from character_kit import add_wall, walk

    stops = []
    for radius in (0.3, 0.5):
        world = make_world()
        face = add_wall(world)
        char = make_char(world, radius=radius)
        settle(char)
        walk(char, (4.5, 0, 0), 2.0)
        stops.append(face - (char.position[0] + radius))
    assert stops == [pytest.approx(SKIN, abs=1e-3)] * 2


# --- 1b..1d: валидация -------------------------------------------------------------

@pytest.mark.parametrize(("radius", "height"),
                         [(0.4, 0.7), (0.5, 0.5), (0.4, 0.8), (0.4, 0.0)])
def test_height_not_above_diameter_is_rejected(radius: float, height: float) -> None:
    """``height <= 2*radius`` (включая сферу) — ошибка с понятным сообщением."""
    with pytest.raises(ValueError, match="height must be") as err:
        CharacterController(make_world(), radius=radius, height=height)
    assert "height" in str(err.value)
    assert "radius" in str(err.value)
    assert f"height={height}" in str(err.value)
    assert f"radius={radius}" in str(err.value)


@pytest.mark.parametrize("radius", [0.0, -0.1])
def test_non_positive_radius_is_rejected(radius: float) -> None:
    with pytest.raises(ValueError):
        CharacterController(make_world(), radius=radius, height=1.8)


@pytest.mark.parametrize("kwargs", [{"skin_width": -0.01}, {"skin_width": 0.0},
                                    {"step_offset": -0.1}])
def test_non_positive_skin_or_negative_step_offset_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        CharacterController(make_world(), **kwargs)


def test_controller_itself_accepts_any_slope_limit() -> None:
    """Сам контроллер ``slope_limit`` не проверяет: ограничение — в адаптере."""
    for limit in (-10.0, 120.0):
        CharacterController(make_world(), slope_limit=limit)


# --- 1e: смена в рантайме через компонент --------------------------------------------

def _scene_hero(**kwargs):
    scene = Scene()
    floor = scene.create_game_object("Floor", position=Vector3(0, -0.5, 0),
                                     scale=Vector3(20, 1, 20))
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    hero = scene.create_game_object("Hero", position=Vector3(0, 0.92, 0))
    comp = hero.add_component(CharacterControllerComponent, **kwargs)
    for _ in range(10):
        scene.fixed_update(DT)
    return scene, hero, comp


def _capsule_extent(scene, hero) -> tuple[float, float]:
    handle = scene.physics.handle_of(hero)
    collider = scene.physics.world.get_colliders(handle)[0]
    lo, hi = collider.world_aabb
    return float(hi[1] - lo[1]), float(hi[0] - lo[0])


def test_component_runtime_height_change_rebuilds_capsule() -> None:
    scene, hero, comp = _scene_hero()
    assert _capsule_extent(scene, hero) == pytest.approx((1.8, 0.8))
    comp.height = 1.2
    comp.radius = 0.3
    scene.fixed_update(DT)
    assert _capsule_extent(scene, hero) == pytest.approx((1.2, 0.6))


def test_runtime_change_keeps_transform_position_and_aabb_in_broadphase() -> None:
    scene, hero, comp = _scene_hero()
    comp.height = 1.2
    scene.fixed_update(DT)
    handle = scene.physics.handle_of(hero)
    world = scene.physics.world
    lo, hi = world.get_colliders(handle)[0].world_aabb
    centre_y = float((lo[1] + hi[1]) / 2)
    assert centre_y == pytest.approx(hero.transform.world_position.y, abs=0.05)
    # новый AABB виден запросам broadphase: луч сверху попадает в верх капсулы
    hit = world.raycast((0, 5, 0), (0, -1, 0), 10.0)
    assert hit is not None and hit.handle == handle
    assert hit.point[1] == pytest.approx(hi[1], abs=0.02)


@pytest.mark.parametrize(("field", "value"), [("height", 0.5), ("height", 0.8),
                                              ("radius", 0.0), ("radius", -0.2)])
def test_runtime_invalid_size_is_error_and_not_registered(caplog, field: str,
                                                          value: float) -> None:
    """F8: ERROR с именем объекта и причиной; ``move`` -> RuntimeError с причиной."""
    scene, _hero, comp = _scene_hero()
    setattr(comp, field, value)
    with caplog.at_level(logging.WARNING):
        scene.fixed_update(DT)
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "Hero" in errors[0].getMessage()
    assert field in errors[0].getMessage()
    with pytest.raises(RuntimeError, match="компонент не зарегистрирован") as err:
        comp.move((1, 0, 0))
    assert field in str(err.value)


def test_error_is_logged_once_and_object_recovers_after_fix(caplog) -> None:
    scene, hero, comp = _scene_hero()
    comp.height = 0.5
    with caplog.at_level(logging.ERROR):
        for _ in range(5):
            scene.fixed_update(DT)
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1
    comp.height = 1.8
    scene.fixed_update(DT)
    assert scene.physics.handle_of(hero) is not None
    comp.move((0.1, 0, 0))  # снова зарегистрирован


def test_component_without_registration_error_still_reports_missing_component() -> None:
    scene = Scene()
    go = scene.create_game_object("Plain")
    scene.fixed_update(DT)
    with pytest.raises(RuntimeError, match="нет CharacterControllerComponent"):
        scene.physics.is_grounded(go)


def test_runtime_zero_skin_is_clamped_with_warning(caplog) -> None:
    scene, hero, comp = _scene_hero()
    comp.skin_width = 0.0
    with caplog.at_level(logging.WARNING):
        scene.fixed_update(DT)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "skin_width" in warnings[0].getMessage()
    assert "Hero" in warnings[0].getMessage()
    for _ in range(30):
        scene.fixed_update(DT)
    ctrl = scene.physics._bindings[hero.id].character
    assert ctrl.skin_width == pytest.approx(1e-3)
    comp.move((1.0, 0.0, 0.0))  # зарегистрирован и двигается
    assert hero.transform.world_position.x > 0.5


@pytest.mark.parametrize(("value", "expected"), [(-10.0, 0.0), (120.0, 90.0),
                                                 (0.0, 0.0), (90.0, 90.0),
                                                 (45.0, 45.0)])
def test_runtime_slope_limit_is_clamped_with_warning(caplog, value: float,
                                                     expected: float) -> None:
    scene, hero, comp = _scene_hero()
    comp.slope_limit = value
    with caplog.at_level(logging.WARNING):
        scene.fixed_update(DT)
    warned = any("slope_limit" in r.getMessage() for r in caplog.records)
    assert warned is (value != expected)
    ctrl = scene.physics._bindings[hero.id].character
    assert ctrl.slope_limit == pytest.approx(expected)


def test_direct_attribute_change_on_controller_does_not_resize_shape() -> None:
    """Размеры читаются только при создании: изменение атрибута форму не трогает."""
    char = make_char(make_world())
    char.height = 1.0
    char.radius = 0.2
    assert char.shape.radius == 0.4
    assert char.shape.half_height == pytest.approx(0.5)


# --- 1f: взаимодействие с ColliderComponent -------------------------------------------

def test_controller_uses_own_capsule_and_ignores_collider_component(caplog) -> None:
    scene = Scene()
    hero = scene.create_game_object("Hero", position=Vector3(0, 5, 0),
                                    scale=Vector3(3, 3, 3))
    hero.add_component(ColliderComponent, shape="box", size=(4, 4, 4))
    hero.add_component(CharacterControllerComponent)
    with caplog.at_level(logging.WARNING):
        scene.fixed_update(DT)
    handle = scene.physics.handle_of(hero)
    colliders = scene.physics.world.get_colliders(handle)
    assert len(colliders) == 1
    assert isinstance(colliders[0].shape, CapsuleShape)
    # размеры - из полей контроллера (0.4 / 1.8) × масштаб объекта 3; box-коллайдер
    # (4 × 4 × 4) в размерах капсулы не участвует
    assert _capsule_extent(scene, hero) == pytest.approx((5.4, 2.4))
    assert not caplog.records  # предупреждения о проигнорированном коллайдере нет
    body = scene.physics.world.get_body(handle)
    assert body.body_type == "kinematic"


def test_removing_controller_falls_back_to_collider_component() -> None:
    scene = Scene()
    go = scene.create_game_object("Obj", position=Vector3(0, 5, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    ctrl = go.add_component(CharacterControllerComponent)
    scene.fixed_update(DT)
    handle = scene.physics.handle_of(go)
    assert isinstance(scene.physics.world.get_colliders(handle)[0].shape, CapsuleShape)
    go.remove_component(ctrl)
    scene.fixed_update(DT)
    handle = scene.physics.handle_of(go)
    assert not isinstance(scene.physics.world.get_colliders(handle)[0].shape,
                          CapsuleShape)
