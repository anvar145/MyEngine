"""Масштаб объекта и размеры капсулы ``CharacterController``.

Капсула берёт размеры из мирового масштаба объекта по формуле коллайдера
(``ColliderComponent`` капсулы), раздельно:

* ``radius`` умножается на ``max(|sx|, |sz|)``, ``height`` - на ``|sy|``;
* ``|s| < 1e-3`` поднимается до ``1e-3`` с одним предупреждением на объект;
* ``height <= 2 * radius`` после масштаба -> ``height := 2 * radius + eps`` с одним
  предупреждением (не ошибка);
* ``skin_width``, ``step_offset``, ``ground_check_distance`` и скорости масштаб не
  затрагивает;
* при смене размеров (масштаб объекта/родителя или поля компонента) двойник
  пересоздаётся **с ногами на месте**: нижняя точка капсулы не двигается, центр
  смещается на ``delta_height / 2``; ``vertical_velocity`` и
  ``horizontal_velocity`` переносятся; мелкие правки (< 1e-6) двойника не трогают.

Подробности - ``docs/CORE.md`` §7.2.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from engine.core import physics_adapter
from engine.core.math3d import Quaternion, Vector3
from engine.core.physics_adapter import scaled_capsule_size
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
)
from engine.core.scene import Scene
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0
REST_Y = 0.92  # центр капсулы 0.4/1.8 на полу: height / 2 + skin_width
EPS_HEIGHT = 2e-3  # запас height над 2 * radius при выравнивании


def _scene(floor: bool = True) -> Scene:
    scene = Scene()
    if floor:
        ground = scene.create_game_object(
            "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(40, 1, 40)
        )
        ground.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    return scene


def _hero(scene: Scene, y: float = REST_Y, scale=(1, 1, 1), parent=None, **fields):
    go = scene.create_game_object(
        "Hero", parent, position=Vector3(0, y, 0), scale=Vector3(*scale)
    )
    comp = go.add_component(CharacterControllerComponent, **fields)
    return go, comp


def _run(scene: Scene, steps: int) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)


def _ctrl(scene: Scene, hero):
    scene.physics.sync_structure()
    return scene.physics._bindings[hero.id].character


def _extent(scene: Scene, hero) -> tuple[float, float]:
    """(высота, диаметр) капсулы по AABB коллайдера в мире."""
    handle = scene.physics.handle_of(hero)
    lo, hi = scene.physics.world.get_colliders(handle)[0].world_aabb
    return float(hi[1] - lo[1]), float(hi[0] - lo[0])


def _bottom(scene: Scene, hero) -> float:
    handle = scene.physics.handle_of(hero)
    lo, _ = scene.physics.world.get_colliders(handle)[0].world_aabb
    return float(lo[1])


@pytest.fixture
def builds(monkeypatch):
    """Счётчик созданных контроллеров (вызов проходит к настоящему классу)."""
    calls = [0]
    original = physics_adapter.CharacterController

    def counting(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(physics_adapter, "CharacterController", counting)
    return calls


# --- формула ------------------------------------------------------------------

@pytest.mark.parametrize(
    ("scale", "radius", "height"),
    [
        ((1, 1, 1), 0.4, 1.8),
        ((2, 2, 2), 0.8, 3.6),
        ((2, 1, 2), 0.8, 1.8),
        ((0.5, 0.5, 0.5), 0.2, 0.9),
        ((1, 3, 1), 0.4, 5.4),
        ((2, 2, 3), 1.2, 3.6),  # радиус - по большей из осей X/Z
        ((3, 2, 2), 1.2, 3.6),
        ((-2, 1, -2), 0.8, 1.8),  # зеркало: масштаб берётся по модулю
        ((1, -2, 1), 0.4, 3.6),
    ],
)
def test_object_scale_sets_capsule_size(scale, radius, height):
    scene = _scene()
    hero, _ = _hero(scene, y=4.0, scale=scale)
    _run(scene, 2)

    ctrl = _ctrl(scene, hero)
    assert ctrl.radius == pytest.approx(radius)
    assert ctrl.height == pytest.approx(height)
    assert _extent(scene, hero) == pytest.approx((height, 2 * radius))


def test_parent_scale_applies_to_child_character():
    scene = _scene()
    parent = scene.create_game_object(
        "Parent", position=Vector3(0, 4, 0), scale=Vector3(2, 2, 2)
    )
    hero, _ = _hero(scene, y=0.0, parent=parent)
    _run(scene, 2)

    ctrl = _ctrl(scene, hero)
    assert (ctrl.radius, ctrl.height) == pytest.approx((0.8, 3.6))


def test_parent_and_own_scale_multiply():
    scene = _scene()
    parent = scene.create_game_object(
        "Parent", position=Vector3(0, 4, 0), scale=Vector3(2, 2, 2)
    )
    hero, _ = _hero(scene, y=0.0, scale=(0.5, 0.5, 0.5), parent=parent)
    _run(scene, 2)

    ctrl = _ctrl(scene, hero)
    assert (ctrl.radius, ctrl.height) == pytest.approx((0.4, 1.8))


@pytest.mark.parametrize(
    ("radius", "height", "scale"),
    [(0.4, 1.8, (2, 2, 2)), (0.3, 1.4, (1, 2, 3)), (0.5, 3.0, (0.7, 1.9, 0.2))],
)
def test_character_capsule_matches_capsule_collider(radius, height, scale):
    """Капсула персонажа и капсульный ``ColliderComponent`` считаются одинаково."""
    scene = _scene(floor=False)
    ghost = scene.create_game_object(
        "Ghost", position=Vector3(0, 50, 0), scale=Vector3(*scale)
    )
    ghost.add_component(
        ColliderComponent, shape="capsule", radius=radius, height=height
    )
    hero, _ = _hero(
        scene, y=100.0, scale=scale, radius=radius, height=height, use_gravity=False
    )
    _run(scene, 2)

    assert _extent(scene, hero) == pytest.approx(_extent(scene, ghost))


@pytest.mark.parametrize(
    ("radius", "height", "scale", "expected"),
    [
        (0.4, 1.8, (1, 1, 1), (0.4, 1.8, False)),
        (0.4, 1.8, (2, 1, 2), (0.8, 1.8, False)),
        (0.4, 1.8, (4, 1, 4), (1.6, 3.2 + EPS_HEIGHT, True)),
        (0.4, 1.8, (-4, 1, 4), (1.6, 3.2 + EPS_HEIGHT, True)),
        (0.4, 1.8, (-4, 1, -4), (1.6, 3.2 + EPS_HEIGHT, True)),  # зеркало по X и Z
        (0.4, 1.8, (1, -2, 1), (0.4, 3.6, False)),  # зеркало по Y: |sy|
        (0.4, 1.8, (0, 1, 0), (0.001, 1.8, False)),  # ось не меньше 1e-3
    ],
)
def test_scaled_capsule_size(radius, height, scale, expected):
    got = scaled_capsule_size(radius, height, np.array(scale, dtype=float))
    assert got[:2] == pytest.approx(expected[:2])
    assert got[2] is expected[2]


# --- крайние случаи -------------------------------------------------------------

def test_zero_scale_axis_is_clamped_with_one_warning(caplog):
    scene = _scene()
    hero, _ = _hero(scene, scale=(0, 1, 1))
    with caplog.at_level(logging.WARNING):
        _run(scene, 30)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Hero" in warnings[0].getMessage()
    assert "масштаб" in warnings[0].getMessage()
    assert scene.physics.handle_of(hero) is not None
    ctrl = _ctrl(scene, hero)
    assert (ctrl.radius, ctrl.height) == pytest.approx((0.4, 1.8))


def test_zero_height_scale_is_clamped_and_object_stays_in_world(caplog):
    scene = _scene()
    hero, _ = _hero(scene, scale=(1, 0, 1))
    with caplog.at_level(logging.WARNING):
        _run(scene, 30)

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert scene.physics.handle_of(hero) is not None
    ctrl = _ctrl(scene, hero)
    assert ctrl.height == pytest.approx(2 * ctrl.radius + EPS_HEIGHT)


def test_height_not_above_diameter_after_scale_is_raised_with_one_warning(caplog):
    scene = _scene()
    hero, _ = _hero(scene, y=3.0, scale=(4, 1, 4))  # r = 1.6, h = 1.8 <= 2r
    with caplog.at_level(logging.WARNING):
        _run(scene, 60)

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Hero" in warnings[0].getMessage()
    assert "height" in warnings[0].getMessage()
    ctrl = _ctrl(scene, hero)
    assert ctrl.radius == pytest.approx(1.6)
    assert ctrl.height == pytest.approx(3.2 + EPS_HEIGHT)
    assert scene.physics.is_grounded(hero) is True


def test_own_height_not_above_diameter_is_still_an_error(caplog):
    """Ошибка в самих полях компонента масштабом не лечится."""
    scene = _scene()
    hero, _comp = _hero(scene, scale=(2, 2, 2), height=0.5)
    with caplog.at_level(logging.WARNING):
        _run(scene, 2)

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "height" in errors[0].getMessage()
    assert scene.physics.handle_of(hero) is None


def test_other_parameters_are_not_scaled():
    scene = _scene()
    hero, comp = _hero(scene, scale=(2, 2, 2))
    _run(scene, 2)

    ctrl = _ctrl(scene, hero)
    assert ctrl.skin_width == pytest.approx(comp.skin_width)
    assert ctrl.step_offset == pytest.approx(comp.step_offset)
    assert ctrl.slope_limit == pytest.approx(comp.slope_limit)
    assert ctrl.ground_check_distance == pytest.approx(comp.ground_check_distance)
    assert ctrl.move_speed == pytest.approx(comp.move_speed)
    assert ctrl.jump_velocity == pytest.approx(comp.jump_velocity)


def test_scaled_character_stands_on_ground_with_skin_gap():
    scene = _scene()
    hero, _ = _hero(scene, y=5.0, scale=(2, 2, 2))
    _run(scene, 120)

    assert scene.physics.is_grounded(hero) is True
    assert hero.transform.world_position.y == pytest.approx(1.82, abs=2e-3)
    assert _bottom(scene, hero) == pytest.approx(0.02, abs=2e-3)


# --- смена масштаба на лету ------------------------------------------------------

def test_scale_change_rebuilds_twin_with_new_size():
    scene = _scene()
    hero, _ = _hero(scene)
    _run(scene, 10)

    hero.transform.scale = Vector3(2, 2, 2)
    _run(scene, 2)

    ctrl = _ctrl(scene, hero)
    assert (ctrl.radius, ctrl.height) == pytest.approx((0.8, 3.6))
    assert _extent(scene, hero) == pytest.approx((3.6, 1.6))


def test_parent_scale_change_rebuilds_twin():
    scene = _scene()
    # центр героя совпадает с началом координат родителя: масштаб его не сдвигает
    parent = scene.create_game_object("Parent", position=Vector3(0, REST_Y, 0))
    hero, _ = _hero(scene, y=0.0, parent=parent)
    _run(scene, 10)

    parent.transform.scale = Vector3(1, 0.5, 1)
    _run(scene, 2)

    ctrl = _ctrl(scene, hero)
    assert (ctrl.radius, ctrl.height) == pytest.approx((0.4, 0.9))
    assert _bottom(scene, hero) == pytest.approx(0.02, abs=1e-3)


@pytest.mark.parametrize("new_scale", [(1, 2, 1), (1, 0.5, 1), (2, 2, 2)])
def test_scale_change_keeps_feet_in_place(new_scale):
    scene = _scene()
    hero, _ = _hero(scene, y=5.0, use_gravity=False)
    _run(scene, 5)
    feet = _bottom(scene, hero)
    old_height = _ctrl(scene, hero).height

    hero.transform.scale = Vector3(*new_scale)
    _run(scene, 2)

    ctrl = _ctrl(scene, hero)
    assert ctrl.height == pytest.approx(old_height * new_scale[1])
    assert _bottom(scene, hero) == pytest.approx(feet, abs=1e-6)
    # центр сдвинулся на половину изменения высоты и записан в трансформ
    shift = (ctrl.height - old_height) / 2
    assert hero.transform.world_position.y == pytest.approx(5.0 + shift, abs=1e-4)
    assert ctrl.position[1] == pytest.approx(5.0 + shift, abs=1e-4)


def test_scale_change_of_grounded_hero_does_not_lift_or_sink_feet():
    scene = _scene()
    hero, _ = _hero(scene)
    _run(scene, 30)
    assert _bottom(scene, hero) == pytest.approx(0.02, abs=1e-3)

    hero.transform.scale = Vector3(1, 2, 1)
    _run(scene, 30)

    assert _bottom(scene, hero) == pytest.approx(0.02, abs=1e-3)
    assert scene.physics.is_grounded(hero) is True
    assert hero.transform.world_position.y == pytest.approx(1.82, abs=2e-3)


def test_component_height_change_keeps_feet_in_place():
    scene = _scene()
    hero, comp = _hero(scene, y=5.0, use_gravity=False)
    _run(scene, 5)
    feet = _bottom(scene, hero)

    comp.height = 2.4
    _run(scene, 2)

    assert _bottom(scene, hero) == pytest.approx(feet, abs=1e-6)
    assert hero.transform.world_position.y == pytest.approx(5.3, abs=1e-4)


def test_radius_change_keeps_center():
    scene = _scene()
    hero, comp = _hero(scene, y=5.0, use_gravity=False)
    _run(scene, 5)

    comp.radius = 0.6
    _run(scene, 2)

    assert _ctrl(scene, hero).radius == pytest.approx(0.6)
    assert hero.transform.world_position.y == pytest.approx(5.0, abs=1e-6)


def test_scale_change_carries_velocities():
    scene = _scene()
    hero, _ = _hero(scene, y=8.0)
    _run(scene, 20)
    old = _ctrl(scene, hero)
    old.horizontal_velocity = np.array([1.5, 0.0, -0.5])
    vertical = old.vertical_velocity
    assert vertical < -1.0

    hero.transform.scale = Vector3(2, 2, 2)
    scene.physics.sync_in()

    new = _ctrl(scene, hero)
    assert new is not old
    assert new.vertical_velocity == pytest.approx(vertical)
    assert new.horizontal_velocity == pytest.approx([1.5, 0.0, -0.5])


def test_component_edit_carries_velocities_too():
    scene = _scene()
    hero, comp = _hero(scene, y=8.0)
    _run(scene, 20)
    vertical = _ctrl(scene, hero).vertical_velocity

    comp.radius = 0.5
    scene.physics.sync_in()

    assert _ctrl(scene, hero).vertical_velocity == pytest.approx(vertical)


# --- двойник не пересоздаётся без причины ---------------------------------------

def test_static_scale_does_not_rebuild_twin(builds):
    scene = _scene()
    _hero(scene, scale=(2, 2, 2))
    _run(scene, 120)

    assert builds[0] == 1


def test_scale_change_rebuilds_twin_once(builds):
    scene = _scene()
    hero, _ = _hero(scene)
    _run(scene, 10)
    hero.transform.scale = Vector3(1, 2, 1)
    _run(scene, 120)

    assert builds[0] == 2


def test_tiny_scale_noise_does_not_rebuild_twin(builds):
    scene = _scene()
    hero, _ = _hero(scene)
    _run(scene, 10)

    hero.transform.scale = Vector3(1 + 1e-8, 1 + 1e-8, 1 + 1e-8)
    _run(scene, 10)
    assert builds[0] == 1

    hero.transform.scale = Vector3(1, 1.001, 1)
    _run(scene, 10)
    assert builds[0] == 2


def test_rotation_and_movement_do_not_rebuild_twin(builds):
    scene = _scene()
    hero, _ = _hero(scene, scale=(2, 1, 2))
    _run(scene, 5)

    for k in range(60):
        hero.transform.rotation = Quaternion.from_axis_angle(
            Vector3(0, 1, 0), 3.0 * k
        )
        scene.physics.update_character(hero, (1.0, 0.0, 0.0))
        scene.fixed_update(DT)

    assert builds[0] == 1


def test_scale_change_does_not_break_resting_fast_path(monkeypatch):
    calls = [0]
    original = PhysicsWorld.shapecast

    def counting(self, *args, **kwargs):
        calls[0] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PhysicsWorld, "shapecast", counting)
    scene = _scene()
    hero, _ = _hero(scene)
    _run(scene, 10)
    hero.transform.scale = Vector3(2, 2, 2)
    _run(scene, 30)  # пересоздан и осел
    pose = _ctrl(scene, hero).position.tobytes()
    calls[0] = 0

    _run(scene, 60)

    assert calls[0] == 0
    assert _ctrl(scene, hero).position.tobytes() == pose
