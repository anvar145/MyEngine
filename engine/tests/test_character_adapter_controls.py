"""CharacterControllerComponent в сцене: пассивная гравитация, живые поля, ввод.

Уровень компонента и адаптера (без скриптов): ``scene.fixed_update`` шагает
физику; персонажа, которого за шаг не двигал скрипт, адаптер ведёт сам
(гравитация, ``use_gravity``/``gravity_scale``), а ``move_speed``,
``jump_velocity``, ``sprint``, ``air_control`` читаются при каждом вызове.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from engine.core.math3d import Quaternion, Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
)
from engine.core.scene import Scene
from engine.physics.character import CharacterController
from engine.physics.collider import Collider
from engine.physics.shape import BoxShape
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0
REST_Y = 0.92


def _run(scene: Scene, steps: int) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)
        scene.update(DT)


def _scene(y: float = REST_Y, **cc):
    scene = Scene()
    floor = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(40, 1, 40)
    )
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go = scene.create_game_object("Hero", position=Vector3(0, y, 0))
    comp = go.add_component(CharacterControllerComponent, **cc)
    return scene, go, comp


def _ctrl(scene: Scene, go) -> CharacterController:
    scene.physics.sync_structure()
    return scene.physics._bindings[go.id].character


def _y(go) -> float:
    return go.transform.world_position.y


# --- пассивная гравитация ---------------------------------------------------

def test_character_spawned_in_air_falls_without_any_script():
    scene, go, _ = _scene(y=5.0)
    _run(scene, 70)
    assert scene.physics.is_grounded(go)
    assert _y(go) == pytest.approx(REST_Y, abs=0.01)


def test_grounded_character_stays_put_without_any_script():
    scene, go, _ = _scene()
    _run(scene, 120)
    assert scene.physics.is_grounded(go)
    assert _y(go) == pytest.approx(REST_Y, abs=0.005)


def test_use_gravity_false_keeps_character_in_the_air():
    scene, go, _ = _scene(y=5.0, use_gravity=False)
    _run(scene, 70)
    assert _y(go) == pytest.approx(5.0)


def test_use_gravity_toggle_is_live():
    scene, go, comp = _scene(y=5.0, use_gravity=False)
    _run(scene, 10)
    handle = scene.physics.handle_of(go)
    assert _y(go) == pytest.approx(5.0)
    comp.use_gravity = True
    _run(scene, 70)
    assert _y(go) == pytest.approx(REST_Y, abs=0.01)
    assert scene.physics.handle_of(go) == handle


def test_gravity_scale_zero_then_live_edit_makes_it_fall():
    scene, go, comp = _scene(y=5.0, gravity_scale=0.0)
    _run(scene, 20)
    assert _y(go) == pytest.approx(5.0)
    comp.gravity_scale = 1.0
    _run(scene, 70)
    assert _y(go) == pytest.approx(REST_Y, abs=0.01)


def test_gravity_scale_slows_the_fall():
    def fallen(scale: float) -> float:
        scene, go, _ = _scene(y=30.0, gravity_scale=scale)
        _run(scene, 30)
        return 30.0 - _y(go)

    assert fallen(0.5) == pytest.approx(0.5 * fallen(1.0), rel=0.02)


def test_scripted_character_does_not_get_a_second_gravity_step():
    """Если скрипт двигал персонажа, пассивный шаг этого кадра пропускается."""
    scene, go, _ = _scene(y=30.0)
    world = PhysicsWorld(gravity=(0, -9.81, 0))
    world.add_static(Collider(BoxShape((50, 0.5, 50))), (0, -0.5, 0))
    reference = CharacterController(world, (0, 30.0, 0))
    for _ in range(30):
        scene.physics.update_character(go, (0, 0, 0))
        scene.fixed_update(DT)
        reference.simple_move((0, 0, 0))
        world.step()
    assert _y(go) == pytest.approx(float(reference.position[1]), abs=1e-6)


def test_raw_move_also_counts_as_scripted_and_adds_no_gravity():
    scene, go, comp = _scene(y=5.0)
    for _ in range(30):
        comp.move((0.0, 0.0, 0.0))
        scene.fixed_update(DT)
    assert _y(go) == pytest.approx(5.0)


def test_character_falls_again_when_script_stops_moving_it():
    scene, go, comp = _scene(y=5.0)
    for _ in range(10):
        comp.move((0.0, 0.0, 0.0))
        scene.fixed_update(DT)
    assert _y(go) == pytest.approx(5.0)
    _run(scene, 70)
    assert _y(go) == pytest.approx(REST_Y, abs=0.01)


# --- живые поля -------------------------------------------------------------

def _speed(scene, comp, direction=(1.0, 0.0, 0.0)) -> float:
    moved = comp.move_input(direction)
    scene.fixed_update(DT)
    return moved[0] / DT


def test_move_input_uses_move_speed_and_edit_is_live():
    scene, go, comp = _scene()
    _run(scene, 5)
    handle = scene.physics.handle_of(go)
    assert _speed(scene, comp) == pytest.approx(4.5, rel=0.01)
    comp.move_speed = 9.0
    assert _speed(scene, comp) == pytest.approx(9.0, rel=0.01)
    assert scene.physics.handle_of(go) == handle


def test_jump_without_argument_uses_jump_velocity_and_edit_is_live():
    scene, go, comp = _scene()
    _run(scene, 5)
    comp.jump_velocity = 7.5
    assert comp.jump() is True
    assert _ctrl(scene, go).vertical_velocity == pytest.approx(7.5)


def test_jump_with_argument_keeps_working_as_before():
    scene, go, comp = _scene()
    _run(scene, 5)
    assert comp.jump(3.0) is True
    assert _ctrl(scene, go).vertical_velocity == pytest.approx(3.0)


def test_jump_is_refused_without_gravity():
    scene, _, comp = _scene(use_gravity=False)
    _run(scene, 5)
    assert comp.jump() is False


def test_ground_check_distance_edit_is_live():
    scene, go, comp = _scene(y=0.9 + 0.02 + 0.08, use_gravity=False)
    _run(scene, 3)
    assert not scene.physics.is_grounded(go)
    comp.ground_check_distance = 0.1
    comp.move((0.0, 0.0, 0.0))
    assert scene.physics.is_grounded(go)


def test_snap_to_ground_edit_reaches_the_controller():
    scene, go, comp = _scene()
    _run(scene, 3)
    assert _ctrl(scene, go).snap_to_ground is True
    comp.snap_to_ground = False
    comp.move((0.0, 0.0, 0.0))
    assert _ctrl(scene, go).snap_to_ground is False


def test_invalid_live_value_is_clamped_with_one_warning(caplog):
    scene, go, comp = _scene()
    _run(scene, 3)
    comp.move_speed = -5.0
    with caplog.at_level(logging.WARNING, logger="engine.core.physics_adapter"):
        assert _speed(scene, comp) == pytest.approx(0.0, abs=1e-6)
        _speed(scene, comp)
        _run(scene, 2)
    warnings = [r for r in caplog.records if "move_speed" in r.getMessage()]
    assert len(warnings) == 1
    assert _ctrl(scene, go).move_speed == 0.0


def test_non_numeric_live_value_falls_back_to_default(caplog):
    scene, go, comp = _scene()
    _run(scene, 3)
    comp.move_speed = "fast"
    with caplog.at_level(logging.WARNING, logger="engine.core.physics_adapter"):
        _speed(scene, comp)
    assert _ctrl(scene, go).move_speed == 4.5
    assert any("move_speed" in r.getMessage() for r in caplog.records)


def test_constructor_values_reach_the_controller():
    custom = {
        "move_speed": 6.5, "sprint_multiplier": 2.0, "jump_velocity": 7.5,
        "use_gravity": False, "gravity_scale": 0.5, "ground_check_distance": 0.08,
        "air_control": 0.75, "snap_to_ground": False,
    }
    scene, go, _ = _scene(**custom)
    ctrl = _ctrl(scene, go)
    for name, value in custom.items():
        assert getattr(ctrl, name) == value, name


# --- спринт и воздушный контроль --------------------------------------------

def test_sprint_multiplies_move_speed_until_switched_off():
    scene, _, comp = _scene()
    _run(scene, 5)
    comp.sprint(True)
    assert comp.is_sprinting
    assert _speed(scene, comp) == pytest.approx(4.5 * 1.6, rel=0.01)
    assert _speed(scene, comp) == pytest.approx(4.5 * 1.6, rel=0.01)  # сохраняется
    comp.sprint(False)
    assert _speed(scene, comp) == pytest.approx(4.5, rel=0.01)


def test_sprint_uses_live_multiplier():
    scene, _, comp = _scene(sprint_multiplier=2.0)
    _run(scene, 5)
    comp.sprint(True)
    comp.sprint_multiplier = 3.0
    assert _speed(scene, comp) == pytest.approx(13.5, rel=0.01)


def test_sprint_does_not_touch_explicit_simple_move():
    scene, _, comp = _scene()
    _run(scene, 5)
    comp.sprint(True)
    moved = comp.simple_move((3.0, 0.0, 0.0))
    assert moved[0] / DT == pytest.approx(3.0, rel=0.01)


def test_sprint_state_survives_live_edits():
    scene, _, comp = _scene()
    _run(scene, 3)
    comp.sprint(True)
    comp.move_speed = 5.0
    _run(scene, 2)
    assert comp.is_sprinting


def test_set_air_control_overrides_field_until_cleared():
    scene, _, comp = _scene(y=30.0, air_control=0.3)
    comp.set_air_control(1.0)
    assert comp.air_control_override == 1.0
    assert comp.air_control == 0.3  # поле не тронуто
    assert _speed(scene, comp) == pytest.approx(4.5, rel=0.01)
    comp.set_air_control(None)
    assert comp.air_control_override is None
    follow = _speed(scene, comp)  # дальше сближение на долю 0.3 от разницы
    assert follow == pytest.approx(4.5, rel=0.01)  # уже на полной скорости
    comp.move_speed = 0.0
    assert _speed(scene, comp) == pytest.approx(4.5 * 0.7, rel=0.02)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_set_air_control_rejects_out_of_range(bad):
    _, _, comp = _scene()
    with pytest.raises(ValueError, match="air_control"):
        comp.set_air_control(bad)


# --- выравнивание по земле --------------------------------------------------

def _slope_scene(degrees: float, alignment: bool, yaw: float = 0.0):
    scene = Scene()
    slope = scene.create_game_object(
        "Slope",
        position=Vector3(0, -0.5, 0),
        rotation=Quaternion.from_axis_angle(Vector3.forward(), degrees),
        scale=Vector3(40, 1, 40),
    )
    slope.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go = scene.create_game_object(
        "Hero", position=Vector3(0, 0.95, 0),
        rotation=Quaternion.from_axis_angle(Vector3.up(), yaw),
    )
    comp = go.add_component(CharacterControllerComponent, alignment_to_ground=alignment)
    return scene, go, comp


def _up_of(go) -> Vector3:
    return go.transform.world_rotation.rotate_vector(Vector3.up())


def _normal_of(scene, go) -> np.ndarray:
    return _ctrl(scene, go).ground_normal


@pytest.mark.parametrize("degrees", [15.0, 30.0])
def test_alignment_tilts_transform_to_ground_normal(degrees):
    scene, go, _ = _slope_scene(degrees, alignment=True)
    _run(scene, 90)
    assert scene.physics.is_grounded(go)
    normal = _normal_of(scene, go)
    up = _up_of(go)
    assert (up.x, up.y, up.z) == pytest.approx(tuple(normal), abs=1e-4)
    expected = math.radians(degrees)
    assert up.y == pytest.approx(math.cos(expected), abs=0.02)


def test_alignment_keeps_yaw():
    yaw = 70.0
    scene, go, _ = _slope_scene(20.0, alignment=True, yaw=yaw)
    _run(scene, 90)
    rot = go.transform.world_rotation
    # рыскание = поворот вокруг Y компоненты "twist" кватерниона
    twist = Quaternion(0.0, rot.y, 0.0, rot.w).normalized
    expected = Quaternion.from_axis_angle(Vector3.up(), yaw)
    assert abs(twist.dot(expected)) == pytest.approx(1.0, abs=1e-4)


def test_alignment_off_leaves_rotation_alone():
    scene, go, _ = _slope_scene(30.0, alignment=False)
    _run(scene, 90)
    up = _up_of(go)
    assert (up.x, up.y, up.z) == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)


def test_alignment_does_not_change_the_physical_capsule():
    scene, go, _ = _slope_scene(30.0, alignment=True)
    _run(scene, 90)
    ctrl = _ctrl(scene, go)
    assert ctrl.shape.radius == pytest.approx(0.4)
    assert ctrl.height == pytest.approx(1.8)
    p = go.transform.world_position
    assert (p.x, p.y, p.z) == pytest.approx(tuple(ctrl.position), abs=1e-5)


def test_alignment_is_not_applied_in_the_air():
    scene = Scene()
    go = scene.create_game_object("Hero", position=Vector3(0, 50, 0))
    go.add_component(CharacterControllerComponent, alignment_to_ground=True)
    _run(scene, 10)
    up = _up_of(go)
    assert (up.x, up.y, up.z) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_switching_alignment_off_restores_upright_pose_keeping_yaw():
    scene, go, comp = _slope_scene(30.0, alignment=True, yaw=40.0)
    _run(scene, 90)
    assert _up_of(go).y < 0.95
    comp.alignment_to_ground = False
    _run(scene, 3)
    up = _up_of(go)
    assert (up.x, up.y, up.z) == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    rot = go.transform.world_rotation
    expected = Quaternion.from_axis_angle(Vector3.up(), 40.0)
    assert abs(rot.dot(expected)) == pytest.approx(1.0, abs=1e-4)
