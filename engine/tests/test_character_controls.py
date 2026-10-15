"""CharacterController (ядро): параметры движения, пассивная гравитация, ввод.

Поля ``move_speed``/``sprint_multiplier``/``jump_velocity``/``use_gravity``/
``gravity_scale``/``ground_check_distance``/``air_control``/``snap_to_ground``
и методы ``step`` (гравитация без ввода), ``move_input`` (нормированное
направление × скорость, бег, воздушный контроль), ``jump()`` без аргумента.
"""

from __future__ import annotations

import numpy as np
import pytest
from character_kit import DT, add_slope, foot_y, make_char, make_world, settle

from engine.physics.character import CharacterController

REST_Y = 0.92  # height/2 + skin_width для значений по умолчанию

DEFAULTS = {
    "move_speed": 4.5,
    "sprint_multiplier": 1.6,
    "jump_velocity": 5.0,
    "use_gravity": True,
    "gravity_scale": 1.0,
    "ground_check_distance": 0.03,
    "air_control": 0.3,
    "snap_to_ground": True,
}
INVALID = [
    ("move_speed", -0.1),
    ("move_speed", float("nan")),
    ("sprint_multiplier", 0.99),
    ("jump_velocity", 0.0),
    ("jump_velocity", -1.0),
    ("gravity_scale", -0.01),
    ("ground_check_distance", 0.0),
    ("air_control", -0.01),
    ("air_control", 1.01),
]


def _fall(char: CharacterController, seconds: float) -> None:
    """Пассивные шаги: вход скрипта отсутствует, работает только гравитация."""
    for _ in range(round(seconds * 60)):
        char.step()
        char.world.step()


def _speed_x(displacement: np.ndarray, dt: float = DT) -> float:
    return float(displacement[0]) / dt


# --- поля и валидация -------------------------------------------------------

@pytest.mark.parametrize(("name", "value"), DEFAULTS.items())
def test_default_value(name, value):
    char = make_char(make_world())
    assert getattr(char, name) == value


def test_constructor_accepts_custom_values():
    custom = {
        "move_speed": 6.0, "sprint_multiplier": 2.0, "jump_velocity": 7.0,
        "use_gravity": False, "gravity_scale": 0.5, "ground_check_distance": 0.1,
        "air_control": 0.8, "snap_to_ground": False,
    }
    char = make_char(make_world(), **custom)
    for name, value in custom.items():
        assert getattr(char, name) == value


@pytest.mark.parametrize(("name", "value"), INVALID)
def test_constructor_rejects_invalid(name, value):
    with pytest.raises(ValueError, match=name):
        make_char(make_world(), **{name: value})


# --- пассивная гравитация (step) --------------------------------------------

def test_step_lands_character_spawned_above_ground():
    char = make_char(make_world(), y=5.0)
    assert not char.grounded
    _fall(char, 1.2)
    assert char.grounded
    assert char.position[1] == pytest.approx(REST_Y, abs=0.01)


def test_step_keeps_grounded_character_still():
    char = make_char(make_world(), y=REST_Y)
    assert char.grounded
    start = char.position.copy()
    _fall(char, 1.0)
    assert char.grounded
    assert np.allclose(char.position, start, atol=1e-3)


def test_step_uses_world_fixed_dt_by_default_and_explicit_dt():
    a = make_char(make_world(), y=5.0)
    b = make_char(make_world(), y=5.0)
    a.step()
    b.step(DT)
    assert np.allclose(a.position, b.position)


def test_use_gravity_false_does_not_fall():
    char = make_char(make_world(), y=5.0, use_gravity=False)
    _fall(char, 1.0)
    assert char.position[1] == pytest.approx(5.0)
    assert char.vertical_velocity == 0.0


def test_use_gravity_false_also_applies_to_simple_move():
    char = make_char(make_world(), y=5.0, use_gravity=False)
    for _ in range(30):
        char.simple_move((1.0, 0.0, 0.0))
        char.world.step()
    assert char.position[1] == pytest.approx(5.0)
    assert char.position[0] > 0.4


def test_use_gravity_false_forbids_jump():
    char = make_char(make_world(), y=REST_Y, use_gravity=False)
    assert char.grounded
    assert char.jump(5.0) is False
    assert char.vertical_velocity == 0.0


def _fallen(scale: float, seconds: float = 0.5) -> float:
    char = make_char(make_world(), y=20.0, gravity_scale=scale)
    start = float(char.position[1])
    _fall(char, seconds)
    return start - float(char.position[1])


def test_gravity_scale_scales_fall_distance():
    base = _fallen(1.0)
    assert base == pytest.approx(0.5 * 9.81 * 0.25, rel=0.1)
    assert _fallen(2.0) == pytest.approx(2.0 * base, rel=0.02)
    assert _fallen(0.5) == pytest.approx(0.5 * base, rel=0.02)


def test_gravity_scale_zero_means_no_gravity():
    assert _fallen(0.0) == pytest.approx(0.0, abs=1e-9)


def test_gravity_scale_also_applies_to_simple_move():
    def drop(scale: float) -> float:
        char = make_char(make_world(), y=20.0, gravity_scale=scale)
        for _ in range(30):
            char.simple_move((0.0, 0.0, 0.0))
            char.world.step()
        return 20.0 - float(char.position[1])

    assert drop(2.0) == pytest.approx(2.0 * drop(1.0), rel=0.02)


def test_gravity_follows_world_gravity_times_scale():
    """Гравитация персонажа - ``world.gravity[1] * gravity_scale``."""
    def drop(world_gravity: float, scale: float) -> float:
        char = make_char(make_world(world_gravity), y=50.0, gravity_scale=scale)
        _fall(char, 0.5)
        return 50.0 - float(char.position[1])

    assert drop(-20.0, 1.0) == pytest.approx(drop(-10.0, 2.0), rel=0.01)


# --- проба земли ------------------------------------------------------------

def _hovering(gap: float, **kwargs) -> CharacterController:
    """Персонаж без гравитации на расстоянии ``gap`` от нижнего полюса до пола."""
    char = make_char(make_world(), use_gravity=False, **kwargs)
    char.teleport((0.0, 0.9 + char.skin_width + gap, 0.0))
    return char


def test_default_probe_is_skin_plus_three_centimetres():
    assert _hovering(0.02).grounded
    assert not _hovering(0.05).grounded


def test_larger_ground_check_distance_reaches_farther():
    assert not _hovering(0.08).grounded
    assert _hovering(0.08, ground_check_distance=0.1).grounded


def test_smaller_ground_check_distance_reaches_less():
    assert _hovering(0.02).grounded
    assert not _hovering(0.02, ground_check_distance=0.01).grounded


# --- прилипание к земле -----------------------------------------------------

def _descend(snap: bool) -> int:
    """Число кадров без контакта при спуске по склону 30° с ходьбой."""
    world = make_world()
    add_slope(world, 30.0, length=12.0, x0=2.0)
    char = make_char(world, x=9.0, y=6.0, snap_to_ground=snap)
    for _ in range(120):
        char.step()
        world.step()
    assert char.grounded
    airborne = 0
    for _ in range(40):
        char.simple_move((-4.5, 0.0, 0.0))
        world.step()
        airborne += 0 if char.grounded else 1
    return airborne


def test_snap_to_ground_keeps_contact_while_descending():
    assert _descend(snap=True) == 0


def test_without_snap_to_ground_character_leaves_slope_while_descending():
    assert _descend(snap=False) > 0


# --- jump() без аргумента ---------------------------------------------------

def test_jump_without_argument_uses_jump_velocity():
    char = make_char(make_world(), jump_velocity=7.5)
    settle(char)
    assert char.jump() is True
    assert char.vertical_velocity == pytest.approx(7.5)


def test_jump_with_argument_overrides_jump_velocity():
    char = make_char(make_world(), jump_velocity=7.5)
    settle(char)
    assert char.jump(3.0) is True
    assert char.vertical_velocity == pytest.approx(3.0)


def test_jump_rejects_non_positive_speed():
    char = make_char(make_world())
    settle(char)
    with pytest.raises(ValueError):
        char.jump(0.0)


# --- move_input -------------------------------------------------------------

def test_move_input_scales_direction_by_move_speed():
    char = make_char(make_world())
    settle(char)
    assert _speed_x(char.move_input((1.0, 0.0, 0.0))) == pytest.approx(4.5, rel=0.01)


def test_move_input_uses_custom_move_speed():
    char = make_char(make_world(), move_speed=8.0)
    settle(char)
    assert _speed_x(char.move_input((1.0, 0.0, 0.0))) == pytest.approx(8.0, rel=0.01)


def test_move_input_zero_direction_stands_still():
    char = make_char(make_world())
    settle(char)
    assert np.allclose(char.move_input((0.0, 0.0, 0.0))[[0, 2]], 0.0)


def test_move_input_diagonal_is_not_faster():
    char = make_char(make_world())
    settle(char)
    moved = char.move_input((1.0, 0.0, 1.0))
    assert np.hypot(moved[0], moved[2]) / DT == pytest.approx(4.5, rel=0.01)


def test_move_input_partial_stick_deflection_is_proportional():
    char = make_char(make_world())
    settle(char)
    assert _speed_x(char.move_input((0.5, 0.0, 0.0))) == pytest.approx(2.25, rel=0.01)


def test_move_input_ignores_vertical_component():
    char = make_char(make_world())
    settle(char)
    moved = char.move_input((1.0, 7.0, 0.0))
    assert _speed_x(moved) == pytest.approx(4.5, rel=0.01)
    assert abs(moved[1]) < 1e-3


def test_move_input_sprint_multiplies_speed():
    char = make_char(make_world())
    settle(char)
    moved = char.move_input((1.0, 0.0, 0.0), sprint=True)
    assert _speed_x(moved) == pytest.approx(4.5 * 1.6, rel=0.01)


def test_move_input_sprint_uses_custom_multiplier():
    char = make_char(make_world(), sprint_multiplier=2.0)
    settle(char)
    moved = char.move_input((1.0, 0.0, 0.0), sprint=True)
    assert _speed_x(moved) == pytest.approx(9.0, rel=0.01)


def test_move_input_applies_gravity_and_lands():
    char = make_char(make_world(), y=5.0)
    for _ in range(80):
        char.move_input((0.0, 0.0, 0.0))
        char.world.step()
    assert char.grounded
    assert char.position[1] == pytest.approx(REST_Y, abs=0.01)


# --- воздушный контроль -----------------------------------------------------

def _airborne(**kwargs) -> CharacterController:
    char = make_char(make_world(), y=30.0, **kwargs)
    assert not char.grounded
    return char


def test_air_control_blends_toward_wanted_velocity():
    char = _airborne()
    first = _speed_x(char.move_input((1.0, 0.0, 0.0)))
    second = _speed_x(char.move_input((1.0, 0.0, 0.0)))
    assert first == pytest.approx(0.3 * 4.5, rel=0.01)
    assert second == pytest.approx(first + (4.5 - first) * 0.3, rel=0.01)


def test_air_control_one_gives_full_control():
    char = _airborne(air_control=1.0)
    assert _speed_x(char.move_input((1.0, 0.0, 0.0))) == pytest.approx(4.5, rel=0.01)


def test_air_control_zero_keeps_momentum():
    char = _airborne(air_control=0.0)
    assert _speed_x(char.move_input((1.0, 0.0, 0.0))) == pytest.approx(0.0, abs=1e-6)


def test_air_control_argument_overrides_field():
    char = _airborne(air_control=0.3)
    moved = char.move_input((1.0, 0.0, 0.0), air_control=1.0)
    assert _speed_x(moved) == pytest.approx(4.5, rel=0.01)
    assert char.air_control == 0.3


def test_air_control_is_frame_rate_independent():
    fine = _airborne()
    fine.move_input((1.0, 0.0, 0.0))
    fine.move_input((1.0, 0.0, 0.0))
    coarse = _airborne()
    moved = coarse.move_input((1.0, 0.0, 0.0), dt=2 * DT)
    assert _speed_x(moved, 2 * DT) == pytest.approx(
        fine.horizontal_velocity[0], rel=0.01
    )


def test_momentum_is_kept_when_leaving_the_ground():
    char = make_char(make_world(), air_control=0.0)
    settle(char)
    for _ in range(10):
        char.move_input((1.0, 0.0, 0.0))
        char.world.step()
    assert char.jump()
    moved = char.move_input((0.0, 0.0, 0.0))
    assert _speed_x(moved) == pytest.approx(4.5, rel=0.01)


def test_ground_input_is_immediate_regardless_of_air_control():
    char = make_char(make_world(), air_control=0.0)
    settle(char)
    assert _speed_x(char.move_input((1.0, 0.0, 0.0))) == pytest.approx(4.5, rel=0.01)


def test_explicit_simple_move_ignores_air_control_and_sprint():
    """Явная скорость - «как раньше»: ни бега, ни смешивания в воздухе."""
    char = _airborne(air_control=0.0)
    moved = char.simple_move((3.0, 0.0, 0.0))
    assert _speed_x(moved) == pytest.approx(3.0, rel=0.01)


def test_character_stays_on_floor_after_walking() -> None:
    char = make_char(make_world())
    settle(char)
    for _ in range(60):
        char.move_input((1.0, 0.0, 0.0))
        char.world.step()
    assert foot_y(char) == pytest.approx(0.02, abs=0.005)
