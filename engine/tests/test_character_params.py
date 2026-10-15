"""Параметры ``CharacterControllerComponent``: поля, валидация, «живые» правки."""

from __future__ import annotations

import logging

import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    clamp_character_field,
)
from engine.core.scene import Scene

DT = 1.0 / 60.0

NEW_DEFAULTS = {
    "move_speed": 4.5,
    "sprint_multiplier": 1.6,
    "jump_velocity": 5.0,
    "use_gravity": True,
    "gravity_scale": 1.0,
    "ground_check_distance": 0.03,
    "air_control": 0.3,
    "snap_to_ground": True,
    "alignment_to_ground": False,
}
# Значения, отличные от дефолтов, - чтобы поймать «поле не сохраняется».
NEW_CUSTOM = {
    "move_speed": 6.5,
    "sprint_multiplier": 2.0,
    "jump_velocity": 7.5,
    "use_gravity": False,
    "gravity_scale": 0.5,
    "ground_check_distance": 0.08,
    "air_control": 0.75,
    "snap_to_ground": False,
    "alignment_to_ground": True,
}
INVALID = [
    ("move_speed", -0.1),
    ("move_speed", float("nan")),
    ("move_speed", float("inf")),
    ("sprint_multiplier", 0.99),
    ("sprint_multiplier", 0.0),
    ("jump_velocity", 0.0),
    ("jump_velocity", -3.0),
    ("gravity_scale", -0.01),
    ("ground_check_distance", 0.0),
    ("ground_check_distance", -0.03),
    ("air_control", -0.01),
    ("air_control", 1.01),
    ("air_control", float("nan")),
]
BOUNDARY = [
    ("move_speed", 0.0),
    ("sprint_multiplier", 1.0),
    ("gravity_scale", 0.0),
    ("air_control", 0.0),
    ("air_control", 1.0),
    ("jump_velocity", 1e-6),
    ("ground_check_distance", 1e-6),
]


def _run(scene: Scene, steps: int) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)
        scene.update(DT)


def _hero_on_floor(**cc):
    scene = Scene()
    floor = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(40, 1, 40)
    )
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go = scene.create_game_object("Hero", position=Vector3(0, 0.93, 0))
    comp = go.add_component(CharacterControllerComponent, **cc)
    return scene, go, comp


# --- значения по умолчанию и конструктор ------------------------------------

@pytest.mark.parametrize(("name", "value"), NEW_DEFAULTS.items())
def test_new_field_default(name, value):
    got = getattr(CharacterControllerComponent(), name)
    assert got == value and type(got) is type(value), name


@pytest.mark.parametrize(("name", "value"), NEW_CUSTOM.items())
def test_new_field_is_accepted_by_constructor(name, value):
    comp = CharacterControllerComponent(**{name: value})
    assert getattr(comp, name) == value


@pytest.mark.parametrize(("name", "value"), INVALID)
def test_constructor_rejects_invalid_value(name, value):
    with pytest.raises(ValueError, match=name):
        CharacterControllerComponent(**{name: value})


@pytest.mark.parametrize(("name", "value"), BOUNDARY)
def test_constructor_accepts_boundary_value(name, value):
    assert getattr(CharacterControllerComponent(**{name: value}), name) == value


def test_flags_are_coerced_to_bool():
    comp = CharacterControllerComponent(
        use_gravity=0, snap_to_ground=1, alignment_to_ground=1
    )
    assert comp.use_gravity is False
    assert comp.snap_to_ground is True
    assert comp.alignment_to_ground is True


def test_old_positional_arguments_keep_their_order():
    """Новые поля идут после старых: вызовы со старой сигнатурой не ломаются."""
    comp = CharacterControllerComponent(0.5, 2.0, 0.25, 40.0, 0.03, 2, 0xF)
    assert (comp.radius, comp.height, comp.step_offset) == (0.5, 2.0, 0.25)
    assert (comp.slope_limit, comp.skin_width, comp.layer, comp.mask) == (
        40.0, 0.03, 2, 0xF,
    )


# --- clamp_character_field ---------------------------------------------------

@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("move_speed", -2.0, 0.0),
        ("move_speed", 3.0, 3.0),
        ("sprint_multiplier", 0.5, 1.0),
        ("jump_velocity", -1.0, 1e-3),
        ("jump_velocity", 0.0, 1e-3),
        ("gravity_scale", -1.0, 0.0),
        ("ground_check_distance", 0.0, 1e-3),
        ("air_control", 2.0, 1.0),
        ("air_control", -2.0, 0.0),
    ],
)
def test_clamp_character_field(name, value, expected):
    assert clamp_character_field(name, value) == expected


@pytest.mark.parametrize(("name", "value"), BOUNDARY)
def test_clamp_leaves_valid_value_untouched(name, value):
    """Допустимое значение (даже меньше 1e-3 у открытого минимума) не меняется."""
    assert clamp_character_field(name, value) == value


def test_clamp_non_finite_falls_back_to_default():
    assert clamp_character_field("move_speed", float("nan")) == 4.5
    assert clamp_character_field("air_control", float("inf")) == 0.3


# --- «живые» поля не пересоздают двойник -------------------------------------

def test_live_field_edit_does_not_rebuild_controller():
    scene, go, comp = _hero_on_floor()
    _run(scene, 3)
    handle = scene.physics.handle_of(go)
    for name, value in NEW_CUSTOM.items():
        setattr(comp, name, value)
    _run(scene, 1)
    assert scene.physics.handle_of(go) == handle


def test_geometry_edit_still_rebuilds_controller():
    scene, go, comp = _hero_on_floor()
    _run(scene, 3)
    handle = scene.physics.handle_of(go)
    comp.radius = 0.35
    _run(scene, 1)
    assert scene.physics.handle_of(go) != handle


def test_live_edit_keeps_vertical_velocity():
    """Правка live-поля в падении не обнуляет вертикальную скорость."""
    scene, go, comp = _hero_on_floor()
    go.transform.position = Vector3(0, 6.0, 0)
    scene.physics.update_character(go, (0, 0, 0))
    scene.physics.update_character(go, (0, 0, 0))
    ctrl = scene.physics._bindings[go.id].character
    before = ctrl.vertical_velocity
    assert before < 0.0
    comp.move_speed = 9.0
    scene.physics.sync_structure()
    assert scene.physics._bindings[go.id].character.vertical_velocity == before


# --- сериализация новых полей -----------------------------------------------

def test_to_dict_contains_new_fields_with_values():
    data = CharacterControllerComponent(**NEW_CUSTOM).to_dict()
    for name, value in NEW_CUSTOM.items():
        assert data[name] == value and type(data[name]) is type(value), name


def test_from_dict_round_trip_new_fields():
    comp = CharacterControllerComponent(**NEW_CUSTOM)
    restored = CharacterControllerComponent.from_dict(comp.to_dict())
    for name, value in NEW_CUSTOM.items():
        assert getattr(restored, name) == value, name


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("move_speed", -5.0, 0.0),
        ("sprint_multiplier", 0.2, 1.0),
        ("jump_velocity", -1.0, 1e-3),
        ("gravity_scale", -2.0, 0.0),
        ("ground_check_distance", 0.0, 1e-3),
        ("air_control", 3.0, 1.0),
    ],
)
def test_from_dict_clamps_invalid_value_and_warns(caplog, name, value, expected):
    with caplog.at_level(logging.WARNING, logger="engine.core.physics_components"):
        comp = CharacterControllerComponent.from_dict({name: value})
    assert getattr(comp, name) == expected
    assert any(name in rec.getMessage() for rec in caplog.records)


def test_from_dict_non_finite_uses_default_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="engine.core.physics_components"):
        comp = CharacterControllerComponent.from_dict({"move_speed": float("nan")})
    assert comp.move_speed == 4.5
    assert any("move_speed" in rec.getMessage() for rec in caplog.records)


def test_from_dict_valid_values_do_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="engine.core.physics_components"):
        CharacterControllerComponent.from_dict(
            CharacterControllerComponent(**NEW_CUSTOM).to_dict()
        )
    assert not caplog.records
