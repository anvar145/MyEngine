"""Физические компоненты на дескрипторах: формат v1 не меняется."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
)
from engine.core.reflection import FieldKind, registry

_ROOT = Path(__file__).resolve().parents[2]
_SCENES = sorted((_ROOT / "projects").glob("*/Assets/Scenes/*.scene.json"))


# --- эталон: прежние to_dict (скопированы до миграции) -----------------------

def _legacy_rigidbody(c) -> dict:
    return {
        "body_type": c.body_type, "mass": c.mass,
        "linear_damping": c.linear_damping, "angular_damping": c.angular_damping,
        "gravity_scale": c.gravity_scale, "use_gravity": c.use_gravity,
        "freeze_rotation": c.freeze_rotation, "can_sleep": c.can_sleep,
    }


def _legacy_collider(c) -> dict:
    data = {
        "shape": c.shape, "size": list(c.size), "radius": c.radius,
        "height": c.height, "offset": list(c.offset),
        "is_trigger": c.is_trigger, "friction": c.friction,
        "restitution": c.restitution, "layer": c.layer, "mask": c.mask,
    }
    if tuple(c.rotation_offset) != (0.0, 0.0, 0.0, 1.0):
        data["rotation_offset"] = list(c.rotation_offset)
    if c.friction_combine != "multiply":
        data["friction_combine"] = c.friction_combine
    if c.restitution_combine != "max":
        data["restitution_combine"] = c.restitution_combine
    return data


_CC_NAMES = (
    "radius", "height", "step_offset", "slope_limit", "skin_width", "layer",
    "mask", "move_speed", "sprint_multiplier", "jump_velocity", "use_gravity",
    "gravity_scale", "ground_check_distance", "air_control", "snap_to_ground",
    "alignment_to_ground",
)


def _legacy_character(c) -> dict:
    return {n: getattr(c, n) for n in _CC_NAMES}


def _samples():
    yield RigidBodyComponent(), _legacy_rigidbody
    yield RigidBodyComponent("kinematic", 3.5, 0.1, 0.2, 0.5, False, True, False), (
        _legacy_rigidbody
    )
    yield ColliderComponent(), _legacy_collider
    yield ColliderComponent(
        "capsule", (2, 3, 4), 0.7, 3.0, (0.1, 0.2, 0.3), True, 0.9, 0.4, 5, 0xF0F0,
        (0.0, 0.7071, 0.0, 0.7071), "max", "min",
    ), _legacy_collider
    yield CharacterControllerComponent(), _legacy_character
    yield CharacterControllerComponent(
        0.5, 2.0, 0.4, 30.0, 0.05, 3, 0xFF, 6.0, 2.0, 7.0, False, 0.5, 0.1, 0.9,
        False, True,
    ), _legacy_character


@pytest.mark.parametrize("pair", list(_samples()), ids=lambda p: type(p[0]).__name__)
def test_serialization_matches_legacy_bit_exact(pair) -> None:
    comp, legacy = pair
    assert json.dumps(registry.serialize(comp)) == json.dumps(legacy(comp))
    assert json.dumps(comp.to_dict()) == json.dumps(legacy(comp))


def test_field_order_is_constructor_order() -> None:
    assert [d.name for d in registry.fields(CharacterControllerComponent)] == list(
        _CC_NAMES
    )
    assert [d.name for d in registry.fields(RigidBodyComponent)] == list(
        _legacy_rigidbody(RigidBodyComponent())
    )


@pytest.mark.parametrize("pair", list(_samples()), ids=lambda p: type(p[0]).__name__)
def test_round_trip(pair) -> None:
    comp, legacy = pair
    back = type(comp).from_dict(json.loads(json.dumps(comp.to_dict())))
    assert legacy(back) == legacy(comp)
    again = registry.deserialize(type(comp), registry.serialize(comp))
    assert legacy(again) == legacy(comp)


def test_defaults_match_previous() -> None:
    cc = CharacterControllerComponent()
    assert (cc.radius, cc.height, cc.step_offset, cc.slope_limit) == (
        0.4, 1.8, 0.3, 45.0,
    )
    assert (cc.move_speed, cc.sprint_multiplier, cc.jump_velocity) == (4.5, 1.6, 5.0)
    assert (cc.ground_check_distance, cc.air_control) == (0.03, 0.3)
    assert cc.mask == 0xFFFFFFFF and cc.snap_to_ground and not cc.alignment_to_ground
    col = ColliderComponent()
    assert (col.shape, col.size, col.radius, col.height) == ("box", (1.0,) * 3, 0.5, 2.0)
    assert (col.friction, col.restitution, col.layer) == (0.5, 0.0, 1)
    assert (col.friction_combine, col.restitution_combine) == ("multiply", "max")
    assert registry.for_name(ColliderComponent, "mask").kind is FieldKind.MASK32


def test_character_live_flags_from_descriptors() -> None:
    live = {d.name for d in registry.fields(CharacterControllerComponent) if d.live}
    assert live == {
        "move_speed", "sprint_multiplier", "jump_velocity", "gravity_scale",
        "ground_check_distance", "air_control", "use_gravity", "snap_to_ground",
        "alignment_to_ground",
    }
    assert not any(d.live for d in registry.fields(RigidBodyComponent))
    assert not any(d.live for d in registry.fields(ColliderComponent))


@pytest.mark.parametrize(
    ("name", "bad", "message"),
    [
        ("move_speed", -1.0, "move_speed=-1.0: допустимо значение >= 0"),
        ("sprint_multiplier", 0.5, "sprint_multiplier=0.5: допустимо значение >= 1"),
        ("jump_velocity", 0.0, "jump_velocity=0.0: допустимо значение > 0"),
        ("air_control", 2.0, "air_control=2.0: допустимо значение >= 0 и <= 1"),
        ("gravity_scale", float("nan"), "gravity_scale=nan"),
    ],
)
def test_character_constructor_errors(name, bad, message) -> None:
    with pytest.raises(ValueError) as info:
        CharacterControllerComponent(**{name: bad})
    assert message in str(info.value)


def test_character_from_dict_clamps_with_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        cc = CharacterControllerComponent.from_dict(
            {"move_speed": -3, "jump_velocity": 0, "air_control": 9, "radius": None}
        )
    assert (cc.move_speed, cc.jump_velocity, cc.air_control) == (0.0, 1e-3, 1.0)
    assert cc.radius == 0.4
    assert "move_speed=-3.0 недопустим, используется 0.0" in caplog.text


def test_collider_and_rigidbody_store_values_as_is() -> None:
    col = ColliderComponent(radius=-1.0, friction=-2.0, shape="cone")
    assert (col.radius, col.friction, col.shape) == (-1.0, -2.0, "cone")
    body = RigidBodyComponent(mass=-5.0)
    assert body.mass == -5.0


def test_setattr_notifications_unchanged() -> None:
    from engine.core.scene import Scene

    scene = Scene()
    obj = scene.create_game_object("c")
    cc = obj.add_component(CharacterControllerComponent())
    obj.add_component(ColliderComponent())
    physics = scene.physics
    physics.step(1 / 60)
    cc.move_speed = 9.0  # живое поле
    assert cc.move_speed == 9.0
    cc.radius = 0.5  # геометрия: помечает на пересоздание
    assert cc.radius == 0.5


def test_unknown_keys_in_scene_are_ignored() -> None:
    comp = ColliderComponent.from_dict({"shape": "sphere", "extra": 1})
    assert comp.shape == "sphere"


@pytest.mark.skipif(not _SCENES, reason="нет проектных сцен")
@pytest.mark.parametrize("path", _SCENES, ids=lambda p: f"{p.parent.parent.parent.name}/{p.name}")
def test_project_scenes_physics_components_round_trip(path: Path) -> None:
    classes = {
        "RigidBodyComponent": RigidBodyComponent,
        "ColliderComponent": ColliderComponent,
        "CharacterControllerComponent": CharacterControllerComponent,
    }
    found = 0

    def walk(node) -> None:
        nonlocal found
        if isinstance(node, dict):
            cls = classes.get(node.get("type"))
            if cls is not None:
                comp = registry.deserialize(cls, node)
                assert comp.to_dict() == cls.from_dict(node).to_dict()
                found += 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    text = path.read_text(encoding="utf-8")
    walk(json.loads(text))
    assert (found > 0) == any(name in text for name in classes)
