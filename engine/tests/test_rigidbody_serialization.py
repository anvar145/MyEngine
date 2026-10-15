"""Верификация RigidBody: сериализация RigidBodyComponent в .scene.json."""

from __future__ import annotations

import json

import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.physics.body import RigidBody
from engine.scene_io import load_scene, save_scene, scene_from_dict, scene_to_dict

DT = 1.0 / 60.0

FIELDS = (
    "body_type", "mass", "linear_damping", "angular_damping",
    "gravity_scale", "use_gravity", "freeze_rotation", "can_sleep",
)
CUSTOM = {
    "body_type": "kinematic",
    "mass": 7.25,
    "linear_damping": 0.7,
    "angular_damping": 1.3,
    "gravity_scale": -0.5,
    "use_gravity": False,
    "freeze_rotation": True,
    "can_sleep": False,
}
DEFAULTS = {
    "body_type": "dynamic",
    "mass": 1.0,
    "linear_damping": 0.05,
    "angular_damping": 0.05,
    "gravity_scale": 1.0,
    "use_gravity": True,
    "freeze_rotation": False,
    "can_sleep": True,
}


def _scene_with(**kw) -> Scene:
    scene = Scene()
    go = scene.create_game_object("Box", position=Vector3(0, 5, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go.add_component(RigidBodyComponent, **kw)
    return scene


def _component(scene: Scene) -> RigidBodyComponent:
    comp = scene.find("Box").get_component(RigidBodyComponent)
    assert comp is not None
    return comp


def test_to_dict_covers_every_field() -> None:
    comp = RigidBodyComponent()
    assert set(comp.to_dict()) == set(FIELDS) == set(RigidBodyComponent.__slots__)


def test_round_trip_every_field_non_default() -> None:
    data = json.loads(json.dumps(scene_to_dict(_scene_with(**CUSTOM))))
    loaded = _component(scene_from_dict(data))
    for name in FIELDS:
        assert getattr(loaded, name) == CUSTOM[name], name


def test_round_trip_through_scene_json_file(tmp_path) -> None:
    path = tmp_path / "rb.scene.json"
    save_scene(_scene_with(**CUSTOM), path)
    loaded = _component(load_scene(path))
    assert loaded.to_dict() == CUSTOM


def test_defaults_are_written_explicitly() -> None:
    """Факт: значения по умолчанию в JSON пишутся, а не опускаются."""
    data = scene_to_dict(_scene_with())
    obj = next(o for o in data["objects"] if o["name"] == "Box")
    entry = next(c for c in obj["components"] if c["type"] == "RigidBodyComponent")
    saved = {k: v for k, v in entry.items() if k != "type"}
    assert saved == DEFAULTS


def test_missing_fields_fall_back_to_component_defaults() -> None:
    comp = RigidBodyComponent.from_dict({})
    assert comp.to_dict() == DEFAULTS


@pytest.mark.parametrize("name", FIELDS)
def test_single_missing_field_gets_default(name: str) -> None:
    data = dict(CUSTOM)
    del data[name]
    comp = RigidBodyComponent.from_dict(data)
    assert getattr(comp, name) == DEFAULTS[name]
    for other in FIELDS:
        if other != name:
            assert getattr(comp, other) == CUSTOM[other]


def test_component_defaults_equal_rigidbody_defaults() -> None:
    body = RigidBody()
    comp = RigidBodyComponent()
    for name in ("mass", "linear_damping", "angular_damping", "gravity_scale",
                 "use_gravity", "freeze_rotation", "can_sleep", "body_type"):
        assert getattr(body, name) == getattr(comp, name), name


def test_extra_unknown_keys_are_ignored() -> None:
    comp = RigidBodyComponent.from_dict({"mass": 3.0, "future": 1})
    assert comp.mass == 3.0


def test_loaded_scene_simulates_like_original() -> None:
    def fall(scene: Scene) -> float:
        for _ in range(60):
            scene.fixed_update(DT)
            scene.update(DT)
        return scene.find("Box").transform.world_position.y

    original = _scene_with(gravity_scale=2.0, linear_damping=0.3, mass=3.0)
    clone = scene_from_dict(json.loads(json.dumps(scene_to_dict(original))))
    assert fall(original) == pytest.approx(fall(clone), abs=1e-6)
    assert fall(clone) < 5.0 - 1.0


def test_physics_params_survive_to_world_body_after_load() -> None:
    clone = scene_from_dict(
        json.loads(json.dumps(scene_to_dict(_scene_with(**{**CUSTOM, "body_type": "dynamic"}))))
    )
    clone.fixed_update(DT)
    comp = _component(clone)
    body = clone.physics.world.get_body(comp.handle)
    assert body.mass == 7.25
    assert body.linear_damping == 0.7
    assert body.angular_damping == 1.3
    assert body.gravity_scale == -0.5
    assert body.use_gravity is False
    assert body.freeze_rotation is True
    assert body.can_sleep is False
