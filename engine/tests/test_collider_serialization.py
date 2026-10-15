"""Верификация Collider: сериализация ColliderComponent в .scene.json."""

from __future__ import annotations

import json

import numpy as np
import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.scene_io import load_scene, save_scene, scene_from_dict, scene_to_dict

DT = 1.0 / 60.0

FIELDS = ("shape", "size", "radius", "height", "offset", "is_trigger",
          "friction", "restitution", "layer", "mask")

NON_DEFAULT = {
    "shape": "capsule",
    "size": (1.5, 2.5, 3.5),
    "radius": 0.7,
    "height": 3.3,
    "offset": (0.25, -0.5, 1.75),
    "is_trigger": True,
    "friction": 0.123,
    "restitution": 0.456,
    "layer": 0b10100,
    "mask": 0b00110,
}


def _scene_with(**kwargs) -> Scene:
    scene = Scene()
    go = scene.create_game_object("Obj", position=Vector3(0, 3, 0))
    go.add_component(ColliderComponent, **kwargs)
    return scene


def _collider(scene: Scene) -> ColliderComponent:
    comp = scene.find("Obj").get_component(ColliderComponent)
    assert comp is not None
    return comp


def _roundtrip(scene: Scene) -> Scene:
    return scene_from_dict(json.loads(json.dumps(scene_to_dict(scene))))


# --- запись ---------------------------------------------------------------------

def test_to_dict_writes_all_ten_fields() -> None:
    data = ColliderComponent().to_dict()
    assert set(data) == set(FIELDS)


def test_default_rotation_offset_and_combine_are_not_written() -> None:
    data = ColliderComponent().to_dict()
    for key in ("rotation_offset", "friction_combine", "restitution_combine"):
        assert key not in data


def test_non_default_rotation_offset_and_combine_survive_round_trip() -> None:
    quat = (0.0, 0.7071067811865476, 0.0, 0.7071067811865476)
    scene = _scene_with(rotation_offset=quat, friction_combine="min",
                        restitution_combine="average")
    assert {"rotation_offset", "friction_combine", "restitution_combine"} <= set(
        _collider(scene).to_dict())
    loaded = _collider(_roundtrip(scene))
    assert tuple(loaded.rotation_offset) == quat
    assert loaded.friction_combine == "min"
    assert loaded.restitution_combine == "average"


def test_default_values_in_dict() -> None:
    assert ColliderComponent().to_dict() == {
        "shape": "box", "size": [1.0, 1.0, 1.0], "radius": 0.5, "height": 2.0,
        "offset": [0.0, 0.0, 0.0], "is_trigger": False, "friction": 0.5,
        "restitution": 0.0, "layer": 1, "mask": 0xFFFFFFFF,
    }


def test_scene_dict_contains_collider_entry() -> None:
    scene = _scene_with(**NON_DEFAULT)
    components = scene_to_dict(scene)["objects"][0]["components"]
    entry = next(c for c in components if c["type"] == "ColliderComponent")
    assert entry["shape"] == "capsule"
    assert entry["is_trigger"] is True


# --- round-trip ---------------------------------------------------------------------

@pytest.mark.parametrize("field", FIELDS)
def test_each_field_survives_round_trip(field: str) -> None:
    scene = _scene_with(**{field: NON_DEFAULT[field]})
    loaded = _collider(_roundtrip(scene))
    value = getattr(loaded, field)
    expected = NON_DEFAULT[field]
    if isinstance(expected, tuple):
        assert tuple(value) == expected
    else:
        assert value == expected
        assert type(value) is type(expected)


def test_all_fields_together_survive_round_trip_through_file(tmp_path) -> None:
    scene = _scene_with(**NON_DEFAULT)
    path = save_scene(scene, tmp_path / "level.scene.json")
    loaded = _collider(load_scene(path))
    for field in FIELDS:
        value = getattr(loaded, field)
        expected = NON_DEFAULT[field]
        assert (tuple(value) if isinstance(expected, tuple) else value) == expected


@pytest.mark.parametrize("shape", ["box", "sphere", "capsule"])
def test_each_shape_name_survives_round_trip(shape: str) -> None:
    assert _collider(_roundtrip(_scene_with(shape=shape))).shape == shape


def test_large_and_unusual_layer_mask_values_survive() -> None:
    scene = _scene_with(layer=1 << 40, mask=0xFFFFFFFF)
    loaded = _collider(_roundtrip(scene))
    assert loaded.layer == 1 << 40
    assert loaded.mask == 0xFFFFFFFF


def test_float_precision_is_kept() -> None:
    scene = _scene_with(friction=0.1, restitution=1 / 3, radius=1e-7)
    loaded = _collider(_roundtrip(scene))
    assert loaded.friction == 0.1
    assert loaded.restitution == 1 / 3
    assert loaded.radius == 1e-7


def test_double_round_trip_is_stable() -> None:
    scene = _scene_with(**NON_DEFAULT)
    once = scene_to_dict(_roundtrip(scene))
    twice = scene_to_dict(_roundtrip(_roundtrip(scene)))
    assert once == twice


def test_loaded_scene_simulates_like_original() -> None:
    def build() -> Scene:
        scene = Scene()
        floor = scene.create_game_object("Floor", position=Vector3(0, -5, 0),
                                         scale=Vector3(100, 10, 100))
        floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1),
                            restitution=0.3)
        ball = scene.create_game_object("Ball", position=Vector3(0, 3, 0))
        ball.add_component(ColliderComponent, shape="sphere", radius=0.4,
                           offset=(0.2, 0.0, 0.0), restitution=0.6, friction=0.2)
        ball.add_component(RigidBodyComponent)
        return scene

    def positions(scene: Scene) -> np.ndarray:
        for _ in range(240):
            scene.fixed_update(DT)
        ball = scene.find("Ball")
        return np.array([ball.transform.world_position.x,
                         ball.transform.world_position.y,
                         ball.transform.world_position.z])

    original = build()
    reloaded = _roundtrip(build())
    assert np.array_equal(positions(original), positions(reloaded))


# --- чтение: дефолты и устойчивость ---------------------------------------------

DEFAULTS = {
    "shape": "box", "size": (1.0, 1.0, 1.0), "radius": 0.5, "height": 2.0,
    "offset": (0.0, 0.0, 0.0), "is_trigger": False, "friction": 0.5,
    "restitution": 0.0, "layer": 1, "mask": 0xFFFFFFFF,
}


@pytest.mark.parametrize("field", FIELDS)
def test_missing_field_falls_back_to_default(field: str) -> None:
    data = ColliderComponent(**{k: NON_DEFAULT[k] for k in FIELDS}).to_dict()
    del data[field]
    loaded = ColliderComponent.from_dict(data)
    value = getattr(loaded, field)
    expected = DEFAULTS[field]
    assert (tuple(value) if isinstance(expected, tuple) else value) == expected
    # остальные поля остались заданными
    for other in FIELDS:
        if other != field:
            got = getattr(loaded, other)
            want = NON_DEFAULT[other]
            assert (tuple(got) if isinstance(want, tuple) else got) == want


def test_empty_dict_gives_all_defaults() -> None:
    loaded = ColliderComponent.from_dict({})
    for field, expected in DEFAULTS.items():
        value = getattr(loaded, field)
        assert (tuple(value) if isinstance(expected, tuple) else value) == expected


def test_extra_fields_are_ignored() -> None:
    data = ColliderComponent().to_dict()
    data.update({"future": 1, "type": "x"})
    loaded = ColliderComponent.from_dict(data)
    assert loaded.to_dict() == ColliderComponent().to_dict()


def test_old_scene_without_collider_fields_loads() -> None:
    scene = _scene_with()
    data = scene_to_dict(scene)
    entry = next(c for c in data["objects"][0]["components"]
                 if c["type"] == "ColliderComponent")
    for key in list(entry):
        if key != "type":
            del entry[key]
    loaded = _collider(scene_from_dict(data))
    assert loaded.shape == "box" and loaded.mask == 0xFFFFFFFF


def test_null_vector_fields_use_defaults() -> None:
    loaded = ColliderComponent.from_dict({"size": None, "offset": None})
    assert tuple(loaded.size) == (1.0, 1.0, 1.0)
    assert tuple(loaded.offset) == (0.0, 0.0, 0.0)


def test_numeric_strings_and_ints_are_coerced() -> None:
    loaded = ColliderComponent.from_dict(
        {"radius": "0.25", "friction": 1, "layer": "4", "is_trigger": 1})
    assert loaded.radius == 0.25
    assert loaded.friction == 1.0 and isinstance(loaded.friction, float)
    assert loaded.layer == 4
    assert loaded.is_trigger is True


@pytest.mark.parametrize("field", FIELDS)
def test_null_field_falls_back_to_default(field: str) -> None:
    loaded = ColliderComponent.from_dict({field: None})
    value = getattr(loaded, field)
    expected = DEFAULTS[field]
    assert (tuple(value) if isinstance(expected, tuple) else value) == expected


def test_scene_json_with_null_radius_loads_default_radius() -> None:
    data = scene_to_dict(_scene_with(shape="sphere", radius=2.0))
    entry = next(c for c in data["objects"][0]["components"]
                 if c["type"] == "ColliderComponent")
    entry["radius"] = None
    loaded = _collider(scene_from_dict(json.loads(json.dumps(data))))
    assert loaded.radius == 0.5


def test_null_rigidbody_fields_fall_back_to_defaults() -> None:
    fields = ("body_type", "mass", "linear_damping", "angular_damping",
              "gravity_scale", "use_gravity", "freeze_rotation", "can_sleep")
    loaded = RigidBodyComponent.from_dict(dict.fromkeys(fields))
    assert loaded.to_dict() == RigidBodyComponent().to_dict()


def test_null_new_collider_fields_use_defaults() -> None:
    loaded = ColliderComponent.from_dict(
        {"rotation_offset": None, "friction_combine": None,
         "restitution_combine": None})
    assert tuple(loaded.rotation_offset) == (0.0, 0.0, 0.0, 1.0)
    assert (loaded.friction_combine, loaded.restitution_combine) == ("multiply", "max")


def test_short_vector_is_kept_as_is_and_does_not_crash_simulation() -> None:
    """size из двух чисел читается; адаптер берёт ``size[:3]`` и падает в ValueError,
    поэтому объект пропускается с предупреждением (сцена остальная живёт)."""
    scene = Scene()
    go = scene.create_game_object("Obj", position=Vector3(0, 3, 0))
    comp = ColliderComponent.from_dict({"size": [1.0, 2.0]})
    go.add_component(comp)
    rb = go.add_component(RigidBodyComponent)
    scene.fixed_update(DT)
    assert tuple(comp.size) == (1.0, 2.0)
    assert rb.handle is None


# --- неизвестная форма -------------------------------------------------------------

def test_unknown_shape_loads_but_object_is_skipped_at_simulation(caplog) -> None:
    scene = Scene()
    go = scene.create_game_object("Obj", position=Vector3(0, 3, 0))
    go.add_component(ColliderComponent, shape="cone")
    go.add_component(RigidBodyComponent)
    loaded = _roundtrip(scene)
    assert _collider(loaded).shape == "cone"  # сериализация форму не проверяет
    with caplog.at_level("WARNING"):
        loaded.fixed_update(DT)
    assert loaded.find("Obj").get_component(RigidBodyComponent).handle is None
    assert any("пропущен" in r.message for r in caplog.records)


def test_empty_shape_string_is_also_unknown() -> None:
    assert ColliderComponent.from_dict({"shape": ""}).shape == ""
