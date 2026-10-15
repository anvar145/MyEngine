"""Верификация CharacterController: сериализация компонента в ``.scene.json``."""

from __future__ import annotations

import json

import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
)
from engine.core.scene import Scene
from engine.scene_io import load_scene, save_scene, scene_from_dict, scene_to_dict

FIELDS = ("radius", "height", "step_offset", "slope_limit", "skin_width",
          "layer", "mask", "move_speed", "sprint_multiplier", "jump_velocity",
          "use_gravity", "gravity_scale", "ground_check_distance", "air_control",
          "snap_to_ground", "alignment_to_ground")
CUSTOM = {
    "radius": 0.35,
    "height": 1.7,
    "step_offset": 0.45,
    "slope_limit": 50.0,
    "skin_width": 0.03,
    "layer": 4,
    "mask": 0x80000006,
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
DEFAULTS = {
    "radius": 0.4,
    "height": 1.8,
    "step_offset": 0.3,
    "slope_limit": 45.0,
    "skin_width": 0.02,
    "layer": 1,
    "mask": 0xFFFFFFFF,
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


def _scene_with(**kw) -> Scene:
    scene = Scene()
    go = scene.create_game_object("Hero", position=Vector3(0, 1, 0))
    go.add_component(CharacterControllerComponent, **kw)
    return scene


def _component(scene: Scene) -> CharacterControllerComponent:
    comp = scene.find("Hero").get_component(CharacterControllerComponent)
    assert comp is not None
    return comp


def test_to_dict_covers_every_field():
    comp = CharacterControllerComponent()
    assert set(comp.to_dict()) == set(FIELDS)
    # Приватные слоты (``_sprinting`` и т. п.) - рантайм-состояние, не поля сцены.
    public = {s for s in CharacterControllerComponent.__slots__ if s[0] != "_"}
    assert set(comp.to_dict()) == public


def test_to_dict_writes_defaults_explicitly():
    assert CharacterControllerComponent().to_dict() == DEFAULTS


def test_defaults_match_documented_values():
    comp = CharacterControllerComponent()
    for name, value in DEFAULTS.items():
        assert getattr(comp, name) == value, name


def test_component_round_trip_keeps_all_values():
    comp = CharacterControllerComponent(**CUSTOM)
    restored = CharacterControllerComponent.from_dict(comp.to_dict())
    assert restored.to_dict() == CUSTOM


def test_dict_is_json_serializable_and_stable():
    comp = CharacterControllerComponent(**CUSTOM)
    text = json.dumps(comp.to_dict())
    again = CharacterControllerComponent.from_dict(json.loads(text))
    assert again.to_dict() == CUSTOM
    assert json.dumps(again.to_dict()) == text


def test_mask_bit_31_survives_json():
    comp = CharacterControllerComponent(mask=0x80000000)
    restored = CharacterControllerComponent.from_dict(
        json.loads(json.dumps(comp.to_dict()))
    )
    assert restored.mask == 0x80000000


def test_scene_round_trip_in_memory():
    scene = _scene_with(**CUSTOM)
    restored = scene_from_dict(scene_to_dict(scene))
    assert _component(restored).to_dict() == CUSTOM


def test_scene_round_trip_through_file(tmp_path):
    scene = _scene_with(**CUSTOM)
    path = tmp_path / "hero.scene.json"
    save_scene(scene, path)
    restored = load_scene(path)
    assert _component(restored).to_dict() == CUSTOM


def test_scene_file_contains_all_sixteen_fields(tmp_path):
    scene = _scene_with()
    path = tmp_path / "hero.scene.json"
    save_scene(scene, path)
    text = path.read_text(encoding="utf-8")
    for name in FIELDS:
        assert f'"{name}"' in text, name


def test_missing_fields_take_defaults():
    comp = CharacterControllerComponent.from_dict({})
    assert comp.to_dict() == DEFAULTS


@pytest.mark.parametrize("name", FIELDS)
def test_single_missing_field_takes_its_default(name):
    data = dict(CUSTOM)
    del data[name]
    comp = CharacterControllerComponent.from_dict(data)
    assert getattr(comp, name) == DEFAULTS[name]
    for other in FIELDS:
        if other != name:
            assert getattr(comp, other) == CUSTOM[other]


@pytest.mark.parametrize("name", FIELDS)
def test_null_field_takes_default(name):
    data = dict(CUSTOM)
    data[name] = None
    comp = CharacterControllerComponent.from_dict(data)
    assert getattr(comp, name) == DEFAULTS[name]


def test_unknown_fields_are_ignored():
    data = dict(CUSTOM, bogus_field=7.0, ground_velocity=9.0)
    comp = CharacterControllerComponent.from_dict(data)
    assert comp.to_dict() == CUSTOM
    assert not hasattr(comp, "bogus_field")


def test_scene_loads_object_without_character_fields_with_defaults():
    scene = _scene_with(**CUSTOM)
    data = scene_to_dict(scene)
    entry = _entry(data)
    for name in FIELDS:
        del entry[name]  # остаётся только {"type": ...}
    restored = scene_from_dict(data)
    assert _component(restored).to_dict() == DEFAULTS


def test_scene_json_entry_has_type_and_all_fields():
    entry = _entry(scene_to_dict(_scene_with(**CUSTOM)))
    assert entry["type"] == "CharacterControllerComponent"
    assert {k: v for k, v in entry.items() if k != "type"} == CUSTOM


def _entry(data: dict) -> dict:
    """Запись компонента персонажа в списке ``components`` первого объекта."""
    return next(
        c for c in data["objects"][0]["components"]
        if c["type"] == "CharacterControllerComponent"
    )


# --- невалидные значения в JSON: фиксируем текущее поведение ----------------

def test_non_numeric_value_fails_to_load_with_value_error():
    with pytest.raises(ValueError):
        CharacterControllerComponent.from_dict({"radius": "abc"})


def test_negative_radius_loads_but_is_rejected_at_runtime():
    """Компонент валидацию не делает; ERROR/отказ регистрации — в адаптере."""
    scene = _scene_with(radius=-0.5)
    restored = scene_from_dict(scene_to_dict(scene))
    assert _component(restored).radius == -0.5
    with pytest.raises(RuntimeError, match="не зарегистрирован"):
        restored.physics.is_grounded(restored.find("Hero"))


def test_numeric_strings_are_coerced():
    comp = CharacterControllerComponent.from_dict({"radius": "0.5", "layer": "3"})
    assert comp.radius == 0.5
    assert comp.layer == 3


def test_negative_mask_is_kept_as_is_not_wrapped():
    comp = CharacterControllerComponent.from_dict({"mask": -1})
    assert comp.mask == -1  # без нормализации к 32 битам (документируем)


def test_collider_and_character_on_same_object_both_round_trip():
    scene = Scene()
    go = scene.create_game_object("Hero")
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go.add_component(CharacterControllerComponent, **CUSTOM)
    restored = scene_from_dict(scene_to_dict(scene))
    hero = restored.find("Hero")
    assert hero.get_component(ColliderComponent) is not None
    assert hero.get_component(CharacterControllerComponent).to_dict() == CUSTOM
