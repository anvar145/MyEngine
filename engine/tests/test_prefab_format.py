"""Формат ``*.prefab.json``: запись, чтение, валидация, ссылки ``local``."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from engine.core.components import MeshRenderer
from engine.core.event_bus import EventBus
from engine.core.math3d import Vector3
from engine.core.scene import Scene
from engine.prefab import (
    PrefabError,
    build_prefab_objects,
    dump_prefab,
    load_prefab_file,
    prefab_from_dict,
    prefab_to_dict,
    validate_prefab_data,
)
from engine.scene_io import scene_to_dict, serialize_objects
from engine.scripting import ScriptComponent, ScriptRuntime

SCRIPT = "target: GameObject = None\n"


def _enemy(tmp_path: Path):
    """Сцена с поддеревом Enemy -> (Weapon, Shield); Weapon ссылается на Shield."""
    runtime = ScriptRuntime(project_root=tmp_path, auto_reload=False)
    script_path = tmp_path / "s.py"
    script_path.write_text(SCRIPT, encoding="utf-8")
    scene = Scene("s", event_bus=EventBus())
    scene.create_game_object("Filler")
    root = scene.create_game_object("Enemy", position=Vector3(1, 2, 3), tag="Mob")
    weapon = scene.create_game_object("Weapon", parent=root)
    shield = scene.create_game_object("Shield", parent=root)
    weapon.add_component(MeshRenderer(mesh="cube"))
    script = weapon.add_component(ScriptComponent(script_path, runtime=runtime))
    script.set_field("target", shield)
    return scene, root, weapon, shield


def _uid_counter():
    counter = iter(range(1000))
    mapping: dict[str, str] = {}

    def uid_for(local_id: str) -> str:
        if local_id not in mapping:
            mapping[local_id] = f"{next(counter):016x}"
        return mapping[local_id]

    return uid_for


class TestSerialize:
    def test_shape(self, tmp_path: Path) -> None:
        _, root, _, _ = _enemy(tmp_path)
        data = prefab_to_dict(root)
        assert data["format"] == "myengine.prefab"
        assert data["version"] == 1
        assert data["name"] == "Enemy"
        assert data["root"] == "L0"
        ids = [entry["local_id"] for entry in data["objects"]]
        assert ids == ["L0", "L1", "L2"]
        assert "parent" not in data["objects"][0]
        assert data["objects"][1]["parent"] == {"local": "L0"}
        assert data["objects"][2]["parent"] == {"local": "L0"}

    def test_no_uid_anywhere(self, tmp_path: Path) -> None:
        _, root, weapon, shield = _enemy(tmp_path)
        text = dump_prefab(prefab_to_dict(root))
        for obj in (root, weapon, shield):
            assert obj.uid not in text
        assert '"uid"' not in text

    def test_script_reference_is_local(self, tmp_path: Path) -> None:
        _, root, _, _ = _enemy(tmp_path)
        data = prefab_to_dict(root)
        fields = data["objects"][1]["components"][1]["fields"]
        assert fields["target"] == {"local": "L2"}

    def test_reference_outside_prefab_is_dropped(self, tmp_path: Path) -> None:
        scene, root, weapon, _ = _enemy(tmp_path)
        script = weapon.get_component(ScriptComponent)
        script.set_field("target", scene.find("Filler"))
        data = prefab_to_dict(root)
        assert data["objects"][1]["components"][1]["fields"].get("target") is None

    def test_repeated_dump_is_byte_identical(self, tmp_path: Path) -> None:
        _, root, _, _ = _enemy(tmp_path)
        first = dump_prefab(prefab_to_dict(root))
        for _ in range(10):
            assert dump_prefab(prefab_to_dict(root)) == first

    def test_scene_serialization_unchanged_by_shared_code(
        self, tmp_path: Path
    ) -> None:
        scene, _, _, _ = _enemy(tmp_path)
        assert scene_to_dict(scene)["objects"] == serialize_objects(scene.objects)

    def test_unknown_ref_key(self) -> None:
        with pytest.raises(ValueError, match="ref_key"):
            serialize_objects([], ref_key="bad")


class TestRoundTrip:
    def test_build_and_serialize_again(self, tmp_path: Path) -> None:
        _, root, _, _ = _enemy(tmp_path)
        data = json.loads(dump_prefab(prefab_to_dict(root)))
        prefab = prefab_from_dict(data)

        target = Scene("t", event_bus=EventBus())
        built = build_prefab_objects(target, prefab, _uid_counter())
        assert list(built) == ["L0", "L1", "L2"]
        new_root = built["L0"]
        assert new_root.name == "Enemy" and new_root.tag == "Mob"
        assert new_root.parent is None
        assert built["L1"].parent is new_root and built["L2"].parent is new_root
        assert new_root.transform.position == Vector3(1, 2, 3)
        script = built["L1"].get_component(ScriptComponent)
        assert script is not None
        assert script.to_dict()["fields"]["target"] == {"uid": built["L2"].uid}

        again = prefab_to_dict(new_root)
        assert again == data
        assert dump_prefab(again) == dump_prefab(data)

    def test_file_round_trip(self, tmp_path: Path) -> None:
        _, root, _, _ = _enemy(tmp_path)
        data = prefab_to_dict(root)
        path = tmp_path / "enemy.prefab.json"
        path.write_text(dump_prefab(data), encoding="utf-8")
        prefab = load_prefab_file(path, guid="g1")
        assert prefab.guid == "g1" and prefab.name == "Enemy"
        assert prefab.to_dict() == data

    def test_uids_come_from_callback(self, tmp_path: Path) -> None:
        _, root, _, _ = _enemy(tmp_path)
        prefab = prefab_from_dict(prefab_to_dict(root))
        scene = Scene("t", event_bus=EventBus())
        built = build_prefab_objects(scene, prefab, lambda local: f"{local[1:]:0>16}")
        assert built["L0"].uid == "0" * 16
        assert built["L2"].uid == "0" * 15 + "2"


def _valid() -> dict:
    return {
        "format": "myengine.prefab",
        "version": 1,
        "name": "P",
        "root": "L0",
        "objects": [
            {"local_id": "L0", "name": "A", "components": []},
            {"local_id": "L1", "name": "B", "parent": {"local": "L0"},
             "components": []},
        ],
    }


class TestValidation:
    def test_valid(self) -> None:
        validate_prefab_data(_valid())

    @pytest.mark.parametrize(
        ("mutate", "text"),
        [
            (lambda d: d.update(format="x"), "формат"),
            (lambda d: d.update(version=2), "версия"),
            (lambda d: d.update(root="L9"), "root"),
            (lambda d: d.update(objects=[]), "нет объектов"),
            (lambda d: d["objects"][1].update(local_id="L0"), "не уникальны"),
            (lambda d: d["objects"][1].update(parent={"local": "L7"}), "несуществующ"),
            (lambda d: d["objects"][1].pop("parent"), "без parent"),
            (
                lambda d: d["objects"][0].update(parent={"local": "L1"}),
                "без parent",
            ),
            (
                lambda d: d["objects"][1]["components"].append(
                    {"type": "ScriptComponent", "fields": {"t": {"local": "L5"}}}
                ),
                "ссылка",
            ),
        ],
    )
    def test_invalid(self, mutate, text: str) -> None:
        data = copy.deepcopy(_valid())
        mutate(data)
        with pytest.raises(PrefabError, match=text):
            validate_prefab_data(data)

    def test_hierarchy_cycle(self) -> None:
        data = _valid()
        data["objects"].append(
            {"local_id": "L2", "name": "C", "parent": {"local": "L3"}}
        )
        data["objects"].append(
            {"local_id": "L3", "name": "D", "parent": {"local": "L2"}}
        )
        with pytest.raises(PrefabError):
            validate_prefab_data(data)

    def test_broken_json_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.prefab.json"
        path.write_text("{не json", encoding="utf-8")
        with pytest.raises(PrefabError, match="JSON"):
            load_prefab_file(path)

    def test_not_an_object(self) -> None:
        with pytest.raises(PrefabError):
            validate_prefab_data([])
