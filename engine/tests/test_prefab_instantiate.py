"""Prefab: instantiate, вложенные префабы, циклы, детерминизм uid."""

from __future__ import annotations

import json

import pytest

from engine.core.math3d import Quaternion, Vector3
from engine.core.prefab_components import PrefabInstance
from engine.core.scene import Scene
from engine.prefab import (
    Prefab,
    PrefabCycleError,
    PrefabError,
    prefab_from_dict,
    set_default_prefab_loader,
)
from engine.scene_io import load_scene, save_scene, scene_to_dict
from engine.scripting import ScriptComponent


def _entry(local_id: str, parent: str | None = None, components=()) -> dict:
    entry: dict = {
        "local_id": local_id,
        "name": local_id,
        "tag": "Untagged",
        "layer": 1,
        "active": True,
        "transform": {
            "position": [1.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "components": list(components),
    }
    if parent is not None:
        entry["parent"] = {"local": parent}
    return entry


def _marker(guid: str) -> dict:
    return {
        "type": "PrefabInstance",
        "prefab_guid": guid,
        "prefab_path": f"{guid}.prefab.json",
        "overrides": [],
    }


def _script(tmp_path) -> dict:
    """Скрипт с полем ``target``, ссылающимся на объект ``L2`` префаба."""
    path = tmp_path / "s.py"
    path.write_text("target: GameObject = None\n", encoding="utf-8")
    return {
        "type": "ScriptComponent",
        "path": str(path),
        "fields": {"target": {"local": "L2"}},
    }


def _prefab(guid: str, entries: list[dict]) -> Prefab:
    return prefab_from_dict(
        {
            "format": "myengine.prefab",
            "version": 1,
            "name": guid,
            "root": "L0",
            "objects": entries,
        },
        guid,
    )


def _three(guid: str) -> Prefab:
    return _prefab(guid, [_entry("L0"), _entry("L1", "L0"), _entry("L2", "L0")])


class Library:
    """Словарь префабов + загрузчик для Scene.prefab_loader."""

    def __init__(self, *prefabs: Prefab) -> None:
        self.items = {p.guid: p for p in prefabs}

    def __call__(self, guid: str, path: str) -> Prefab:
        if guid in self.items:
            return self.items[guid]
        raise PrefabError(f"нет префаба {guid!r}")


def _scene(*prefabs: Prefab) -> Scene:
    scene = Scene("s")
    scene.prefab_loader = Library(*prefabs)
    return scene


def _uids(scene: Scene) -> list[str]:
    return [o.uid for o in scene.objects]


class TestInstantiate:
    def test_returns_root_with_marker(self) -> None:
        a = _three("a")
        scene = _scene(a)
        root = scene.instantiate(a)
        assert root.name == "L0"
        assert len(scene.objects) == 3
        assert [c.name for c in root.children] == ["L1", "L2"]
        marker = root.get_component(PrefabInstance)
        assert marker is not None and marker.prefab_guid == "a"
        assert marker.overrides == []

    def test_parent(self) -> None:
        a = _three("a")
        scene = _scene(a)
        holder = scene.create_game_object("holder")
        root = scene.instantiate(a, parent=holder)
        assert root.parent is holder
        assert root in holder.children

    def test_position_and_rotation(self) -> None:
        a = _three("a")
        scene = _scene(a)
        rotation = Quaternion.from_axis_angle(Vector3(0.0, 1.0, 0.0), 90.0)
        root = scene.instantiate(
            a, position=Vector3(5.0, 6.0, 7.0), rotation=rotation
        )
        assert root.transform.position.to_tuple() == (5.0, 6.0, 7.0)
        assert root.transform.rotation.to_tuple() == pytest.approx(
            rotation.to_tuple()
        )

    def test_string_reference_uses_loader(self) -> None:
        a = _three("a")
        scene = _scene(a)
        assert scene.instantiate("a").name == "L0"

    def test_uids_do_not_clash_with_scene(self) -> None:
        a = _three("a")
        scene = _scene(a)
        scene.create_game_object("x")
        before = set(_uids(scene))
        scene.instantiate(a)
        new = set(_uids(scene)) - before
        assert len(new) == 3

    def test_two_instances_have_distinct_uids(self) -> None:
        a = _three("a")
        scene = _scene(a)
        first = scene.instantiate(a)
        second = scene.instantiate(a)
        assert len(set(_uids(scene))) == 6
        assert first.uid != second.uid

    def test_collision_skips_serial(self) -> None:
        a = _three("a")
        scene = _scene(a)
        first = scene.instantiate(a)
        scene._spawn_serial = 0  # как после загрузки сцены
        second = scene.instantiate(a)
        assert second.uid != first.uid
        assert len(set(_uids(scene))) == 6

    def test_refs_resolve_inside_instance(self, tmp_path) -> None:
        a = _prefab("a", [_entry("L0"), _entry("L1", "L0", [_script(tmp_path)]), _entry("L2", "L0")])
        scene = _scene(a)
        first = scene.instantiate(a)
        second = scene.instantiate(a)
        for root in (first, second):
            script = root.children[0].get_component(ScriptComponent)
            assert script.to_dict()["fields"]["target"] == {"uid": root.children[1].uid}

    def test_unknown_string_raises(self) -> None:
        with pytest.raises(PrefabError):
            _scene().instantiate("nope")

    def test_foreign_parent_rejected(self) -> None:
        a = _three("a")
        other = Scene("o").create_game_object("p")
        with pytest.raises(ValueError):
            _scene(a).instantiate(a, parent=other)


class TestNested:
    def test_b_includes_a(self) -> None:
        a = _three("a")
        b = _prefab(
            "b",
            [
                _entry("L0"),
                _entry("L1", "L0", [_marker("a")]),
                _entry("L2", "L0"),
                _entry("L3", "L0"),
            ],
        )
        scene = _scene(a, b)
        root = scene.instantiate(b)
        # B: 4 записи, одна из них заместитель A (3 объекта) -> 3 + 3
        assert len(scene.objects) == 6
        nested = [o for o in scene.objects if o.get_component(PrefabInstance)]
        assert {o.get_component(PrefabInstance).prefab_guid for o in nested} == {
            "a",
            "b",
        }
        placeholder = next(
            o for o in scene.objects
            if o.get_component(PrefabInstance).prefab_guid == "a"
        )
        assert placeholder.parent is root
        assert [c.name for c in placeholder.children] == ["L1", "L2"]

    def test_c_b_a_nine_objects(self) -> None:
        a = _three("a")
        own = [_entry("L0"), _entry("L2", "L0"), _entry("L3", "L0")]
        b = _prefab("b", [*own, _entry("L1", "L0", [_marker("a")])])
        c = _prefab("c", [*own, _entry("L1", "L0", [_marker("b")])])
        scene = _scene(a, b, c)
        scene.instantiate(c)
        # 3 своих объекта c + (3 своих b + 3 объекта a) = 9
        assert len(scene.objects) == 9
        assert len(set(_uids(scene))) == 9

    def test_placeholder_overrides_transform_and_name(self) -> None:
        a = _three("a")
        placeholder = _entry("L1", "L0", [_marker("a")])
        placeholder["name"] = "Weapon"
        placeholder["transform"]["position"] = [0.0, 2.0, 0.0]
        b = _prefab("b", [_entry("L0"), placeholder])
        scene = _scene(a, b)
        scene.instantiate(b)
        weapon = scene.find("Weapon")
        assert weapon is not None
        assert weapon.transform.position.to_tuple() == (0.0, 2.0, 0.0)

    def test_refs_in_nested_resolve_to_own_instance(self, tmp_path) -> None:
        # В A скрипт на L1 ссылается на L2; у каждого экземпляра B своя цель.
        a = _prefab(
            "a",
            [_entry("L0"), _entry("L1", "L0", [_script(tmp_path)]), _entry("L2", "L0")],
        )
        b = _prefab("b", [_entry("L0"), _entry("L1", "L0", [_marker("a")])])
        scene = _scene(a, b)
        for _ in range(2):
            outer = scene.instantiate(b)
            inner = outer.children[0]
            script = inner.children[0].get_component(ScriptComponent)
            target = script.to_dict()["fields"]["target"]
            assert target == {"uid": inner.children[1].uid}
        assert len({o.uid for o in scene.objects}) == len(scene.objects) == 8

    def test_depth_nine_is_error(self) -> None:
        chain = [_three("p0")]
        for i in range(1, 9):
            chain.append(
                _prefab(f"p{i}", [_entry("L0"), _entry("L1", "L0", [_marker(f"p{i-1}")])])
            )
        # p8 -> p7 -> ... -> p0: 9 префабов
        scene = _scene(*chain)
        with pytest.raises(PrefabCycleError):
            scene.instantiate(chain[-1])
        assert len(scene.objects) == 0

    def test_depth_eight_is_ok(self) -> None:
        chain = [_three("p0")]
        for i in range(1, 8):
            chain.append(
                _prefab(f"p{i}", [_entry("L0"), _entry("L1", "L0", [_marker(f"p{i-1}")])])
            )
        scene = _scene(*chain)
        scene.instantiate(chain[-1])
        assert len(scene.objects) > 0

    def test_cycle_a_b_a(self) -> None:
        a = _prefab("a", [_entry("L0"), _entry("L1", "L0", [_marker("b")])])
        b = _prefab("b", [_entry("L0"), _entry("L1", "L0", [_marker("a")])])
        scene = _scene(a, b)
        with pytest.raises(PrefabCycleError):
            scene.instantiate(a)
        assert len(scene.objects) == 0


class TestDeterminism:
    def _nested(self) -> tuple[Prefab, Prefab]:
        a = _three("a")
        b = _prefab(
            "b",
            [_entry("L0"), _entry("L1", "L0", [_marker("a")]), _entry("L2", "L0")],
        )
        return a, b

    def _run(self) -> list[str]:
        a, b = self._nested()
        scene = _scene(a, b)
        scene.instantiate(b)
        scene.instantiate(b, position=Vector3(1.0, 0.0, 0.0))
        scene.instantiate(a)
        return _uids(scene)

    def test_two_runs_in_one_process(self) -> None:
        assert self._run() == self._run()

    def test_roundtrip_save_load(self, tmp_path) -> None:
        a, b = self._nested()
        scene = _scene(a, b)
        scene.instantiate(b)
        scene.instantiate(a)
        path = tmp_path / "s.scene.json"
        save_scene(scene, path)
        first = path.read_bytes()
        set_default_prefab_loader(Library(a, b))
        try:
            loaded = load_scene(path)
        finally:
            set_default_prefab_loader(None)
        assert _uids(loaded) == _uids(scene)
        loaded.prefab_loader = Library(a, b)
        save_scene(loaded, path)
        assert path.read_bytes() == first
        markers = [
            o.get_component(PrefabInstance)
            for o in loaded.objects
            if o.get_component(PrefabInstance)
        ]
        assert len(markers) == 3

    def test_new_instance_after_load_avoids_collision(self, tmp_path) -> None:
        a = _three("a")
        scene = _scene(a)
        scene.instantiate(a)
        path = tmp_path / "s.scene.json"
        save_scene(scene, path)
        set_default_prefab_loader(Library(a))
        try:
            loaded = load_scene(path)
        finally:
            set_default_prefab_loader(None)
        loaded.prefab_loader = Library(a)
        loaded.instantiate(a)
        assert len(set(_uids(loaded))) == 6

    def test_scene_dict_is_valid_json(self) -> None:
        a, b = self._nested()
        scene = _scene(a, b)
        scene.instantiate(b)
        text = json.dumps(scene_to_dict(scene))
        assert '"PrefabInstance"' in text


class TestCollapse:
    def test_nested_instance_collapses_when_saved_as_prefab(self) -> None:
        from engine.prefab import prefab_to_dict

        a = _three("a")
        b = _prefab("b", [_entry("L0"), _entry("L1", "L0", [_marker("a")])])
        scene = _scene(a, b)
        root = scene.instantiate(b)
        # корень экземпляра b сам несёт PrefabInstance -> одна запись
        data = prefab_to_dict(root)
        assert len(data["objects"]) == 1
        holder = scene.create_game_object("holder")
        child = scene.instantiate(b, parent=holder)
        data = prefab_to_dict(holder)
        assert [o["name"] for o in data["objects"]] == ["holder", "L0"]
        assert child.get_component(PrefabInstance) is not None
