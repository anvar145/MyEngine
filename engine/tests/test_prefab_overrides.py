"""Prefab overrides: дифф при сохранении, применение при загрузке, revert."""

from __future__ import annotations

import json
import logging

import pytest

from engine.core.math3d import Vector3
from engine.core.prefab_components import PrefabInstance
from engine.core.scene import Scene
from engine.prefab import (
    PrefabError,
    capture_instances,
    find_instance,
    get_overrides,
    restore_instances,
    revert_override,
)
from engine.prefab_overrides import (
    apply_overrides,
    diff_entry,
    merge_stale,
    override_key,
)
from engine.scene_io import scene_from_dict, scene_to_dict
from engine.tests.test_prefab_instantiate import (
    Library,
    _entry,
    _marker,
    _prefab,
    _script,
    _three,
)


def _reload(scene: Scene, library: Library) -> Scene:
    data = json.loads(json.dumps(scene_to_dict(scene)))
    fresh = Scene("s")
    fresh.prefab_loader = library
    return scene_from_dict(data, into=fresh)


def _mesh(color: list[float]) -> dict:
    return {"type": "MeshRenderer", "color": color}


class TestPureFunctions:
    def test_diff_entry_fields(self) -> None:
        base = _entry("L1", components=[_mesh([1, 1, 1])])
        live = _entry("L1", components=[_mesh([1, 0, 0])])
        live["name"] = "renamed"
        live["transform"]["position"] = [2.0, 0.0, 0.0]
        diff = diff_entry("L1", base, live)
        keys = {(r["component"], r.get("field")) for r in diff}
        assert keys == {
            ("Root", "name"),
            ("Transform", "position"),
            ("MeshRenderer", "color"),
        }

    def test_skip_transform(self) -> None:
        base = _entry("L0")
        live = _entry("L0")
        live["transform"]["position"] = [9.0, 0.0, 0.0]
        assert diff_entry("L0", base, live, skip_transform=True) == []

    def test_added_component_and_unset(self) -> None:
        base = _entry("L1", components=[{"type": "A", "x": 1, "y": 2}])
        live = _entry("L1", components=[{"type": "A", "x": 1}, {"type": "B", "z": 3}])
        diff = diff_entry("L1", base, live)
        assert {"target": "L1", "component": "A", "field": "y", "unset": True} in diff
        added = [r for r in diff if r.get("add")]
        assert added == [
            {"target": "L1", "component": "B", "add": True, "value": {"z": 3}}
        ]

    def test_apply_marks_stale_and_revives(self) -> None:
        entries = [{"uid": "u1", "components": [{"type": "A", "x": 1}]}]
        paths = {"u1": "L1"}
        good = {"target": "L1", "component": "A", "field": "x", "value": 5}
        lost = {"target": "L9", "component": "A", "field": "x", "value": 1}
        result = apply_overrides(entries, paths, [good, lost])
        assert entries[0]["components"][0]["x"] == 5
        assert "stale" not in result[0]
        assert result[1]["stale"] is True
        # устаревшая запись при следующем проходе оживает, когда цель есть
        paths["u9"] = "L9"
        entries.append({"uid": "u9", "components": [{"type": "A", "x": 0}]})
        again = apply_overrides(entries, paths, result)
        assert "stale" not in again[1]
        assert entries[1]["components"][0]["x"] == 1

    def test_merge_stale_keeps_previous_only(self) -> None:
        diff = [{"target": "L1", "component": "A", "field": "x", "value": 1}]
        previous = [
            {"target": "L1", "component": "A", "field": "x", "value": 0, "stale": True},
            {"target": "L2", "component": "A", "field": "x", "value": 0, "stale": True},
            {"target": "L3", "component": "A", "field": "x", "value": 0},
        ]
        merged = merge_stale(diff, previous)
        assert [override_key(r)[0] for r in merged] == ["L1", "L2"]


class TestRoundTrip:
    def _setup(self, tmp_path):
        script = _script(tmp_path)
        a = _prefab(
            "a",
            [
                _entry("L0"),
                _entry("L1", "L0", [_mesh([1.0, 1.0, 1.0])]),
                _entry("L2", "L0", [script]),
            ],
        )
        return a, Library(a)

    def test_five_overrides_byte_identical(self, tmp_path) -> None:
        a, library = self._setup(tmp_path)
        scene = Scene("s")
        scene.prefab_loader = library
        root = scene.instantiate(a)
        l1 = root.children[0]
        l1.transform.position = Vector3(4.0, 5.0, 6.0)
        l1.name = "renamed"
        mesh = next(c for c in l1.components if type(c).__name__ == "MeshRenderer")
        mesh.color = Vector3(1.0, 0.0, 0.0)
        first = scene_to_dict(scene)
        marker = [
            c
            for e in first["objects"]
            for c in e["components"]
            if c["type"] == "PrefabInstance"
        ]
        assert len(first["objects"]) == 1
        assert marker[0]["instance_uid"]
        assert len(marker[0]["overrides"]) >= 3
        second = scene_to_dict(_reload(scene, library))
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_values_survive_reload(self, tmp_path) -> None:
        a, library = self._setup(tmp_path)
        scene = Scene("s")
        scene.prefab_loader = library
        root = scene.instantiate(a)
        root.children[0].name = "renamed"
        fresh = _reload(scene, library)
        names = sorted(o.name for o in fresh.objects)
        assert names == ["L0", "L2", "renamed"]
        assert [o.uid for o in fresh.objects] == [o.uid for o in scene.objects]

    def test_no_overrides_for_clean_instance(self, tmp_path) -> None:
        a, library = self._setup(tmp_path)
        scene = Scene("s")
        scene.prefab_loader = library
        root = scene.instantiate(a)
        assert get_overrides(root) == []

    def test_root_transform_not_an_override(self, tmp_path) -> None:
        a, library = self._setup(tmp_path)
        scene = Scene("s")
        scene.prefab_loader = library
        root = scene.instantiate(a, position=Vector3(9.0, 9.0, 9.0))
        assert get_overrides(root) == []
        fresh = _reload(scene, library)
        assert fresh.objects[0].transform.position.to_tuple() == (9.0, 9.0, 9.0)

    def test_added_child_is_plain_object(self, tmp_path) -> None:
        a, library = self._setup(tmp_path)
        scene = Scene("s")
        scene.prefab_loader = library
        root = scene.instantiate(a)
        extra = scene.create_game_object("extra")
        extra.set_parent(root)
        data = scene_to_dict(scene)
        assert len(data["objects"]) == 2
        assert data["objects"][1]["name"] == "extra"
        assert data["objects"][1]["parent"] == {"uid": root.uid}
        fresh = _reload(scene, library)
        assert len(fresh.objects) == 4

    def test_a42_marker_without_instance_uid_still_loads(self) -> None:
        a = _three("a")
        library = Library(a)
        data = {
            "format": "myengine.scene",
            "version": 2,
            "name": "s",
            "objects": [
                {
                    "uid": "0123456789abcdef",
                    "name": "X",
                    "tag": "Untagged",
                    "layer": 1,
                    "active": True,
                    "transform": _entry("L0")["transform"],
                    "components": [_marker("a")],
                }
            ],
        }
        scene = Scene("s")
        scene.prefab_loader = library
        scene_from_dict(data, into=scene)
        assert len(scene.objects) == 1


class TestStale:
    def test_stale_survives_and_revives(self, tmp_path, caplog) -> None:
        a = _three("a")
        library = Library(a)
        scene = Scene("s")
        scene.prefab_loader = library
        scene.instantiate(a)
        data = scene_to_dict(scene)
        marker = data["objects"][0]["components"][0]
        marker["overrides"] = [
            {"target": "L7", "component": "Root", "field": "name", "value": "z"}
        ]
        fresh = Scene("s")
        fresh.prefab_loader = library
        with caplog.at_level(logging.WARNING, logger="engine.prefab"):
            scene_from_dict(json.loads(json.dumps(data)), into=fresh)
        assert sum("не применились" in r.message for r in caplog.records) == 1
        saved = scene_to_dict(fresh)["objects"][0]["components"][0]["overrides"]
        assert saved[0]["stale"] is True
        # префаб получает L7 - запись оживает
        b = _prefab(
            "a",
            [_entry("L0"), _entry("L1", "L0"), _entry("L2", "L0"), _entry("L7", "L0")],
        )
        library2 = Library(b)
        again = Scene("s")
        again.prefab_loader = library2
        scene_from_dict(json.loads(json.dumps(scene_to_dict(fresh))), into=again)
        assert "z" in [o.name for o in again.objects]
        final = scene_to_dict(again)["objects"][0]["components"][0]["overrides"]
        assert "stale" not in final[0]


class TestNested:
    def _nested(self):
        inner = _prefab("inner", [_entry("L0"), _entry("leaf", "L0")])
        outer_entries = [
            _entry("L0"),
            _entry("P", "L0", [_marker("inner")]),
        ]
        outer = _prefab("outer", outer_entries)
        return inner, outer, Library(inner, outer)

    def test_inner_leaf_override_roundtrip(self) -> None:
        _inner, outer, library = self._nested()
        scene = Scene("s")
        scene.prefab_loader = library
        root = scene.instantiate(outer)
        leaf = next(o for o in scene.objects if o.name == "leaf")
        leaf.name = "changed"
        overrides = get_overrides(root)
        assert [(r["target"], r["field"]) for r in overrides] == [
            ("P/leaf", "name")
        ]
        fresh = _reload(scene, library)
        assert "changed" in [o.name for o in fresh.objects]
        found = find_instance(next(o for o in fresh.objects if o.name == "changed"),
                              loader=library)
        assert found is not None and found[1] == "P/leaf"

    def test_slash_in_local_id_rejected(self) -> None:
        with pytest.raises(PrefabError):
            _prefab("bad", [_entry("L0"), _entry("a/b", "L0")])


class TestRevert:
    def test_revert_name_and_transform(self) -> None:
        a = _three("a")
        library = Library(a)
        scene = Scene("s")
        scene.prefab_loader = library
        root = scene.instantiate(a)
        child = root.children[0]
        child.name = "zzz"
        child.transform.position = Vector3(7.0, 7.0, 7.0)
        overrides = get_overrides(root)
        assert len(overrides) == 2
        for rec in overrides:
            revert_override(root, rec)
        assert child.name == "L1"
        assert child.transform.position.to_tuple() == (1.0, 0.0, 0.0)
        assert get_overrides(root) == []

    def test_revert_component_field(self, tmp_path) -> None:
        a = _prefab(
            "a", [_entry("L0"), _entry("L1", "L0", [_mesh([1.0, 1.0, 1.0])])]
        )
        library = Library(a)
        scene = Scene("s")
        scene.prefab_loader = library
        root = scene.instantiate(a)
        mesh = next(
            c for c in root.children[0].components if type(c).__name__ == "MeshRenderer"
        )
        mesh.color = Vector3(0.0, 0.0, 1.0)
        (rec,) = get_overrides(root)
        revert_override(root, rec)
        assert get_overrides(root) == []

    def test_apply_to_asset_changes_base(self) -> None:
        a = _three("a")
        library = Library(a)
        scene = Scene("s")
        scene.prefab_loader = library
        root = scene.instantiate(a)
        root.children[0].name = "new"
        (rec,) = get_overrides(root)
        a.apply_override_to_asset(
            rec["target"], rec["component"], rec["field"], rec["value"]
        )
        assert get_overrides(root) == []


class TestCaptureRestore:
    def test_rebuild_keeps_overrides(self) -> None:
        a = _three("a")
        library = Library(a)
        scene = Scene("s")
        scene.prefab_loader = library
        holder = scene.create_game_object("holder")
        root = scene.instantiate(a, parent=holder)
        root.children[1].name = "keep"
        uids = sorted(o.uid for o in scene.objects)
        snapshot = capture_instances(scene, "a")
        assert len(snapshot) == 1
        # новая версия префаба: у L1 новая позиция
        entries = [_entry("L0"), _entry("L1", "L0"), _entry("L2", "L0")]
        entries[1]["transform"]["position"] = [3.0, 0.0, 0.0]
        library.items["a"] = _prefab("a", entries)
        calls: list[tuple[int, int]] = []
        restore_instances(
            scene, snapshot, progress=lambda i, n: calls.append((i, n))
        )
        assert calls == [(1, 1)]
        assert sorted(o.uid for o in scene.objects) == uids
        by_name = {o.name: o for o in scene.objects}
        assert "keep" in by_name
        assert by_name["L1"].transform.position.to_tuple() == (3.0, 0.0, 0.0)
        assert by_name["L0"].parent is by_name["holder"]


def test_marker_is_component() -> None:
    assert PrefabInstance("g").overrides == []
