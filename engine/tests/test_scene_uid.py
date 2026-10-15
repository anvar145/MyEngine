"""``uid`` объектов: индекс сцены, формат v2, миграция v1, бэкап ``.v1.bak``."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from engine.core.event_bus import EventBus
from engine.core.game_object import GameObject, is_valid_uid
from engine.core.scene import Scene
from engine.player.snapshot import SceneSnapshot
from engine.scene_io import (
    SCENE_VERSION,
    load_scene,
    migrated_uid,
    save_scene,
    scene_from_dict,
    scene_to_dict,
)
from engine.scripting import ScriptComponent, ScriptRuntime

REPO = Path(__file__).resolve().parents[2]
MAIN_SCENE = REPO / "projects" / "PlayModeTest" / "Assets" / "Scenes" / "main.scene.json"

SCRIPT = """
target: GameObject = None
def on_start(self):
    pass
"""


def _entry(oid: int, name: str, parent: int | None = None, **extra) -> dict:
    return {
        "id": oid,
        "name": name,
        "tag": "Untagged",
        "layer": 1,
        "active": True,
        "parent": parent,
        "transform": {
            "position": [oid, 0, 0],
            "rotation": [0, 0, 0, 1],
            "scale": [1, 1, 1],
        },
        "components": extra.get("components", []),
    }


def _v1(objects: list[dict]) -> dict:
    return {"format": "myengine.scene", "version": 1, "name": "Old", "objects": objects}


class TestUid:
    def test_random_valid_and_unique(self) -> None:
        scene = Scene("s")
        uids = {scene.create_game_object(f"o{i}").uid for i in range(200)}
        assert len(uids) == 200
        assert all(is_valid_uid(uid) for uid in uids)

    @pytest.mark.parametrize("bad", ["", "abc", "g" * 16, "A" * 16, "0" * 17, 5])
    def test_setter_rejects_invalid(self, bad) -> None:
        obj = Scene("s").create_game_object("o")
        with pytest.raises(ValueError, match="uid"):
            obj.uid = bad

    def test_setter_updates_index_and_rejects_duplicate(self) -> None:
        scene = Scene("s")
        a = scene.create_game_object("a")
        b = scene.create_game_object("b")
        old = a.uid
        a.uid = "0123456789abcdef"
        assert scene.find_by_uid("0123456789abcdef") is a
        assert scene.find_by_uid(old) is None
        with pytest.raises(ValueError, match="занят"):
            b.uid = "0123456789abcdef"
        assert scene.find_by_uid(b.uid) is b

    def test_orphan_object_has_uid(self) -> None:
        assert is_valid_uid(GameObject("free").uid)


class TestIndex:
    def test_create_and_destroy(self) -> None:
        scene = Scene("s")
        obj = scene.create_game_object("o")
        assert scene.find_by_uid(obj.uid) is obj
        uid = obj.uid
        scene.destroy_game_object(obj, immediate=True)
        assert scene.find_by_uid(uid) is None

    def test_deferred_destroy_and_children(self) -> None:
        scene = Scene("s")
        parent = scene.create_game_object("p")
        child = scene.create_game_object("c", parent=parent)
        scene.destroy_game_object(parent)
        scene.process_pending()
        assert scene.find_by_uid(parent.uid) is None
        assert scene.find_by_uid(child.uid) is None

    def test_clear_empties_index(self) -> None:
        scene = Scene("s")
        for i in range(5):
            scene.create_game_object(f"o{i}")
        scene.clear()
        assert scene.find_by_uid("0" * 16) is None
        assert not scene._by_uid  # noqa: SLF001

    def test_adopt_conflicting_uid_reissued(self, caplog) -> None:
        scene = Scene("s")
        first = scene.create_game_object("first")
        orphan = GameObject("orphan")
        orphan._uid = first.uid  # noqa: SLF001
        with caplog.at_level(logging.ERROR):
            scene.add_game_object(orphan)
        assert orphan.uid != first.uid
        assert scene.find_by_uid(first.uid) is first
        assert scene.find_by_uid(orphan.uid) is orphan
        assert any("занят" in r.getMessage() for r in caplog.records)

    def test_index_not_serialized(self) -> None:
        scene = Scene("s")
        scene.create_game_object("o")
        text = json.dumps(scene_to_dict(scene))
        assert "_by_uid" not in text


class TestFormatV2:
    def test_written_shape(self) -> None:
        scene = Scene("s")
        parent = scene.create_game_object("p")
        child = scene.create_game_object("c", parent=parent)
        data = scene_to_dict(scene)
        assert data["version"] == SCENE_VERSION == 2
        p, c = data["objects"]
        assert p["uid"] == parent.uid and "id" not in p and p["parent"] is None
        assert c["uid"] == child.uid and c["parent"] == {"uid": parent.uid}

    def test_round_trip_is_identical(self) -> None:
        scene = Scene("s")
        a = scene.create_game_object("a")
        scene.create_game_object("b", parent=a)
        first = scene_to_dict(scene)
        second = scene_to_dict(scene_from_dict(json.loads(json.dumps(first))))
        assert first == second

    def test_v2_uses_file_uids(self) -> None:
        data = scene_to_dict(_populated())
        loaded = scene_from_dict(data)
        assert [o.uid for o in loaded.objects] == [o["uid"] for o in data["objects"]]

    def test_newer_version_rejected(self) -> None:
        data = scene_to_dict(Scene("s"))
        data["version"] = 3
        with pytest.raises(ValueError, match="новой"):
            scene_from_dict(data)

    def test_duplicate_uid_in_file(self, caplog) -> None:
        data = scene_to_dict(_populated())
        data["objects"][1]["uid"] = data["objects"][0]["uid"]
        with caplog.at_level(logging.ERROR):
            loaded = scene_from_dict(data)
        assert len(loaded) == 3
        assert len({o.uid for o in loaded.objects}) == 3
        assert any("занят" in r.getMessage() for r in caplog.records)

    def test_invalid_uid_in_file(self, caplog) -> None:
        data = scene_to_dict(_populated())
        data["objects"][0]["uid"] = "zzz"
        with caplog.at_level(logging.ERROR):
            loaded = scene_from_dict(data)
        assert len(loaded) == 3
        assert all(is_valid_uid(o.uid) for o in loaded.objects)
        assert any("некорректный uid" in r.getMessage() for r in caplog.records)


def _populated() -> Scene:
    scene = Scene("s", event_bus=EventBus())
    a = scene.create_game_object("a")
    scene.create_game_object("b", parent=a)
    scene.create_game_object("c")
    return scene


class TestMigrationV1:
    def test_migrated_uid_is_stable_blake2b(self) -> None:
        expected = hashlib.blake2b(b"v1:7", digest_size=8).hexdigest()
        assert migrated_uid(7) == expected
        assert is_valid_uid(migrated_uid(7))
        assert migrated_uid(7) != migrated_uid(8)

    def test_loading_twice_gives_same_uids(self) -> None:
        data = _v1([_entry(1, "a"), _entry(2, "b", parent=1), _entry(5, "c")])
        first = scene_from_dict(json.loads(json.dumps(data)))
        second = scene_from_dict(json.loads(json.dumps(data)))
        uids = [o.uid for o in first.objects]
        assert uids == [o.uid for o in second.objects]
        assert uids == [migrated_uid(1), migrated_uid(2), migrated_uid(5)]

    def test_hierarchy_restored_and_written_as_v2(self) -> None:
        data = _v1([_entry(1, "a"), _entry(2, "b", parent=1)])
        scene = scene_from_dict(data)
        a, b = scene.objects
        assert b.parent is a
        out = scene_to_dict(scene)
        assert out["version"] == 2
        assert out["objects"][1]["parent"] == {"uid": migrated_uid(1)}
        assert all("id" not in o for o in out["objects"])
        # повторная миграция записанного файла - те же uid и тот же результат
        assert scene_to_dict(scene_from_dict(out)) == out

    def test_v1_loading_does_not_touch_source_dict(self) -> None:
        data = _v1([_entry(1, "a")])
        before = json.dumps(data, sort_keys=True)
        scene_from_dict(data)
        assert json.dumps(data, sort_keys=True) == before

    def test_script_reference_v1_and_v2(self, tmp_path: Path) -> None:
        script = tmp_path / "s.py"
        script.write_text(SCRIPT, encoding="utf-8")
        runtime = ScriptRuntime(project_root=tmp_path, auto_reload=False)

        def with_ref(ref: dict, v1: bool) -> dict:
            comp = {
                "type": "ScriptComponent",
                "path": str(script),
                "fields": {"target": ref},
            }
            objects = [_entry(1, "owner", components=[comp]), _entry(2, "goal")]
            if v1:
                return _v1(objects)
            data = _v1(objects)
            data["version"] = 2
            return data

        for ref, v1 in (({"object": 2}, True), ({"object": 2}, False)):
            ScriptComponent.default_runtime = runtime  # type: ignore[attr-defined]
            scene = scene_from_dict(with_ref(ref, v1))
            owner = scene.objects[0].get_component(ScriptComponent)
            assert owner is not None
            owner_fields = owner.to_dict()["fields"]
            assert owner_fields["target"] == {"uid": scene.objects[1].uid}

    def test_main_scene_deterministic_and_unchanged(self) -> None:
        if not MAIN_SCENE.is_file():
            pytest.skip("нет projects/PlayModeTest")
        before = hashlib.sha256(MAIN_SCENE.read_bytes()).hexdigest()
        first = [o.uid for o in load_scene(MAIN_SCENE).objects]
        second = [o.uid for o in load_scene(MAIN_SCENE).objects]
        assert first == second and len(set(first)) == len(first)
        assert hashlib.sha256(MAIN_SCENE.read_bytes()).hexdigest() == before


class TestBackup:
    def _write_v1(self, path: Path) -> bytes:
        text = json.dumps(_v1([_entry(1, "a")]), indent=2)
        path.write_text(text, encoding="utf-8")
        return path.read_bytes()

    def test_first_save_makes_backup_and_writes_v2(self, tmp_path, caplog) -> None:
        target = tmp_path / "a.scene.json"
        original = self._write_v1(target)
        scene = load_scene(target)
        with caplog.at_level(logging.INFO):
            save_scene(scene, target)
        backup = tmp_path / "a.scene.json.v1.bak"
        assert backup.read_bytes() == original
        assert json.loads(target.read_text("utf-8"))["version"] == 2
        assert any("бэкап" in r.getMessage() for r in caplog.records)

    def test_backup_not_overwritten(self, tmp_path) -> None:
        target = tmp_path / "a.scene.json"
        backup = tmp_path / "a.scene.json.v1.bak"
        backup.write_text("самая старая", encoding="utf-8")
        self._write_v1(target)
        save_scene(load_scene(target), target)
        assert backup.read_text("utf-8") == "самая старая"

    def test_second_save_of_v2_makes_no_new_backup(self, tmp_path) -> None:
        target = tmp_path / "a.scene.json"
        self._write_v1(target)
        scene = load_scene(target)
        save_scene(scene, target)
        (tmp_path / "a.scene.json.v1.bak").unlink()
        save_scene(scene, target)
        assert not (tmp_path / "a.scene.json.v1.bak").exists()

    def test_new_file_has_no_backup(self, tmp_path) -> None:
        target = tmp_path / "new.scene.json"
        save_scene(_populated(), target)
        assert not list(tmp_path.glob("*.bak"))

    def test_save_as_to_other_path_has_no_backup(self, tmp_path) -> None:
        source = tmp_path / "a.scene.json"
        self._write_v1(source)
        save_scene(load_scene(source), tmp_path / "b.scene.json")
        assert not list(tmp_path.glob("*.bak"))
        assert json.loads(source.read_text("utf-8"))["version"] == 1

    def test_newer_target_is_overwritten_without_backup(self, tmp_path, caplog) -> None:
        target = tmp_path / "a.scene.json"
        target.write_text(json.dumps({"version": 9}), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            save_scene(_populated(), target)
        assert not list(tmp_path.glob("*.bak"))
        assert json.loads(target.read_text("utf-8"))["version"] == 2
        assert any("бэкап не создан" in r.getMessage() for r in caplog.records)

    def test_corrupt_target_is_overwritten(self, tmp_path) -> None:
        target = tmp_path / "a.scene.json"
        target.write_text("{не json", encoding="utf-8")
        save_scene(_populated(), target)
        assert json.loads(target.read_text("utf-8"))["version"] == 2


class TestSnapshot:
    def test_freeze_restore_keeps_uids(self) -> None:
        scene = _populated()
        uids = [o.uid for o in scene.objects]
        snapshot = SceneSnapshot.freeze(scene)
        snapshot.restore(scene)
        assert [o.uid for o in scene.objects] == uids
        assert scene.find_by_uid(uids[1]).parent is scene.find_by_uid(uids[0])

    def test_json_round_trip(self) -> None:
        scene = _populated()
        snapshot = SceneSnapshot.freeze(scene)
        again = SceneSnapshot.from_json(snapshot.to_json())
        assert again.data == snapshot.data
