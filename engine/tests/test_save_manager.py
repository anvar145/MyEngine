"""SaveManager: формат слота, round-trip, атомарность (A7a п.2)."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from engine.core.component import Component
from engine.core.math3d import Vector3
from engine.core.reflection import component_fields, field
from engine.core.save import (
    SaveContext,
    SaveCorruptedError,
    SaveManager,
)
from engine.core.scene import Scene
from engine.scripting.behaviour import Behaviour, export


@component_fields
class Health(Component):
    hp: int = field(100, save=True, range=(0, 1000))
    max_hp: int = 100  # не сохраняется: значение из сцены
    pos: tuple[float, float, float] = field((0.0, 0.0, 0.0), save=True)


class Inventory(Component):
    def __init__(self) -> None:
        super().__init__()
        self.items: list[str] = []

    def on_save(self, ctx: SaveContext) -> None:
        ctx.set("items", list(self.items))
        ctx.set("meta.count", len(self.items))

    def on_load(self, ctx: SaveContext) -> None:
        self.items = ctx.get("items", [])


class Mover(Behaviour):
    speed: float = export(5.0, save=True)
    label: str = export("a")
    home: Vector3 = export(Vector3(), save=True)


class _BehaviourHost(Component):
    """Дублёр ScriptComponent: отдаёт ``behaviour`` как настоящий."""

    def __init__(self) -> None:
        super().__init__()
        self.behaviour = Mover()


def _build() -> Scene:
    scene = Scene("level1")
    for i in range(5):
        obj = scene.create_game_object(f"o{i}")
        obj.uid = f"{i:016x}"
    objs = scene.objects
    objs[0].add_component(Health())
    objs[1].add_component(Health())
    objs[2].add_component(Inventory())
    objs[3].add_component(Inventory())
    objs[4].add_component(_BehaviourHost())
    return scene


def _mgr(tmp_path: Path, version: int = 1) -> SaveManager:
    return SaveManager(tmp_path / "saves", version=version)


def test_round_trip_save_load_save_identical(tmp_path: Path) -> None:
    scene = _build()
    o = scene.objects
    o[0].get_component(Health).hp = 42
    o[1].get_component(Health).pos = (1.0, 2.0, 3.0)
    o[2].get_component(Inventory).items = ["sword", "key"]
    host = o[4].get_component(_BehaviourHost)
    host.behaviour.speed = 9.5
    host.behaviour.home = Vector3(1, 2, 3)
    scene.state.set("quest.stage", 3)
    scene.state.set("name", "Аня")
    mgr = _mgr(tmp_path)
    mgr.save("a", scene, scene.state)

    fresh = _build()
    mgr.load("a", fresh)
    fo = fresh.objects
    assert fo[0].get_component(Health).hp == 42
    assert fo[0].get_component(Health).max_hp == 100
    assert fo[1].get_component(Health).pos == (1.0, 2.0, 3.0)
    assert fo[2].get_component(Inventory).items == ["sword", "key"]
    assert fo[3].get_component(Inventory).items == []
    b = fo[4].get_component(_BehaviourHost).behaviour
    assert b.speed == 9.5 and b.home == Vector3(1, 2, 3) and b.label == "a"
    assert fresh.state.get("quest.stage") == 3 and fresh.state.get("name") == "Аня"

    mgr.save("b", fresh, fresh.state)
    with zipfile.ZipFile(mgr.slot("a").path) as za, zipfile.ZipFile(
        mgr.slot("b").path
    ) as zb:
        for entry in ("state.json", "objects.json", "version"):
            assert za.read(entry) == zb.read(entry)


def test_save_only_marked_fields(tmp_path: Path) -> None:
    scene = _build()
    mgr = _mgr(tmp_path)
    slot = mgr.save("s", scene, scene.state)
    with zipfile.ZipFile(slot.path) as z:
        objects = json.loads(z.read("objects.json"))
    health = objects["0000000000000000"]["components"]["Health#0"]
    assert set(health["fields"]) == {"hp", "pos"}
    assert objects["0000000000000003"]["components"]["Inventory#0"]["data"] == {
        "items": [],
        "meta": {"count": 0},
    }


def test_zip_layout_and_meta(tmp_path: Path) -> None:
    scene = _build()
    mgr = _mgr(tmp_path, version=7)
    slot = mgr.save("s", scene, scene.state, playtime_sec=12.5, screenshot=b"PNG")
    with zipfile.ZipFile(slot.path) as z:
        assert z.namelist() == [
            "version", "meta.json", "state.json", "objects.json", "screenshot.png",
        ]
        assert z.read("version") == b"7"
        assert z.testzip() is None
    assert slot.meta.version == 7 and slot.meta.scene_name == "level1"
    assert slot.meta.playtime_sec == 12.5
    assert slot.meta.screenshot_path == "screenshot.png"
    assert mgr.slot("s").meta == slot.meta
    assert not list((tmp_path / "saves").glob("*.tmp"))


def test_list_slots_delete_and_missing(tmp_path: Path) -> None:
    scene = _build()
    mgr = _mgr(tmp_path)
    assert mgr.list_slots() == [] and mgr.slot("x") is None
    mgr.save("one", scene, scene.state)
    mgr.save("two", scene, scene.state)
    assert {s.name for s in mgr.list_slots()} == {"one", "two"}
    assert mgr.delete("one") and not mgr.delete("one")
    assert [s.name for s in mgr.list_slots()] == ["two"]
    with pytest.raises(FileNotFoundError):
        mgr.load("nope", scene)


@pytest.mark.parametrize("name", ["", "../x", "a/b", "a.b", "a\b"])
def test_bad_slot_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        _mgr(tmp_path).slot(name)


def test_missing_uid_warns_once(tmp_path: Path, caplog) -> None:
    scene = _build()
    scene.objects[0].get_component(Health).hp = 1
    scene.objects[1].get_component(Health).hp = 2
    mgr = _mgr(tmp_path)
    mgr.save("s", scene, scene.state)
    other = Scene("level1")
    other.create_game_object("only").uid = "ffffffffffffffff"
    with caplog.at_level("WARNING", logger="engine.core.save"):
        mgr.load("s", other)
    warnings = [r for r in caplog.records if "нет в сцене" in r.getMessage()]
    assert len(warnings) == 1


def test_atomic_write_keeps_old_slot(tmp_path: Path, monkeypatch) -> None:
    scene = _build()
    scene.objects[0].get_component(Health).hp = 11
    mgr = _mgr(tmp_path)
    mgr.save("s", scene, scene.state)
    before = mgr.slot("s").path.read_bytes()

    scene.objects[0].get_component(Health).hp = 99

    def boom(*_a, **_k):
        raise OSError("диск отвалился")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        mgr.save("s", scene, scene.state)
    monkeypatch.undo()
    assert mgr.slot("s").path.read_bytes() == before
    assert not list((tmp_path / "saves").glob("*.tmp"))


def test_failure_while_writing_leaves_no_tmp(tmp_path: Path, monkeypatch) -> None:
    scene = _build()
    mgr = _mgr(tmp_path)

    def boom(*_a, **_k):
        raise OSError("fsync")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        mgr.save("s", scene, scene.state)
    assert mgr.slot("s") is None
    assert not list((tmp_path / "saves").glob("*"))


def test_corrupted_file_detected(tmp_path: Path) -> None:
    scene = _build()
    mgr = _mgr(tmp_path)
    slot = mgr.save("s", scene, scene.state)
    data = bytearray(slot.path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    slot.path.write_bytes(bytes(data))
    with pytest.raises(SaveCorruptedError):
        mgr.load("s", scene)
    (tmp_path / "saves" / "junk.save").write_bytes(b"not a zip")
    assert "junk" not in {s.name for s in mgr.list_slots()}
