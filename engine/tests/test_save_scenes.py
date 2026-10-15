"""Слот сохранения с несколькими сценами: ``meta.scenes`` (A8b п.5)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from engine.core.component import Component
from engine.core.reflection import component_fields, field
from engine.core.save import SaveManager, SaveScene
from engine.core.save.slot import SaveSlotMeta
from engine.core.scene import Scene
from engine.core.scene_manager import SceneManager


@component_fields
class Counter(Component):
    value: int = field(0, save=True, range=(0, 1000))


def _make(name: str, uid: str) -> Scene:
    scene = Scene(name)
    obj = scene.create_game_object("obj")
    obj.uid = uid
    obj.add_component(Counter())
    return scene


def _loader(path: Path) -> Scene:
    # имя файла без суффиксов -> имя сцены; uid разные для каждой сцены
    stem = path.name.removesuffix(".scene.json")
    return _make(stem, f"{abs(hash(stem)) % 10**15:016x}")


def _project(tmp_path: Path) -> tuple[Path, list[Path]]:
    root = tmp_path / "proj"
    scenes_dir = root / "Assets" / "Scenes"
    scenes_dir.mkdir(parents=True)
    files = []
    for name in ("main", "hud", "music"):
        file = scenes_dir / f"{name}.scene.json"
        file.write_text("{}", encoding="utf-8")
        files.append(file)
    return root, files


def _counter(scene: Scene) -> Counter:
    return scene.objects[0].get_component(Counter)


def _manager(root: Path) -> SceneManager:
    return SceneManager(_loader, root=root)


def test_round_trip_two_scenes(tmp_path: Path) -> None:
    root, (main, hud, _music) = _project(tmp_path)
    saves = SaveManager(tmp_path / "saves", version=1)
    manager = _manager(root)
    manager.load(main)
    manager.load(hud, additive=True)
    for scene in manager.all_scenes:
        scene.save_manager = saves
    _counter(manager.get(main)).value = 7
    _counter(manager.get(hud)).value = 42

    slot = manager.get(main).save("s")

    assert slot.meta.scenes == (
        SaveScene("Assets/Scenes/main.scene.json", False),
        SaveScene("Assets/Scenes/hud.scene.json", True),
    )
    assert slot.meta.scene_name == slot.meta.scenes[0].path
    with zipfile.ZipFile(slot.path) as archive:
        raw = json.loads(archive.read("meta.json"))
    assert raw["scenes"][1] == {"path": "Assets/Scenes/hud.scene.json", "additive": True}

    # Новая сессия: чистый менеджер, сцены пересобираются из шаблона.
    fresh = _manager(root)
    restored = fresh.restore(saves, "s")
    assert restored.meta.scenes == slot.meta.scenes
    assert fresh.active_path == main
    assert fresh.loaded == [main, hud]
    assert list(fresh.additive_scenes) == [hud]
    assert _counter(fresh.get(main)).value == 7
    assert _counter(fresh.get(hud)).value == 42


def test_restore_replaces_scene_set_and_keeps_order(tmp_path: Path) -> None:
    root, (main, hud, music) = _project(tmp_path)
    saves = SaveManager(tmp_path / "saves", version=1)
    manager = _manager(root)
    manager.load(main)
    manager.load(music, additive=True)
    manager.load(hud, additive=True)
    manager.get(main).save_manager = saves
    manager.get(main).save("s")

    other = _manager(root)
    other.load(hud)  # другая активная
    other.load(music, additive=True)
    other.restore(saves, "s")
    assert other.active_path == main
    assert other.loaded == [main, music, hud]
    assert list(other.additive_scenes) == [music, hud]


def test_restore_noop_when_set_matches(tmp_path: Path) -> None:
    root, (main, hud, _music) = _project(tmp_path)
    saves = SaveManager(tmp_path / "saves", version=1)
    manager = _manager(root)
    manager.load(main)
    manager.load(hud, additive=True)
    manager.get(main).save_manager = saves
    manager.get(main).save("s")
    before = {p: manager.get(p) for p in manager.loaded}
    manager.restore(saves, "s")
    assert {p: manager.get(p) for p in manager.loaded} == before


def test_scene_without_manager_keeps_old_meta(tmp_path: Path) -> None:
    scene = _make("level1", "0" * 16)
    saves = SaveManager(tmp_path / "saves", version=1)
    slot = saves.save("s", scene, scene.state)
    assert slot.meta.scene_name == "level1"
    assert slot.meta.scenes == (SaveScene("level1", False),)


def test_a7a_save_without_scenes_is_read_as_single_scene(tmp_path: Path) -> None:
    saves = SaveManager(tmp_path / "saves", version=1)
    scene = _make("level1", "0" * 16)
    slot = saves.save("s", scene, scene.state)
    # Переписываем meta.json в формат A7a: без поля scenes.
    with zipfile.ZipFile(slot.path) as archive:
        entries = {n: archive.read(n) for n in archive.namelist()}
    meta = json.loads(entries["meta.json"])
    del meta["scenes"]
    entries["meta.json"] = json.dumps(meta).encode("utf-8")
    with zipfile.ZipFile(slot.path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)

    old = saves.slot("s")
    assert old is not None
    assert old.meta.scenes == (SaveScene("level1", False),)
    read_slot, _state, _objects = saves.read("s")
    assert read_slot.meta.scenes == (SaveScene("level1", False),)
    saves.load("s", scene)  # без исключений


def test_meta_rejects_bad_scenes() -> None:
    from engine.core.save import SaveCorruptedError

    raw = {"timestamp": 1, "version": 1, "scene_name": "a", "scenes": [{"x": 1}]}
    with pytest.raises(SaveCorruptedError):
        SaveSlotMeta.from_dict(raw)


def test_missing_scene_file_raises(tmp_path: Path) -> None:
    root, (main, hud, _music) = _project(tmp_path)
    saves = SaveManager(tmp_path / "saves", version=1)
    manager = _manager(root)
    manager.load(main)
    manager.load(hud, additive=True)
    manager.get(main).save_manager = saves
    manager.get(main).save("s")
    hud.unlink()
    with pytest.raises(FileNotFoundError):
        _manager(root).restore(saves, "s")
