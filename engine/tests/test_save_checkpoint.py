"""Каталог сейвов, чекпоинты, автосейв, быстрые слоты (A7b)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.core.save import (
    AUTOSAVE_SLOT,
    QUICK_SLOT,
    AutoSavePolicy,
    CheckpointComponent,
    CheckpointReached,
    SafeZone,
    SaveManager,
    SaveStatus,
    autosave_before_switch,
    default_save_root,
)
from engine.core.scene import Scene
from engine.core.scene_manager import SceneManager
from engine.physics.events import TriggerEnter, TriggerExit


def _enter(scene: Scene, trigger, other, cls=TriggerEnter) -> None:
    scene.events.emit(cls(1, 2, trigger, other))


def _world(tmp_path: Path):
    scene = Scene("level")
    scene.save_manager = SaveManager(tmp_path / "saves", version=1)
    zone = scene.create_game_object("Zone")
    player = scene.create_game_object("Hero")
    player.tag = "player"
    return scene, zone, player


# ---------------------------------------------------------------- каталог
def _project(tmp_path: Path, name: str = "Game") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "p.project.json").write_text(json.dumps({"name": name}), "utf-8")
    return tmp_path


def test_editor_root_is_project_saves(tmp_path: Path) -> None:
    assert default_save_root(_project(tmp_path), player=False) == tmp_path / "Saves"


def test_player_root_uses_appdata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "ad"))
    root = default_save_root(_project(tmp_path / "proj"), player=True)
    assert root == tmp_path / "ad" / "Game" / "saves"


def test_player_root_without_appdata_falls_back(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.delenv("APPDATA", raising=False)
    project = _project(tmp_path)
    assert default_save_root(project, player=True) == project / "Saves"


def test_player_root_linux_and_macos(tmp_path: Path, monkeypatch) -> None:
    project = _project(tmp_path / "proj", "G")
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    assert default_save_root(project, player=True) == (
        tmp_path / "home" / ".local" / "share" / "G" / "saves"
    )
    monkeypatch.setattr("sys.platform", "darwin")
    assert default_save_root(project, player=True) == (
        tmp_path / "home" / "Library" / "Application Support" / "G" / "saves"
    )


# -------------------------------------------------------------- чекпоинты
def test_checkpoint_saves_once_for_player(tmp_path: Path) -> None:
    scene, zone, player = _world(tmp_path)
    zone.add_component(CheckpointComponent("cp1"))
    reached: list[str] = []
    scene.events.subscribe(CheckpointReached, lambda e: reached.append(e.slot_name))
    _enter(scene, zone, player)
    _enter(scene, zone, player)  # повторный вход без изменений
    assert reached == ["cp1"]
    assert scene.save_manager.slot("cp1") is not None
    scene.state.set("gold", 5)
    _enter(scene, zone, player)  # состояние изменилось
    assert reached == ["cp1", "cp1"]


def test_checkpoint_not_idempotent_always_saves(tmp_path: Path) -> None:
    scene, zone, player = _world(tmp_path)
    zone.add_component(CheckpointComponent("cp", idempotent=False))
    reached: list[str] = []
    scene.events.subscribe(CheckpointReached, lambda e: reached.append(e.slot_name))
    _enter(scene, zone, player)
    _enter(scene, zone, player)
    assert len(reached) == 2


def test_checkpoint_ignores_non_player_and_other_triggers(tmp_path: Path) -> None:
    scene, zone, player = _world(tmp_path)
    zone.add_component(CheckpointComponent("cp"))
    other = scene.create_game_object("Rock")
    _enter(scene, zone, other)
    _enter(scene, other, player)  # триггер - не наш объект
    assert scene.save_manager.slot("cp") is None


def test_checkpoint_without_manager_is_noop() -> None:
    scene = Scene("x")
    zone = scene.create_game_object("Z")
    player = scene.create_game_object("P")
    player.tag = "Player"
    zone.add_component(CheckpointComponent("cp"))
    _enter(scene, zone, player)  # без исключений


def test_checkpoint_detach_unsubscribes(tmp_path: Path) -> None:
    scene, zone, player = _world(tmp_path)
    comp = zone.add_component(CheckpointComponent("cp"))
    zone.remove_component(comp)
    _enter(scene, zone, player)
    assert scene.save_manager.slot("cp") is None


# ---------------------------------------------------------------- автосейв
def test_timer_requires_safe_zone(tmp_path: Path) -> None:
    scene, zone, player = _world(tmp_path)
    zone.add_component(SafeZone())
    policy = zone.add_component(AutoSavePolicy(interval_sec=1.0))
    policy.on_update(1.0)
    assert scene.save_manager.slot(AUTOSAVE_SLOT) is None  # игрока в зоне нет
    _enter(scene, zone, player)
    policy.on_update(1.0)
    assert scene.save_manager.slot(AUTOSAVE_SLOT) is not None
    scene.save_manager.delete(AUTOSAVE_SLOT)
    scene.save_manager.hashes.clear()
    _enter(scene, zone, player, TriggerExit)
    policy.on_update(1.0)
    assert scene.save_manager.slot(AUTOSAVE_SLOT) is None


def test_timer_without_zone_requirement(tmp_path: Path) -> None:
    scene, zone, _ = _world(tmp_path)
    policy = zone.add_component(
        AutoSavePolicy(interval_sec=2.0, require_safe_zone=False)
    )
    policy.on_update(1.0)
    assert scene.save_manager.slot(AUTOSAVE_SLOT) is None
    policy.on_update(1.0)
    assert scene.save_manager.slot(AUTOSAVE_SLOT) is not None


def test_no_autosave_flag(tmp_path: Path) -> None:
    scene, zone, _ = _world(tmp_path)
    scene.save_manager.autosave_enabled = False
    policy = zone.add_component(
        AutoSavePolicy(interval_sec=1.0, require_safe_zone=False)
    )
    policy.on_update(1.0)
    assert scene.save_manager.slot(AUTOSAVE_SLOT) is None


def test_scene_change_autosaves_old_scene(tmp_path: Path) -> None:
    old, zone, _ = _world(tmp_path)
    zone.add_component(AutoSavePolicy())
    new = Scene("next")
    new.save_manager = old.save_manager
    assert autosave_before_switch(old, new)
    assert old.save_manager.slot(AUTOSAVE_SLOT).meta.scene_name == "level"
    assert not autosave_before_switch(None, new)
    assert not autosave_before_switch(new, new)


def test_scene_change_skipped_when_busy_or_off(tmp_path: Path) -> None:
    old, zone, _ = _world(tmp_path)
    zone.add_component(AutoSavePolicy(on_scene_change=False))
    assert not autosave_before_switch(old, Scene("n"))
    zone.get_component(AutoSavePolicy).on_scene_change = True
    old.save_manager.busy = True
    assert not autosave_before_switch(old, Scene("n"))


def test_manager_hook_runs_before_activate(tmp_path: Path) -> None:
    first, zone, _ = _world(tmp_path)
    zone.add_component(AutoSavePolicy())
    second = Scene("second")
    scenes = {Path("a"): first, Path("b"): second}
    manager = SceneManager(lambda path: scenes[path])
    manager.before_activate.append(autosave_before_switch)
    manager.load(Path("a"))
    assert first.save_manager.slot(AUTOSAVE_SLOT) is None
    manager.load(Path("b"))
    assert first.save_manager.slot(AUTOSAVE_SLOT) is not None


# ---------------------------------------------------------------- F5 / F9
def test_quick_save_and_load(tmp_path: Path) -> None:
    scene, _, _ = _world(tmp_path)
    status: list[str] = []
    scene.events.subscribe(SaveStatus, lambda e: status.append(e.text))
    scene.state.set("k", 1)
    assert scene.quick_save() is not None
    scene.state.set("k", 2)
    assert scene.quick_load() is not None
    assert scene.state.get("k") == 1
    assert status == ["Сохранено", "Загружено"]


def test_quick_load_falls_back_to_autosave(tmp_path: Path) -> None:
    scene, _, _ = _world(tmp_path)
    status: list[str] = []
    scene.events.subscribe(SaveStatus, lambda e: status.append(e.text))
    assert scene.quick_load() is None and status == ["Нет сохранения"]
    scene.state.set("k", 7)
    scene.save(AUTOSAVE_SLOT)
    scene.state.set("k", 0)
    assert scene.quick_load().name == AUTOSAVE_SLOT
    assert scene.state.get("k") == 7
    assert scene.save_manager.slot(QUICK_SLOT) is None


@pytest.mark.parametrize("cls", [CheckpointComponent, SafeZone, AutoSavePolicy])
def test_components_round_trip(cls) -> None:
    comp = cls()
    assert cls.from_dict(comp.to_dict()).to_dict() == comp.to_dict()
