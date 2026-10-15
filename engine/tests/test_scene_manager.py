"""SceneManager: загрузка, активная сцена, события, выгрузка."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.scene import Scene
from engine.core.scene_manager import SceneManager
from engine.core.ui_events import SceneActivated, SceneLoaded, SceneUnloaded


def _manager():
    manager = SceneManager(lambda path: Scene(path.stem))
    log: list[object] = []
    for kind in (SceneLoaded, SceneActivated, SceneUnloaded):
        manager.events.subscribe(kind, log.append)
    return manager, log


def test_load_activates_and_emits() -> None:
    manager, log = _manager()
    scene = manager.load(Path("a"))
    assert manager.active_scene is scene
    assert [type(e).__name__ for e in log] == ["SceneLoaded", "SceneActivated"]
    assert log[0].name == "a"


def test_load_without_activation_keeps_active() -> None:
    manager, _ = _manager()
    first = manager.load(Path("a"))
    manager.load(Path("b"), activate=False)
    assert manager.active_scene is first
    manager.set_active(Path("b"))
    assert manager.active_scene.name == "b"


def test_unload_shuts_down_and_clears_active() -> None:
    manager, log = _manager()
    scene = manager.load(Path("a"))
    log.clear()
    manager.unload(Path("a"))
    assert manager.active_scene is None and not manager.loaded
    assert [type(e).__name__ for e in log] == ["SceneUnloaded"]
    assert len(scene) == 0


def test_reload_replaces_scene() -> None:
    manager, _ = _manager()
    first = manager.load(Path("a"))
    second = manager.load(Path("a"))
    assert second is not first and manager.active_scene is second


def test_set_active_unknown_raises() -> None:
    manager, _ = _manager()
    with pytest.raises(KeyError):
        manager.set_active(Path("x"))
