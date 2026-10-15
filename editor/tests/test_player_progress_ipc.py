"""Плеер шлёт прогресс загрузки сцены в IPC."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from engine.core.ui_events import SceneLoadProgress
from engine.player.app import PlayerApp


class _Server:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def event(self, name: str, **fields) -> None:
        self.events.append((name, fields))


def test_forward_progress_sends_ipc_events() -> None:
    app = PlayerApp.__new__(PlayerApp)
    app.scenes = SimpleNamespace(all_scenes=[])
    app.server = _Server()
    app._forward_progress(SceneLoadProgress(None, 0.451, "build", "x/a.scene.json"))
    app._forward_progress(SceneLoadProgress(None, 1.0, "activate", "x/a.scene.json"))
    assert app.server.events == [
        ("scene_load_progress", {"progress": 0.451, "phase": "build"}),
        ("scene_load_progress", {"progress": 1.0, "phase": "activate"}),
        ("scene_loaded", {"path": Path("a.scene.json").name}),
    ]
