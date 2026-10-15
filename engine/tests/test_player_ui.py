"""UI в плеере (headless): клик переключает сцену, кнопка выхода останавливает игру."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from engine.core.scene import Scene
from engine.core.ui_components import UIButton
from engine.scene_io import save_scene
from engine.scripting.script_component import ScriptComponent

ROOT = Path(__file__).resolve().parents[2]

QUIT_SCRIPT = """\
from engine.core.ui_events import QuitRequested, UIEventFired


def on_start(self):
    def handler(event):
        if event.event_name == "quit":
            self.scene.events.emit(QuitRequested())

    self.scene.events.subscribe(UIEventFired, handler)
"""


def _button(scene: Scene, on_click: str) -> None:
    scene.create_game_object("Play").add_component(
        UIButton,
        text="Play",
        position=(100.0, 100.0),
        size=(200.0, 60.0),
        on_click=on_click,
    )


def _project(tmp_path: Path, on_click: str, with_quit_script: bool = False) -> Path:
    scenes = tmp_path / "Assets" / "Scenes"
    scenes.mkdir(parents=True)
    menu = Scene("Menu")
    menu.create_game_object("MenuMarker")
    _button(menu, on_click)
    if with_quit_script:
        scripts = tmp_path / "Assets" / "Scripts"
        scripts.mkdir(parents=True)
        script = scripts / "quit.py"
        script.write_text(QUIT_SCRIPT, encoding="utf-8")
        menu.create_game_object("Host").add_component(
            ScriptComponent, path=str(script)
        )
    save_scene(menu, scenes / "menu.scene.json")
    game = Scene("Game")
    game.create_game_object("GameMarker")
    save_scene(game, scenes / "game.scene.json")
    return tmp_path


def _run(project: Path, *extra: str) -> tuple[subprocess.CompletedProcess, dict]:
    state = project / "state.json"
    command = [
        sys.executable,
        "-m",
        "engine.player",
        "--project",
        str(project),
        "--scene",
        "menu.scene.json",
        "--headless",
        "--dump-state",
        str(state),
        *extra,
    ]
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    done = subprocess.run(
        command,
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=180,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done, json.loads(state.read_text("utf-8"))


def _names(state: dict) -> set[str]:
    return {entry["name"] for entry in state["objects"]}


def test_click_switches_scene(tmp_path: Path) -> None:
    project = _project(tmp_path, "load_scene:game.scene.json")
    _, state = _run(project, "--frames", "10", "--click-at", "150,120")
    names = _names(state)
    assert "GameMarker" in names and "MenuMarker" not in names


def test_click_outside_keeps_scene(tmp_path: Path) -> None:
    project = _project(tmp_path, "load_scene:game.scene.json")
    _, state = _run(project, "--frames", "10", "--click-at", "900,600")
    assert "MenuMarker" in _names(state)


def test_quit_button_stops_player(tmp_path: Path) -> None:
    project = _project(tmp_path, "event:quit", with_quit_script=True)
    _, state = _run(project, "--frames", "200", "--click-at", "150,120")
    assert state["frames"] < 200
