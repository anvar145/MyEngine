"""Сохранения в плеере (headless): F5/F9, чекпоинты, автосейв, флаги (A7b)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from engine.core.save import AutoSavePolicy, CheckpointComponent
from engine.core.scene import Scene
from engine.scene_io import save_scene
from engine.scripting.script_component import ScriptComponent

ROOT = Path(__file__).resolve().parents[2]

HERO = '''\
from engine.scripting import Behaviour, export


class Hero(Behaviour):
    hp: int = export(10, save=True)
    frame = 0

    def on_update(self, delta_time):
        self.frame += 1
        if self.frame == 5:
            self.hp = 5
        elif self.frame == 20:
            self.hp = 99
        elif self.frame in (25, 45):
            self.log(f"HP={self.hp}")
'''


def _project(tmp_path: Path, *policies: object, hero: bool = True) -> Path:
    scenes = tmp_path / "Assets" / "Scenes"
    scenes.mkdir(parents=True)
    (tmp_path / "game.project.json").write_text(
        json.dumps({"name": "G", "scene": "Assets/Scenes/level.scene.json"}), "utf-8"
    )
    scene = Scene("Level")
    host = scene.create_game_object("Host")
    for policy in policies:
        host.add_component(policy)
    if hero:
        scripts = tmp_path / "Assets" / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "hero.py").write_text(HERO, encoding="utf-8")
        host.add_component(ScriptComponent, path=str(scripts / "hero.py"))
    save_scene(scene, scenes / "level.scene.json")
    return tmp_path


def _run(project: Path, saves: Path, *extra: str) -> str:
    command = [
        sys.executable,
        "-m",
        "engine.player",
        "--project",
        str(project),
        "--headless",
        "--saves-dir",
        str(saves),
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
    return done.stdout + done.stderr


def test_quick_save_and_load_restore_field(tmp_path: Path) -> None:
    project = _project(tmp_path / "p")
    saves = tmp_path / "saves"
    out = _run(
        project,
        saves,
        "--frames",
        "50",
        "--quick-save-at",
        "10",
        "--quick-load-at",
        "30",
    )
    assert (saves / "_quick.save").is_file()
    # до загрузки hp = 99 (кадр 25), после - снова 5 (кадр 45)
    assert out.index("HP=99") < out.index("HP=5")


def test_quick_load_without_slot_is_harmless(tmp_path: Path) -> None:
    project = _project(tmp_path / "p")
    out = _run(project, tmp_path / "saves", "--frames", "40", "--quick-load-at", "30")
    assert "HP=99" in out and "HP=5" not in out


def test_autosave_timer_and_no_autosave_flag(tmp_path: Path) -> None:
    policy = AutoSavePolicy(interval_sec=0.5, require_safe_zone=False)
    project = _project(tmp_path / "p", policy, hero=False)
    saves = tmp_path / "saves"
    _run(project, saves, "--frames", "90")
    assert (saves / "_autosave.save").is_file()
    off = tmp_path / "saves_off"
    _run(project, off, "--frames", "90", "--no-autosave")
    assert not (off / "_autosave.save").exists()


def test_checkpoint_component_is_loaded_from_scene(tmp_path: Path) -> None:
    project = _project(tmp_path / "p", CheckpointComponent("cp"), hero=False)
    # без игрока в триггере слот не создаётся, сцена с компонентом грузится
    saves = tmp_path / "saves"
    _run(project, saves, "--frames", "10")
    assert not (saves / "cp.save").exists()
