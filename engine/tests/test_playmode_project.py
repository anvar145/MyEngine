"""Проект ``projects/PlayModeTest`` в плеере: кадр, состояние, скрипты (headless)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects" / "PlayModeTest"
SCENE = "Assets/Scenes/main.scene.json"

pytestmark = pytest.mark.skipif(
    not (PROJECT / "Assets" / "Scenes" / "main.scene.json").is_file(),
    reason="проект PlayModeTest не собран (scripts/seed_playmode_test.py)",
)


def _run_player(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "engine.player",
        "--project",
        str(PROJECT),
        "--scene",
        SCENE,
        "--headless",
        *extra,
    ]
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    return subprocess.run(
        command,
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=180,
    )


def _distinct_colors(path: Path, top: bool) -> int:
    """Число различных цветов (квантование по 5 бит) в половине кадра."""
    from PySide6.QtGui import QImage

    image = QImage(str(path)).convertToFormat(QImage.Format.Format_RGB32)
    assert not image.isNull(), "скриншот не прочитался"
    half = image.height() // 2
    rows = range(0, half) if top else range(half, image.height())
    colors = set()
    for y in rows:
        for x in range(0, image.width(), 4):
            pixel = image.pixel(x, y)
            red, green, blue = pixel >> 16 & 255, pixel >> 8 & 255, pixel & 255
            colors.add((red >> 5, green >> 5, blue >> 5))
    return len(colors)


def test_scene_frame_has_sky_and_objects(tmp_path: Path) -> None:
    """Небо сверху (≥5 цветов), объекты и пол снизу (≥3 цветов)."""
    shot = tmp_path / "frame.png"
    done = _run_player(tmp_path, "--frames", "20", "--screenshot", str(shot))
    assert done.returncode == 0, done.stdout + done.stderr
    assert _distinct_colors(shot, top=True) >= 5
    assert _distinct_colors(shot, top=False) >= 3


def test_dump_state_shows_loaded_scripts(tmp_path: Path) -> None:
    """После 120 кадров все скрипты загружены без ошибок, объекты на местах."""
    state_path = tmp_path / "state.json"
    done = _run_player(tmp_path, "--frames", "120", "--dump-state", str(state_path))
    assert done.returncode == 0, done.stdout + done.stderr
    state = json.loads(state_path.read_text("utf-8"))
    assert state["frames"] == 120
    scripts = [s for o in state["objects"] for s in o["scripts"]]
    assert {s["name"] for s in scripts} == {
        "orbit_camera.py",
        "spinner.py",
        "collector.py",
    }
    assert all(s["loaded"] and not s["error"] for s in scripts)
    names = {o["name"] for o in state["objects"]}
    assert {"Camera", "Sun", "Floor", "Checker cube", "Metal sphere"} <= names
