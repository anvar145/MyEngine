"""Сцена с секцией ``post`` включает HDR в плеере без поля ``hdr`` в проекте (B8)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects" / "PlayModeTest"
MESSAGE = "HDR включён автоматически"


def _run(scene: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "engine.player", "--project", str(PROJECT),
         "--scene", scene, "--headless", "--no-audio", "--frames", "2"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120, cwd=ROOT, env={
            **{k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"},
            "PYTHONIOENCODING": "utf-8",
        },
    )
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr


@pytest.mark.skipif(not (PROJECT / "Assets/Scenes/post_demo.scene.json").is_file(),
                    reason="нет PlayModeTest")
def test_scene_with_post_enables_hdr_and_scene_without_does_not() -> None:
    project = json.loads(next(PROJECT.glob("*.project.json")).read_text("utf-8"))
    assert "hdr" not in project
    assert MESSAGE in _run("post_demo.scene.json")
    assert MESSAGE not in _run("main.scene.json")


def _project_copy(tmp_path: Path, hdr: bool | None) -> Path:
    """Копия PlayModeTest без Library с заданным ``hdr`` в проекте."""
    import shutil

    target = tmp_path / "proj"
    shutil.copytree(PROJECT / "Assets", target / "Assets")
    data = json.loads(next(PROJECT.glob("*.project.json")).read_text("utf-8"))
    data.pop("hdr", None)
    if hdr is not None:
        data["hdr"] = hdr
    (target / "g.project.json").write_text(json.dumps(data), encoding="utf-8")
    return target


@pytest.mark.skipif(not (PROJECT / "Assets/Scenes/post_demo.scene.json").is_file(),
                    reason="нет PlayModeTest")
@pytest.mark.parametrize("hdr", [False, True])
def test_explicit_hdr_is_not_overridden_by_scene(tmp_path: Path, hdr: bool) -> None:
    project = _project_copy(tmp_path, hdr)
    result = subprocess.run(
        [sys.executable, "-m", "engine.player", "--project", str(project),
         "--scene", "post_demo.scene.json", "--headless", "--no-audio",
         "--frames", "2"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=180, cwd=ROOT, env={
            **{k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"},
            "PYTHONIOENCODING": "utf-8",
        },
    )
    out = result.stdout + result.stderr
    assert MESSAGE not in out
    assert ("hdr: false" in out) is (not hdr)
