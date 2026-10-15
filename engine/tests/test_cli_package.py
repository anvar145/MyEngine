"""A13a: сборка пака проектом (``myengine package``), валидация, игра из пака."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from engine.cli import main as cli_main
from engine.cli.package import PackageError, build_pak, collect_files
from engine.core.resources.pak import PakReader
from engine.core.scene import Scene
from engine.player.pak_locator import PakNotFoundError, find_pak
from engine.scene_io import read_scene_data, save_scene
from engine.scripting import ScriptComponent, ScriptRuntime

ROOT = Path(__file__).resolve().parents[2]
OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
SCRIPT = "def on_start(self):\n    self.log('HELLO_FROM_PAK')\n"
SCRIPT_KEY = "Assets/Scripts/hello.py"


def _make_project(root: Path) -> Path:
    (root / "Assets" / "Scripts").mkdir(parents=True)
    (root / "Assets" / "Scenes").mkdir()
    (root / "Settings").mkdir()
    (root / "Assets" / "tri.obj").write_text(OBJ, encoding="utf-8")
    script = root / SCRIPT_KEY
    script.write_text(SCRIPT, encoding="utf-8")
    (root / "Settings" / "audio.json").write_text("{}", encoding="utf-8")
    (root / "Demo.project.json").write_text('{"name": "Demo"}', encoding="utf-8")
    scene = Scene("main")
    obj = scene.create_game_object("O")
    obj.add_component(ScriptComponent(script, runtime=ScriptRuntime(project_root=root))
    )
    target = save_scene(scene, root / "Assets" / "Scenes" / "main.scene.json")
    data = json.loads(target.read_text(encoding="utf-8"))
    _relativize(data)
    target.write_text(json.dumps(data), encoding="utf-8")
    return root


def _relativize(node: object) -> None:
    if isinstance(node, dict):
        if str(node.get("path", "")).endswith(".py"):
            node["path"] = SCRIPT_KEY
        for value in node.values():
            _relativize(value)
    elif isinstance(node, list):
        for value in node:
            _relativize(value)


def test_collect_files(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "p")
    keys = set(collect_files(project, project / "Assets"))
    assert keys == {
        "Assets/Scenes/main.scene.json",
        SCRIPT_KEY,
        "Settings/audio.json",
        "Demo.project.json",
    }


def test_build_pak_contains_files_and_build_info(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "p")
    result = build_pak(project, tmp_path / "out", quality="low")
    assert result.report.errors == []
    with PakReader(result.pak) as pak:
        assert pak.read_file(SCRIPT_KEY) == (project / SCRIPT_KEY).read_bytes()
        build = json.loads(pak.read_file("build.json"))
        assert build["quality"] == "low"
        assert len(pak) >= 1


def test_validation_reports_broken_ref_and_missing_script(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "p")
    scene = project / "Assets" / "Scenes" / "main.scene.json"
    data = json.loads(scene.read_text(encoding="utf-8"))
    data["objects"][0]["mesh_ref"] = {"guid": "ab" * 16, "path": "x.obj"}
    scene.write_text(json.dumps(data), encoding="utf-8")
    (project / SCRIPT_KEY).unlink()
    with pytest.raises(PackageError) as info:
        build_pak(project, tmp_path / "out")
    text = str(info.value)
    assert "битая ссылка" in text
    assert "нет скрипта" in text


def test_cli_returns_error_code(tmp_path: Path, capsys) -> None:
    code = cli_main(["package", "--project", str(tmp_path), "--pak-only"])
    assert code == 1
    assert "Assets" in capsys.readouterr().err


def test_find_pak_modes(tmp_path: Path) -> None:
    pak = tmp_path / "x.pak"
    pak.write_bytes(b"x")
    assert find_pak(pak, tmp_path, frozen=True) == pak
    assert find_pak(None, tmp_path, frozen=False) is None
    with pytest.raises(PakNotFoundError):
        find_pak(None, tmp_path / "nope", frozen=True)


def test_runtime_source_resolver_and_scene_reader(tmp_path: Path) -> None:
    runtime = ScriptRuntime(project_root=tmp_path)
    runtime.set_source_resolver(
        lambda key: SCRIPT.encode() if key == SCRIPT_KEY else None
    )
    assert runtime.read_virtual(SCRIPT_KEY) == SCRIPT
    assert runtime.read_virtual("Assets/Scripts/none.py") is None
    data = read_scene_data(tmp_path / "s.scene.json", lambda _p: '{"a": 1}')
    assert data == {"a": 1}


def test_game_runs_from_pak_without_assets(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "p")
    result = build_pak(project, tmp_path / "out")
    empty = tmp_path / "game"
    empty.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "engine.player",
            "--project",
            str(empty),
            "--pak",
            str(result.pak),
            "--headless",
            "--frames",
            "5",
            "--saves-dir",
            str(tmp_path / "saves"),
        ],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=180,
    )
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    assert "HELLO_FROM_PAK" in output
    assert not (empty / "Assets").exists()
