"""A13b: ``build.json`` пака, приоритет CLI и кэш шейдеров в каталогах пользователя."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from engine.cli.package import build_pak
from engine.player.app import PlayerApp, parse_args
from engine.tests.test_cli_package import _make_project


def _app(tmp_path: Path, pak: Path, *extra: str) -> PlayerApp:
    empty = tmp_path / "game"
    empty.mkdir(exist_ok=True)
    args = parse_args(
        ["--project", str(empty), "--pak", str(pak), "--headless", *extra]
    )
    return PlayerApp(args)


@pytest.fixture
def pak_low(tmp_path: Path) -> Path:
    project = _make_project(tmp_path / "p")
    return build_pak(project, tmp_path / "out", quality="low").pak


def test_quality_from_build_json(tmp_path: Path, pak_low: Path) -> None:
    app = _app(tmp_path, pak_low)
    assert app.args.quality == "low"
    assert app.build_info["quality"] == "low"
    app.vfs.close()


def test_cli_quality_overrides_build_json(tmp_path: Path, pak_low: Path) -> None:
    app = _app(tmp_path, pak_low, "--quality", "high")
    assert app.args.quality == "high"
    app.vfs.close()


def test_default_quality_without_pak_info(tmp_path: Path) -> None:
    args = parse_args(["--project", str(tmp_path), "--headless"])
    assert PlayerApp(args).args.quality == "medium"


def test_shader_cache_in_user_cache(tmp_path: Path, pak_low: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "local"))
    monkeypatch.setenv("HOME", str(tmp_path / "local"))
    app = _app(tmp_path, pak_low, "--user-dirs")
    cache = app._shader_cache_dir()
    assert cache is not None
    assert cache.name == "shader_cache"
    assert cache.is_dir()
    assert str(tmp_path / "local") in str(cache)
    app.vfs.close()


def test_shader_cache_dev_mode_stays_in_library(tmp_path: Path, pak_low: Path) -> None:
    app = _app(tmp_path, pak_low)
    assert app._shader_cache_dir() == app.library_dir / "ShaderCache"
    app.vfs.close()


def test_unwritable_cache_disables_cache(
    tmp_path: Path, pak_low: Path, monkeypatch, caplog
) -> None:
    app = _app(tmp_path, pak_low, "--user-dirs")
    monkeypatch.setattr("engine.player.app.is_writable_dir", lambda _p: False)
    with caplog.at_level(logging.WARNING):
        assert app._shader_cache_dir() is None
    warnings = [r for r in caplog.records if "Кэш шейдеров отключён" in r.message]
    assert len(warnings) == 1
    app.vfs.close()


def test_pak_extract_goes_to_user_cache(
    tmp_path: Path, pak_low: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "local"))
    app = _app(tmp_path, pak_low, "--user-dirs")
    reader = app.vfs.reader
    guid = next(iter(reader._assets))
    reader.resolve(guid)
    assert (tmp_path / "local") in reader.cache_dir.parents
    assert reader.cache_dir.parent.name == "pak_extract"
    extracted = reader.cache_dir
    app.vfs.close()
    assert not extracted.exists()


@pytest.mark.parametrize("spec", ["main", "main.scene.json", "Scenes/main.scene.json"])
def test_scene_spec_resolves_in_pak(tmp_path: Path, pak_low: Path, spec: str) -> None:
    """``--scene main`` и ``--scene main.scene.json`` находят одну сцену в паке."""
    app = _app(tmp_path, pak_low)
    assert app._resolve_scene(spec).name == "main.scene.json"
    app.vfs.close()
