"""Каталог сохранений по умолчанию (A7b п.1)."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

__all__ = ["default_save_root", "game_name", "project_save_version"]


def _project_json_texts(root: Path, vfs: Any = None) -> Iterator[str]:
    """Тексты ``*.project.json``: сначала с диска, затем из пака (``vfs``)."""
    for file in sorted(root.glob("*.project.json")):
        try:
            yield file.read_text(encoding="utf-8")
        except OSError:
            continue
    if vfs is None:
        return
    for key in vfs.list_files(""):
        if "/" in key or not key.endswith(".project.json"):
            continue
        data = vfs.read_file(key)
        if data is not None:
            yield data.decode("utf-8", errors="replace")


def _project_json_field(root: Path, vfs: Any, field: str) -> Iterator[Any]:
    for text in _project_json_texts(root, vfs):
        try:
            yield json.loads(text).get(field)
        except (ValueError, AttributeError):
            continue


def game_name(project_path: str | Path, vfs: Any = None) -> str:
    """Имя игры из ``*.project.json`` (поле ``name``), иначе имя каталога.

    Args:
        project_path: Корень проекта.
        vfs: Источник файлов проекта (пак), если проекта нет на диске.
    """
    root = Path(project_path)
    for name in _project_json_field(root, vfs, "name"):
        if isinstance(name, str) and name.strip():
            return name.strip()
    return root.resolve().name


def project_save_version(project_path: str | Path, vfs: Any = None) -> int:
    """Версия схемы сейвов из ``*.project.json`` (``save_version``), по умолчанию 1."""
    for value in _project_json_field(Path(project_path), vfs, "save_version"):
        if isinstance(value, int) and value >= 1:
            return value
    return 1


def _user_data_dir() -> Path | None:
    """Каталог данных пользователя ОС или ``None``, если его не определить."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        return Path(appdata) if appdata else None
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    base = os.environ.get("XDG_DATA_HOME")
    return Path(base) if base else Path.home() / ".local" / "share"


def default_save_root(
    project_path: str | Path, *, player: bool, vfs: Any = None
) -> Path:
    """Каталог сохранений по умолчанию.

    Плеер: каталог данных пользователя ``<game_name>/saves`` (Windows -
    ``%APPDATA%``, Linux - ``~/.local/share``, macOS -
    ``~/Library/Application Support``); без него (нет ``APPDATA``) -
    ``<project>/Saves``. Редактор: ``<project>/Saves``.

    Args:
        project_path: Корень проекта.
        player: ``True`` для плеера, ``False`` для редактора.
        vfs: Источник файлов проекта (пак) для имени игры.
    """
    root = Path(project_path)
    if player:
        base = _user_data_dir()
        if base is not None:
            return base / game_name(root, vfs) / "saves"
    return root / "Saves"
