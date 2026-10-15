"""Каталоги пользователя ОС для записи данных игры (A13b).

Собранная игра лежит в каталоге, куда писать нельзя (``Program Files``),
поэтому сейвы, логи и кэши идут в каталоги пользователя.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

__all__ = ["is_writable_dir", "user_cache_root", "user_data_root"]


def user_data_root(game_name: str) -> Path:
    """Данные игры: ``%APPDATA%/<game>``, ``~/.local/share/<game>``, ``~/Library/Application Support/<game>``."""  # noqa: E501
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / game_name


def user_cache_root(game_name: str) -> Path:
    """Кэш игры: ``%LOCALAPPDATA%/<game>``, ``~/.cache/<game>``, ``~/Library/Caches/<game>``."""  # noqa: E501
    if sys.platform == "win32":
        base = Path(
            os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        )
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / game_name


def is_writable_dir(path: Path) -> bool:
    """Можно ли создать каталог и писать в него (проверка временным файлом)."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            pass
    except OSError:
        return False
    return True
