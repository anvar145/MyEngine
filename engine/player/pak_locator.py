"""Поиск ``data.pak`` собранной игры (A13a п.5)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["PAK_NAME", "PakNotFoundError", "find_pak", "is_frozen"]

_log = logging.getLogger("engine.player")

#: Имя пака рядом с исполняемым файлом собранной игры.
PAK_NAME = "data.pak"


class PakNotFoundError(FileNotFoundError):
    """В собранной игре нет ``data.pak``."""


def is_frozen() -> bool:
    """Запущена ли программа как собранный exe (PyInstaller)."""
    return bool(getattr(sys, "frozen", False))


def _candidates(project: Path, explicit: str | Path | None) -> list[Path]:
    found: list[Path] = []
    if explicit:
        found.append(Path(explicit))
        return found
    exe_dir = Path(sys.executable).resolve().parent
    found.append(exe_dir / PAK_NAME)
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        found.append(Path(bundle) / PAK_NAME)
    found.append(exe_dir / project.name / PAK_NAME)
    return found


def find_pak(
    explicit: str | Path | None, project: Path, *, frozen: bool | None = None
) -> Path | None:
    """Найти пак собранной игры.

    Порядок: ``--pak``; ``<exe_dir>/data.pak`` (onedir); ``sys._MEIPASS/data.pak``
    (onefile); ``<exe_dir>/<имя проекта>/data.pak``. Вне собранной игры
    (``frozen`` ложно) ищется только явный путь.

    Args:
        explicit: Значение ``--pak``.
        project: Корень проекта.
        frozen: Принудительно задать режим «собранная игра» (для тестов).

    Returns:
        Путь к паку или ``None`` вне собранной игры без ``--pak``.

    Raises:
        PakNotFoundError: Собранная игра, а пака нет ни в одном месте.
    """
    if frozen is None:
        frozen = is_frozen()
    if explicit:
        return Path(explicit)
    if not frozen:
        return None
    options = _candidates(project, None)
    for option in options:
        if option.is_file():
            return option
    text = "Нет data.pak, игра не собрана. Искали: " + ", ".join(
        str(o) for o in options
    )
    _log.error(text)
    raise PakNotFoundError(text)
