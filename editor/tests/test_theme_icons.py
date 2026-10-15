"""Тесты темы и иконок: файлы иконок на месте, тема их не отключает."""

from __future__ import annotations

import re
from pathlib import Path

from editor.theme import ICONS_DIR, load_stylesheet

EDITOR_ROOT = Path(__file__).resolve().parents[1]

#: Вызовы ``icon("имя")`` в исходниках редактора.
_ICON_CALL = re.compile(r'icon\("([^"]+)"\)')


def used_icon_names() -> set[str]:
    """Собрать имена иконок, которые запрашивает код редактора.

    Returns:
        Множество имён без расширения.
    """
    names: set[str] = set()
    for path in EDITOR_ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        names.update(_ICON_CALL.findall(path.read_text(encoding="utf-8")))
    return names


def test_icons_directory_exists() -> None:
    assert ICONS_DIR.is_dir(), f"нет каталога иконок: {ICONS_DIR}"


def test_every_used_icon_has_a_file() -> None:
    """Каждое ``icon("имя")`` обязано иметь ``assets/icons/имя.svg``.

    Отсутствующий файл даёт пустую иконку: кнопка выглядит невидимой.
    """
    used = used_icon_names()
    assert used, "в исходниках редактора не найдено ни одного вызова icon()"
    missing = sorted(
        name for name in used if not (ICONS_DIR / f"{name}.svg").is_file()
    )
    assert not missing, f"нет SVG-файлов для иконок: {missing}"


def test_every_icon_file_is_readable_svg() -> None:
    """Файлы иконок — корректный SVG (иначе QIcon вернёт пустую иконку)."""
    for path in sorted(ICONS_DIR.glob("*.svg")):
        text = path.read_text(encoding="utf-8")
        assert "<svg" in text, f"{path.name} не похож на SVG"
        assert 'viewBox="' in text, f"у {path.name} нет viewBox"


def test_theme_keeps_dock_title_icons() -> None:
    """Тема не должна отключать кнопки закрытия и открепления док-панелей.

    ``titlebar-close-icon: none`` в QSS убирает значок, и кнопка становится
    невидимой, хотя остаётся кликабельной.
    """
    stylesheet = load_stylesheet()
    assert stylesheet, "тема не загрузилась"
    assert "titlebar-close-icon: none" not in stylesheet
    assert "titlebar-normal-icon: none" not in stylesheet
