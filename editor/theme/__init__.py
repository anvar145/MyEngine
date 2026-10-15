"""Тема оформления редактора: токены палитры и загрузка QSS.

Все цвета собраны в одном месте (:data:`PALETTE`) и синхронизированы с
``dark.qss.tpl``. Токены нужны и коду: например, цвет подсветки выделенного
объекта во вьюпорте берётся отсюда, чтобы 3D и интерфейс выглядели цельно.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

__all__ = [
    "ICONS_DIR",
    "ICON_SIZE",
    "PALETTE",
    "SPACING",
    "apply_theme",
    "icon",
    "load_stylesheet",
]

_log = logging.getLogger("editor.theme")

THEME_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THEME_DIR.parents[1]
ICONS_DIR = PROJECT_ROOT / "assets" / "icons"

#: Палитра интерфейса: тёплые нейтральные тона без синего, терракотовый акцент.
#: Единственный источник цветов: ``dark.qss.tpl`` подставляет их как ``@имя``.
PALETTE: dict[str, str] = {
    "window": "#141312",
    "panel": "#1a1918",
    "surface": "#1e1d1b",
    "surface_alt": "#262523",
    "hover": "#232220",
    "border": "#2c2a27",
    "border_light": "#3a3835",
    "text": "#e8e4de",
    "text_muted": "#8f8a82",
    "text_dim": "#5f5b55",
    "accent": "#d97757",
    "accent_hover": "#e28a6d",
    "accent_soft": "#3a2a23",
    "on_accent": "#1a1918",
    "warning": "#e0b25c",
    "error": "#e0675f",
    "success": "#7fb08a",
}

#: Шкала отступов и размеры иконок (px).
SPACING = (4, 8, 12, 16)
ICON_SIZE = 18
#: Толщина линии иконок Lucide (по умолчанию в файлах 2).
ICON_STROKE = 1.75

#: Цвет рамки выделения в 3D-вьюпорте (совпадает с акцентом интерфейса).
SELECTION_COLOR = PALETTE["warning"]

_ICON_CACHE: dict[tuple[str, str | None], QIcon] = {}


def load_stylesheet(name: str = "dark.qss.tpl") -> str:
    """Прочитать шаблон темы и подставить цвета палитры.

    Токены ``@имя`` заменяются значениями :data:`PALETTE` один раз при запуске,
    поэтому в рантайме стоимости нет.

    Args:
        name: Имя файла-шаблона в каталоге темы.

    Returns:
        Текст таблицы стилей; при ошибке чтения возвращается пустая строка
        (интерфейс останется работоспособным, просто без оформления).
    """
    path = THEME_DIR / name
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _log.error("Не удалось прочитать тему '%s': %s", path, exc)
        return ""
    # Длинные имена первыми: «@text_muted» не должен ломаться заменой «@text».
    for token in sorted(PALETTE, key=len, reverse=True):
        text = text.replace(f"@{token}", PALETTE[token])
    return text.replace("@icons", ICONS_DIR.as_posix())


def apply_theme(application: object, name: str = "dark.qss.tpl") -> bool:
    """Применить тему к приложению.

    Args:
        application: Экземпляр ``QApplication``.
        name: Имя файла темы.

    Returns:
        ``True``, если стиль применён.
    """
    stylesheet = load_stylesheet(name)
    if not stylesheet:
        return False
    application.setStyleSheet(stylesheet)  # type: ignore[attr-defined]
    _log.debug("Тема '%s' применена (%d символов)", name, len(stylesheet))
    return True


def _render_svg(svg: str, color: str, size: int) -> QPixmap:
    """Нарисовать SVG Lucide заданным цветом в квадратный ``QPixmap``."""
    data = svg.replace("currentColor", color).replace(
        'stroke-width="2"', f'stroke-width="{ICON_STROKE}"'
    )
    renderer = QSvgRenderer(data.encode("utf-8"))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return pixmap


def icon(name: str, color: str | None = None) -> QIcon:
    """Загрузить SVG-иконку из ``assets/icons`` (с кэшированием).

    Иконки Lucide рисуются ``currentColor``: цвет подставляется из палитры.
    Без явного ``color`` иконка получает состояния: обычная — ``text``,
    выключенная — ``text_dim``, включённая (checked) — ``accent``.

    Args:
        name: Имя иконки без расширения (``"cube"``, ``"delete"``, ...).
        color: Цвет (hex или ключ :data:`PALETTE`); ``None`` — по состояниям.

    Returns:
        Иконка; если файла нет — пустая иконка (кнопка останется с текстом).
    """
    key = (name, color)
    cached = _ICON_CACHE.get(key)
    if cached is not None:
        return cached
    path = ICONS_DIR / f"{name}.svg"
    result = QIcon()
    if path.is_file():
        svg = path.read_text(encoding="utf-8")
        size = ICON_SIZE * 2
        if color is not None:
            result.addPixmap(_render_svg(svg, PALETTE.get(color, color), size))
        else:
            normal = _render_svg(svg, PALETTE["text"], size)
            result.addPixmap(normal, QIcon.Mode.Normal, QIcon.State.Off)
            result.addPixmap(
                _render_svg(svg, PALETTE["accent"], size),
                QIcon.Mode.Normal,
                QIcon.State.On,
            )
            result.addPixmap(
                _render_svg(svg, PALETTE["text_dim"], size), QIcon.Mode.Disabled
            )
    else:
        _log.debug("Иконка '%s' не найдена (%s)", name, path)
    _ICON_CACHE[key] = result
    return result
