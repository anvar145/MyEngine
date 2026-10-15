"""Бэкенд ввода Qt: ``keyPressEvent``/``keyReleaseEvent`` → :class:`InputState`.

Работает только пока включён флаг ``enabled`` (Play Mode). Горячие клавиши
редактора не фильтруются: события передаются дальше как есть (решение A5 Q3).
Сейчас Play Mode идёт в отдельном процессе плеера с окном GLFW, поэтому
потребителя у бэкенда нет; он готов к встроенному вьюпорту.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeyEvent

from engine.core.input import InputState

__all__ = ["QtInputBackend", "qt_key_name"]

_SPECIAL: dict[int, str] = {
    Qt.Key_Space: "space",
    Qt.Key_Escape: "escape",
    Qt.Key_Return: "enter",
    Qt.Key_Enter: "enter",
    Qt.Key_Tab: "tab",
    Qt.Key_Backspace: "backspace",
    Qt.Key_Right: "right",
    Qt.Key_Left: "left",
    Qt.Key_Down: "down",
    Qt.Key_Up: "up",
    Qt.Key_Shift: "shift",
    Qt.Key_Control: "ctrl",
    Qt.Key_Alt: "alt",
}


def qt_key_name(key: int) -> str | None:
    """Имя клавиши для маппинга (``"w"``, ``"space"``, ``"f2"``) или ``None``."""
    if Qt.Key_A <= key <= Qt.Key_Z:
        return chr(ord("a") + key - Qt.Key_A)
    if Qt.Key_0 <= key <= Qt.Key_9:
        return chr(ord("0") + key - Qt.Key_0)
    if Qt.Key_F1 <= key <= Qt.Key_F12:
        return f"f{key - Qt.Key_F1 + 1}"
    return _SPECIAL.get(int(key))


class QtInputBackend(QObject):
    """Фильтр событий Qt, пишущий клавиши и кнопки мыши в ``state``.

    Args:
        state: Общее состояние ввода.
        parent: Владелец QObject.
    """

    def __init__(self, state: InputState, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.state = state
        #: Пока ``False`` события игнорируются (вне Play Mode).
        self.enabled = False

    def poll(self) -> None:
        """События приходят от Qt; опрашивать нечего."""

    def set_enabled(self, enabled: bool) -> None:
        """Включить или выключить приём; при выключении клавиши сбрасываются."""
        if self.enabled and not enabled:
            self.state.clear()
        self.enabled = enabled

    def handle_key(self, key: int, down: bool, auto_repeat: bool = False) -> None:
        """Обработать клавишу (автоповтор игнорируется)."""
        if not self.enabled or auto_repeat:
            return
        name = qt_key_name(key)
        if name is None:
            return
        if down:
            self.state.key_down(name)
        else:
            self.state.key_up(name)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Читает key-события и не потребляет их (хоткеи редактора живы)."""
        kind = event.type()
        if kind in (QEvent.KeyPress, QEvent.KeyRelease):
            if isinstance(event, QKeyEvent):
                down = kind == QEvent.KeyPress
                self.handle_key(event.key(), down, event.isAutoRepeat())
        elif kind == QEvent.FocusOut:
            self.state.clear()
        return False
