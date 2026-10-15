"""Панель консоли: логи движка и сообщения редактора.

Панель подключается к стандартному модулю :mod:`logging` через собственный
обработчик, поэтому в неё попадают и сообщения рендерера, и ошибки скриптов —
без правок в самом движке.
"""

from __future__ import annotations

import logging
from collections import deque

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from editor.panels.base import DockPanel
from editor.theme import PALETTE

__all__ = ["ConsolePanel", "QtLogHandler"]

#: Сколько строк хранить в консоли (старые вытесняются).
MAX_LINES = 2000

LEVEL_KEYS: tuple[tuple[str, int], ...] = (
    ("console.levels", logging.DEBUG),
    ("console.levels", logging.INFO),
    ("console.levels", logging.WARNING),
    ("console.levels", logging.ERROR),
)


class QtLogHandler(logging.Handler, QObject):
    """Обработчик :mod:`logging`, транслирующий записи в Qt-сигнал.

    Signals:
        message_logged: Запись лога (текст, числовой уровень).
    """

    message_logged = Signal(str, int)

    def __init__(self, level: int = logging.INFO) -> None:
        logging.Handler.__init__(self, level)
        QObject.__init__(self)
        # Уровень ставим и логгеру: корневой логгер по умолчанию пропускает
        # только WARNING и выше, поэтому INFO-сообщения скриптов (print, лог
        # движка) до обработчика просто не доходили.
        self.setLevel(level)
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        """Отправить запись в консоль (вызывается из любого потока)."""
        try:
            self.message_logged.emit(self.format(record), record.levelno)
        except Exception:  # noqa: BLE001 - логирование не должно ломать приложение
            self.handleError(record)


class ConsolePanel(DockPanel):
    """Консоль с уровнями и историей сообщений."""

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__("panel.console", context, parent)
        self._history: deque[tuple[str, int]] = deque(maxlen=MAX_LINES)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.level_combo = QComboBox(container)
        self.clear_button = QPushButton(container)
        self.clear_button.clicked.connect(self.clear)
        controls.addWidget(self.level_combo, 1)
        controls.addWidget(self.clear_button)
        layout.addLayout(controls)

        self.view = QPlainTextEdit(container)
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(MAX_LINES)
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.view, 1)
        self.setWidget(container)

        self._handler = QtLogHandler()
        self._handler.message_logged.connect(self.append)
        logging.getLogger("engine").addHandler(self._handler)
        logging.getLogger("editor").addHandler(self._handler)

        self.level_combo.addItem("", logging.DEBUG)
        self.level_combo.addItem("", logging.INFO)
        self.level_combo.addItem("", logging.WARNING)
        self.level_combo.addItem("", logging.ERROR)
        self.level_combo.setCurrentIndex(1)
        self.level_combo.currentIndexChanged.connect(self._apply_level)
        self.retranslate()

    # ------------------------------------------------------------------ #
    # Локализация
    # ------------------------------------------------------------------ #
    def retranslate(self, *_args: object) -> None:
        """Перевести подписи панели."""
        super().retranslate()
        self.clear_button.setText(self.tr("console.clear"))
        labels = {
            0: "DEBUG",
            1: "INFO",
            2: "WARNING",
            3: "ERROR",
        }
        for index, text in labels.items():
            self.level_combo.setItemText(index, f"{self.tr('console.levels')}: {text}")

    # ------------------------------------------------------------------ #
    # Сообщения
    # ------------------------------------------------------------------ #
    def append(self, text: str, level: int = logging.INFO) -> None:
        """Добавить строку в консоль.

        Args:
            text: Текст сообщения.
            level: Уровень логирования (влияет на цвет).
        """
        self._history.append((text, level))
        if level < self._minimum_level():
            return
        self.view.appendHtml(self._format_line(text, level))
        self.view.moveCursor(QTextCursor.End)
        # Курсор в конце строки уводит вид вправо и обрезает начало записей.
        self.view.horizontalScrollBar().setValue(0)

    def clear(self) -> None:
        """Очистить консоль и историю."""
        self._history.clear()
        self.view.clear()

    @property
    def history(self) -> tuple[tuple[str, int], ...]:
        """История сообщений (для тестов и отчётов)."""
        return tuple(self._history)

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #
    def _minimum_level(self) -> int:
        """Текущий порог отображения."""
        data = self.level_combo.currentData()
        return int(data) if data is not None else logging.INFO

    def _apply_level(self, *_args: object) -> None:
        """Перерисовать историю под новый порог."""
        self.view.clear()
        for text, level in self._history:
            if level >= self._minimum_level():
                self.view.appendHtml(self._format_line(text, level))

    @staticmethod
    def _format_line(text: str, level: int) -> str:
        """Раскрасить строку по уровню (в цветах темы)."""
        colors = {
            logging.DEBUG: PALETTE["text_muted"],
            logging.INFO: PALETTE["text"],
            logging.WARNING: PALETTE["warning"],
            logging.ERROR: PALETTE["error"],
            logging.CRITICAL: PALETTE["error"],
        }
        color = QColor(colors.get(level, PALETTE["text"]))
        escaped = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        return f'<span style="color:{color.name()}">{escaped}</span>'

    def closeEvent(self, event) -> None:  # noqa: N802 - API Qt
        """Отключить обработчик логов при закрытии панели."""
        logging.getLogger("engine").removeHandler(self._handler)
        logging.getLogger("editor").removeHandler(self._handler)
        super().closeEvent(event)
