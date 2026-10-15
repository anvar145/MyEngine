"""Диалог поиска и замены по открытому скрипту.

Диалог немодальный: он живёт рядом с редактором, поэтому поиск можно уточнять,
продолжая править код. Вся работа с текстом выполняется через
:class:`QPlainTextEdit`, а диалог лишь собирает параметры и показывает результат.

Example:
    >>> dialog = FindReplaceDialog(editor, translator=context.tr)
    >>> dialog.show()
    >>> dialog.find_next()
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["FindReplaceDialog", "TextTarget", "resolve_text_target"]

#: Источник текста для поиска: сам редактор или панель с активной вкладкой.
TextTarget = QPlainTextEdit | object


def resolve_text_target(target: object) -> QPlainTextEdit:
    """Привести цель поиска к конкретному редактору.

    Диалог можно открыть и для одного :class:`QPlainTextEdit`, и для панели со
    вкладками: во втором случае поиск идёт по активной вкладке.

    Args:
        target: Редактор или объект с методом ``current_editor()``.

    Returns:
        Редактор, в котором нужно искать.

    Raises:
        TypeError: Если цель не предоставляет текстовый редактор.
    """
    if isinstance(target, QPlainTextEdit):
        return target
    current = getattr(target, "current_editor", None)
    if callable(current):
        editor = current()
        if isinstance(editor, QPlainTextEdit):
            return editor
    raise TypeError(f"не удалось получить редактор из {target!r}")


class FindReplaceDialog(QDialog):
    """Окно «Найти и заменить» для редактора скриптов.

    Args:
        target: Редактор (``QPlainTextEdit``) или панель с вкладками
            (объект с методом ``current_editor()``).
        parent: Родительское окно.
        translator: Функция перевода (обычно ``context.tr``).

    Signals:
        status_message: Текст для статусной строки редактора.
    """

    status_message = Signal(str)

    def __init__(
        self,
        target: TextTarget,
        parent: QWidget | None = None,
        *,
        translator: Callable[..., str] | None = None,
    ) -> None:
        super().__init__(parent)
        self._target = target
        self._translate = translator or (lambda key, **kwargs: key)
        self._editor: QPlainTextEdit | None = None
        self.setObjectName("script_find_dialog")
        self.setModal(False)
        self.setMinimumWidth(460)
        self._last_searched = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.find_edit = QLineEdit(self)
        self.replace_edit = QLineEdit(self)
        self.find_label = QLabel(self)
        self.replace_label = QLabel(self)
        grid.addWidget(self.find_label, 0, 0)
        grid.addWidget(self.find_edit, 0, 1, 1, 3)
        grid.addWidget(self.replace_label, 1, 0)
        grid.addWidget(self.replace_edit, 1, 1, 1, 3)
        layout.addLayout(grid)

        self.case_box = QCheckBox(self)
        self.word_box = QCheckBox(self)
        options = QHBoxLayout()
        options.setSpacing(14)
        options.addWidget(self.case_box)
        options.addWidget(self.word_box)
        options.addStretch(1)
        layout.addLayout(options)

        self.find_next_button = QPushButton(self)
        self.find_prev_button = QPushButton(self)
        self.replace_button = QPushButton(self)
        self.replace_all_button = QPushButton(self)
        self.close_button = QPushButton(self)
        self.replace_button.setProperty("accent", True)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self.find_next_button)
        buttons.addWidget(self.find_prev_button)
        buttons.addSpacing(10)
        buttons.addWidget(self.replace_button)
        buttons.addWidget(self.replace_all_button)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self.result_label = QLabel(self)
        self.result_label.setProperty("role", "hint")
        layout.addWidget(self.result_label)

        self.find_next_button.clicked.connect(self.find_next)
        self.find_prev_button.clicked.connect(self.find_previous)
        self.replace_button.clicked.connect(self.replace_current)
        self.replace_all_button.clicked.connect(self.replace_all)
        self.close_button.clicked.connect(self.hide)
        self.find_edit.returnPressed.connect(self.find_next)
        self.replace_edit.returnPressed.connect(self.replace_current)
        self.find_edit.textChanged.connect(self._reset_result)
        self.case_box.toggled.connect(self._reset_result)
        self.word_box.toggled.connect(self._reset_result)

        self.retranslate()

    # ------------------------------------------------------------------ #
    # Цель поиска
    # ------------------------------------------------------------------ #
    @property
    def editor(self) -> QPlainTextEdit:
        """Редактор, в котором выполняется поиск.

        Для панели со вкладками это активная вкладка, поэтому поиск всегда идёт
        по тому коду, который пользователь видит.

        Raises:
            TypeError: Если цель поиска не предоставила редактор.
        """
        return resolve_text_target(self._target)

    def set_target(self, target: TextTarget) -> None:
        """Заменить источник текста (например, при смене вкладки).

        Args:
            target: Редактор или панель с методом ``current_editor()``.
        """
        self._target = target

    # ------------------------------------------------------------------ #
    # Локализация
    # ------------------------------------------------------------------ #
    def set_translator(self, translator: Callable[..., str]) -> None:
        """Заменить функцию перевода и перерисовать подписи.

        Args:
            translator: Функция перевода (``context.tr``).
        """
        self._translate = translator
        self.retranslate()

    def retranslate(self) -> None:
        """Перевести подписи диалога."""
        self.setWindowTitle(self.tr("script.find.title"))
        self.find_label.setText(self.tr("script.find.what"))
        self.replace_label.setText(self.tr("script.find.with"))
        self.find_edit.setPlaceholderText(self.tr("script.find.placeholder"))
        self.replace_edit.setPlaceholderText(self.tr("script.find.replace_placeholder"))
        self.case_box.setText(self.tr("script.find.case"))
        self.word_box.setText(self.tr("script.find.word"))
        self.find_next_button.setText(self.tr("script.find.next"))
        self.find_prev_button.setText(self.tr("script.find.previous"))
        self.replace_button.setText(self.tr("script.find.replace"))
        self.replace_all_button.setText(self.tr("script.find.replace_all"))
        self.close_button.setText(self.tr("common.close"))

    def tr(self, key: str, **kwargs: object) -> str:  # noqa: A003 - привычное имя Qt
        """Перевести строку через переданный переводчик."""
        return self._translate(key, **kwargs)

    # ------------------------------------------------------------------ #
    # Поиск
    # ------------------------------------------------------------------ #
    def find_flags(self) -> QTextDocument.FindFlag:
        """Собрать флаги поиска по состоянию галочек."""
        flags = QTextDocument.FindFlag(0)
        if self.case_box.isChecked():
            flags |= QTextDocument.FindFlag.FindCaseSensitively
        if self.word_box.isChecked():
            flags |= QTextDocument.FindFlag.FindWholeWords
        return flags

    def _flags(self) -> QTextDocument.FindFlag:
        """Внутренний синоним :meth:`find_flags` (историческое имя)."""
        return self.find_flags()

    def find_next(self) -> bool:
        """Найти следующее вхождение (с переходом в начало документа).

        Returns:
            ``True``, если вхождение найдено и выделено.
        """
        return self._find(backward=False)

    def find_previous(self) -> bool:
        """Найти предыдущее вхождение.

        Returns:
            ``True``, если вхождение найдено и выделено.
        """
        return self._find(backward=True)

    def _find(self, *, backward: bool) -> bool:
        """Выполнить поиск в нужном направлении.

        Args:
            backward: Искать назад.

        Returns:
            ``True``, если вхождение найдено.
        """
        needle = self.find_edit.text()
        if not needle:
            self._set_result(self.tr("script.find.empty"))
            return False
        flags = self.find_flags()
        editor = self.editor
        if backward:
            flags |= QTextDocument.FindFlag.FindBackward
            origin = editor.textCursor()
            if origin.hasSelection():
                origin.setPosition(origin.selectionStart())
                editor.setTextCursor(origin)
        found = editor.find(needle, flags)
        if not found and not backward:
            # Продолжаем поиск с начала документа.
            cursor = editor.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            editor.setTextCursor(cursor)
            found = editor.find(needle, flags)
        if found:
            self._last_searched = needle
            self._set_result(self.tr("script.find.found", text=needle))
        else:
            self._set_result(self.tr("script.find.not_found", text=needle))
        return bool(found)

    # ------------------------------------------------------------------ #
    # Замена
    # ------------------------------------------------------------------ #
    def replace_current(self) -> bool:
        """Заменить текущее выделение и перейти к следующему вхождению.

        Returns:
            ``True``, если замена выполнена (даже если следующее вхождение не
            найдено — иначе кнопка «Заменить» выглядела бы сломанной).
        """
        editor = self.editor
        cursor = editor.textCursor()
        if not cursor.hasSelection():
            return self.find_next()
        needle = self.find_edit.text()
        if not needle or cursor.selectedText() != needle:
            return self.find_next()
        cursor.insertText(self.replace_edit.text())
        editor.setTextCursor(cursor)
        self.find_next()
        return True

    def replace_all(self) -> int:
        """Заменить все вхождения в документе.

        Returns:
            Число выполненных замен.
        """
        needle = self.find_edit.text()
        if not needle:
            self._set_result(self.tr("script.find.empty"))
            return 0
        replacement = self.replace_edit.text()
        flags = self.find_flags()
        document = self.editor.document()
        cursor = QTextCursor(document)
        cursor.beginEditBlock()
        count = 0
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        while True:
            cursor = document.find(needle, cursor, flags)
            if cursor.isNull():
                break
            cursor.insertText(replacement)
            count += 1
        cursor.endEditBlock()
        self._set_result(self.tr("script.find.replaced", count=count))
        return count

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    def set_find_text(self, text: str) -> None:
        """Заполнить поле поиска (например, выделенным в редакторе текстом).

        Args:
            text: Строка для поиска.
        """
        self.find_edit.setText(text)
        self.find_edit.selectAll()

    def result_text(self) -> str:
        """Текущий текст строки результата (для тестов)."""
        return self.result_label.text()

    def _set_result(self, text: str) -> None:
        """Показать результат и продублировать его в статусную строку."""
        self.result_label.setText(text)
        self.status_message.emit(text)

    def _reset_result(self, *_args: object) -> None:
        """Очистить строку результата при смене запроса."""
        self.result_label.clear()
        self._last_searched = ""

    def show_for(self, text: str = "") -> None:
        """Показать диалог, при необходимости подставив выделенный текст.

        Args:
            text: Готовый запрос; пустая строка — взять выделение редактора.
        """
        selection = text or self.editor.textCursor().selectedText()
        if selection and "\u2029" not in selection:
            self.set_find_text(selection)
        self.show()
        self.raise_()
        self.activateWindow()
        self.find_edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def hideEvent(self, event) -> None:  # noqa: N802 - API Qt
        """Вернуть фокус редактору при закрытии окна.

        Args:
            event: Событие скрытия.
        """
        with contextlib.suppress(TypeError):
            # Цель поиска могла исчезнуть: например, закрыли последнюю вкладку.
            self.editor.setFocus(Qt.FocusReason.OtherFocusReason)
        super().hideEvent(event)
