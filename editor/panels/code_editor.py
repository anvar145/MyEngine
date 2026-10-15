"""Виджет кода для редактора скриптов.

:class:`CodeEditor` — это ``QPlainTextEdit`` с тем, что отличает редактор кода
от обычного поля ввода:

* нумерация строк — внешняя панель :class:`LineNumberArea` слева от текста;
* автоотступ после ``:``, умный ``Backspace`` и Tab/Shift+Tab по выделению;
* парные скобки: автозакрытие, пропуск закрывающей и подсветка пары;
* автодополнение через ``QCompleter`` — по Ctrl+Space и при вводе имени;
* подсветка синтаксиса :class:`~engine.scripting.python_highlighter.PythonHighlighter`.

Example:
    >>> editor = CodeEditor()
    >>> editor.set_text("print('привет')\\n")
    >>> editor.highlighter is not None
    True
"""

from __future__ import annotations

import re

from PySide6.QtCore import QRect, QSize, QStringListModel, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QKeyEvent,
    QPainter,
    QPaintEvent,
    QResizeEvent,
    QTextCursor,
    QTextFormat,
)
from PySide6.QtWidgets import QCompleter, QPlainTextEdit, QTextEdit, QWidget

from engine.scripting.python_highlighter import PythonHighlighter
from engine.scripting.syntax import (
    ENGINE_API_ATTRIBUTES,
    completion_words,
)

__all__ = ["CODE_FONT_FAMILIES", "CodeEditor", "LineNumberArea", "code_font"]

#: Предпочтительные моноширинные шрифты (используется первый найденный).
CODE_FONT_FAMILIES: tuple[str, ...] = (
    "JetBrains Mono",
    "JetBrainsMono Nerd Font",
    "Cascadia Mono",
    "Cascadia Code",
    "Fira Code",
    "Consolas",
    "DejaVu Sans Mono",
    "Menlo",
    "Monaco",
    "Courier New",
)

#: Размер шрифта кода (пункты).
CODE_FONT_SIZE = 11

#: Ширина отступа в пробелах.
TAB_WIDTH = 4

#: Скобки, которые закрываются автоматически (кавычки — тоже «скобки»).
BRACKET_PAIRS: dict[str, str] = {"(": ")", "[": "]", "{": "}", '"': '"', "'": "'"}

#: Закрывающие символы, которые «пропускаются» при вводе поверх.
CLOSING_CHARS = frozenset(BRACKET_PAIRS.values())

#: Кавычки: у них нет дополнительного отступа, но есть проверка контекста.
QUOTES = frozenset({'"', "'"})

#: Ключевые слова, после которых отступ уменьшается на шаг.
DEDENT_KEYWORDS = ("return", "pass", "break", "continue", "raise")

#: Цвета оформления редактора (согласованы с тёмной темой).
EDITOR_COLORS: dict[str, str] = {
    "background": "#141312",
    "gutter_background": "#171615",
    "gutter_text": "#5f5b55",
    "gutter_text_current": "#c9c4bc",
    "current_line": "#1c1b19",
    "matched_bracket": "#4a3a30",
    "unmatched_bracket": "#7a3a35",
}

#: Минимальная ширина панели номеров (в «цифрах»).
MIN_LINE_NUMBER_DIGITS = 3

#: Стиль всплывающего списка подсказок (тёмная тема).
COMPLETER_STYLE = (
    "QListView {"
    "  background-color: #1e1d1b; border: 1px solid #3a3835;"
    "  color: #e8e4de; padding: 4px; outline: none;"
    "}"
    "QListView::item { padding: 3px 8px; border-radius: 5px; }"
    "QListView::item:selected { background-color: #3a2a23; color: #e8e4de; }"
)

#: Имя слева от курсора (кандидат в префикс автодополнения).
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")

#: Ведущие пробелы строки (будущий отступ).
_INDENT_RE = re.compile(r"^[ \t]*")

#: Строка открывает блок, если заканчивается двоеточием.
_BLOCK_RE = re.compile(r":\s*$")


def code_font() -> QFont:
    """Подобрать моноширинный шрифт для кода.

    Returns:
        Шрифт из :data:`CODE_FONT_FAMILIES` или системный моноширинный.
    """
    available = set(QFontDatabase.families())
    for family in CODE_FONT_FAMILIES:
        if family in available:
            font = QFont(family, CODE_FONT_SIZE)
            font.setStyleHint(QFont.StyleHint.Monospace)
            font.setFixedPitch(True)
            return font
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setPointSize(CODE_FONT_SIZE)
    # Без системных шрифтов (offscreen) Qt отдаёт «monospace» без подсказки.
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    return font


def matching_bracket(text: str, position: int) -> tuple[int, int, bool] | None:
    """Найти пару скобок рядом с позицией курсора.

    Args:
        text: Текст документа целиком (без учёта строк).
        position: Позиция курсора.

    Returns:
        Тройку «индекс открывающей, индекс закрывающей, пара найдена» или
        ``None``, если рядом с курсором скобки нет. Если пара не закрыта,
        оба индекса указывают на одиночную скобку, а флаг равен ``False``.
    """
    for index in (position, position - 1):
        if not 0 <= index < len(text):
            continue
        char = text[index]
        if char in "([{":
            partner = _scan_forward(text, index, char)
            if partner is None:
                return index, index, False
            return index, partner, True
        if char in ")]}":
            partner = _scan_backward(text, index, char)
            if partner is None:
                return index, index, False
            return partner, index, True
    return None


def _scan_forward(text: str, index: int, opener: str) -> int | None:
    """Найти закрывающую скобку для открывающей в позиции ``index``."""
    closer = {"(": ")", "[": "]", "{": "}"}[opener]
    depth = 0
    for scan in range(index, len(text)):
        char = text[scan]
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return scan
    return None


def _scan_backward(text: str, index: int, closer: str) -> int | None:
    """Найти открывающую скобку для закрывающей в позиции ``index``."""
    opener = {")": "(", "]": "[", "}": "{"}[closer]
    depth = 0
    for scan in range(index, -1, -1):
        char = text[scan]
        if char == closer:
            depth += 1
        elif char == opener:
            depth -= 1
            if depth == 0:
                return scan
    return None


class LineNumberArea(QWidget):
    """Панель нумерации строк слева от редактора.

    Args:
        editor: Редактор кода, которому принадлежит панель.
    """

    def __init__(self, editor: CodeEditor) -> None:
        super().__init__(editor)
        self._editor = editor
        self.setObjectName("script_line_numbers")
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def sizeHint(self) -> QSize:  # noqa: N802 - API Qt
        """Ширина панели под текущее число строк."""
        return QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - API Qt
        """Перерисовать номера строк."""
        self._editor.paint_line_numbers(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - API Qt
        """Поставить курсор в строку, по которой кликнули.

        Args:
            event: Событие мыши.
        """
        self._editor.select_line_at(event.position().y())
        super().mousePressEvent(event)


class CodeEditor(QPlainTextEdit):
    """Редактор Python-кода: подсветка, отступы, скобки и подсказки.

    Args:
        parent: Родительский виджет.
        api_names: Имена API движка для подсветки и подсказок.

    Signals:
        cursor_position_changed: Курсор переместился (строка, столбец).
        save_requested: Нажат Ctrl+S.
        run_requested: Нажат F5.
        stop_requested: Нажат Shift+F5.
        reload_requested: Нажат F6.
    """

    cursor_position_changed = Signal(int, int)
    save_requested = Signal()
    run_requested = Signal()
    stop_requested = Signal()
    reload_requested = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        api_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("script_code_editor")
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setFont(code_font())
        self.setTabChangesFocus(False)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * TAB_WIDTH)
        self.setCursorWidth(2)

        self.api_names: tuple[str, ...] = api_names or ENGINE_API_ATTRIBUTES
        self.highlighter = PythonHighlighter(self.document())
        self.highlighter.set_api_names(self.api_names)

        self._line_numbers_enabled = True
        self._completion_enabled = True
        self._completer_visible = False
        self._line_number_area = LineNumberArea(self)

        self._words: list[str] = []
        self._model = QStringListModel(self)
        self._completer = self._make_completer()
        self._refresh_words()

        self._completion_timer = QTimer(self)
        self._completion_timer.setSingleShot(True)
        self._completion_timer.setInterval(120)
        self._completion_timer.timeout.connect(self._maybe_complete)

        self.blockCountChanged.connect(self._update_line_number_area_width)
        self.updateRequest.connect(self._update_line_number_area)
        self.cursorPositionChanged.connect(self._on_cursor_moved)
        self.textChanged.connect(self._on_text_changed)

        self._update_line_number_area_width()
        self._refresh_highlights()

    # ------------------------------------------------------------------ #
    # Текст
    # ------------------------------------------------------------------ #
    def text(self) -> str:
        """Текст документа целиком."""
        return self.toPlainText()

    def set_text(self, text: str) -> None:
        """Заменить содержимое документа.

        Флаг изменения намеренно сохраняется: обычно текст подставляют
        программно — после поиска с заменой, правки или установки шаблона, — и
        редактор должен показать, что файл отличается от версии на диске.
        Загрузка свежего файла сбрасывает флаг явно (:meth:`mark_saved`).

        Сигналы документа на время замены блокируются: ``setPlainText`` сам
        управляет флагом изменения и сбрасывает его в конце.

        Args:
            text: Новый исходный код.
        """
        document = self.document()
        modified = document.isModified()
        document.blockSignals(True)
        try:
            self.setPlainText(text)
        finally:
            document.blockSignals(False)
        document.setModified(modified)
        self.textChanged.emit()

    def is_modified(self) -> bool:
        """``True``, если документ изменён с момента сохранения."""
        return self.document().isModified()

    def mark_saved(self) -> None:
        """Сбросить флаг изменения (после сохранения файла)."""
        self.document().setModified(False)

    def indent_for_new_line(self) -> str:
        """Определить отступ новой строки.

        Правила простые и предсказуемые: отступ берётся у ближайшей непустой
        строки выше, после строки с ``:`` добавляется один шаг, а после
        ``return``/``pass``/``break``/``continue``/``raise`` шаг убирается
        (пользователь, скорее всего, выходит из блока).

        Returns:
            Строка пробелов для начала новой строки.
        """
        line = self._context_line()
        stripped = line.rstrip()
        indent = _INDENT_RE.match(line).group(0)  # type: ignore[union-attr]
        if _BLOCK_RE.search(stripped):
            return indent + " " * TAB_WIDTH
        if stripped.endswith(DEDENT_KEYWORDS):
            return indent[: max(0, len(indent) - TAB_WIDTH)]
        return indent

    def _context_line(self) -> str:
        """Текст строки, задающей отступ для новой строки.

        Если курсор стоит в пустой строке (обычная ситуация сразу после
        ``Enter``), берётся ближайшая непустая строка выше: иначе отступ всегда
        был бы пустым.

        Returns:
            Текст строки-образца (пустая строка, если выше ничего нет).
        """
        block = self.textCursor().block()
        if block.text().strip():
            return block.text()
        candidate = block.previous()
        while candidate.isValid():
            if candidate.text().strip():
                return candidate.text()
            candidate = candidate.previous()
        return ""

    # ------------------------------------------------------------------ #
    # Нумерация строк
    # ------------------------------------------------------------------ #
    def set_line_numbers(self, enabled: bool) -> None:
        """Включить или выключить панель нумерации.

        Args:
            enabled: ``True`` — показывать номера строк слева.
        """
        self._line_numbers_enabled = bool(enabled)
        self._line_number_area.setVisible(self._line_numbers_enabled)
        self._update_line_number_area_width()

    def line_numbers_enabled(self) -> bool:
        """Включена ли панель нумерации."""
        return self._line_numbers_enabled

    def line_number_area_width(self) -> int:
        """Ширина панели номеров под текущее число строк."""
        digits = max(MIN_LINE_NUMBER_DIGITS, len(str(max(1, self.blockCount()))))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def paint_line_numbers(self, event: QPaintEvent) -> None:
        """Нарисовать номера строк, попавшие в область перерисовки.

        Args:
            event: Событие перерисовки панели.
        """
        painter = QPainter(self._line_number_area)
        painter.fillRect(event.rect(), QColor(EDITOR_COLORS["gutter_background"]))
        painter.setFont(self.font())

        block = self.firstVisibleBlock()
        number = block.blockNumber()
        offset = self.contentOffset()
        top = round(self.blockBoundingGeometry(block).translated(offset).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        current_line = self.textCursor().blockNumber()
        height = self.fontMetrics().height()
        width = self._line_number_area.width() - 6

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                color = (
                    EDITOR_COLORS["gutter_text_current"]
                    if number == current_line
                    else EDITOR_COLORS["gutter_text"]
                )
                painter.setPen(QColor(color))
                painter.drawText(
                    0,
                    top,
                    width,
                    height,
                    int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                    str(number + 1),
                )
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            number += 1

    def select_line_at(self, y: float) -> None:
        """Поставить курсор в строку по вертикальной координате.

        Args:
            y: Координата Y внутри панели номеров.
        """
        block = self.firstVisibleBlock()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        while block.isValid():
            height = self.blockBoundingRect(block).height()
            if top <= y <= top + height:
                self.setTextCursor(QTextCursor(block))
                return
            top += height
            block = block.next()

    # ------------------------------------------------------------------ #
    # Автодополнение
    # ------------------------------------------------------------------ #
    def set_completion_enabled(self, enabled: bool) -> None:
        """Включить или выключить автодополнение.

        Args:
            enabled: ``True`` — показывать подсказки.
        """
        self._completion_enabled = bool(enabled)
        if not self._completion_enabled:
            self.hide_completions()

    def completion_words(self) -> list[str]:
        """Словарь автодополнения для текущего документа."""
        return list(self._words)

    def completion_prefix(self) -> str:
        """Имя слева от курсора — будущий префикс дополнения."""
        cursor = self.textCursor()
        line = cursor.block().text()
        match = _WORD_RE.search(line[: cursor.positionInBlock()])
        return match.group(0) if match else ""

    def show_completions(self) -> None:
        """Показать окно подсказок для текущего префикса."""
        if not self._completion_enabled:
            return
        prefix = self.completion_prefix()
        if not prefix:
            return
        self._completer.setCompletionPrefix(prefix)
        if self._completer.completionCount() == 0:
            self.hide_completions()
            return
        popup = self._completer.popup()
        popup.setFont(self.font())
        popup.setMinimumWidth(max(240, self.width() // 3))
        self._completer.setCurrentRow(0)
        rect = self.cursorRect()
        rect.setWidth(
            popup.sizeHintForColumn(0)
            + popup.verticalScrollBar().sizeHint().width()
            + 24
        )
        self._completer.complete(rect)
        self._completer_visible = not popup.isHidden()

    def hide_completions(self) -> None:
        """Скрыть окно подсказок."""
        self._completer_visible = False
        self._completer.popup().hide()

    def insert_completion(self, completion: str) -> None:
        """Вставить выбранное слово вместо набранного префикса.

        Args:
            completion: Выбранное слово (может содержать ``self.``).
        """
        if not self._completer_visible:
            return
        selected = completion.split(".")[-1]
        cursor = self.textCursor()
        cursor.movePosition(
            QTextCursor.MoveOperation.Left,
            QTextCursor.MoveMode.KeepAnchor,
            len(self._completer.completionPrefix()),
        )
        cursor.insertText(selected)
        self.setTextCursor(cursor)
        self.hide_completions()

    # ------------------------------------------------------------------ #
    # Подсветка выделений
    # ------------------------------------------------------------------ #
    def _refresh_highlights(self) -> None:
        """Пересобрать дополнительные выделения (строка и пара скобок)."""
        selections: list[QTextEdit.ExtraSelection] = []
        current = self._make_selection(self.textCursor(), EDITOR_COLORS["current_line"])
        current.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        selections.append(current)

        bracket = matching_bracket(self.toPlainText(), self.textCursor().position())
        if bracket is not None:
            first, second, matched = bracket
            color = (
                EDITOR_COLORS["matched_bracket"]
                if matched
                else EDITOR_COLORS["unmatched_bracket"]
            )
            for index in {first, second}:
                selections.append(self._make_selection(self._cursor_at(index), color))
        self.setExtraSelections(selections)

    def _make_selection(
        self, cursor: QTextCursor, color: str
    ) -> QTextEdit.ExtraSelection:
        """Собрать выделение с заданным цветом фона.

        Args:
            cursor: Позиция выделения.
            color: Цвет фона в формате ``#rrggbb``.

        Returns:
            Готовое дополнительное выделение.
        """
        selection = QTextEdit.ExtraSelection()
        selection.format.setBackground(QColor(color))
        selection.cursor = cursor
        return selection

    def _cursor_at(self, position: int) -> QTextCursor:
        """Курсор, выделяющий один символ в позиции ``position``."""
        cursor = QTextCursor(self.document())
        cursor.setPosition(position)
        cursor.movePosition(
            QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor
        )
        return cursor

    # ------------------------------------------------------------------ #
    # События
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - API Qt
        """Обработать клавишу: отступы, скобки, подсказки, горячие клавиши.

        Args:
            event: Событие клавиатуры.
        """
        key = event.key()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        if self._completer_visible and key in (
            Qt.Key.Key_Enter,
            Qt.Key.Key_Return,
            Qt.Key.Key_Tab,
            Qt.Key.Key_Escape,
            Qt.Key.Key_Backtab,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        ):
            # Список подсказок обрабатывает клавиши сам.
            event.ignore()
            return

        if ctrl and key == Qt.Key.Key_S:
            self.save_requested.emit()
            return
        if ctrl and key == Qt.Key.Key_Space:
            self.show_completions()
            return
        if ctrl and key == Qt.Key.Key_Slash:
            self.toggle_comment()
            return
        if key == Qt.Key.Key_F5:
            (self.stop_requested if shift else self.run_requested).emit()
            return
        if key == Qt.Key.Key_F6:
            self.reload_requested.emit()
            return
        if key == Qt.Key.Key_Tab and not ctrl:
            self.indent_selection()
            return
        if key == Qt.Key.Key_Backtab:
            self.unindent_selection()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._insert_new_line():
            return
        if key == Qt.Key.Key_Backspace and not ctrl and self._smart_backspace():
            return
        text = event.text()
        if text and not ctrl and (
            self._skip_closing_char(text) or self._autoclose_bracket(text)
        ):
            return
        super().keyPressEvent(event)

    def _insert_new_line(self) -> bool:
        """Вставить перевод строки с отступом предыдущей.

        Returns:
            ``True``, если строка вставлена этим обработчиком.
        """
        cursor = self.textCursor()
        if cursor.hasSelection():
            return False
        cursor.insertText("\n" + self.indent_for_new_line())
        self.setTextCursor(cursor)
        return True

    def indent_selection(self) -> None:
        """Увеличить отступ выделенных строк (или вставить табуляцию)."""
        cursor = self.textCursor()
        pad = " " * TAB_WIDTH
        if not cursor.hasSelection():
            cursor.insertText(pad)
            self.setTextCursor(cursor)
            return
        self._shift_lines(cursor, pad)

    def unindent_selection(self) -> None:
        """Уменьшить отступ выделенных строк (или текущей строки)."""
        cursor = self.textCursor()
        if not cursor.hasSelection():
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        self._shift_lines(cursor, None)

    def toggle_comment(self) -> None:
        """Закомментировать или раскомментировать выделенные строки.

        Если все непустые строки выделения уже закомментированы — комментарии
        снимаются, иначе добавляется ``# ``. Пустые строки не трогаются.
        """
        cursor = self.textCursor()
        if not cursor.hasSelection():
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        document = self.document()
        start_block = document.findBlock(cursor.selectionStart()).blockNumber()
        end_block = document.findBlock(cursor.selectionEnd()).blockNumber()

        blocks = [
            document.findBlockByNumber(number)
            for number in range(start_block, end_block + 1)
        ]
        meaningful = [block for block in blocks if block.text().strip()]
        if not meaningful:
            return
        uncomment = all(block.text().lstrip().startswith("#") for block in meaningful)

        edit = QTextCursor(document)
        edit.beginEditBlock()
        for block in reversed(blocks):
            text = block.text()
            if not text.strip():
                continue
            line_cursor = QTextCursor(block)
            if not uncomment:
                line_cursor.insertText("# ")
                continue
            stripped = text.lstrip()
            offset = len(text) - len(stripped)
            line_cursor.movePosition(
                QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.MoveAnchor, offset
            )
            length = 2 if stripped.startswith("# ") else 1
            line_cursor.movePosition(
                QTextCursor.MoveOperation.Right,
                QTextCursor.MoveMode.KeepAnchor,
                length,
            )
            line_cursor.removeSelectedText()
        edit.endEditBlock()
        self.setTextCursor(cursor)

    def _shift_lines(self, cursor: QTextCursor, pad: str | None) -> None:
        """Сдвинуть отступ у всех строк выделения одним действием.

        Строки обходятся от последней к первой: правка начала документа не
        сдвигает позиции ещё не обработанных строк, поэтому выделение не
        «уезжает». Редактирование идёт в одном ``editBlock``, так что
        отменяется одним нажатием Ctrl+Z.

        Args:
            cursor: Курсор с выделением.
            pad: Пробелы для добавления; ``None`` — убрать один шаг отступа.
        """
        document = self.document()
        start_block = document.findBlock(cursor.selectionStart()).blockNumber()
        end_block = document.findBlock(cursor.selectionEnd()).blockNumber()
        # Выделение, заканчивающееся переводом строки, включает и следующий
        # (пустой) блок — его сдвигать не нужно, иначе появляется «висячий»
        # отступ в конце текста.
        if end_block > start_block and not document.findBlockByNumber(end_block).text():
            end_block -= 1
        cursor.beginEditBlock()
        for number in range(end_block, start_block - 1, -1):
            block = document.findBlockByNumber(number)
            line = block.text()
            prefix = _INDENT_RE.match(line).group(0)  # type: ignore[union-attr]
            line_cursor = QTextCursor(block)
            if pad is None:
                remove = min(len(prefix), TAB_WIDTH)
                if not remove:
                    continue
                line_cursor.movePosition(
                    QTextCursor.MoveOperation.Right,
                    QTextCursor.MoveMode.KeepAnchor,
                    remove,
                )
                line_cursor.removeSelectedText()
            else:
                line_cursor.insertText(pad)
        cursor.endEditBlock()
        self.setTextCursor(cursor)

    def _smart_backspace(self) -> bool:
        """Удалить отступ целиком, если курсор стоит на его границе.

        Returns:
            ``True``, если клавиша обработана.
        """
        cursor = self.textCursor()
        if cursor.hasSelection():
            return False
        column = cursor.positionInBlock()
        if column == 0 or column % TAB_WIDTH != 0:
            return False
        if cursor.block().text()[:column].strip():
            return False
        cursor.beginEditBlock()
        cursor.movePosition(
            QTextCursor.MoveOperation.Left,
            QTextCursor.MoveMode.KeepAnchor,
            TAB_WIDTH,
        )
        cursor.removeSelectedText()
        cursor.endEditBlock()
        self.setTextCursor(cursor)
        return True

    def _skip_closing_char(self, text: str) -> bool:
        """Пропустить закрывающий символ, если он уже вставлен парой.

        Args:
            text: Введённый текст.

        Returns:
            ``True``, если клавиша обработана.
        """
        if text not in CLOSING_CHARS:
            return False
        cursor = self.textCursor()
        if cursor.hasSelection():
            return False
        line = cursor.block().text()
        column = cursor.positionInBlock()
        if column < len(line) and line[column] == text:
            cursor.movePosition(QTextCursor.MoveOperation.Right)
            self.setTextCursor(cursor)
            return True
        return False

    def _autoclose_bracket(self, text: str) -> bool:
        """Закрыть скобку или кавычку автоматически.

        Args:
            text: Введённый символ.

        Returns:
            ``True``, если символ обработан.
        """
        if text not in BRACKET_PAIRS:
            return False
        cursor = self.textCursor()
        closing = BRACKET_PAIRS[text]
        if cursor.hasSelection():
            selected = cursor.selectedText()
            cursor.insertText(f"{text}{selected}{closing}")
            self.setTextCursor(cursor)
            return True
        if text in QUOTES:
            line = cursor.block().text()
            column = cursor.positionInBlock()
            previous = line[column - 1] if column > 0 else ""
            # Кавычка в середине слова или строки — обычный символ.
            if previous.isalnum() or previous in QUOTES:
                return False
            if column < len(line) and line[column] in CLOSING_CHARS:
                return False
        cursor.insertText(text + closing)
        cursor.movePosition(QTextCursor.MoveOperation.Left)
        self.setTextCursor(cursor)
        return True

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - API Qt
        """Пересчитать геометрию панели номеров.

        Args:
            event: Событие изменения размера.
        """
        super().resizeEvent(event)
        self._reposition_line_numbers()

    # ------------------------------------------------------------------ #
    # Внутренние обработчики
    # ------------------------------------------------------------------ #
    def _make_completer(self) -> QCompleter:
        """Собрать ``QCompleter`` с тёмным всплывающим списком."""
        completer = QCompleter(self)
        completer.setWidget(self)
        completer.setModel(self._model)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseSensitive)
        popup = completer.popup()
        popup.setObjectName("script_completer_popup")
        popup.setStyleSheet(COMPLETER_STYLE)
        completer.activated.connect(self.insert_completion)
        return completer

    def _refresh_words(self) -> None:
        """Обновить словарь подсказок (Python + API движка)."""
        self._words = completion_words(self.api_names)
        self._model.setStringList(self._words)

    def _update_line_number_area_width(self, *_args: object) -> None:
        """Пересчитать отступ текста и геометрию панели номеров."""
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)
        self._reposition_line_numbers()

    def _reposition_line_numbers(self) -> None:
        """Поставить панель номеров по левому краю области текста."""
        if not self._line_numbers_enabled:
            return
        rect = self.contentsRect()
        self._line_number_area.setGeometry(
            QRect(rect.left(), rect.top(), self.line_number_area_width(), rect.height())
        )
        self._line_number_area.raise_()

    def _update_line_number_area(self, rect, dy: int) -> None:
        """Прокрутить панель номеров вместе с текстом.

        Args:
            rect: Изменившаяся область документа.
            dy: Вертикальный сдвиг (0 при перерисовке).
        """
        if not self._line_numbers_enabled:
            return
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(
                0, rect.y(), self._line_number_area.width(), rect.height()
            )
        if rect.contains(self.viewport().rect()):
            self._update_line_number_area_width()

    def _on_cursor_moved(self) -> None:
        """Обновить подсветку и сообщить позицию курсора."""
        self._refresh_highlights()
        cursor = self.textCursor()
        self.cursor_position_changed.emit(
            cursor.blockNumber() + 1, cursor.positionInBlock() + 1
        )

    def _on_text_changed(self) -> None:
        """Пересобрать словарь подсказок и, возможно, показать их."""
        self._refresh_words()
        if self._completion_enabled and not self._completer_visible:
            self._completion_timer.start()

    def _maybe_complete(self) -> None:
        """Показать подсказки, если пользователь набирает имя."""
        if len(self.completion_prefix()) >= 2:
            self.show_completions()
