"""Подсветка синтаксиса Python для редактора скриптов.

Модуль реализует :class:`PythonHighlighter` — наследник ``QSyntaxHighlighter``,
который раскрашивает исходник в цвета тёмной темы редактора. Подсветка
построена на правилах: каждое правило — регулярное выражение и стиль, а
многострочные строки обрабатываются конечным автоматом через состояния блока
(``setCurrentBlockState``), поэтому docstring и ``'''``-строки не «рвут»
подсветку на середине.

Имена, ключевые слова и цвета живут в :mod:`engine.scripting.syntax` — этот
модуль Qt не импортирует, им пользуются и слой редактора, и тесты.

Example:
    >>> highlighter = PythonHighlighter(editor.document())
    >>> highlighter.set_api_names(["game_object", "scene", "input"])
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from engine.scripting.syntax import (
    BUILTIN_NAMES,
    ENGINE_API_ATTRIBUTES,
    KEYWORDS,
    TOKEN_COLORS,
)

# PySide6 — необязательная зависимость движка (extra ``editor``): пакет
# ``engine.scripting`` обязан импортироваться и в headless-окружении, поэтому
# отсутствие Qt не считается ошибкой — подсветка просто не создаётся.
try:  # pragma: no cover - ветка зависит от окружения
    from PySide6.QtCore import QRegularExpression
    from PySide6.QtGui import (
        QColor,
        QFont,
        QSyntaxHighlighter,
        QTextCharFormat,
        QTextDocument,
    )

    QT_AVAILABLE = True
except ImportError:  # pragma: no cover - headless-окружение без PySide6
    QT_AVAILABLE = False

    def QSyntaxHighlighter(*_args: object, **_kwargs: object) -> Any:  # noqa: N802
        """Заглушка базового класса, когда PySide6 недоступен."""
        raise ImportError("для подсветки синтаксиса нужен PySide6")

__all__ = [
    "QT_AVAILABLE",
    "PythonHighlighter",
    "STATE_NONE",
    "STATE_TRIPLE_DOUBLE",
    "STATE_TRIPLE_SINGLE",
]

#: Имена, которые считаются «своими» для скрипта (выделяются красноватым).
SELF_NAMES: tuple[str, ...] = ("self", "cls")

#: Состояния блоков для многострочных строк.
STATE_NONE = -1
STATE_TRIPLE_DOUBLE = 1
STATE_TRIPLE_SINGLE = 2


def _names_pattern(names: Iterable[str]) -> str:
    """Собрать регулярное выражение «любое из имён» с границами слова."""
    return r"\b(?:" + "|".join(names) + r")\b"


def _keyword_pattern() -> str:
    """Регулярное выражение для ключевых слов Python."""
    return _names_pattern(KEYWORDS)


#: Правила подсветки: (регулярное выражение, имя токена).
#:
#: Порядок важен: комментарии и строки идут первыми, чтобы ключевые слова
#: внутри литералов не перекрашивались.
RULES: tuple[tuple[str, str], ...] = (
    (r"#[^\n]*", "comment"),
    (r'"""(?:[^"]|"(?!""))*(?:"""|$)', "string"),
    (r"'''(?:[^']|'(?!''))*(?:'''|$)", "string"),
    (r'"(?:[^"\\\n]|\\.)*"', "string"),
    (r"'(?:[^'\\\n]|\\.)*'", "string"),
    (r"\b(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\b", "number"),
    (r"\b0[xX][0-9a-fA-F_]+\b", "number"),
    (r"\b[0-9][0-9_]*\b", "number"),
    (r"@[A-Za-z_][A-Za-z0-9_.]*", "decorator"),
    (r"\b(?:def|class)\s+[A-Za-z_][A-Za-z0-9_]*", "definition"),
    (_keyword_pattern(), "keyword"),
    (_names_pattern(SELF_NAMES), "self"),
    (_names_pattern(BUILTIN_NAMES), "builtin"),
    (r"[+\-*/%=<>!&|^~]+", "operator"),
)


def _format_for(token: str) -> QTextCharFormat:
    """Собрать формат текста для токена.

    Args:
        token: Имя токена из :data:`~engine.scripting.syntax.TOKEN_COLORS`.

    Returns:
        Настроенный формат (цвет, а для ключевых слов — ещё и насыщенность).
    """
    char_format = QTextCharFormat()
    char_format.setForeground(QColor(TOKEN_COLORS[token]))
    if token in ("keyword", "definition", "decorator"):
        char_format.setFontWeight(QFont.Weight.DemiBold)
    elif token == "comment":
        char_format.setFontItalic(True)
    return char_format


if QT_AVAILABLE:

    class PythonHighlighter(QSyntaxHighlighter):
        """Подсветка синтаксиса Python с поддержкой API движка.

        Args:
            document: Документ, к которому привязывается подсветка.

        Attributes:
            api_names: Имена API движка, выделяемые особым цветом после ``self.``.
        """

        def __init__(self, document: QTextDocument) -> None:
            super().__init__(document)
            self.api_names: set[str] = set(ENGINE_API_ATTRIBUTES)
            self._formats: dict[str, QTextCharFormat] = {
                token: _format_for(token) for token in TOKEN_COLORS
            }
            self._rules: list[tuple[QRegularExpression, QTextCharFormat]] = [
                (QRegularExpression(pattern), self._formats[token])
                for pattern, token in RULES
            ]
            self._api_format = self._formats["api"]
            self._api_pattern = QRegularExpression(
                r"self\.([A-Za-z_][A-Za-z0-9_]*)"
            )

        # -------------------------------------------------------------- #
        # Настройка
        # -------------------------------------------------------------- #
        def set_api_names(self, names: Iterable[str]) -> None:
            """Задать имена API движка, которые нужно подсвечивать.

            Args:
                names: Набор имён (``game_object``, ``scene``, ...).
            """
            self.api_names = {str(name) for name in names}
            self.rehighlight()

        # -------------------------------------------------------------- #
        # Подсветка
        # -------------------------------------------------------------- #
        def highlightBlock(self, text: str) -> None:  # noqa: N802 - API Qt
            """Раскрасить один блок (строку) документа.

            Args:
                text: Текст блока без завершающего перевода строки.
            """
            for pattern, char_format in self._rules:
                iterator = pattern.globalMatch(text)
                while iterator.hasNext():
                    match = iterator.next()
                    length = match.capturedLength()
                    if length > 0:
                        self.setFormat(match.capturedStart(), length, char_format)
            self._highlight_api(text)
            self._highlight_multiline_strings(text)

        # -------------------------------------------------------------- #
        # Внутреннее
        # -------------------------------------------------------------- #
        def _highlight_api(self, text: str) -> None:
            """Подсветить ``self.<api>`` цветом API движка.

            Args:
                text: Текст блока.
            """
            iterator = self._api_pattern.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                attribute = match.captured(1)
                if attribute in self.api_names:
                    self.setFormat(
                        match.capturedStart(1), len(attribute), self._api_format
                    )

        def _highlight_multiline_strings(self, text: str) -> None:
            """Продолжить или закрыть многострочную строку.

            Понимает ``\"\"\"`` и ``'''``: если строка началась в предыдущем
            блоке и не закрылась, весь блок остаётся строкой, а состояние
            переносится на следующий блок.

            Args:
                text: Текст блока.
            """
            self.setCurrentBlockState(STATE_NONE)
            previous = self.previousBlockState()
            if previous not in (STATE_TRIPLE_DOUBLE, STATE_TRIPLE_SINGLE):
                return
            delimiter = '"""' if previous == STATE_TRIPLE_DOUBLE else "'''"
            index = text.find(delimiter)
            if index < 0:
                self.setFormat(0, len(text), self._formats["string"])
                self.setCurrentBlockState(previous)
                return
            self.setFormat(0, index + len(delimiter), self._formats["string"])

else:

    class PythonHighlighter:  # type: ignore[no-redef]
        """Заглушка подсветки: PySide6 не установлен.

        Raises:
            ImportError: Всегда — сообщает, что нужен extra ``editor``.
        """

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise ImportError(
                "подсветка синтаксиса требует PySide6: "
                "установите extra 'editor' (pip install myengine[editor])"
            )

