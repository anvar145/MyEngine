"""Данные подсветки и автодополнения Python-скриптов (без Qt).

Модуль намеренно не тянет PySide6: имена ключевых слов, встроенных функций и
публичного API движка нужны и слою редактора (подсветка, ``QCompleter``), и
тестам. Цвета хранятся рядом с токенами и сходятся с палитрой тёмной темы
``editor.theme.PALETTE``.

Example:
    >>> "on_update" in ENGINE_API
    True
    >>> "print" in completion_words()
    True
"""

from __future__ import annotations

import builtins
import keyword
from collections.abc import Iterable

__all__ = [
    "BUILTIN_NAMES",
    "ENGINE_API",
    "ENGINE_API_ATTRIBUTES",
    "KEYWORDS",
    "TOKEN_COLORS",
    "completion_words",
]

#: Цвета токенов (согласованы с тёмной темой редактора).
TOKEN_COLORS: dict[str, str] = {
    "keyword": "#d97757",
    "builtin": "#7fb08a",
    "definition": "#e8c07a",
    "string": "#a8c98a",
    "number": "#e0b25c",
    "comment": "#6f6a62",
    "decorator": "#e0905f",
    "self": "#d98a8a",
    "api": "#c9a27a",
    "operator": "#a39e95",
}

#: Ключевые слова Python (``True``/``False``/``None`` приходят из builtins).
KEYWORDS: tuple[str, ...] = tuple(keyword.kwlist)

#: Встроенные имена, которые имеет смысл дополнять в редакторе.
BUILTIN_NAMES: tuple[str, ...] = tuple(sorted(dir(builtins)))

#: Публичный API движка, доступный скрипту от лица ``self``.
ENGINE_API: tuple[str, ...] = (
    "self.game_object",
    "self.scene",
    "self.transform",
    "self.input",
    "self.physics",
    "self.time",
    "self.log",
    "self.name",
    "self.path",
    "self.instance",
    "self.delta_time",
    "self.debug",
    "self.warning",
    "self.error",
    "self.on_start",
    "self.on_update",
    "self.on_destroy",
    "self.on_reload",
)

#: Имена API без префикса ``self.`` — то, что видит ``QCompleter`` после точки.
ENGINE_API_ATTRIBUTES: tuple[str, ...] = (
    "debug",
    "delta_time",
    "error",
    "game_object",
    "input",
    "instance",
    "is_key_down",
    "pressed",
    "held",
    "released",
    "axis",
    "axis2",
    "checkpoint",
    "vec2",
    "contexts",
    "log",
    "name",
    "path",
    "physics",
    "pressed_keys",
    "scene",
    "time",
    "transform",
    "warning",
    "raycast",
    "overlap_sphere",
    "set_gravity",
    "apply_impulse",
    "is_grounded",
    "move_character",
    "simple_move",
    "jump",
)

#: Обратные вызовы жизненного цикла, которые поддерживает движок.
LIFECYCLE_HOOKS: tuple[str, ...] = (
    "on_start",
    "on_update",
    "on_fixed_update",
    "on_destroy",
    "on_reload",
)


def completion_words(*extra: Iterable[str]) -> list[str]:
    """Собрать словарь автодополнения.

    Args:
        *extra: Дополнительные наборы имён (например, имена из текущего файла).

    Returns:
        Отсортированный список уникальных слов для ``QCompleter``.
    """
    words: set[str] = set(KEYWORDS)
    words.update(BUILTIN_NAMES)
    words.update(ENGINE_API)
    words.update(ENGINE_API_ATTRIBUTES)
    words.update(LIFECYCLE_HOOKS)
    for group in extra:
        words.update(name for name in group if name)
    return sorted(words)
