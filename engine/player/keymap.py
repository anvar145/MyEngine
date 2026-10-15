"""Соответствие кодов клавиш GLFW именам, которые видят скрипты."""

from __future__ import annotations

from collections.abc import Iterable

from engine.core.input import InputState
from engine.core.input.keys import KEY_F2, glfw_key_name

__all__ = ["KEY_F2", "glfw_key_name", "sync_input"]


def sync_input(
    state: InputState, keys: Iterable[int], extra: Iterable[str] = ()
) -> None:
    """Привести ``state`` к набору удерживаемых клавиш GLFW.

    Args:
        state: Общее состояние ввода скриптов.
        keys: Коды GLFW удерживаемых клавиш.
        extra: Имена клавиш, считающиеся нажатыми всегда (``--hold-keys``).
    """
    names = {name for key in keys if (name := glfw_key_name(key))}
    names.update(str(k).lower() for k in extra)
    state.pressed_keys.clear()
    state.pressed_keys.update(names)
