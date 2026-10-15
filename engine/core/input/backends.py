"""Бэкенды ввода: источники событий для :class:`InputState`.

Один интерфейс :class:`InputBackend`; реализации живут там, где есть нужная
библиотека: GLFW - в ``engine.player``, Qt - в ``editor``. Здесь - только
протокол и :class:`ManualInputBackend` для тестов, IPC и headless.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from engine.core.input.state import InputState

__all__ = ["InputBackend", "ManualInputBackend"]


class InputBackend(Protocol):
    """Источник ввода, пишущий в общий :class:`InputState`."""

    state: InputState

    def poll(self) -> None:
        """Обновить состояние из устройства (вызывается раз в кадр)."""


class ManualInputBackend:
    """Ручной бэкенд: события задаются кодом.

    Args:
        state: Состояние, в которое пишутся события (по умолчанию - новое).
    """

    def __init__(self, state: InputState | None = None) -> None:
        self.state = state if state is not None else InputState()
        self._held: set[str] = set()

    def poll(self) -> None:
        """Ничего не опрашивает: события приходят вызовами методов."""

    def key_down(self, *keys: str) -> None:
        """Нажать клавиши."""
        for key in keys:
            self.state.key_down(key)

    def key_up(self, *keys: str) -> None:
        """Отпустить клавиши."""
        for key in keys:
            self.state.key_up(key)

    def feed_key(self, name: str, down: bool = True) -> None:
        """Нажать (``down=True``) или отпустить клавишу по имени."""
        if down:
            self.state.key_down(name)
        else:
            self.state.key_up(name)

    def hold(self, keys: Iterable[str]) -> None:
        """Удерживать ровно ``keys`` (``--hold-keys``), раз в кадр.

        Клавиши, нажатые другими источниками, не трогаются: отпускаются только
        те, что этот бэкенд сам нажал раньше.
        """
        wanted = {str(k).lower() for k in keys}
        for key in sorted(self._held - wanted):
            self.state.key_up(key)
        for key in sorted(wanted - self._held):
            self.state.key_down(key)
        # Повторная инъекция каждый кадр: клавиша могла быть снята окном.
        self.state.pressed_keys.update(wanted)
        self._held = wanted

    def mouse_button(self, name: str, down: bool) -> None:
        """Нажать или отпустить кнопку мыши."""
        self.state.mouse_button(name, down)

    def mouse_move(self, dx: float, dy: float) -> None:
        """Сместить мышь."""
        self.state.mouse_move(dx, dy)

    def gamepad_button(self, name: str, down: bool) -> None:
        """Нажать или отпустить кнопку геймпада."""
        self.state.gamepad_button(name, down)

    def gamepad_axis(self, name: str, value: float) -> None:
        """Выставить значение оси геймпада."""
        self.state.gamepad_axis(name, value)

    def end_frame(self) -> None:
        """Завершить кадр."""
        self.state.end_frame()
