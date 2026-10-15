"""Бэкенд ввода GLFW: колбэки окна → :class:`InputState`."""

from __future__ import annotations

from engine.core.input import InputState
from engine.core.input.keys import glfw_key_name

__all__ = ["GlfwInputBackend"]


class GlfwInputBackend:
    """Превращает события GLFW в ``key_down``/``key_up``/``mouse_*``.

    Окно (:class:`~engine.player.window.GlfwWindow`) вызывает ``on_*`` из своих
    колбэков. Геймпад подключается через ``state.gamepad_*`` (п.4, пока только
    структура).

    Args:
        state: Общее состояние ввода.
    """

    def __init__(self, state: InputState) -> None:
        self.state = state
        self._last_cursor: tuple[float, float] | None = None

    def poll(self) -> None:
        """События приходят колбэками; опрашивать нечего."""

    def on_key(self, key: int, down: bool) -> None:
        """Клавиша GLFW нажата или отпущена."""
        name = glfw_key_name(key)
        if name is None:
            return
        if down:
            self.state.key_down(name)
        else:
            self.state.key_up(name)

    def on_mouse_button(self, name: str, down: bool) -> None:
        """Кнопка мыши нажата или отпущена."""
        self.state.mouse_button(name, down)

    def on_cursor(self, x: float, y: float) -> None:
        """Курсор в точке ``(x, y)``: накапливается дельта кадра."""
        if self._last_cursor is not None:
            self.state.mouse_move(x - self._last_cursor[0], y - self._last_cursor[1])
        self._last_cursor = (x, y)

    def on_cursor_leave(self) -> None:
        """Курсор покинул окно: следующая дельта не должна «прыгать»."""
        self._last_cursor = None

    def end_frame(self) -> None:
        """Конец кадра игры."""
        self.state.end_frame()
