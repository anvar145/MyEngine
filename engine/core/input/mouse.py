"""Состояние мыши: позиция, кнопки и очередь событий за кадр.

Данные без Qt и GL. Источник (GLFW-окно плеера, IPC редактора, тест) вызывает
:meth:`MouseState.move` / :meth:`MouseState.button`; потребитель (``UISystem``)
забирает накопленные события через :meth:`MouseState.drain` в порядке прихода,
поэтому нажатие и отпускание в одном кадре не теряются.

Координаты — пиксели окна (цели), начало слева сверху.

Example:
    >>> mouse = MouseState()
    >>> mouse.move(10, 20)
    >>> mouse.button("left", True)
    >>> [event.kind for event in mouse.drain()]
    ['move', 'down']
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MOUSE_BUTTONS", "MouseEvent", "MouseState"]

#: Допустимые имена кнопок.
MOUSE_BUTTONS: tuple[str, ...] = ("left", "right", "middle")


@dataclass(frozen=True, slots=True)
class MouseEvent:
    """Событие мыши.

    Attributes:
        kind: ``move``, ``down`` или ``up``.
        x: Позиция курсора в пикселях окна.
        y: Позиция курсора в пикселях окна.
        button: Имя кнопки для ``down``/``up``, иначе ``""``.
    """

    kind: str
    x: float
    y: float
    button: str = ""


class MouseState:
    """Текущее состояние мыши и очередь необработанных событий.

    Attributes:
        x: Последняя позиция по X (пиксели окна).
        y: Последняя позиция по Y.
        inside: Находится ли курсор в окне.
    """

    def __init__(self) -> None:
        self.x = -1.0
        self.y = -1.0
        self.inside = False
        self._down: set[str] = set()
        self._events: list[MouseEvent] = []

    @property
    def position(self) -> tuple[float, float]:
        """Позиция курсора ``(x, y)``."""
        return (self.x, self.y)

    def is_down(self, button: str = "left") -> bool:
        """Удерживается ли кнопка."""
        return button in self._down

    def move(self, x: float, y: float) -> None:
        """Курсор сместился в ``(x, y)``."""
        self.x, self.y = float(x), float(y)
        self.inside = True
        self._events.append(MouseEvent("move", self.x, self.y))

    def leave(self) -> None:
        """Курсор покинул окно."""
        self.inside = False
        self.x = self.y = -1.0
        self._events.append(MouseEvent("move", self.x, self.y))

    def button(self, name: str, down: bool) -> None:
        """Кнопка нажата или отпущена в текущей позиции.

        Raises:
            ValueError: Если имя кнопки неизвестно.
        """
        if name not in MOUSE_BUTTONS:
            raise ValueError(f"Неизвестная кнопка мыши: {name!r}")
        if down:
            self._down.add(name)
        else:
            self._down.discard(name)
        self._events.append(
            MouseEvent("down" if down else "up", self.x, self.y, name)
        )

    def drain(self) -> list[MouseEvent]:
        """Забрать накопленные события (очередь очищается)."""
        events, self._events = self._events, []
        return events

    def __len__(self) -> int:
        return len(self._events)
