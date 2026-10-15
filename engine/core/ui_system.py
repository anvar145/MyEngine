"""Ввод UI: наведение, нажатие и клик по кнопкам, события через ``scene.events``.

Без GL и Qt. Раскладку берёт из общей :class:`UILayout` (ту же, что рисует
рендер), мышь — из :class:`MouseState` в пикселях окна. Размер окна и масштаб
задаёт ``screen`` (``mouse_to_ui``, ``canvas_size``).

Порядок за вызов :meth:`UISystem.update`: события мыши по очереди (перемещение →
hover, нажатие → press, отпускание → release/click), затем проверка наведения для
текущей позиции (кнопка могла исчезнуть или стать неактивной).

Example:
    >>> system = UISystem(scene, screen, mouse)
    >>> mouse.move(100, 50)
    >>> system.update()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from engine.core.input import MouseState
from engine.core.ui_components import UIButton, UIText
from engine.core.ui_events import (
    SceneLoadRequested,
    SceneUnloadRequested,
    UIClick,
    UIEventFired,
    UIHoverEnter,
    UIHoverExit,
    UIPress,
    UIRelease,
)
from engine.core.ui_layout import UILayout, UILayoutItem

if TYPE_CHECKING:
    from engine.core.scene import Scene

__all__ = ["UISystem"]

_log = logging.getLogger("engine.core.ui_system")


class _ScreenLike(Protocol):
    @property
    def canvas_size(self) -> tuple[float, float]: ...

    def mouse_to_ui(self, x_px: float, y_px: float) -> tuple[float, float]: ...


def _contains(rect: tuple[float, float, float, float], x: float, y: float) -> bool:
    return rect[0] <= x < rect[0] + rect[2] and rect[1] <= y < rect[1] + rect[3]


class UISystem:
    """Обработка мыши для UI сцены.

    Args:
        scene: Сцена с виджетами (события уходят в ``scene.events``).
        screen: Экран UI (``UIScreen``): перевод мыши в единицы UI.
        mouse: Состояние мыши.
        layout: Общая раскладка (по умолчанию своя).
        measure: Функция размера текста, если раскладка общая с рендером.

    Attributes:
        hovered: ``GameObject.id`` кнопки под курсором или ``None``.
        pressed: ``GameObject.id`` кнопки, на которой нажата левая кнопка.
    """

    def __init__(
        self,
        scene: Scene,
        screen: _ScreenLike,
        mouse: MouseState,
        layout: UILayout | None = None,
        measure: Callable[[UIText], tuple[float, float]] | None = None,
    ) -> None:
        self.scene = scene
        self.screen = screen
        self.mouse = mouse
        self.layout = UILayout() if layout is None else layout
        self.measure = measure
        self.hovered: int | None = None
        self.pressed: int | None = None
        self._buttons: dict[int, UIButton] = {}

    # ------------------------------------------------------------------ #
    def update(self) -> None:
        """Обработать накопленные события мыши и обновить состояния кнопок."""
        for event in self.mouse.drain():
            if event.kind == "move":
                self._refresh_hover(event.x, event.y)
            elif event.kind == "down" and event.button == "left":
                self._refresh_hover(event.x, event.y)
                self._press()
            elif event.kind == "up" and event.button == "left":
                self._refresh_hover(event.x, event.y)
                self._release()
        # Кнопка под курсором могла исчезнуть или стать выключенной.
        self._refresh_hover(self.mouse.x, self.mouse.y)

    # ------------------------------------------------------------------ #
    def _hit(self, x_px: float, y_px: float) -> UILayoutItem | None:
        """Верхняя доступная кнопка под точкой (пиксели окна)."""
        items = self.layout.update(self.scene, self.screen.canvas_size, self.measure)
        self._buttons = {
            item.game_object.id: item.element
            for item in items
            if isinstance(item.element, UIButton)
        }
        x, y = self.screen.mouse_to_ui(x_px, y_px)
        for item in reversed(items):
            element = item.element
            if not isinstance(element, UIButton):
                continue
            if not (item.enabled and element.interactive):
                continue
            if not _contains(item.rect, x, y):
                continue
            if item.clip is not None and not _contains(item.clip, x, y):
                continue
            return item
        return None

    def _refresh_hover(self, x_px: float, y_px: float) -> None:
        hit = self._hit(x_px, y_px) if self.mouse.inside else None
        new = None if hit is None else hit.game_object.id
        if new != self.hovered:
            old = self.hovered
            self.hovered = new
            if old is not None:
                self._emit(UIHoverExit, old)
            if new is not None:
                self._emit(UIHoverEnter, new)
        self._sync_states()

    def _press(self) -> None:
        if self.hovered is None:
            return
        self.pressed = self.hovered
        self._emit(UIPress, self.pressed)
        self._sync_states()

    def _release(self) -> None:
        pressed, self.pressed = self.pressed, None
        if pressed is None:
            return
        self._emit(UIRelease, pressed)
        if pressed == self.hovered:
            self._emit(UIClick, pressed)
            self._run_action(pressed)
        self._sync_states()

    def _sync_states(self) -> None:
        """Выставить ``UIButton.state`` (только при реальном изменении)."""
        for widget_id, button in self._buttons.items():
            if widget_id == self.pressed and widget_id == self.hovered:
                state = "pressed"
            elif widget_id == self.hovered:
                state = "hover"
            else:
                state = "normal"
            if button.state != state:
                button.state = state

    def _name(self, widget_id: int) -> str:
        game_object = self.scene.get_game_object(widget_id)
        return "" if game_object is None else game_object.name

    def _emit(self, event_type: type, widget_id: int) -> None:
        self.scene.events.emit(event_type(widget_id, self._name(widget_id)))

    def _run_action(self, widget_id: int) -> None:
        button = self._buttons.get(widget_id)
        if button is None or not button.on_click:
            return
        action = button.on_click
        if action.startswith("event:"):
            self.scene.events.emit(UIEventFired(action[6:], widget_id))
        elif action.startswith("load_scene:"):
            self.scene.events.emit(SceneLoadRequested(action[11:]))
        elif action.startswith("load_additive:"):
            self.scene.events.emit(SceneLoadRequested(action[14:], additive=True))
        elif action.startswith("unload_scene:"):
            self.scene.events.emit(SceneUnloadRequested(action[13:]))
        else:
            _log.debug("UI: действие '%s' пока не поддерживается", action)
