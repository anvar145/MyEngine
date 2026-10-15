"""Раскладка UI-иерархии: прямоугольники, слои, прозрачность, отсечение.

Без GL и Qt: общий расчёт для рендера (``engine.rendering.ui``) и ввода
(``engine.core.ui_system``). Родителем виджета считается ближайший предок с
``UIElement`` (иначе — холст); позиция ``Transform`` игнорируется.

Порядок рисования — по ``(суммарный sorting_layer, порядок обхода)``: дети
рисуются поверх родителя, поздние соседи — поверх ранних.

Example:
    >>> layout = UILayout()
    >>> for item in layout.update(scene, (1280.0, 720.0)):
    ...     print(item.game_object.name, item.rect)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from engine.core.components import TextRenderer
from engine.core.ui_components import UIElement, UIText

if TYPE_CHECKING:
    from engine.core.game_object import GameObject
    from engine.core.scene import Scene

__all__ = ["UILayout", "UILayoutItem", "resolve_rect"]

Rect = tuple[float, float, float, float]

#: Доли ширины/высоты родителя для каждого якоря.
_ANCHOR_FRACTIONS: dict[str, tuple[float, float]] = {
    "top_left": (0.0, 0.0),
    "top_center": (0.5, 0.0),
    "top_right": (1.0, 0.0),
    "middle_left": (0.0, 0.5),
    "middle_center": (0.5, 0.5),
    "middle_right": (1.0, 0.5),
    "bottom_left": (0.0, 1.0),
    "bottom_center": (0.5, 1.0),
    "bottom_right": (1.0, 1.0),
}


def resolve_rect(
    anchor: str,
    pivot: tuple[float, float],
    position: tuple[float, float],
    size: tuple[float, float],
    parent: Rect,
) -> Rect:
    """Прямоугольник ``(x, y, w, h)``: левый верх = якорь + position − pivot·size."""
    fx, fy = _ANCHOR_FRACTIONS[anchor]
    px, py, pw, ph = parent
    return (
        px + fx * pw + position[0] - pivot[0] * size[0],
        py + fy * ph + position[1] - pivot[1] * size[1],
        size[0],
        size[1],
    )


def _intersect(a: Rect, b: Rect) -> Rect:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    return (left, top, max(0.0, right - left), max(0.0, bottom - top))


@dataclass(frozen=True)
class UILayoutItem:
    """Рассчитанный виджет.

    Attributes:
        game_object: Объект-владелец.
        element: Компонент виджета.
        rect: Прямоугольник ``(x, y, w, h)`` в единицах UI.
        clip: Прямоугольник отсечения или ``None``.
        layer: Суммарный слой сортировки.
        opacity: Итоговая непрозрачность (произведение по цепочке).
        order: Номер в порядке обхода.
        enabled: Включён ли весь путь до корня (для ввода).
    """

    game_object: GameObject
    element: UIElement
    rect: Rect
    clip: Rect | None
    layer: int
    opacity: float
    order: int
    enabled: bool


class UILayout:
    """Кэшируемая раскладка UI сцены.

    Пересчёт происходит только при смене структуры, активности, полей UI,
    текстовых параметров или размера холста.
    """

    def __init__(self) -> None:
        self._token: object = None
        self._items: list[UILayoutItem] = []

    def update(
        self,
        scene: Scene,
        canvas: tuple[float, float],
        measure: Callable[[UIText], tuple[float, float]] | None = None,
    ) -> list[UILayoutItem]:
        """Вернуть раскладку, пересчитав при изменениях.

        Args:
            scene: Сцена.
            canvas: Размер холста в единицах UI.
            measure: Размер текста по содержимому (для ``size == (0, 0)``).

        Returns:
            Видимые виджеты в порядке рисования (снизу вверх).
        """
        token = (
            id(scene),
            scene.structure_version,
            scene.activity_version,
            UIElement.revision,
            TextRenderer.revision,
            canvas,
            measure,
        )
        if token == self._token:
            return self._items
        items: list[UILayoutItem] = []
        canvas_rect: Rect = (0.0, 0.0, float(canvas[0]), float(canvas[1]))
        for root in scene.root_objects:
            self._walk(root, canvas_rect, None, 0, 1.0, True, items, measure)
        # Сортировка устойчивая: при равных слоях сохраняется порядок обхода.
        items.sort(key=lambda item: item.layer)
        self._items = items
        self._token = token
        return items

    def _walk(
        self,
        game_object: GameObject,
        parent_rect: Rect,
        parent_clip: Rect | None,
        parent_layer: int,
        parent_opacity: float,
        parent_enabled: bool,
        items: list[UILayoutItem],
        measure: Callable[[UIText], tuple[float, float]] | None,
    ) -> None:
        if not game_object.active_in_hierarchy:
            return
        element = _find_element(game_object)
        rect, clip = parent_rect, parent_clip
        layer, opacity, enabled = parent_layer, parent_opacity, parent_enabled
        if element is not None:
            if not element.visible:
                return
            size = element.size
            if size == (0.0, 0.0) and isinstance(element, UIText) and measure:
                size = measure(element)
            rect = resolve_rect(
                element.anchor, element.pivot, element.position, size, parent_rect
            )
            layer = parent_layer + element.sorting_layer
            opacity = parent_opacity * element.opacity
            enabled = parent_enabled and element.enabled
            clip = parent_clip
            if element.clip:
                clip = rect if clip is None else _intersect(clip, rect)
            items.append(
                UILayoutItem(
                    game_object,
                    element,
                    rect,
                    clip,
                    layer,
                    opacity,
                    len(items),
                    enabled,
                )
            )
        for child in game_object.children:
            self._walk(child, rect, clip, layer, opacity, enabled, items, measure)


def _find_element(game_object: GameObject) -> UIElement | None:
    for component in game_object.components:
        if isinstance(component, UIElement):
            return component
    return None
