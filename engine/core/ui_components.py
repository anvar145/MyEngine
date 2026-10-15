"""UI-компоненты: данные экранных виджетов (панель, картинка, текст, кнопка).

Только данные: ни OpenGL, ни Qt. Экранную раскладку и рисование выполняет
``engine.rendering.ui``, мышь и события — ``engine.core.ui_system``. Позиция
``GameObject`` на экран не влияет: объект служит контейнером иерархии, а
``sorting_layer`` и ``opacity`` детей накладываются на значения родителя.

Валидация выполняется при любой записи поля (в том числе из инспектора), поэтому
в компоненте не бывает неверных ``anchor``/``pivot``/``opacity``/``size``.

Example:
    >>> panel = scene.create_game_object("Panel").add_component(UIPanel)
    >>> panel.size = (200.0, 100.0)
"""

from __future__ import annotations

from collections.abc import Callable

from engine.core.component import Component
from engine.core.components import TextRenderer

__all__ = [
    "UI_ANCHORS",
    "UIButton",
    "UIElement",
    "UIImage",
    "UIPanel",
    "UIText",
]

#: Допустимые якоря.
UI_ANCHORS: tuple[str, ...] = (
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)

#: Допустимые префиксы ``UIButton.on_click``.
CLICK_PREFIXES: tuple[str, ...] = (
    "event:",
    "load_scene:",
    "load_additive:",
    "unload_scene:",
    "script:",
)


def _floats(value: object, count: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(v) for v in value)  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        raise ValueError(f"{name}: ожидаются числа") from None
    if len(result) != count:
        raise ValueError(f"{name}: ожидается {count} числа(ел)")
    return result


def _position(value: object) -> tuple[float, float]:
    return _floats(value, 2, "position")  # type: ignore[return-value]


def _size(value: object) -> tuple[float, float]:
    width, height = _floats(value, 2, "size")
    if width < 0 or height < 0:
        raise ValueError("size не может быть отрицательным")
    return (width, height)


def _pivot(value: object) -> tuple[float, float]:
    x, y = _floats(value, 2, "pivot")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("pivot должен быть в диапазоне [0, 1]")
    return (x, y)


def _anchor(value: object) -> str:
    if value not in UI_ANCHORS:
        raise ValueError(f"anchor должен быть одним из {UI_ANCHORS}")
    return str(value)


def _opacity(value: object) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not 0.0 <= number <= 1.0:
        raise ValueError("opacity должна быть в диапазоне [0, 1]")
    return number


def _rgba(value: object) -> tuple[float, float, float, float]:
    color = _floats(value, 4, "color")
    if any(not 0.0 <= c <= 1.0 for c in color):
        raise ValueError("компоненты цвета должны быть в диапазоне [0, 1]")
    return color  # type: ignore[return-value]


def _uv_rect(value: object) -> tuple[float, float, float, float]:
    return _floats(value, 4, "uv_rect")  # type: ignore[return-value]


def _non_negative(value: object) -> float:
    number = float(value)  # type: ignore[arg-type]
    if number < 0:
        raise ValueError("значение не может быть отрицательным")
    return number


def _positive_int(value: object) -> int:
    number = int(value)  # type: ignore[call-overload]
    if number <= 0:
        raise ValueError("значение должно быть положительным")
    return number


def _on_click(value: object) -> str:
    text = str(value)
    if text and not text.startswith(CLICK_PREFIXES):
        raise ValueError(f"on_click должен начинаться с одного из {CLICK_PREFIXES}")
    return text


#: Нормализаторы полей: имя → функция, возвращающая допустимое значение.
_NORMALIZERS: dict[str, Callable[[object], object]] = {
    "position": _position,
    "size": _size,
    "anchor": _anchor,
    "pivot": _pivot,
    "visible": bool,
    "sorting_layer": int,  # type: ignore[dict-item]
    "clip": bool,
    "opacity": _opacity,
    "color": _rgba,
    "border_color": _rgba,
    "border_width": _non_negative,
    "texture_path": str,
    "uv_rect": _uv_rect,
    "normal_color": _rgba,
    "hover_color": _rgba,
    "pressed_color": _rgba,
    "text_color": _rgba,
    "font_path": str,
    "size_px": _positive_int,
    "on_click": _on_click,
    "interactive": bool,
}

_TUPLE_FIELDS = frozenset(
    {
        "position",
        "size",
        "pivot",
        "color",
        "border_color",
        "uv_rect",
        "normal_color",
        "hover_color",
        "pressed_color",
        "text_color",
    }
)


def _slots(cls: type) -> tuple[str, ...]:
    """Все публичные слоты класса вместе с родительскими."""
    names: list[str] = ["enabled"] if issubclass(cls, UIElement) else []
    for klass in reversed(cls.__mro__):
        for name in getattr(klass, "__slots__", ()):
            if not name.startswith("_") and name not in names:
                names.append(name)
    return tuple(names)


class UIElement(Component):
    """Общие поля экранного виджета.

    Args:
        position: Смещение от якоря в единицах UI.
        size: Размер ``(w, h)``; ``0`` у текста означает «по содержимому».
        anchor: Точка родителя, к которой привязан виджет.
        pivot: Точка внутри виджета (0..1), которая ставится в якорь.
        visible: Рисовать ли виджет (и его детей).
        enabled: Реагирует ли на мышь.
        sorting_layer: Слой (больше — выше; складывается с родительским).
        clip: Обрезать содержимое по границам виджета.
        opacity: Непрозрачность 0..1 (перемножается с родительской).

    Raises:
        ValueError: Если значение недопустимо.
    """

    __slots__ = (
        "anchor",
        "clip",
        "opacity",
        "pivot",
        "position",
        "size",
        "sorting_layer",
        "visible",
    )

    #: Глобальный счётчик правок любых UI-компонентов (для кэшей рендера).
    revision = 0

    def __init__(
        self,
        position: tuple[float, float] = (0.0, 0.0),
        size: tuple[float, float] = (0.0, 0.0),
        anchor: str = "top_left",
        pivot: tuple[float, float] = (0.0, 0.0),
        *,
        visible: bool = True,
        enabled: bool = True,
        sorting_layer: int = 0,
        clip: bool = False,
        opacity: float = 1.0,
    ) -> None:
        super().__init__()
        self.position = position
        self.size = size
        self.anchor = anchor
        self.pivot = pivot
        self.visible = visible
        self.enabled = enabled
        self.sorting_layer = sorting_layer
        self.clip = clip
        self.opacity = opacity

    def _attach(self, game_object) -> None:
        """Привязка к объекту; ``enabled`` из конструктора не сбрасывается."""
        enabled = self._enabled
        super()._attach(game_object)
        self._enabled = enabled

    def __setattr__(self, name: str, value: object) -> None:
        normalizer = _NORMALIZERS.get(name)
        if normalizer is not None:
            value = normalizer(value)
        super().__setattr__(name, value)
        UIElement.revision += 1

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент в словарь."""
        data: dict[str, object] = {}
        for name in _slots(type(self)):
            value = getattr(self, name)
            data[name] = list(value) if name in _TUPLE_FIELDS else value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]):
        """Восстановить компонент; отсутствующие поля берутся по умолчанию."""
        kwargs = {name: data[name] for name in _slots(cls) if name in data}
        return cls(**kwargs)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} anchor={self.anchor} size={self.size}>"


class UIPanel(UIElement):
    """Залитый прямоугольник с необязательной рамкой.

    Args:
        color: Цвет заливки RGBA.
        border_color: Цвет рамки RGBA.
        border_width: Толщина рамки в единицах UI (0 — без рамки).
    """

    __slots__ = ("border_color", "border_width", "color")

    def __init__(
        self,
        *,
        color: tuple[float, float, float, float] = (0.15, 0.15, 0.2, 0.9),
        border_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        border_width: float = 0.0,
        size: tuple[float, float] = (100.0, 100.0),
        **element: object,
    ) -> None:
        super().__init__(size=size, **element)  # type: ignore[arg-type]
        self.color = color
        self.border_color = border_color
        self.border_width = border_width


class UIImage(UIElement):
    """Картинка из файла проекта (путь относительно корня проекта).

    Args:
        texture_path: Путь к PNG/JPEG; пустой — белый квад.
        uv_rect: Прямоугольник текстуры ``(u0, v0, u1, v1)``; ``v0`` — верх.
        color: Множитель цвета RGBA (tint).
    """

    __slots__ = ("color", "texture_path", "uv_rect")

    def __init__(
        self,
        *,
        texture_path: str = "",
        uv_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        size: tuple[float, float] = (100.0, 100.0),
        **element: object,
    ) -> None:
        super().__init__(size=size, **element)  # type: ignore[arg-type]
        self.texture_path = texture_path
        self.uv_rect = uv_rect
        self.color = color


class UIText(UIElement):
    """Текст UI: ``UIElement`` + экземпляр :class:`TextRenderer` (вариант А).

    Поля текста (строка, шрифт, размер, цвет, alpha, выравнивание, перенос,
    обводка, жирность) не дублируются: берутся из ``text``. ``billboard`` и
    ``units_per_pixel`` в UI не используются; ``wrap_width`` — в единицах UI.
    Размер ``(0, 0)`` — по границам текста; иначе текст выравнивается внутри
    прямоугольника виджета по ``align``/``vertical_align``.

    Args:
        text: Текстовые параметры (``TextRenderer`` или просто строка).
    """

    __slots__ = ("text",)

    def __init__(self, text: TextRenderer | str = "Text", **element: object) -> None:
        super().__init__(**element)  # type: ignore[arg-type]
        if isinstance(text, str):
            text = TextRenderer(text=text, billboard=False)
        self.text = text

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент (текст — вложенным словарём)."""
        data = super().to_dict()
        data["text"] = self.text.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]):
        """Восстановить компонент."""
        kwargs = {name: data[name] for name in _slots(cls) if name in data}
        text = kwargs.pop("text", None)
        component = cls(**kwargs)  # type: ignore[arg-type]
        if isinstance(text, dict):
            component.text = TextRenderer.from_dict(text)
        return component


class UIButton(UIElement):
    """Кнопка: фон по состоянию и центрированная подпись.

    Args:
        normal_color: Цвет в покое.
        hover_color: Цвет под курсором.
        pressed_color: Цвет при нажатии.
        text: Подпись.
        text_color: Цвет подписи RGBA.
        font_path: Шрифт (``""`` — по умолчанию).
        size_px: Размер шрифта в пикселях.
        on_click: ``event:<имя>`` — событие ``UIEventFired``;
            ``load_scene:<путь>`` — переход на сцену; ``script:`` зарезервирован.
        interactive: Реагирует ли на мышь (в дополнение к ``enabled``).

    Attributes:
        state: Служебное состояние ``normal``/``hover``/``pressed`` (не
            сериализуется; меняет ``UISystem``).
    """

    __slots__ = (
        "font_path",
        "hover_color",
        "interactive",
        "normal_color",
        "on_click",
        "pressed_color",
        "size_px",
        "state",
        "text",
        "text_color",
    )

    def __init__(
        self,
        *,
        normal_color: tuple[float, float, float, float] = (0.25, 0.3, 0.45, 1.0),
        hover_color: tuple[float, float, float, float] = (0.35, 0.42, 0.62, 1.0),
        pressed_color: tuple[float, float, float, float] = (0.15, 0.2, 0.32, 1.0),
        text: str = "Button",
        text_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        font_path: str = "",
        size_px: int = 24,
        on_click: str = "",
        interactive: bool = True,
        size: tuple[float, float] = (160.0, 40.0),
        **element: object,
    ) -> None:
        super().__init__(size=size, **element)  # type: ignore[arg-type]
        self.normal_color = normal_color
        self.hover_color = hover_color
        self.pressed_color = pressed_color
        self.text = str(text)
        self.text_color = text_color
        self.font_path = font_path
        self.size_px = size_px
        self.on_click = on_click
        self.interactive = interactive
        self.state = "normal"

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент (без служебного ``state``)."""
        data = super().to_dict()
        del data["state"]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]):
        """Восстановить компонент."""
        kwargs = {
            name: data[name] for name in _slots(cls) if name in data and name != "state"
        }
        return cls(**kwargs)  # type: ignore[arg-type]

    @property
    def current_color(self) -> tuple[float, float, float, float]:
        """Цвет фона для текущего состояния."""
        if self.state == "pressed":
            return self.pressed_color
        if self.state == "hover":
            return self.hover_color
        return self.normal_color
