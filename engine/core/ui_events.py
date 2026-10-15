"""События UI, перехода между сценами и выхода.

Все — обычные классы-события для ``scene.events`` (:class:`EventBus`), подписка
по типу: ``self.scene.events.subscribe(UIClick, handler)``.

``UIEventFired`` несёт имя из ``UIButton.on_click = "event:<имя>"``; остальные
обработчики (``SceneManager``, плеер) фильтруют свои типы.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.core.event_bus import Event

__all__ = [
    "QuitRequested",
    "RenderScaleChanged",
    "SceneActivated",
    "SceneLoadProgress",
    "SceneLoadRequested",
    "SceneUnloadRequested",
    "SceneLoaded",
    "SceneUnloaded",
    "UIClick",
    "UIEventFired",
    "UIHoverEnter",
    "UIHoverExit",
    "UIPress",
    "UIRelease",
]


@dataclass(slots=True)
class UIHoverEnter(Event):
    """Курсор вошёл в кнопку.

    Attributes:
        widget_id: Идентификатор ``GameObject`` виджета.
        name: Имя ``GameObject`` виджета.
    """

    widget_id: int
    name: str = ""


@dataclass(slots=True)
class UIHoverExit(Event):
    """Курсор покинул кнопку (поля как у :class:`UIHoverEnter`)."""

    widget_id: int
    name: str = ""


@dataclass(slots=True)
class UIPress(Event):
    """Левая кнопка нажата на кнопке UI (поля как у :class:`UIHoverEnter`)."""

    widget_id: int
    name: str = ""


@dataclass(slots=True)
class UIRelease(Event):
    """Левая кнопка отпущена после нажатия на этой кнопке."""

    widget_id: int
    name: str = ""


@dataclass(slots=True)
class UIClick(Event):
    """Нажатие и отпускание на одной и той же кнопке."""

    widget_id: int
    name: str = ""


@dataclass(slots=True)
class UIEventFired(Event):
    """Сработал ``on_click = "event:<имя>"``.

    Attributes:
        event_name: Имя из ``on_click`` без префикса.
        widget_id: ``GameObject`` кнопки.
    """

    event_name: str
    widget_id: int = 0


@dataclass(slots=True)
class SceneLoadRequested(Event):
    """Запрошен переход на сцену (``on_click = "load_scene:<путь>"``).

    Attributes:
        path: Путь к файлу сцены (как в ``on_click``).
        additive: Загрузить рядом с активной (``load_additive:<путь>``).
    """

    path: str
    additive: bool = False


@dataclass(slots=True)
class SceneUnloadRequested(Event):
    """Запрошена выгрузка дополнительной сцены (``unload_scene:<путь>``).

    Attributes:
        path: Путь к файлу сцены.
    """

    path: str


@dataclass(slots=True)
class QuitRequested(Event):
    """Запрошен выход из игры (плеер останавливает цикл)."""


@dataclass(slots=True)
class SceneLoaded(Event):
    """Сцена загружена менеджером.

    Attributes:
        name: Имя сцены.
        path: Путь к файлу.
    """

    name: str
    path: str = ""
    additive: bool = False


@dataclass(slots=True)
class SceneUnloaded(Event):
    """Сцена выгружена менеджером (поля как у :class:`SceneLoaded`)."""

    name: str
    path: str = ""
    additive: bool = False


@dataclass(slots=True)
class SceneActivated(Event):
    """Сцена стала активной (поля как у :class:`SceneLoaded`)."""

    name: str
    path: str = ""
    additive: bool = False


@dataclass(slots=True)
class SceneLoadProgress(Event):
    """Шаг асинхронной загрузки сцены (экран загрузки рисует приложение).

    Attributes:
        operation: ``SceneLoadOperation`` (``cancel()``, ``result()``).
        progress: Общий прогресс ``0..1``.
        phase: ``"parse"`` | ``"build"`` | ``"activate"``.
        path: Файл сцены.
    """

    operation: object
    progress: float
    phase: str
    path: str = ""


@dataclass(slots=True)
class RenderScaleChanged(Event):
    """Динамическое разрешение сменило долю разрешения сцены.

    Attributes:
        old: Предыдущий ``render_scale``.
        new: Новый ``render_scale`` (0.25–1.0).
    """

    old: float
    new: float
