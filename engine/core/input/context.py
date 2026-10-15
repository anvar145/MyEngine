"""Контексты ввода: кто сейчас получает действия.

Стек контекстов упорядочен по приоритету. Контекст разрешает действия из
своего набора ``actions`` (``None`` - любые). Если действия в наборе нет, а
контекст ``absorb`` - он поглощает его и нижние контексты ничего не получают;
иначе проверка идёт дальше вниз. Пустой стек разрешает всё.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "SYSTEM_ACTIONS",
    "UI_ACTIONS",
    "InputContext",
    "InputContextStack",
    "cutscene_context",
    "game_context",
    "paused_context",
    "ui_context",
]

#: Действия интерфейса (разрешены в контексте UI).
UI_ACTIONS: frozenset[str] = frozenset({"UIClick", "UICancel"})

#: Действия, которые должны работать поверх любого меню: игрок закрывает
#: меню той же клавишей, которой его открыл (``Pause``).
SYSTEM_ACTIONS: frozenset[str] = frozenset({"Pause"})


@dataclass(frozen=True, slots=True)
class InputContext:
    """Контекст ввода.

    Контекст с ``absorb=True`` блокирует всё, чего нет в его ``actions``, поэтому
    ``ui`` и ``paused`` явно включают ``Pause`` (:data:`SYSTEM_ACTIONS`): иначе
    открытое меню нельзя закрыть той же клавишей. Нужен ещё один сквозной
    набор (скриншот, консоль) - добавьте контекст с высоким приоритетом через
    :meth:`InputContextStack.push`.

    Attributes:
        name: Уникальное имя в стеке.
        priority: Чем больше, тем раньше проверяется.
        actions: Разрешённые действия или ``None`` (все).
        absorb: Поглощать остальные действия (не отдавать нижним контекстам).
    """

    name: str
    priority: int = 0
    actions: frozenset[str] | None = None
    absorb: bool = False


def game_context() -> InputContext:
    """Игровой ввод: приоритет 0, все действия."""
    return InputContext("game", 0, None, False)


def paused_context() -> InputContext:
    """Пауза: приоритет 50, остаются ``Pause``, ``Menu`` и интерфейс."""
    actions = frozenset({"Menu"}) | SYSTEM_ACTIONS | UI_ACTIONS
    return InputContext("paused", 50, actions, True)


def ui_context() -> InputContext:
    """Интерфейс: приоритет 100, только действия UI, остальное поглощается."""
    return InputContext("ui", 100, UI_ACTIONS | SYSTEM_ACTIONS, True)


def cutscene_context(skip: Iterable[str] = ("Pause", "UIClick")) -> InputContext:
    """Катсцена: приоритет 200, доступны только действия пропуска."""
    return InputContext("cutscene", 200, frozenset(skip), True)


class InputContextStack:
    """Стек контекстов.

    Args:
        contexts: Начальные контексты (по умолчанию - ``game``).
    """

    def __init__(self, contexts: Iterable[InputContext] | None = None) -> None:
        self._items: dict[str, InputContext] = {}
        for context in (game_context(),) if contexts is None else contexts:
            self._items[context.name] = context
        self._order: list[InputContext] = []
        self._resort()

    def push(self, context: InputContext) -> None:
        """Добавить контекст (одноимённый заменяется)."""
        self._items[context.name] = context
        self._resort()

    def pop(self, name: str) -> InputContext | None:
        """Убрать контекст по имени; ``None``, если его не было."""
        removed = self._items.pop(name, None)
        self._resort()
        return removed

    def reset(self) -> None:
        """Оставить только ``game`` (например, при загрузке сцены)."""
        self._items = {"game": game_context()}
        self._resort()

    def has(self, name: str) -> bool:
        """Есть ли контекст с таким именем."""
        return name in self._items

    def names(self) -> list[str]:
        """Имена от старшего приоритета к младшему."""
        return [c.name for c in self._order]

    def clear(self) -> None:
        """Убрать все контексты (разрешено всё)."""
        self._items.clear()
        self._resort()

    def allows(self, action: str) -> bool:
        """Получает ли игра действие ``action`` при текущем стеке."""
        for context in self._order:
            if context.actions is None or action in context.actions:
                return True
            if context.absorb:
                return False
        return True

    def _resort(self) -> None:
        self._order = sorted(self._items.values(), key=lambda c: -c.priority)

    def __len__(self) -> int:
        return len(self._items)
