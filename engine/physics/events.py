"""События физики: триггеры и столкновения (публикуются через EventBus)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PhysicsEvent:
    """Событие пары тел: ``a``/``b`` — handle, ``user_a``/``user_b`` — данные."""

    a: int
    b: int
    user_a: Any = None
    user_b: Any = None


@dataclass(frozen=True, slots=True)
class TriggerEnter(PhysicsEvent):
    """Тело вошло в триггер."""


@dataclass(frozen=True, slots=True)
class TriggerStay(PhysicsEvent):
    """Тело остаётся в триггере (каждый шаг)."""


@dataclass(frozen=True, slots=True)
class TriggerExit(PhysicsEvent):
    """Тело покинуло триггер."""


@dataclass(frozen=True, slots=True)
class CollisionEnter(PhysicsEvent):
    """Начало контакта двух непроникаемых тел."""


@dataclass(frozen=True, slots=True)
class CollisionExit(PhysicsEvent):
    """Конец контакта."""
