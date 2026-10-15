"""Шина событий (Event Bus) для связи компонентов, систем и подсистем движка.

Возможности:

* подписка по типу события с **приоритетом** выполнения;
* подписка на **базовый** тип: обработчик получит и события-наследники;
* ``once``-подписки и автоматическая отмена подписок на методы **умерших**
  объектов (слабые ссылки), чтобы не терять ссылки на удалённые компоненты;
* **отложенная** доставка (:meth:`EventBus.emit_deferred` + :meth:`EventBus.flush`)
  — структурные изменения сцены не происходят посреди чужого обработчика;
* изоляция ошибок: падение одного обработчика не мешает остальным.

Example:
    >>> bus = EventBus()
    >>> bus.subscribe(DamageEvent, on_damage)
    >>> bus.emit(DamageEvent(target=entity, amount=10))
    1
"""

from __future__ import annotations

import bisect
import logging
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

__all__ = ["Event", "EventBus", "EventBusStats", "Subscription"]

_log = logging.getLogger("engine.core.events")

EventT = TypeVar("EventT", bound=Any)
Handler = Callable[[Any], None]

MAX_DELIVERY_DEPTH = 16
MAX_FLUSH_EVENTS = 100_000


def _same_handler(left: Handler | None, right: Handler) -> bool:
    """Сравнить два обработчика, корректно учитывая связанные методы."""
    if left is None:
        return False
    if left is right:
        return True
    left_self = getattr(left, "__self__", None)
    right_self = getattr(right, "__self__", None)
    if left_self is None or right_self is None:
        return False
    return left_self is right_self and getattr(left, "__func__", None) is getattr(
        right, "__func__", None
    )


class Event:
    """Базовый класс события.

    Наследоваться не обязательно — шина работает с любым объектом, — но
    наследники дают типизацию, единый вид и возможность подписаться на
    группу событий сразу.
    """

    __slots__ = ()


@dataclass(slots=True)
class EventBusStats:
    """Счётчики шины для отладки и панели профайлера."""

    emitted: int = 0
    delivered: int = 0
    errors: int = 0
    deferred_flushed: int = 0
    max_depth_seen: int = 0
    cancelled_dead_refs: int = 0

    def reset(self) -> None:
        """Обнулить все счётчики."""
        self.emitted = 0
        self.delivered = 0
        self.errors = 0
        self.deferred_flushed = 0
        self.max_depth_seen = 0
        self.cancelled_dead_refs = 0


class Subscription:
    """Регистрация обработчика в шине.

    Подписку можно отменить вручную (:meth:`cancel`) или использовать как
    контекстный менеджер — при выходе она отменяется автоматически.

    Ссылка на обработчик хранится **слабо** (для методов), поэтому подписка не
    продлевает жизнь объекту-подписчику.
    """

    __slots__ = (
        "_alive",
        "_bus",
        "_func_ref",
        "_order",
        "_target_ref",
        "event_type",
        "once",
        "priority",
    )

    def __init__(
        self,
        bus: EventBus,
        event_type: type,
        target_ref: weakref.WeakMethod | None,
        func_ref: Handler | None,
        priority: int,
        once: bool,
        order: int,
    ) -> None:
        self._bus: EventBus | None = bus
        self.event_type = event_type
        self._target_ref = target_ref
        self._func_ref = func_ref
        self.priority = priority
        self.once = once
        self._alive = True
        self._order = order

    @property
    def handler(self) -> Handler | None:
        """Живой обработчик или ``None``, если объект-владелец уже удалён."""
        if self._target_ref is not None:
            return self._target_ref()
        return self._func_ref

    @property
    def alive(self) -> bool:
        """``True``, пока подписка активна."""
        return self._alive and self._bus is not None

    def cancel(self) -> bool:
        """Отменить подписку. Возвращает ``True``, если она была активна."""
        bus = self._bus
        if bus is None or not self._alive:
            return False
        self._alive = False
        bus._remove_subscription(self)  # noqa: SLF001 - подписка часть шины
        self._bus = None
        return True

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.cancel()

    def __repr__(self) -> str:
        state = "alive" if self.alive else "cancelled"
        return (
            f"<Subscription {self.event_type.__name__} "
            f"priority={self.priority} {state}>"
        )


@dataclass(slots=True)
class _Entry:
    """Внутренняя запись шины: подписка и её место в порядке вызова."""

    subscription: Subscription
    priority: int
    once: bool
    order: int
    _sort_key: tuple[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sort_key = (-self.priority, self.order)

    def resolve(self) -> Handler | None:
        """Вернуть живой обработчик или ``None``, если объект-владелец умер."""
        return self.subscription.handler


class EventBus:
    """Типизированная шина событий с приоритетами и отложенной доставкой.

    Args:
        max_delivery_depth: Максимальная вложенность ``emit`` внутри ``emit``;
            защищает от бесконечной рекурсии при взаимных событиях.
    """

    __slots__ = ("_handlers", "_deferred", "_depth", "_order", "stats", "_max_depth")

    def __init__(self, *, max_delivery_depth: int = MAX_DELIVERY_DEPTH) -> None:
        self._handlers: dict[type, list[_Entry]] = {}
        self._deferred: deque[Any] = deque()
        self._depth = 0
        self._order = 0
        self._max_depth = max_delivery_depth
        self.stats = EventBusStats()

    # ------------------------------------------------------------------ #
    # Подписки
    # ------------------------------------------------------------------ #
    def subscribe(
        self,
        event_type: type[EventT],
        handler: Callable[[EventT], None],
        *,
        priority: int = 0,
        once: bool = False,
    ) -> Subscription:
        """Подписаться на события указанного типа.

        Args:
            event_type: Класс события (можно базовый — наследники тоже придут).
            handler: Функция или метод; методы удерживаются слабо, поэтому
                подписка исчезает вместе с объектом-владельцем.
            priority: Чем больше, тем раньше вызывается обработчик.
            once: Отменить подписку после первого вызова.

        Returns:
            Подписку, которой можно отменить регистрацию.
        """
        order = self._order
        self._order += 1

        target_ref: weakref.WeakMethod | None = None
        func_ref: Handler | None = None
        if hasattr(handler, "__self__") and handler.__self__ is not None:
            try:
                target_ref = weakref.WeakMethod(handler)  # type: ignore[arg-type]
            except TypeError:  # pragma: no cover - экзотические callables
                func_ref = handler
        else:
            func_ref = handler

        subscription = Subscription(
            self, event_type, target_ref, func_ref, priority, once, order
        )
        entry = _Entry(subscription, priority, once, order)
        entries = self._handlers.setdefault(event_type, [])
        bisect.insort(entries, entry, key=lambda item: item._sort_key)  # noqa: SLF001
        return subscription

    def unsubscribe(self, subscription: Subscription) -> bool:
        """Отменить конкретную подписку."""
        return subscription.cancel()

    def unsubscribe_handler(self, handler: Handler) -> int:
        """Отменить все подписки указанной функции или метода.

        Returns:
            Количество отменённых подписок.
        """
        cancelled = 0
        for entries in list(self._handlers.values()):
            for entry in list(entries):
                if not _same_handler(entry.resolve(), handler):
                    continue
                if entry.subscription.cancel():
                    cancelled += 1
        return cancelled

    def clear(self, event_type: type | None = None) -> None:
        """Удалить подписки (все или только на указанный тип)."""
        if event_type is None:
            for entries in self._handlers.values():
                for entry in entries:
                    entry.subscription._alive = False  # noqa: SLF001
            self._handlers.clear()
        else:
            for entry in self._handlers.pop(event_type, []):
                entry.subscription._alive = False  # noqa: SLF001

    def handler_count(self, event_type: type | None = None) -> int:
        """Число зарегистрированных обработчиков."""
        if event_type is not None:
            return len(self._handlers.get(event_type, ()))
        return sum(len(entries) for entries in self._handlers.values())

    def _remove_subscription(self, subscription: Subscription) -> None:
        entries = self._handlers.get(subscription.event_type)
        if not entries:
            return
        for index, entry in enumerate(entries):
            if entry.subscription is subscription:
                entries.pop(index)
                break
        if not entries:
            self._handlers.pop(subscription.event_type, None)

    # ------------------------------------------------------------------ #
    # Доставка
    # ------------------------------------------------------------------ #
    def emit(self, event: Any) -> int:
        """Доставить событие немедленно всем подходящим обработчикам.

        Обработчики вызываются в порядке убывания приоритета. Обработчик
        базового класса получает события-наследники.

        Args:
            event: Любой объект-событие; его тип определяет подписчиков.

        Returns:
            Количество успешно вызванных обработчиков.
        """
        self.stats.emitted += 1
        if self._depth >= self._max_depth:
            _log.error(
                "Превышена глубина доставки событий (%d), событие %s отброшено",
                self._max_depth,
                type(event).__name__,
            )
            return 0

        self._depth += 1
        self.stats.max_depth_seen = max(self.stats.max_depth_seen, self._depth)
        delivered = 0
        try:
            for event_type in type(event).__mro__:
                entries = self._handlers.get(event_type)
                if not entries:
                    continue
                for entry in list(entries):
                    handler = entry.resolve()
                    if handler is None:
                        entry.subscription.cancel()
                        self.stats.cancelled_dead_refs += 1
                        continue
                    try:
                        handler(event)
                    except Exception:  # noqa: BLE001 - ошибка подписчика не должна ломать кадр
                        self.stats.errors += 1
                        _log.exception(
                            "Ошибка обработчика %r события %s",
                            handler,
                            type(event).__name__,
                        )
                        continue
                    delivered += 1
                    self.stats.delivered += 1
                    if entry.once:
                        entry.subscription.cancel()
        finally:
            self._depth -= 1
        return delivered

    def emit_deferred(self, event: Any) -> None:
        """Положить событие в очередь; оно будет доставлено в :meth:`flush`."""
        self._deferred.append(event)

    def flush(self, max_events: int | None = None) -> int:
        """Доставить все отложенные события в порядке постановки.

        Args:
            max_events: Верхняя граница числа обработанных событий за вызов.

        Returns:
            Количество доставленных событий (с учётом добавленных в процессе).
        """
        limit = MAX_FLUSH_EVENTS if max_events is None else max(0, max_events)
        processed = 0
        while self._deferred and processed < limit:
            event = self._deferred.popleft()
            self.emit(event)
            processed += 1
            self.stats.deferred_flushed += 1
        if self._deferred and processed >= limit:
            _log.warning(
                "flush остановлен на %d событиях, в очереди осталось %d",
                processed,
                len(self._deferred),
            )
        return processed

    @property
    def deferred_count(self) -> int:
        """Число событий, ожидающих отложенной доставки."""
        return len(self._deferred)

    def __repr__(self) -> str:
        return (
            f"<EventBus handlers={self.handler_count()} "
            f"deferred={self.deferred_count} errors={self.stats.errors}>"
        )
