"""Тесты шины событий."""

from __future__ import annotations

import gc
from dataclasses import dataclass

from engine.core.event_bus import Event, EventBus


@dataclass(slots=True)
class Damage(Event):
    amount: int


@dataclass(slots=True)
class CriticalDamage(Damage):
    multiplier: float = 2.0


class Listener:
    """Объект-подписчик: проверяем и слабые ссылки, и порядок вызовов."""

    def __init__(
        self, name: str, log: list[str], bus: EventBus, priority: int = 0
    ) -> None:
        self.name = name
        self.log = log
        self.subscription = bus.subscribe(Damage, self.on_damage, priority=priority)

    def on_damage(self, event: Damage) -> None:
        self.log.append(f"{self.name}:{event.amount}")


class TestSubscription:
    def test_emit_calls_handler(self) -> None:
        bus = EventBus()
        seen: list[int] = []
        bus.subscribe(Damage, lambda event: seen.append(event.amount))
        assert bus.emit(Damage(5)) == 1
        assert seen == [5]

    def test_priority_order(self) -> None:
        bus = EventBus()
        log: list[str] = []
        listeners = [
            Listener("low", log, bus, priority=0),
            Listener("high", log, bus, priority=10),
            Listener("middle", log, bus, priority=5),
        ]
        bus.emit(Damage(1))
        assert log == ["high:1", "middle:1", "low:1"]
        assert len(listeners) == 3  # подписчики должны жить во время вызова

    def test_once_subscription(self) -> None:
        bus = EventBus()
        seen: list[int] = []
        bus.subscribe(Damage, lambda event: seen.append(event.amount), once=True)
        bus.emit(Damage(1))
        bus.emit(Damage(2))
        assert seen == [1]
        assert bus.handler_count(Damage) == 0

    def test_cancel_and_context_manager(self) -> None:
        bus = EventBus()
        seen: list[int] = []
        subscription = bus.subscribe(Damage, lambda event: seen.append(event.amount))
        assert subscription.cancel() is True
        assert subscription.cancel() is False
        bus.emit(Damage(1))
        with bus.subscribe(Damage, lambda event: seen.append(99)):
            bus.emit(Damage(2))
        bus.emit(Damage(3))
        assert seen == [99]

    def test_unsubscribe_handler(self) -> None:
        bus = EventBus()

        def handler(event: Damage) -> None:
            raise AssertionError("не должен вызываться")

        bus.subscribe(Damage, handler)
        assert bus.unsubscribe_handler(handler) == 1
        assert bus.emit(Damage(1)) == 0

    def test_base_class_receives_derived_event(self) -> None:
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(Event, lambda event: seen.append("event"))
        bus.subscribe(Damage, lambda event: seen.append("damage"))
        bus.emit(CriticalDamage(7))
        assert seen == ["damage", "event"]

    def test_dead_method_subscription_is_dropped(self) -> None:
        bus = EventBus()
        log: list[str] = []
        listener = Listener("temp", log, bus)
        del listener
        gc.collect()
        assert bus.emit(Damage(3)) == 0
        assert bus.stats.cancelled_dead_refs == 1


class TestDelivery:
    def test_error_in_handler_does_not_break_others(self) -> None:
        bus = EventBus()
        seen: list[int] = []

        def broken(event: Damage) -> None:
            raise RuntimeError("boom")

        bus.subscribe(Damage, broken)
        bus.subscribe(Damage, lambda event: seen.append(event.amount))
        assert bus.emit(Damage(4)) == 1
        assert seen == [4]
        assert bus.stats.errors == 1

    def test_deferred_delivery(self) -> None:
        bus = EventBus()
        seen: list[int] = []
        bus.subscribe(Damage, lambda event: seen.append(event.amount))
        bus.emit_deferred(Damage(1))
        bus.emit_deferred(Damage(2))
        assert seen == []
        assert bus.deferred_count == 2
        assert bus.flush() == 2
        assert seen == [1, 2]
        assert bus.deferred_count == 0

    def test_recursion_guard(self) -> None:
        bus = EventBus(max_delivery_depth=3)
        bus.subscribe(Damage, lambda event: bus.emit(Damage(event.amount + 1)))
        assert bus.emit(Damage(0)) == 1
        assert bus.stats.max_depth_seen <= 3

    def test_clear(self) -> None:
        bus = EventBus()
        bus.subscribe(Damage, lambda event: None)
        bus.subscribe(Event, lambda event: None)
        bus.clear(Damage)
        assert bus.handler_count(Damage) == 0
        assert bus.handler_count(Event) == 1
        bus.clear()
        assert bus.handler_count() == 0

    def test_stats_and_repr(self) -> None:
        bus = EventBus()
        bus.subscribe(Damage, lambda event: None)
        bus.emit(Damage(1))
        stats = bus.stats
        assert stats.emitted == 1
        assert stats.delivered == 1
        assert "EventBus" in repr(bus)
        stats.reset()
        assert bus.stats.emitted == 0
