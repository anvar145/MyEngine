"""Scheduler: кучи, порядок, отмена, категории времени, интеграция со Scene."""

from __future__ import annotations

from engine.core.game_loop import GameLoop
from engine.core.scene import Scene
from engine.core.scheduling import ScheduleItem, Scheduler, TimeCategory


class Probe(ScheduleItem):
    """Одноразовая задача, пишущая своё имя в общий журнал."""

    __slots__ = ("log", "name", "advances")

    def __init__(self, log, name, wake, category=TimeCategory.GAME, owner=None):
        super().__init__(wake, category, owner)
        self.log = log
        self.name = name
        self.advances = 0

    def advance(self, now: float) -> bool:
        self.advances += 1
        self.log.append(self.name)
        return False


def _make(sched: Scheduler, log, name, delay, **kwargs):
    category = kwargs.get("category", TimeCategory.GAME)
    return sched.add(Probe(log, name, sched.now(category) + delay, **kwargs))


def test_item_fires_when_time_reached() -> None:
    sched, log = Scheduler(), []
    handle = _make(sched, log, "a", 0.5)
    sched.update_game(0.4)
    assert log == [] and not handle.is_done
    sched.update_game(0.1)
    assert log == ["a"] and handle.is_done and not handle.is_cancelled
    assert sched.count == 0


def test_cancel_before_fire() -> None:
    sched, log = Scheduler(), []
    handle = _make(sched, log, "a", 0.5)
    handle.cancel()
    handle.cancel()  # повторная отмена безвредна
    sched.update_game(1.0)
    assert log == [] and handle.is_cancelled and handle.is_done
    assert sched.count == 0


def test_cancel_after_done_is_noop() -> None:
    sched, log = Scheduler(), []
    handle = _make(sched, log, "a", 0.1)
    sched.update_game(0.2)
    handle.cancel()
    assert not handle.is_cancelled and handle.is_done


def test_heap_untouched_until_due() -> None:
    sched, log = Scheduler(), []
    probes = [
        sched.add(Probe(log, f"p{i}", 100.0 + i)).item for i in range(999)
    ]
    first = sched.add(Probe(log, "first", 0.5)).item
    sched.update_game(0.6)
    assert log == ["first"]
    assert first.advances == 1
    assert all(p.advances == 0 for p in probes)
    assert sched.count == 999


def test_equal_wake_time_fires_in_insertion_order() -> None:
    sched, log = Scheduler(), []
    for name in "abcde":
        _make(sched, log, name, 1.0)
    sched.update_game(1.0)
    assert log == list("abcde")


def test_cancel_all_only_touches_owner() -> None:
    sched, log = Scheduler(), []
    mine, other = object(), object()
    _make(sched, log, "m1", 0.1, owner=mine)
    _make(sched, log, "m2", 0.2, owner=mine)
    _make(sched, log, "o1", 0.3, owner=other)
    _make(sched, log, "n1", 0.4)
    assert sched.cancel_all(mine) == 2
    assert sched.cancel_all(mine) == 0
    assert sched.count == 2
    sched.update_game(1.0)
    assert log == ["o1", "n1"]


def test_clear_cancels_everything_and_calls_on_cancel() -> None:
    class Spy(Probe):
        __slots__ = ("cancelled_calls",)

        def on_cancel(self) -> None:
            self.cancelled_calls = getattr(self, "cancelled_calls", 0) + 1

    sched, log = Scheduler(), []
    spy = sched.add(Spy(log, "s", 1.0)).item
    sched.clear()
    assert spy.cancelled_calls == 1 and sched.count == 0
    sched.update_game(5.0)
    assert log == []


def test_categories_are_independent() -> None:
    sched, log = Scheduler(), []
    _make(sched, log, "game", 1.0)
    _make(sched, log, "ui", 1.0, category=TimeCategory.UI)
    sched.update_game(1.0)
    assert log == ["game"]
    sched.update_ui(1.0)
    assert log == ["game", "ui"]


def test_pause_freezes_game_but_not_ui() -> None:
    sched, log = Scheduler(), []
    _make(sched, log, "game", 0.5)
    _make(sched, log, "ui", 0.5, category=TimeCategory.UI)
    sched.game_paused = True
    for _ in range(10):
        sched.update(0.1, 0.1)
    assert log == ["ui"]
    sched.game_paused = False
    sched.update(0.4, 0.0)
    assert log == ["ui"]  # до срабатывания игрового таймера ещё 0.1
    sched.update(0.11, 0.0)
    assert log == ["ui", "game"]


def test_cancelled_dead_entries_are_compacted() -> None:
    sched, log = Scheduler(), []
    handles = [_make(sched, log, str(i), 10.0 + i) for i in range(200)]
    for handle in handles[:150]:
        handle.cancel()
    assert sched.count == 50
    assert len(sched._heaps[TimeCategory.GAME]) < 200  # noqa: SLF001
    sched.update_game(1000.0)
    assert len(log) == 50


def test_item_can_cancel_itself_from_advance() -> None:
    sched, log = Scheduler(), []

    class SelfCancel(Probe):
        __slots__ = ("handle",)

        def advance(self, now: float) -> bool:
            self.handle.cancel()
            return True

    item = SelfCancel(log, "x", 0.1)
    item.handle = sched.add(item)
    sched.update_game(0.2)
    assert item.handle.is_cancelled and sched.count == 0
    assert len(sched._heaps[TimeCategory.GAME]) == 0  # noqa: SLF001
    assert sched._dead[TimeCategory.GAME] == 0  # noqa: SLF001


# ------------------------------------------------------------ интеграция


def test_scene_update_advances_game_clock_before_components() -> None:
    from engine.core.component import Component

    order: list[str] = []

    class Hook(Component):
        def on_update(self, dt: float) -> None:
            order.append("component")

    scene = Scene("S")
    scene.create_game_object("O").add_component(Hook())
    sched = scene.scheduler
    sched.add(
        type("F", (Probe,), {"__slots__": ()})(order, "timer", 0.01)
    )
    scene.update(0.02)
    assert order == ["timer", "component"]


def test_scene_fixed_update_does_not_move_time() -> None:
    scene = Scene("S")
    log: list[str] = []
    _make(scene.scheduler, log, "a", 0.01)
    for _ in range(5):
        scene.fixed_update(0.02)
    assert log == [] and scene.scheduler.fixed_frame == 5
    scene.update(0.02)
    assert log == ["a"]


def test_time_scale_and_pause_through_game_loop() -> None:
    scene = Scene("S")
    loop = GameLoop(scene, time_scale=0.5)
    log: list[str] = []
    _make(scene.scheduler, log, "game", 0.5)
    _make(scene.scheduler, log, "ui", 0.5, category=TimeCategory.UI)
    for _ in range(5):  # 5 x 0.1 сырых = 0.5 UI, но 0.25 игрового
        loop.step(0.1)
    assert log == ["ui"]
    for _ in range(6):  # запас на накопление float
        loop.step(0.1)
    assert log == ["ui", "game"]

    log.clear()
    _make(scene.scheduler, log, "game2", 0.2)
    _make(scene.scheduler, log, "ui2", 0.2, category=TimeCategory.UI)
    loop.pause()
    for _ in range(10):
        loop.step(0.1)
    assert log == ["ui2"]
    loop.resume()
    for _ in range(5):
        loop.step(0.1)
    assert log == ["ui2", "game2"]


def test_attach_scene_inherits_pause() -> None:
    loop = GameLoop()
    loop.pause()
    scene = Scene("S")
    loop.attach_scene(scene)
    assert scene.scheduler.game_paused
    loop.resume()
    assert not scene.scheduler.game_paused


def test_scene_shutdown_clears_scheduler() -> None:
    scene = Scene("S")
    _make(scene.scheduler, [], "a", 5.0)
    scene.shutdown()
    assert scene.scheduler.count == 0
