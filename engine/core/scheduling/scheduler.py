"""Планировщик задач по времени: две независимые кучи (игровое и UI-время).

Задачи (:class:`ScheduleItem`) лежат в куче ``heapq`` по ключу
``(wake_time, порядок вставки)``: равные моменты срабатывают в порядке добавления,
поэтому поведение детерминировано. Если ничего не созрело, кадр стоит O(1) -
сравнивается только вершина кучи.

Два времени:

* :attr:`TimeCategory.GAME` - игровое время. Растёт на ``dt`` из
  ``Scene.update`` (уже умноженный на ``time_scale``); пока планировщик на паузе
  (:attr:`Scheduler.game_paused`), не растёт совсем;
* :attr:`TimeCategory.UI` - сырое время кадра. Растёт на ``dt`` из
  :meth:`Scheduler.update_ui` независимо от паузы и ``time_scale``.

Время категорий несравнимо, поэтому у каждой своя куча и свои часы.

Фиксированный шаг время **не** двигает (иначе игровое время росло бы дважды за
кадр): ``Scene.fixed_update`` только вызывает :meth:`Scheduler.tick_fixed`,
который считает фиксированные кадры.
"""

from __future__ import annotations

import heapq
import logging
from enum import IntEnum
from typing import Any

__all__ = ["Handle", "ScheduleItem", "Scheduler", "TimeCategory"]

_log = logging.getLogger("engine.core.scheduling")

#: Предел срабатываний одной категории за одно обновление (защита от `every(0)`).
MAX_FIRES_PER_UPDATE = 100_000


class TimeCategory(IntEnum):
    """Какие часы ведут задачу.

    ``IntEnum``: хеш целого считается в C, а не в Python как у ``Enum``
    (категория - ключ словарей в горячем пути кучи).
    """

    GAME = 0
    UI = 1
    #: Игровое время без ``time_scale`` (на паузе стоит); для ``wait_unscaled``.
    #: Ведётся из :meth:`Scheduler.update_ui`.
    GAME_RAW = 2


#: Где ждёт задача после ``advance``: куча по времени, следующий игровой кадр,
#: фиксированный шаг.
QUEUE_HEAP = 0
QUEUE_FRAME = 1
QUEUE_FIXED = 2


class ScheduleItem:
    """Базовая задача планировщика.

    Наследники задают ``wake_time`` и переопределяют :meth:`advance`.

    Attributes:
        wake_time: Момент срабатывания по часам своей категории.
        category: Часы, по которым считается ``wake_time``.
        owner: Владелец (для :meth:`Scheduler.cancel_all`) или ``None``.
        result: Результат завершённой задачи (корутина, ``return x``).
    """

    __slots__ = (
        "_scheduler",
        "cancelled",
        "category",
        "done",
        "order",
        "owner",
        "queue",
        "result",
        "wake_frame",
        "wake_time",
    )

    def __init__(
        self,
        wake_time: float = 0.0,
        category: TimeCategory = TimeCategory.GAME,
        owner: object | None = None,
    ) -> None:
        self.wake_time = float(wake_time)
        self.category = category
        self.owner = owner
        self.order = 0
        #: ``QUEUE_*``: куда ставить задачу, если ``advance`` вернул ``True``.
        self.queue = QUEUE_HEAP
        #: Для ``QUEUE_FRAME``/``QUEUE_FIXED`` - номер кадра пробуждения.
        self.wake_frame = 0
        self.cancelled = False
        self.done = False
        self.result: Any = None
        self._scheduler: Scheduler | None = None

    def is_due(self, now: float) -> bool:
        """Созрела ли задача на момент ``now``."""
        return self.wake_time <= now

    def advance(self, now: float) -> bool:
        """Выполнить срабатывание.

        Args:
            now: Текущее время категории.

        Returns:
            ``True`` - задача остаётся (``wake_time``, ``queue``, ``wake_frame``
            и ``category`` мог измениться), ``False`` - завершена.
        """
        return False

    def on_cancel(self) -> None:
        """Вызывается один раз при отмене до завершения."""


class Handle:
    """Ссылка на задачу: отмена и опрос состояния.

    Держит задачу сильной ссылкой (нужна после завершения ради ``result``).
    """

    __slots__ = ("_item",)

    def __init__(self, item: ScheduleItem) -> None:
        self._item = item

    def cancel(self) -> None:
        """Отменить задачу; повторный вызов и вызов после завершения безвредны."""
        scheduler = self._item._scheduler  # noqa: SLF001
        if scheduler is not None:
            scheduler.cancel(self)

    @property
    def is_done(self) -> bool:
        """Задача завершена или отменена."""
        return self._item.done or self._item.cancelled

    @property
    def is_cancelled(self) -> bool:
        """Задача отменена до завершения."""
        return self._item.cancelled

    @property
    def result(self) -> Any:
        """Результат завершённой задачи (``None``, если нет или не завершена)."""
        return self._item.result

    @property
    def item(self) -> ScheduleItem:
        """Сама задача."""
        return self._item

    def __repr__(self) -> str:
        state = "cancelled" if self._item.cancelled else (
            "done" if self._item.done else "active"
        )
        return f"<Handle {type(self._item).__name__} {state}>"


class Scheduler:
    """Планировщик задач сцены (``Scene.scheduler``)."""

    __slots__ = (
        "_alive",
        "_by_owner",
        "_current",
        "_dead",
        "_fixed_waiters",
        "_frame_waiters",
        "_heaps",
        "_next_order",
        "_now",
        "fixed_frame",
        "frame",
        "game_paused",
    )

    def __init__(self) -> None:
        self._heaps: dict[TimeCategory, list[tuple[float, int, ScheduleItem]]] = {
            category: [] for category in TimeCategory
        }
        self._now: dict[TimeCategory, float] = dict.fromkeys(TimeCategory, 0.0)
        self._dead: dict[TimeCategory, int] = dict.fromkeys(TimeCategory, 0)
        self._frame_waiters: list[ScheduleItem] = []
        self._fixed_waiters: list[ScheduleItem] = []
        self._by_owner: dict[int, dict[ScheduleItem, None]] = {}
        self._next_order = 0
        self._alive = 0
        self._current: ScheduleItem | None = None
        #: Счётчики кадров для ожидания «следующего кадра» (корутины).
        self.frame = 0
        self.fixed_frame = 0
        #: Пока ``True``, игровые часы стоят (ставит ``GameLoop.pause``).
        self.game_paused = False

    # ------------------------------------------------------------------ #
    @property
    def count(self) -> int:
        """Число живых (не отменённых и не завершённых) задач."""
        return self._alive

    def now(self, category: TimeCategory = TimeCategory.GAME) -> float:
        """Текущее время категории (для вычисления ``wake_time`` новых задач)."""
        return self._now[category]

    # ------------------------------------------------------------------ #
    def add(self, item: ScheduleItem) -> Handle:
        """Поставить задачу; ``item.wake_time`` уже задан по часам её категории."""
        if item._scheduler is not None:  # noqa: SLF001
            raise ValueError("задача уже поставлена в планировщик")
        item._scheduler = self  # noqa: SLF001
        self._push(item)
        self._alive += 1
        owner = item.owner
        if owner is not None:
            self._by_owner.setdefault(id(owner), {})[item] = None
        return Handle(item)

    def cancel(self, handle: Handle) -> None:
        """Отменить задачу по ссылке."""
        self._cancel_item(handle._item)  # noqa: SLF001

    def cancel_all(self, owner: object) -> int:
        """Отменить все задачи владельца.

        Returns:
            Сколько задач отменено.
        """
        items = self._by_owner.get(id(owner))
        if not items:
            return 0
        count = 0
        for item in list(items):
            if self._cancel_item(item):
                count += 1
        return count

    def clear(self) -> None:
        """Отменить все задачи (с ``on_cancel``) и сбросить часы."""
        for heap in self._heaps.values():
            for _, _, item in list(heap):
                self._cancel_item(item)
            heap.clear()
        for waiters in (self._frame_waiters, self._fixed_waiters):
            for item in list(waiters):
                self._cancel_item(item)
        self._frame_waiters = []
        self._fixed_waiters = []
        for category in self._now:
            self._now[category] = 0.0
            self._dead[category] = 0
        self._by_owner.clear()
        self._alive = 0
        self.frame = 0
        self.fixed_frame = 0

    def after(self, seconds: float, callback: Any, **kwargs: Any) -> Handle:
        """Одноразовый таймер; см. :func:`engine.core.scheduling.timers.after`."""
        from engine.core.scheduling.timers import after

        return after(self, seconds, callback, **kwargs)

    def every(self, seconds: float, callback: Any, **kwargs: Any) -> Handle:
        """Повторяющийся таймер; см. :func:`engine.core.scheduling.timers.every`."""
        from engine.core.scheduling.timers import every

        return every(self, seconds, callback, **kwargs)

    # ------------------------------------------------------------------ #
    def update(self, dt_game: float, dt_ui: float = 0.0) -> int:
        """Продвинуть обе категории.

        Args:
            dt_game: Игровой ``dt`` (уже с ``time_scale``).
            dt_ui: Сырой ``dt`` кадра.

        Returns:
            Сколько задач сработало.
        """
        return self.update_game(dt_game) + self.update_ui(dt_ui)

    def update_game(self, dt: float) -> int:
        """Продвинуть игровые часы (``Scene.update``); на паузе не двигает.

        Сначала срабатывают задачи кучи, затем ожидающие «следующий кадр» (в
        порядке постановки). Кадровый счётчик на паузе не растёт.
        """
        if self.game_paused:
            return 0
        self.frame += 1
        fired = self._advance(TimeCategory.GAME, dt)
        if self._frame_waiters:
            fired += self._release(self._frame_waiters, QUEUE_FRAME, self.frame)
        return fired

    def update_ui(self, dt: float) -> int:
        """Продвинуть UI-часы (``GameLoop``, каждый кадр, в том числе на паузе).

        Заодно ведёт ``GAME_RAW`` (игровое время без ``time_scale``): оно идёт
        от того же сырого ``dt``, но на паузе стоит.
        """
        fired = self._advance(TimeCategory.UI, dt)
        if not self.game_paused:
            fired += self._advance(TimeCategory.GAME_RAW, dt)
        return fired

    def tick_fixed(self) -> None:
        """Фиксированный шаг: счётчик растёт, ожидающие шаг пробуждаются.

        Игровые часы не двигаются (иначе время росло бы дважды за кадр).
        """
        self.fixed_frame += 1
        if self._fixed_waiters:
            self._release(self._fixed_waiters, QUEUE_FIXED, self.fixed_frame)

    def tween(
        self, target: Any, attr: str, end: Any, duration: float, **kwargs: Any
    ) -> Any:
        """Твин атрибута; см. ``tweens.tween``."""
        from engine.core.scheduling.tweens import tween

        return tween(self, target, attr, end, duration, **kwargs)

    def start_coroutine(self, generator: Any, **kwargs: Any) -> Handle:
        """Запустить корутину; см. ``coroutines.start_coroutine``."""
        from engine.core.scheduling.coroutines import start_coroutine

        return start_coroutine(self, generator, **kwargs)

    # ------------------------------------------------------------------ #
    def _fire(self, item: ScheduleItem, now: float) -> None:
        """``advance`` задачи и возврат в нужную очередь (или завершение)."""
        self._current = item
        try:
            keep = item.advance(now)
        finally:
            self._current = None
        if item.cancelled:  # отменила сама себя из callback
            return
        if keep:
            self._requeue(item)
        else:
            self._finish(item)

    def _requeue(self, item: ScheduleItem) -> None:
        if item.queue == QUEUE_FRAME:
            self._frame_waiters.append(item)
        elif item.queue == QUEUE_FIXED:
            self._fixed_waiters.append(item)
        else:
            self._push(item)

    def _release(self, waiters: list[ScheduleItem], queue: int, counter: int) -> int:
        """Разбудить ожидающих кадр/шаг, чей ``wake_frame`` наступил."""
        due: list[ScheduleItem] = []
        rest: list[ScheduleItem] = []
        for item in waiters:
            if item.cancelled or item.done:
                continue
            (due if item.wake_frame <= counter else rest).append(item)
        # новый список: повторные ожидания во время обхода попадут уже в него
        if queue == QUEUE_FRAME:
            self._frame_waiters = rest
        else:
            self._fixed_waiters = rest
        now = self._now[TimeCategory.GAME]
        fired = 0
        for item in due:
            if not (item.cancelled or item.done):
                fired += 1
                self._fire(item, now)
        return fired

    # ------------------------------------------------------------------ #
    def _push(self, item: ScheduleItem) -> None:
        self._next_order += 1
        item.order = self._next_order
        heapq.heappush(
            self._heaps[item.category], (item.wake_time, item.order, item)
        )

    def _advance(self, category: TimeCategory, dt: float) -> int:
        if dt > 0.0:
            self._now[category] += dt
        heap = self._heaps[category]
        if not heap or heap[0][0] > self._now[category]:
            return 0
        now = self._now[category]
        fired = 0
        while heap and heap[0][0] <= now:
            _, _, item = heapq.heappop(heap)
            if item.cancelled or item.done:
                self._dead[category] -= 1
                continue
            if fired >= MAX_FIRES_PER_UPDATE:
                _log.error(
                    "Планировщик: больше %d срабатываний за обновление, "
                    "остаток отложен (нулевой период у every?)",
                    MAX_FIRES_PER_UPDATE,
                )
                self._push(item)
                break
            fired += 1
            self._fire(item, now)
        return fired

    def _finish(self, item: ScheduleItem) -> None:
        item.done = True
        self._alive -= 1
        self._forget_owner(item)

    def _cancel_item(self, item: ScheduleItem) -> bool:
        if item.cancelled or item.done or item._scheduler is not self:  # noqa: SLF001
            return False
        item.cancelled = True
        self._alive -= 1
        # мёртвая запись остаётся только в куче, и не у исполняемой (она вынута)
        if item.queue == QUEUE_HEAP and item is not self._current:
            self._dead[item.category] += 1
        self._forget_owner(item)
        self._compact(item.category)
        try:
            item.on_cancel()
        except Exception:  # noqa: BLE001 - отмена не должна ронять вызывающего
            _log.exception("Ошибка в on_cancel задачи %s", type(item).__name__)
        return True

    def _forget_owner(self, item: ScheduleItem) -> None:
        owner = item.owner
        if owner is None:
            return
        items = self._by_owner.get(id(owner))
        if items is not None:
            items.pop(item, None)
            if not items:
                del self._by_owner[id(owner)]

    def _compact(self, category: TimeCategory) -> None:
        """Выбросить мёртвые записи, когда их больше половины кучи."""
        heap = self._heaps[category]
        if len(heap) > 64 and self._dead[category] * 2 > len(heap):
            heap[:] = [e for e in heap if not (e[2].cancelled or e[2].done)]
            heapq.heapify(heap)
            self._dead[category] = 0
