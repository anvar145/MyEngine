"""Корутины на генераторах: ``yield`` инструкции ожидания.

Пример::

    def sequence():
        yield Wait(1.0)            # секунда игрового времени (с time_scale)
        yield WaitForFrame()       # следующий кадр update
        yield WaitForFixed(2)      # два фиксированных шага
        yield WaitUntil(lambda: door.open)
        return "готово"            # -> Handle.result

    handle = start_coroutine(scheduler, sequence())

Первый шаг корутины выполняется при ближайшем ``update`` (не внутри
``start_coroutine``). Инструкции - маленькие объекты; :class:`CoroutineItem`
по типу полученной инструкции решает, в какую очередь планировщика встать.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from typing import Any

from engine.core.scheduling.scheduler import (
    QUEUE_FIXED,
    QUEUE_FRAME,
    QUEUE_HEAP,
    Handle,
    ScheduleItem,
    Scheduler,
    TimeCategory,
)

__all__ = [
    "CoroutineItem",
    "Wait",
    "WaitForFixed",
    "WaitForFrame",
    "WaitUntil",
    "start_coroutine",
]

_log = logging.getLogger("engine.core.scheduling")

ErrorHook = Callable[[BaseException], None]


class Wait:
    """Ждать ``seconds`` секунд.

    Attributes:
        seconds: Длительность; ``<= 0`` равно :class:`WaitForFrame`.
        unscaled: Считать игровое время без ``time_scale`` (на паузе всё равно
            стоит). Для UI-корутины игнорируется: её часы всегда сырые.
    """

    __slots__ = ("seconds", "unscaled")

    def __init__(self, seconds: float, unscaled: bool = False) -> None:
        self.seconds = float(seconds)
        self.unscaled = unscaled


class WaitForFrame:
    """Продолжить в следующем кадре ``update`` (не зависит от ``time_scale``)."""

    __slots__ = ()


class WaitForFixed:
    """Продолжить через ``count`` фиксированных шагов."""

    __slots__ = ("count",)

    def __init__(self, count: int = 1) -> None:
        self.count = max(1, int(count))


class WaitUntil:
    """Продолжить, когда ``predicate()`` вернёт истину.

    Предикат проверяется раз в кадр, начиная со следующего.
    """

    __slots__ = ("predicate",)

    def __init__(self, predicate: Callable[[], object]) -> None:
        self.predicate = predicate


class CoroutineItem(ScheduleItem):
    """Задача-корутина: продвигает генератор и ставит его в нужную очередь.

    Разбор ``yield``:

    * :class:`Wait` - в кучу по времени (``GAME``/``GAME_RAW``/``UI``);
    * :class:`WaitForFrame` (и ``None``) - в список ждущих кадра;
    * :class:`WaitForFixed` - в список ждущих фиксированного шага;
    * :class:`WaitUntil` - в список ждущих кадра с проверкой предиката;
    * число - то же, что ``Wait(число)``;
    * всё остальное - ошибка корутины.

    ``return x`` попадает в :attr:`result`. Исключение (в том числе из
    предиката) логируется (или идёт в ``on_error``) и завершает корутину.
    """

    __slots__ = ("_gen", "_predicate", "_scheduler_ref", "on_error")

    def __init__(
        self,
        scheduler: Scheduler,
        generator: Generator[Any, None, Any],
        category: TimeCategory = TimeCategory.GAME,
        owner: object | None = None,
        on_error: ErrorHook | None = None,
    ) -> None:
        super().__init__(scheduler.now(category), category, owner)
        self._gen = generator
        self._predicate: Callable[[], object] | None = None
        self._scheduler_ref = scheduler
        self.on_error = on_error

    def advance(self, now: float) -> bool:
        predicate = self._predicate
        if predicate is not None:
            try:
                ready = predicate()
            except Exception as exc:  # noqa: BLE001 - изоляция пользовательского кода
                self._fail(exc)
                return False
            if not ready:
                self.wake_frame = self._scheduler_ref.frame + 1
                return True
            self._predicate = None
        try:
            instruction = next(self._gen)
        except StopIteration as stop:
            self.result = stop.value
            return False
        except Exception as exc:  # noqa: BLE001 - изоляция пользовательского кода
            self._fail(exc)
            return False
        try:
            self._schedule(instruction)
        except TypeError as exc:
            self._gen.close()
            self._fail(exc)
            return False
        return True

    def on_cancel(self) -> None:
        # finally-блоки генератора выполнятся при закрытии; ошибки не роняют отмену
        gen = self._gen
        if getattr(gen, "gi_running", False):  # отмена из самого генератора
            return
        try:
            gen.close()
        except Exception:  # noqa: BLE001
            _log.exception("Ошибка при закрытии корутины")

    # ------------------------------------------------------------------ #
    def _schedule(self, instruction: object) -> None:
        sched = self._scheduler_ref
        if isinstance(instruction, (int, float)) and not isinstance(
            instruction, bool
        ):
            instruction = Wait(instruction)
        if isinstance(instruction, Wait):
            if instruction.seconds <= 0.0:
                self._wait_frame()
                return
            category = self.category
            if category is not TimeCategory.UI:
                category = (
                    TimeCategory.GAME_RAW
                    if instruction.unscaled
                    else TimeCategory.GAME
                )
            self.category = category
            self.queue = QUEUE_HEAP
            self.wake_time = sched.now(category) + instruction.seconds
        elif instruction is None or isinstance(instruction, WaitForFrame):
            self._wait_frame()
        elif isinstance(instruction, WaitForFixed):
            self.queue = QUEUE_FIXED
            self.wake_frame = sched.fixed_frame + instruction.count
        elif isinstance(instruction, WaitUntil):
            self._predicate = instruction.predicate
            self._wait_frame()
        else:
            raise TypeError(
                f"корутина вернула {type(instruction).__name__}, ожидалась "
                "инструкция ожидания (Wait, WaitForFrame, WaitForFixed, WaitUntil)"
            )

    def _wait_frame(self) -> None:
        self.queue = QUEUE_FRAME
        self.wake_frame = self._scheduler_ref.frame + 1

    def _fail(self, error: BaseException) -> None:
        if self.on_error is not None:
            try:
                self.on_error(error)
                return
            except Exception:  # noqa: BLE001 - обработчик не должен ронять кадр
                _log.exception("Ошибка в обработчике ошибок корутины")
        _log.error("Исключение в корутине %r", self._gen, exc_info=error)


def start_coroutine(
    scheduler: Scheduler,
    generator: Generator[Any, None, Any],
    *,
    owner: object | None = None,
    category: TimeCategory = TimeCategory.GAME,
    on_error: ErrorHook | None = None,
) -> Handle:
    """Запустить корутину.

    Args:
        scheduler: Планировщик.
        generator: Генератор (результат вызова генераторной функции).
        owner: Владелец для ``cancel_all`` (при отмене генератор закрывается).
        category: ``GAME`` или ``UI`` (часы для ``Wait``; кадровые ожидания
            всегда игровые).
        on_error: Обработчик исключения (по умолчанию - лог с трассировкой).

    Returns:
        Ссылка на задачу; ``handle.result`` - значение ``return`` генератора.

    Raises:
        TypeError: Если передан не генератор.
        ValueError: Если ``category`` - ``GAME_RAW``.
    """
    if not hasattr(generator, "send"):
        raise TypeError("start_coroutine ждёт генератор, а не функцию/объект")
    if category is TimeCategory.GAME_RAW:
        raise ValueError("GAME_RAW задаётся через Wait(unscaled=True)")
    item = CoroutineItem(scheduler, generator, category, owner, on_error)
    return scheduler.add(item)
