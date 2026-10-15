"""Таймеры: одноразовый ``after`` и повторяющийся ``every``."""

from __future__ import annotations

import logging
from collections.abc import Callable

from engine.core.scheduling.scheduler import (
    Handle,
    ScheduleItem,
    Scheduler,
    TimeCategory,
)

__all__ = ["EveryItem", "TimerItem", "after", "every"]

_log = logging.getLogger("engine.core.scheduling")

ErrorHook = Callable[[BaseException], None]


def _report(callback: Callable[[], object], error: BaseException,
            on_error: ErrorHook | None) -> None:
    """Сообщить об исключении callback: свой обработчик или лог с трассировкой."""
    if on_error is not None:
        try:
            on_error(error)
            return
        except Exception:  # noqa: BLE001 - обработчик ошибок не должен ронять кадр
            _log.exception("Ошибка в обработчике ошибок таймера")
    _log.error(
        "Исключение в callback таймера %r", callback, exc_info=error
    )


class TimerItem(ScheduleItem):
    """Одноразовый вызов ``callback()`` в момент ``wake_time``."""

    __slots__ = ("callback", "on_error")

    def __init__(
        self,
        wake_time: float,
        callback: Callable[[], object],
        category: TimeCategory = TimeCategory.GAME,
        owner: object | None = None,
        on_error: ErrorHook | None = None,
    ) -> None:
        super().__init__(wake_time, category, owner)
        self.callback = callback
        self.on_error = on_error

    def advance(self, now: float) -> bool:
        try:
            self.callback()
        except Exception as exc:  # noqa: BLE001 - изоляция пользовательского кода
            _report(self.callback, exc, self.on_error)
        return False


class EveryItem(ScheduleItem):
    """Повторяющийся вызов каждые ``period`` секунд.

    Момент ``k``-го срабатывания считается как ``start + k * period``, а не
    накоплением, поэтому период не дрейфует. Если кадр был длиннее периода,
    пропущенные срабатывания выполняются подряд (догоняем).
    """

    __slots__ = ("_start", "_fired", "callback", "on_error", "period")

    def __init__(
        self,
        start: float,
        period: float,
        callback: Callable[[], object],
        category: TimeCategory = TimeCategory.GAME,
        owner: object | None = None,
        on_error: ErrorHook | None = None,
        immediate: bool = False,
    ) -> None:
        # k = 0 - немедленный вызов в момент постановки, иначе первый через period.
        super().__init__(start if immediate else start + period, category, owner)
        self.period = period
        self.callback = callback
        self.on_error = on_error
        self._start = start
        self._fired = 0 if immediate else 1

    def advance(self, now: float) -> bool:
        try:
            self.callback()
        except Exception as exc:  # noqa: BLE001 - every живёт дальше
            _report(self.callback, exc, self.on_error)
        self._fired += 1
        self.wake_time = self._start + self._fired * self.period
        return True


def after(
    scheduler: Scheduler,
    seconds: float,
    callback: Callable[[], object],
    *,
    owner: object | None = None,
    category: TimeCategory = TimeCategory.GAME,
    on_error: ErrorHook | None = None,
) -> Handle:
    """Вызвать ``callback()`` один раз через ``seconds`` секунд.

    Args:
        scheduler: Планировщик.
        seconds: Задержка по часам ``category`` (``0`` - при ближайшем обновлении).
        callback: Вызываемое без аргументов.
        owner: Владелец для ``cancel_all``.
        category: Часы задачи.
        on_error: Обработчик исключения callback (по умолчанию - лог).

    Returns:
        Ссылка на задачу.

    Raises:
        ValueError: Если ``seconds`` отрицательно.
    """
    if seconds < 0.0:
        raise ValueError("seconds не может быть отрицательным")
    item = TimerItem(
        scheduler.now(category) + seconds, callback, category, owner, on_error
    )
    return scheduler.add(item)


def every(
    scheduler: Scheduler,
    seconds: float,
    callback: Callable[[], object],
    *,
    owner: object | None = None,
    category: TimeCategory = TimeCategory.GAME,
    immediate: bool = False,
    on_error: ErrorHook | None = None,
) -> Handle:
    """Вызывать ``callback()`` каждые ``seconds`` секунд, пока не отменят.

    Args:
        scheduler: Планировщик.
        seconds: Период; строго положительный.
        callback: Вызываемое без аргументов.
        owner: Владелец для ``cancel_all``.
        category: Часы задачи.
        immediate: Первый вызов сразу при ближайшем обновлении, а не через период.
        on_error: Обработчик исключения callback (по умолчанию - лог); таймер
            после исключения продолжает работать.

    Returns:
        Ссылка на задачу.

    Raises:
        ValueError: Если ``seconds`` не положительно.
    """
    if seconds <= 0.0:
        raise ValueError("seconds у every должен быть положительным")
    item = EveryItem(
        scheduler.now(category), seconds, callback, category, owner, on_error,
        immediate,
    )
    return scheduler.add(item)
