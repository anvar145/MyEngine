"""Твины: плавное изменение атрибута объекта за время.

Поддерживаются ``float``/``int``, ``Vector3`` (линейно), ``Quaternion``
(slerp) и кортежи чисел (поэлементно). Тип определяется по значению атрибута
в момент создания твина.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from engine.core.math3d import Quaternion, Vector3
from engine.core.scheduling.easing import get_ease
from engine.core.scheduling.scheduler import (
    Handle,
    ScheduleItem,
    Scheduler,
    TimeCategory,
)

__all__ = ["TweenHandle", "TweenItem", "tween"]

_log = logging.getLogger("engine.core.scheduling")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _make_lerp(
    start: Any, end: Any, attr: str
) -> tuple[Any, Callable[[Any, Any, float], Any]]:
    """Проверить ``end`` относительно типа ``start`` и выбрать интерполяцию."""
    if _is_number(start):
        if not _is_number(end):
            raise TypeError(
                f"tween атрибута {attr}: ожидалось число, получено "
                f"{type(end).__name__}"
            )
        return end, lambda a, b, t: a + (b - a) * t
    if isinstance(start, Vector3):
        if not isinstance(end, Vector3):
            raise TypeError(
                f"tween атрибута {attr}: ожидался Vector3, получено "
                f"{type(end).__name__}"
            )
        return end, lambda a, b, t: a.lerp(b, t)
    if isinstance(start, Quaternion):
        if not isinstance(end, Quaternion):
            raise TypeError(
                f"tween атрибута {attr}: ожидался Quaternion, получено "
                f"{type(end).__name__}"
            )
        return end, lambda a, b, t: a.slerp(b, t)
    if isinstance(start, tuple) and all(_is_number(v) for v in start):
        if not isinstance(end, tuple) or len(end) != len(start):
            raise ValueError(
                f"tween атрибута {attr}: длины кортежей должны совпадать "
                f"({len(start)})"
            )
        return end, lambda a, b, t: tuple(
            x + (y - x) * t for x, y in zip(a, b, strict=True)
        )
    raise TypeError(
        f"tween не поддерживает тип {type(start).__name__} для атрибута {attr}"
    )


class TweenItem(ScheduleItem):
    """Задача: каждое обновление ставит ``attr`` в интерполированное значение.

    Attributes:
        target: Объект, чей атрибут меняется.
        attr: Имя атрибута.
        start: Снимок значения в момент создания.
        end: Конечное значение.
        duration: Длительность по часам категории.
        ease: Функция сглаживания.
        on_complete: Вызывается один раз после установки конечного значения.
        on_update: Вызывается после каждой записи значения.
    """

    __slots__ = (
        "_lerp",
        "_start_time",
        "attr",
        "duration",
        "ease",
        "end",
        "on_complete",
        "on_update",
        "start",
        "target",
    )

    def __init__(
        self,
        scheduler: Scheduler,
        target: object,
        attr: str,
        end: Any,
        duration: float,
        ease: Callable[[float], float],
        category: TimeCategory = TimeCategory.GAME,
        owner: object | None = None,
    ) -> None:
        start = getattr(target, attr)
        end, lerp = _make_lerp(start, end, attr)
        now = scheduler.now(category)
        super().__init__(now, category, owner)
        self.target = target
        self.attr = attr
        self.start = start
        self.end = end
        self.duration = float(duration)
        self.ease = ease
        self.on_complete: Callable[[], object] | None = None
        self.on_update: Callable[[Any], object] | None = None
        self._lerp = lerp
        self._start_time = now
        if self.duration <= 0.0:
            setattr(target, attr, end)  # мгновенно; on_complete - при update

    def advance(self, now: float) -> bool:
        if self.duration <= 0.0:
            return self._complete()
        t = (now - self._start_time) / self.duration
        finished = t >= 1.0
        value = self.end if finished else self._lerp(
            self.start, self.end, self.ease(max(t, 0.0))
        )
        try:
            setattr(self.target, self.attr, value)
        except Exception:  # noqa: BLE001 - цель могла быть уничтожена
            _log.exception("Твин: не удалось записать %s", self.attr)
            return False
        self._notify_update(value)
        if finished:
            return self._complete()
        # строго после текущего момента: при dt = 0 повторно не сработает
        self.wake_time = math.nextafter(now, math.inf)
        return True

    def _notify_update(self, value: Any) -> None:
        if self.on_update is None:
            return
        try:
            self.on_update(value)
        except Exception:  # noqa: BLE001 - изоляция пользовательского кода
            _log.exception("Исключение в on_update твина %s", self.attr)
            self.on_update = None  # не засорять лог каждый кадр

    def _complete(self) -> bool:
        callback = self.on_complete
        if callback is not None:
            try:
                callback()
            except Exception:  # noqa: BLE001
                _log.exception("Исключение в on_complete твина %s", self.attr)
        return False


class TweenHandle(Handle):
    """``Handle`` твина с цепочкой ``on_complete`` / ``on_update``."""

    __slots__ = ()

    def on_complete(self, callback: Callable[[], object]) -> TweenHandle:
        """Вызвать ``callback()`` после достижения конечного значения.

        Отмена твина ``on_complete`` не вызывает.
        """
        self._item.on_complete = callback  # type: ignore[attr-defined]
        return self

    def on_update(self, callback: Callable[[Any], object]) -> TweenHandle:
        """Вызывать ``callback(value)`` после каждой записи значения."""
        self._item.on_update = callback  # type: ignore[attr-defined]
        return self


def tween(
    scheduler: Scheduler,
    target: object,
    attr: str,
    end: Any,
    duration: float,
    *,
    ease: str | Callable[[float], float] = "ease_in_out_quad",
    category: TimeCategory = TimeCategory.GAME,
    owner: object | None = None,
) -> TweenHandle:
    """Плавно привести ``target.attr`` к ``end`` за ``duration`` секунд.

    Начальное значение - снимок ``getattr`` в момент вызова; запись значения
    начинается при ближайшем обновлении, последняя запись - ровно ``end``.
    ``duration <= 0``: значение ставится сразу, ``on_complete`` вызывается при
    ближайшем обновлении (как ``after(0, cb)``). Два твина одного атрибута
    разрешены и применяются в порядке добавления.

    Args:
        scheduler: Планировщик.
        target: Объект.
        attr: Имя атрибута (``float``, ``int``, ``Vector3``, ``Quaternion``
            или кортеж чисел).
        end: Конечное значение того же типа.
        duration: Длительность по часам ``category``.
        ease: Имя из ``EASES`` или функция ``f(t)``.
        category: Часы твина.
        owner: Владелец для ``cancel_all``.

    Returns:
        Ссылка на твин с цепочкой ``on_complete``/``on_update``.

    Raises:
        ValueError: Неизвестный ``ease`` или несовпадающая длина кортежей.
        TypeError: Неподдерживаемый тип атрибута или ``end``.
    """
    item = TweenItem(
        scheduler, target, attr, end, duration, get_ease(ease), category, owner
    )
    handle = scheduler.add(item)
    return TweenHandle(handle.item)
