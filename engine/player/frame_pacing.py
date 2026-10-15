"""Темп кадров плеера: ограничение FPS и пауза при свёрнутом окне."""

from __future__ import annotations

import time

__all__ = ["FramePacer"]


class FramePacer:
    """Ограничитель частоты кадров на ``time.perf_counter``.

    Vsync включается на стороне окна (``swap_interval``); этот класс нужен для
    headless-режима и для явного лимита ``--max-fps``.

    Args:
        target_fps: Целевая частота; ``None`` или ``0`` — без ограничения.
    """

    #: Период опроса событий, пока окно свёрнуто (секунды).
    IDLE_SLEEP = 0.05

    def __init__(self, target_fps: float | None = 60.0) -> None:
        self.target_fps = target_fps or None
        self._frame_start = time.perf_counter()

    def begin(self) -> None:
        """Отметить начало кадра."""
        self._frame_start = time.perf_counter()

    def end(self) -> float:
        """Дождаться конца периода кадра.

        Returns:
            Реальную длительность кадра в секундах (с учётом ожидания).
        """
        if self.target_fps:
            period = 1.0 / self.target_fps
            remaining = period - (time.perf_counter() - self._frame_start)
            if remaining > 0.002:
                time.sleep(remaining - 0.001)
            while time.perf_counter() - self._frame_start < period:
                pass
        return time.perf_counter() - self._frame_start

    def idle(self) -> None:
        """Не жечь CPU/GPU, пока рендер приостановлен."""
        time.sleep(self.IDLE_SLEEP)
