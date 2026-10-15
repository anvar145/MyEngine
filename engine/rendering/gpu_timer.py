"""Замер времени GPU через ``GL_TIME_ELAPSED`` (без остановки конвейера).

Запросы идут кольцом: результат читается только у запроса, который драйвер уже
завершил, поэтому ``poll`` никогда не ждёт GPU. Задержка результата — 1–3 кадра.
Нужен динамическому разрешению: время кадра на экране с VSync не показывает
запас, а время GPU показывает.
"""

from __future__ import annotations

import logging

__all__ = ["GpuTimer"]

_log = logging.getLogger("engine.rendering.gpu_timer")

_RING = 4


class GpuTimer:
    """Кольцо запросов ``GL_TIME_ELAPSED``."""

    def __init__(self) -> None:
        self._queries: list[int] = []
        self._pending: list[int] = []  # индексы запросов, ждущих результата
        self._free: list[int] = []
        self._open: int | None = None
        self.available = True
        self._last_ms: float | None = None

    def _ensure(self) -> bool:
        if self._queries or not self.available:
            return self.available
        from OpenGL import GL

        try:
            ids = GL.glGenQueries(_RING)
            if not hasattr(ids, "__iter__"):
                ids = [ids]
            self._queries = [int(i) for i in ids]
        except Exception as exc:  # noqa: BLE001 - драйвер без поддержки
            _log.warning("GL_TIME_ELAPSED недоступен: %s", exc)
            self.available = False
            return False
        self._free = list(range(len(self._queries)))
        return True

    def begin(self) -> None:
        """Начать замер (вызывать перед рисованием кадра)."""
        if not self._ensure() or self._open is not None or not self._free:
            return
        from OpenGL import GL

        index = self._free.pop(0)
        GL.glBeginQuery(GL.GL_TIME_ELAPSED, self._queries[index])
        self._open = index

    def end(self) -> None:
        """Завершить замер (после рисования кадра)."""
        if self._open is None:
            return
        from OpenGL import GL

        GL.glEndQuery(GL.GL_TIME_ELAPSED)
        self._pending.append(self._open)
        self._open = None

    def poll(self) -> float | None:
        """Вернуть время последнего завершённого кадра в мс (или ``None``).

        Returns:
            Свежий результат, если с прошлого вызова завершился хотя бы один
            запрос; иначе ``None`` (старое значение — в :attr:`last_ms`).
        """
        if not self._pending:
            return None
        from OpenGL import GL

        fresh: float | None = None
        while self._pending:
            index = self._pending[0]
            query = self._queries[index]
            if not GL.glGetQueryObjectuiv(query, GL.GL_QUERY_RESULT_AVAILABLE):
                break
            nanoseconds = int(GL.glGetQueryObjectuiv(query, GL.GL_QUERY_RESULT))
            fresh = nanoseconds / 1.0e6
            self._pending.pop(0)
            self._free.append(index)
        if fresh is not None:
            self._last_ms = fresh
        return fresh

    @property
    def last_ms(self) -> float | None:
        """Последний известный результат в мс."""
        return self._last_ms

    def dispose(self) -> None:
        """Удалить запросы."""
        if self._queries:
            from OpenGL import GL

            GL.glDeleteQueries(len(self._queries), self._queries)
        self._queries.clear()
        self._pending.clear()
        self._free.clear()
        self._open = None
