"""Минимальная система фоновых задач с возвратом результата в главный поток.

Тяжёлая работа (импорт ассетов, кук) выполняется в пуле потоков, а реакция на
её завершение — **только** в том потоке, который вызывает :meth:`JobSystem.pump`
(обычно главный). Это позволяет коду главного потока (``AssetDatabase``,
``EventBus``) не заботиться о блокировках.

Example:
    >>> jobs = JobSystem(workers=2)
    >>> job = jobs.submit(heavy, 42, on_done=lambda j: print(j.result()))
    >>> jobs.wait_idle()          # вызывает on_done в текущем потоке
    True
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from typing import Any

__all__ = ["Job", "JobSystem"]

_log = logging.getLogger("engine.core.jobs")


class Job:
    """Дескриптор поставленной в очередь задачи.

    Attributes:
        name: Имя задачи (для логов и отладки).
    """

    __slots__ = ("_future", "_on_done", "name")

    def __init__(
        self,
        name: str,
        on_done: Callable[[Job], None] | None,
    ) -> None:
        self.name = name
        self._on_done = on_done
        self._future: Future[Any] | None = None

    def done(self) -> bool:
        """``True``, если работа завершена (успешно, с ошибкой или отменена)."""
        return self._future is not None and self._future.done()

    @property
    def cancelled(self) -> bool:
        """``True``, если задача была отменена до запуска."""
        return self._future is not None and self._future.cancelled()

    @property
    def error(self) -> BaseException | None:
        """Исключение задачи (``None``, если успешна или ещё не завершена)."""
        future = self._future
        if future is None or not future.done():
            return None
        if future.cancelled():
            return CancelledError()
        return future.exception()

    def result(self, timeout: float | None = None) -> Any:
        """Дождаться и вернуть результат (пробрасывает исключение задачи)."""
        assert self._future is not None  # noqa: S101 - Job создаёт только JobSystem
        return self._future.result(timeout)

    def __repr__(self) -> str:
        state = "done" if self.done() else "running"
        return f"<Job '{self.name}' {state}>"


class JobSystem:
    """Пул потоков с очередью завершённых задач.

    Args:
        workers: Число рабочих потоков.
        name: Префикс имён потоков.
    """

    def __init__(self, workers: int = 2, *, name: str = "JobSystem") -> None:
        self.name = name
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix=name
        )
        self._finished: queue.SimpleQueue[Job] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._pending = 0
        self._closed = False

    @property
    def pending(self) -> int:
        """Сколько задач ещё не доставлено в ``on_done`` (идут или ждут pump)."""
        return self._pending

    def submit(
        self,
        func: Callable[..., Any],
        *args: Any,
        on_done: Callable[[Job], None] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Job:
        """Поставить функцию в очередь на выполнение в рабочем потоке.

        Args:
            func: Выполняемая функция.
            *args: Позиционные аргументы функции.
            on_done: Обратный вызов; исполняется в потоке, вызвавшем
                :meth:`pump`/:meth:`wait_idle`, после завершения задачи.
            name: Имя задачи.
            **kwargs: Именованные аргументы функции.

        Raises:
            RuntimeError: Если система уже остановлена.
        """
        job = Job(name or getattr(func, "__name__", "job"), on_done)
        with self._lock:
            if self._closed:
                raise RuntimeError(f"{self.name} остановлена")
            self._pending += 1
        future = self._executor.submit(func, *args, **kwargs)
        job._future = future  # noqa: SLF001 - Job и JobSystem — одна подсистема
        future.add_done_callback(lambda _future: self._finished.put(job))
        return job

    def pump(self, max_jobs: int | None = None) -> int:
        """Выполнить ``on_done`` завершённых задач в текущем потоке.

        Args:
            max_jobs: Максимум задач за вызов; ``None`` — все готовые.

        Returns:
            Число обработанных задач.
        """
        handled = 0
        while max_jobs is None or handled < max_jobs:
            try:
                job = self._finished.get_nowait()
            except queue.Empty:
                break
            self._dispatch(job)
            handled += 1
        return handled

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Дождаться завершения всех задач, вызывая ``on_done`` в этом потоке.

        Args:
            timeout: Предел ожидания в секундах; ``None`` — без предела.

        Returns:
            ``True``, если очередь опустела до истечения таймаута.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._pending > 0:
            wait = 0.05
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)
            try:
                job = self._finished.get(timeout=wait)
            except queue.Empty:
                continue
            self._dispatch(job)
        return True

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        """Остановить пул.

        Args:
            wait: Дождаться завершения выполняющихся задач.
            cancel_pending: Отменить ещё не начатые задачи.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_pending)
        if wait:
            self.pump()

    def _dispatch(self, job: Job) -> None:
        """Вызвать ``on_done`` задачи; ошибка обработчика не ломает очередь."""
        try:
            if job._on_done is not None:  # noqa: SLF001
                job._on_done(job)  # noqa: SLF001
        except Exception:  # noqa: BLE001 - изоляция ошибок подписчика
            _log.exception("Ошибка обработчика завершения задачи '%s'", job.name)
        finally:
            with self._lock:
                self._pending -= 1

    def __enter__(self) -> JobSystem:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return f"<JobSystem '{self.name}' pending={self._pending}>"
