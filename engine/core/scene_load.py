"""Асинхронная (порционная) загрузка сцены: ``SceneLoadOperation``.

Python не даёт настоящей многопоточности для создания объектов (GIL), поэтому
«асинхронно» здесь значит «порциями по кадрам»:

* **parse** - чтение и разбор файла в рабочем потоке :class:`JobSystem`
  (ввод-вывод и ``json.loads`` освобождают GIL);
* **build** - создание объектов в главном потоке порциями, пока не вышел бюджет
  кадра (``step(budget_ms)``);
* **activate** - регистрация сцены в менеджере и события.

Ядро не знает формат файла: разбор и построение передаются функциями.

Example:
    >>> op = SceneLoadOperation(path, parse, build)
    >>> op.start()
    >>> while not op.step(5.0):
    ...     draw_loading_screen(op.progress)
    >>> scene = op.result()
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engine.core.jobs import Job, JobSystem
    from engine.core.scene import Scene

__all__ = [
    "BUILD_BUDGET_MAX_MS",
    "BUILD_BUDGET_MIN_MS",
    "PHASE_ACTIVATE",
    "PHASE_BUILD",
    "PHASE_PARSE",
    "SceneLoadCancelled",
    "SceneLoadOperation",
    "adaptive_budget_ms",
]

_log = logging.getLogger("engine.core.scene_load")

PHASE_PARSE = "parse"
PHASE_BUILD = "build"
PHASE_ACTIVATE = "activate"

#: Нижняя и верхняя границы адаптивного бюджета build-фазы за кадр.
BUILD_BUDGET_MIN_MS = 2.0
BUILD_BUDGET_MAX_MS = 10.0

#: Доли прогресса: parse, build, activate.
_PARSE_SHARE = 0.05
_ACTIVATE_SHARE = 0.05

BuildFunc = Callable[[Any, Path], Generator[float, None, "Scene"]]


class SceneLoadCancelled(RuntimeError):
    """Результат запрошен у отменённой операции загрузки."""


def adaptive_budget_ms(
    frame_ms: float, load_ms: float = 0.0, target_ms: float = 1000.0 / 60.0
) -> float:
    """Бюджет build-фазы на следующий кадр по замеру предыдущего.

    Свободное время кадра - ``target - (frame - load)``, берётся 80 % и
    зажимается в ``[BUILD_BUDGET_MIN_MS, BUILD_BUDGET_MAX_MS]``.

    Args:
        frame_ms: Полное время прошлого кадра.
        load_ms: Сколько из него заняла загрузка.
        target_ms: Целевая длительность кадра.

    Returns:
        Бюджет в миллисекундах.
    """
    free = target_ms - max(0.0, frame_ms - load_ms)
    return min(BUILD_BUDGET_MAX_MS, max(BUILD_BUDGET_MIN_MS, 0.8 * free))


class SceneLoadOperation:
    """Одна порционная загрузка сцены.

    Args:
        path: Файл сцены.
        parse: ``путь -> данные`` (выполняется в рабочем потоке, если есть
            ``jobs``).
        build: ``(данные, путь) -> генератор`` долей прогресса ``0..1`` с
            готовой сценой в ``StopIteration.value``.
        jobs: Пул потоков; ``None`` - parse выполняется в ``start`` синхронно.
        on_finish: Вызывается в activate-фазе со сценой (регистрация в
            менеджере); исключение считается ошибкой операции.
        on_progress: Вызывается после каждого шага с этой операцией.
    """

    def __init__(
        self,
        path: Path,
        parse: Callable[[Path], Any],
        build: BuildFunc,
        *,
        jobs: JobSystem | None = None,
        on_finish: Callable[[Scene], None] | None = None,
        on_progress: Callable[[SceneLoadOperation], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self._parse = parse
        self._build = build
        self._jobs = jobs
        self._on_finish = on_finish
        self._on_progress = on_progress
        self._phase = PHASE_PARSE
        self._job: Job | None = None
        self._data: Any = None
        self._steps: Generator[float, None, Scene] | None = None
        self._fraction = 0.0
        self._scene: Scene | None = None
        self._started = False
        self._done = False
        self._cancelled = False
        self._error: BaseException | None = None
        #: Суммарное время build-шагов (мс) - для отчётов и бенчмарков.
        self.build_ms = 0.0
        #: Длительность самого долгого шага (мс).
        self.max_step_ms = 0.0
        #: Число вызовов ``step``, сделавших работу.
        self.steps = 0

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    @property
    def phase(self) -> str:
        """``"parse"`` | ``"build"`` | ``"activate"``."""
        return self._phase

    @property
    def progress(self) -> float:
        """Общий прогресс ``0..1``."""
        if self._done and self._error is None and not self._cancelled:
            return 1.0
        if self._phase == PHASE_PARSE:
            return 0.0
        build_span = 1.0 - _PARSE_SHARE - _ACTIVATE_SHARE
        if self._phase == PHASE_BUILD:
            return _PARSE_SHARE + build_span * min(1.0, self._fraction)
        return 1.0 - _ACTIVATE_SHARE

    @property
    def is_done(self) -> bool:
        """Операция закончена (успехом, ошибкой или отменой)."""
        return self._done

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def error(self) -> BaseException | None:
        """Исключение, из-за которого загрузка не удалась."""
        return self._error

    def result(self) -> Scene:
        """Загруженная сцена.

        Raises:
            SceneLoadCancelled: Операция отменена.
            RuntimeError: Операция не завершена.
            Exception: Ошибка загрузки (повторно поднимается).
        """
        if self._cancelled:
            raise SceneLoadCancelled(f"Загрузка '{self.path}' отменена")
        if self._error is not None:
            raise self._error
        if not self._done or self._scene is None:
            raise RuntimeError(f"Загрузка '{self.path}' ещё не завершена")
        return self._scene

    # ------------------------------------------------------------------ #
    # Управление
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Запустить parse (в рабочем потоке, если задан ``jobs``)."""
        if self._started:
            return
        self._started = True
        if self._jobs is not None:
            self._job = self._jobs.submit(
                self._parse, self.path, name=f"parse:{self.path.name}"
            )
            return
        try:
            self._data = self._parse(self.path)
        except Exception as error:  # noqa: BLE001 - ошибка уходит в результат
            self._fail(error)
            return
        self._begin_build()

    def step(self, budget_ms: float = float("inf")) -> bool:
        """Сделать порцию работы не дольше бюджета (плюс один объект).

        Args:
            budget_ms: Бюджет шага в миллисекундах; ``inf`` - до конца
                (синхронная загрузка, ждёт и рабочий поток).

        Returns:
            ``True``, если операция закончена.
        """
        if self._done:
            return True
        if not self._started:
            self.start()
            if self._done:
                return True
        started = time.perf_counter()
        if self._phase == PHASE_PARSE and not self._advance_parse(budget_ms):
            self._notify()
            return self._done
        if self._phase == PHASE_BUILD:
            self._advance_build(started, budget_ms)
        if self._phase == PHASE_ACTIVATE and not self._done:
            self._activate()
        elapsed = (time.perf_counter() - started) * 1000.0
        self.steps += 1
        self.max_step_ms = max(self.max_step_ms, elapsed)
        self._notify()
        return self._done

    def cancel(self) -> None:
        """Отменить загрузку; недостроенная сцена выгружается."""
        if self._done:
            return
        self._cancelled = True
        self._done = True
        self._job = None  # результат parse в рабочем потоке отбрасывается
        if self._steps is not None:
            # GeneratorExit внутри ``iter_scene_from_dict`` выгрузит сцену.
            self._steps.close()
            self._steps = None
        self._data = None
        self._notify()

    # ------------------------------------------------------------------ #
    # Фазы
    # ------------------------------------------------------------------ #
    def _advance_parse(self, budget_ms: float) -> bool:
        job = self._job
        if job is not None:
            if not job.done():
                if budget_ms != float("inf"):
                    return False
                with contextlib.suppress(Exception):  # ошибка разбирается ниже
                    job.result()
            if job.error is not None:
                self._fail(job.error)
                return True
            self._data = job.result()
        self._begin_build()
        return not self._done

    def _begin_build(self) -> None:
        try:
            self._steps = self._build(self._data, self.path)
        except Exception as error:  # noqa: BLE001
            self._fail(error)
            return
        self._phase = PHASE_BUILD

    def _advance_build(self, started: float, budget_ms: float) -> None:
        assert self._steps is not None
        deadline = started + budget_ms / 1000.0
        step_started = time.perf_counter()
        try:
            while True:
                try:
                    self._fraction = next(self._steps)
                except StopIteration as done:
                    self._scene = done.value
                    self._steps = None
                    self._data = None
                    self._phase = PHASE_ACTIVATE
                    break
                if time.perf_counter() >= deadline:
                    break
        except Exception as error:  # noqa: BLE001 - ошибка уходит в результат
            self._steps = None
            self._fail(error)
        finally:
            self.build_ms += (time.perf_counter() - step_started) * 1000.0

    def _activate(self) -> None:
        assert self._scene is not None
        try:
            if self._on_finish is not None:
                self._on_finish(self._scene)
        except Exception as error:  # noqa: BLE001
            self._scene.shutdown()
            self._scene = None
            self._fail(error)
            return
        self._done = True

    def _fail(self, error: BaseException) -> None:
        _log.error("Загрузка сцены '%s' не удалась: %s", self.path, error)
        self._error = error
        self._done = True

    def _notify(self) -> None:
        if self._on_progress is not None:
            try:
                self._on_progress(self)
            except Exception:  # noqa: BLE001 - подписчик не ломает загрузку
                _log.exception("Ошибка обработчика прогресса загрузки")

    def __repr__(self) -> str:
        state = "done" if self._done else self._phase
        return f"<SceneLoadOperation '{self.path.name}' {state} {self.progress:.0%}>"
