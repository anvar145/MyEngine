"""Главный игровой цикл: фиксированный шаг физики + переменный шаг логики.

Классическая схема «fixed timestep with interpolation»:

* физика и всё, что должно быть детерминированным, обновляются фиксированными
  шагами (по умолчанию 1/60 с) — сколько шагов нужно, столько и выполняется;
* игровая логика и рендер обновляются с настоящим ``dt`` кадра;
* :attr:`Time.alpha` — доля незавершённого фиксированного шага, её используют
  для интерполяции положения при отрисовке, чтобы картинка не «дрожала»;
* защита от «спирали смерти» (:attr:`GameLoop.max_substeps`): если машина не
  успевает, лишнее время отбрасывается, а не накапливается бесконечно;
* необязательный бюджет времени на фиксированные шаги (``max_fixed_ms``): шаг
  не начинается, если по оценке его стоимости кадр выйдет за бюджет.

Example:
    >>> loop = GameLoop(scene, fixed_delta=1 / 60, target_fps=60)
    >>> loop.add_render(lambda dt, alpha: renderer.draw(alpha))
    >>> loop.run(max_frames=600)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.scene import Scene

__all__ = ["GameLoop", "LoopStats", "Time"]

_log = logging.getLogger("engine.core.loop")

UpdateHook = Callable[[float], None]
RenderHook = Callable[[float, float], None]


@dataclass(slots=True)
class Time:
    """Текущее время кадра (доступно системам и компонентам).

    Attributes:
        delta: Время кадра с учётом ``time_scale`` (секунды).
        unscaled_delta: Настоящее время кадра без масштаба.
        fixed_delta: Шаг фиксированного обновления.
        elapsed: Суммарно прошедшее игровое время.
        frame: Номер кадра.
        fixed_frame: Общее число выполненных фиксированных шагов.
        alpha: Доля незавершённого фиксированного шага (``0..1``).
        time_scale: Множитель времени (``0`` — пауза логики, ``0.5`` — слоу-мо).
    """

    delta: float = 0.0
    unscaled_delta: float = 0.0
    fixed_delta: float = 1.0 / 60.0
    elapsed: float = 0.0
    frame: int = 0
    fixed_frame: int = 0
    alpha: float = 0.0
    time_scale: float = 1.0


@dataclass(slots=True)
class LoopStats:
    """Статистика цикла для отладочного оверлея и профайлера."""

    frames: int = 0
    fixed_steps: int = 0
    dropped_time: float = 0.0
    overrun_frames: int = 0
    budget_stops: int = 0
    last_fixed_steps: int = 0
    last_fixed_ms: float = 0.0
    last_budget_stop: bool = False
    last_frame_ms: float = 0.0
    smoothed_frame_ms: float = 0.0

    @property
    def fps(self) -> float:
        """Сглаженный FPS (0, пока не накоплена статистика)."""
        if self.smoothed_frame_ms <= 0.0:
            return 0.0
        return 1000.0 / self.smoothed_frame_ms


class GameLoop:
    """Цикл игры: управляет временем, фазами кадра и частотой обновления.

    Args:
        scene: Сцена, фазы которой нужно обновлять (можно задать позже).
        fixed_delta: Шаг фиксированного обновления в секундах.
        max_substeps: Максимум фиксированных шагов за один кадр.
        target_fps: Ограничение частоты кадров; ``None`` — без ограничения.
        time_scale: Множитель игрового времени.
        max_frame_delta: Максимально допустимое время кадра (защита после
            сворачивания окна или длительной паузы).
        smoothing: Коэффициент сглаживания времени кадра (``0..1``).
        continue_on_error: Продолжать кадр при исключении в логике
            (``True`` полезно для редактора, ``False`` — для тестов).
        max_fixed_ms: Бюджет времени фиксированных шагов за кадр (мс) или
            ``None`` — без бюджета. Первый шаг кадра выполняется всегда, каждый
            следующий — только если ``потрачено + оценка шага`` укладывается
            в бюджет (оценка — максимум из сглаженной и последней стоимости).
            Остаток времени не копится (``accumulator %= fixed_delta``), поэтому
            игровое время замедляется, но кадры не удлиняются спиралью.
        clock: Источник времени в секундах (для тестов).
    """

    __slots__ = (
        "_accumulator",
        "_clock",
        "_continue_on_error",
        "_fixed_hooks",
        "_fixed_ema_ms",
        "_fixed_last_ms",
        "_is_running",
        "_last_time",
        "_late_hooks",
        "_max_fixed_ms",
        "_max_frame_delta",
        "_max_substeps",
        "_paused",
        "_render_hooks",
        "_scene_ref",
        "_smoothing",
        "_target_fps",
        "_time_scale",
        "_warn_at",
        "_warn_budget_at",
        "_warn_budget_stops",
        "_warn_overruns",
        "_warn_dropped",
        "_update_hooks",
        "stats",
        "time",
    )

    def __init__(
        self,
        scene: Scene | None = None,
        *,
        fixed_delta: float = 1.0 / 60.0,
        max_substeps: int = 5,
        target_fps: float | None = None,
        time_scale: float = 1.0,
        max_frame_delta: float = 0.25,
        smoothing: float = 0.1,
        continue_on_error: bool = False,
        max_fixed_ms: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if fixed_delta <= 0.0:
            raise ValueError("fixed_delta должен быть положительным")
        if max_substeps < 1:
            raise ValueError("max_substeps должен быть не меньше 1")

        self.time = Time(fixed_delta=fixed_delta, time_scale=time_scale)
        self.stats = LoopStats()

        self._scene_ref: Scene | None = None
        self._accumulator = 0.0
        self._paused = False
        self._is_running = False
        self._last_time = 0.0
        self._max_substeps = int(max_substeps)
        self._max_frame_delta = float(max_frame_delta)
        self._time_scale = float(time_scale)
        self._target_fps = target_fps
        self._smoothing = min(1.0, max(0.0, float(smoothing)))
        self._continue_on_error = bool(continue_on_error)
        self._max_fixed_ms = None if max_fixed_ms is None else float(max_fixed_ms)
        self._clock = clock
        self._fixed_ema_ms = 0.0
        self._fixed_last_ms = 0.0
        self._warn_at = float("-inf")
        self._warn_budget_at = float("-inf")
        self._warn_budget_stops = 0
        self._warn_overruns = 0
        self._warn_dropped = 0.0

        self._fixed_hooks: list[UpdateHook] = []
        self._update_hooks: list[UpdateHook] = []
        self._late_hooks: list[UpdateHook] = []
        self._render_hooks: list[RenderHook] = []

        if scene is not None:
            self.attach_scene(scene)

    # ------------------------------------------------------------------ #
    # Сцена и хуки
    # ------------------------------------------------------------------ #
    @property
    def scene(self) -> Scene | None:
        """Сцена, которую обновляет цикл."""
        return self._scene_ref

    def attach_scene(self, scene: Scene | None) -> None:
        """Назначить (или снять) сцену, обновляемую циклом."""
        self._scene_ref = scene
        if scene is not None:
            scene.scheduler.game_paused = self._paused

    def add_fixed_update(self, hook: UpdateHook) -> None:
        """Добавить колбэк фиксированного шага (вызывается после сцены)."""
        self._fixed_hooks.append(hook)

    def add_update(self, hook: UpdateHook) -> None:
        """Добавить колбэк основной фазы кадра."""
        self._update_hooks.append(hook)

    def add_late_update(self, hook: UpdateHook) -> None:
        """Добавить колбэк поздней фазы кадра."""
        self._late_hooks.append(hook)

    def add_render(self, hook: RenderHook) -> None:
        """Добавить колбэк отрисовки: ``hook(dt, alpha)``."""
        self._render_hooks.append(hook)

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        """``True``, пока выполняется :meth:`run`."""
        return self._is_running

    @property
    def is_paused(self) -> bool:
        """``True``, если симуляция приостановлена."""
        return self._paused

    @property
    def accumulator(self) -> float:
        """Накопленное, но ещё не отработанное время (секунды)."""
        return self._accumulator

    @property
    def time_scale(self) -> float:
        """Текущий множитель игрового времени."""
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value: float) -> None:
        self._time_scale = max(0.0, float(value))
        self.time.time_scale = self._time_scale

    @property
    def target_fps(self) -> float | None:
        """Ограничение частоты кадров или ``None``."""
        return self._target_fps

    @target_fps.setter
    def target_fps(self, value: float | None) -> None:
        self._target_fps = None if value is None else max(1.0, float(value))

    def pause(self) -> None:
        """Приостановить симуляцию (рендер и хуки продолжают вызываться)."""
        self._paused = True
        if self._scene_ref is not None:
            self._scene_ref.scheduler.game_paused = True

    def resume(self) -> None:
        """Продолжить симуляцию."""
        self._paused = False
        if self._scene_ref is not None:
            self._scene_ref.scheduler.game_paused = False

    def stop(self) -> None:
        """Попросить цикл завершиться после текущего кадра."""
        self._is_running = False

    def reset(self) -> None:
        """Сбросить время, аккумулятор и статистику (не трогая сцену)."""
        self.time.elapsed = 0.0
        self.time.frame = 0
        self.time.fixed_frame = 0
        self.time.alpha = 0.0
        self.time.delta = 0.0
        self._accumulator = 0.0
        self.stats = LoopStats()
        self._last_time = 0.0

    # ------------------------------------------------------------------ #
    # Кадр
    # ------------------------------------------------------------------ #
    def step(self, frame_delta: float | None = None) -> Time:
        """Выполнить один кадр цикла.

        Метод детерминирован, если передавать ``frame_delta`` явно, поэтому
        его удобно использовать в тестах и для покадровой отладки.

        Args:
            frame_delta: Время кадра в секундах; ``None`` — измерить реальное.

        Returns:
            Обновлённый :class:`Time` кадра.
        """
        started = self._clock()
        if frame_delta is None:
            raw = started - self._last_time if self._last_time > 0.0 else 0.0
        else:
            raw = float(frame_delta)
        self._last_time = started
        raw = max(0.0, min(raw, self._max_frame_delta))

        self.time.unscaled_delta = raw
        self.time.delta = raw * self._time_scale
        self.time.time_scale = self._time_scale

        self.stats.last_fixed_steps = 0
        self.stats.last_fixed_ms = 0.0
        self.stats.last_budget_stop = False
        if not self._paused:
            self._accumulator += self.time.delta
            steps = 0
            budget = self._max_fixed_ms
            spent_ms = 0.0
            budget_stop = False
            while (
                self._accumulator >= self.time.fixed_delta
                and steps < self._max_substeps
            ):
                if budget is not None and steps > 0:
                    predicted = max(self._fixed_ema_ms, self._fixed_last_ms)
                    if spent_ms + predicted > budget:
                        budget_stop = True
                        break
                t0 = self._clock()
                self._run_fixed(self.time.fixed_delta)
                cost_ms = (self._clock() - t0) * 1000.0
                spent_ms += cost_ms
                self._note_fixed_cost(cost_ms)
                self._accumulator -= self.time.fixed_delta
                steps += 1
                self.time.fixed_frame += 1
                self.stats.fixed_steps += 1
            self.stats.last_fixed_steps = steps
            self.stats.last_fixed_ms = spent_ms
            if budget_stop:
                # Бюджет исчерпан: фазу накопителя сохраняем, остаток отбрасываем.
                self.stats.budget_stops += 1
                self.stats.last_budget_stop = True
                self._warn_budget(spent_ms)
                self.stats.dropped_time += self._accumulator - (
                    self._accumulator % self.time.fixed_delta
                )
                self._accumulator %= self.time.fixed_delta
            elif (
                steps >= self._max_substeps
                and self._accumulator >= self.time.fixed_delta
            ):
                # Спираль смерти: не догоняем бесконечно, часть времени отбрасываем.
                self.stats.dropped_time += self._accumulator
                self.stats.overrun_frames += 1
                self._warn_overrun(self._accumulator)
                self._accumulator = 0.0

        self.time.alpha = min(1.0, self._accumulator / self.time.fixed_delta)
        if self._scene_ref is not None:
            # UI-часы планировщика идут по сырому dt, в том числе на паузе;
            # игровые ведёт Scene.update (на паузе их держит game_paused).
            self._scene_ref.scheduler.update_ui(raw)
        self._run_update(self.time.delta)
        self._run_late(self.time.delta)
        self._run_render(self.time.delta, self.time.alpha)

        self.time.frame += 1
        self.time.elapsed += self.time.delta

        frame_ms = (self._clock() - started) * 1000.0
        self.stats.frames += 1
        self.stats.last_frame_ms = frame_ms
        if self.stats.smoothed_frame_ms <= 0.0:
            self.stats.smoothed_frame_ms = frame_ms
        else:
            alpha = self._smoothing
            self.stats.smoothed_frame_ms = (
                self.stats.smoothed_frame_ms * (1.0 - alpha) + frame_ms * alpha
            )
        return self.time

    def _note_fixed_cost(self, cost_ms: float) -> None:
        """Оценка стоимости фикс-шага: быстро растёт, медленно спадает."""
        self._fixed_last_ms = cost_ms
        if cost_ms >= self._fixed_ema_ms:
            self._fixed_ema_ms = 0.5 * self._fixed_ema_ms + 0.5 * cost_ms
        else:
            self._fixed_ema_ms += 0.1 * (cost_ms - self._fixed_ema_ms)

    def _warn_budget(self, spent_ms: float) -> None:
        """Замедление времени из-за бюджета: сводка не чаще раза в секунду."""
        self._warn_budget_stops += 1
        now = self._clock()
        if now - self._warn_budget_at < 1.0:
            return
        _log.warning(
            "Физика не укладывается в бюджет %.1f мс: игровое время замедлено "
            "(%d кадров за секунду, шаг ≈%.1f мс, последний кадр %.1f мс)",
            self._max_fixed_ms or 0.0,
            self._warn_budget_stops,
            max(self._fixed_ema_ms, self._fixed_last_ms),
            spent_ms,
        )
        self._warn_budget_at = now
        self._warn_budget_stops = 0

    def _warn_overrun(self, dropped: float) -> None:
        """Предупреждение о нехватке времени: первое сразу, далее не чаще 1/с."""
        self._warn_overruns += 1
        self._warn_dropped += dropped
        now = self._clock()
        if now - self._warn_at < 1.0:
            return
        _log.warning(
            "Кадр не успевает: %d раз, пропущено %.3f с симуляции (max_substeps=%d)",
            self._warn_overruns,
            self._warn_dropped,
            self._max_substeps,
        )
        self._warn_at = now
        self._warn_overruns = 0
        self._warn_dropped = 0.0

    def run(self, max_frames: int | None = None) -> int:
        """Крутить цикл в реальном времени до :meth:`stop` или лимита кадров.

        Args:
            max_frames: Остановиться после указанного числа кадров.

        Returns:
            Число выполненных кадров.
        """
        self._is_running = True
        self._last_time = time.perf_counter()
        frames = 0
        period = None if self._target_fps is None else 1.0 / self._target_fps
        try:
            while self._is_running and (max_frames is None or frames < max_frames):
                frame_started = time.perf_counter()
                self.step()
                frames += 1
                if period is not None:
                    remaining = period - (time.perf_counter() - frame_started)
                    if remaining > 0.0:
                        time.sleep(remaining)
        except KeyboardInterrupt:  # pragma: no cover - интерактивный сценарий
            _log.info("Цикл остановлен пользователем (Ctrl+C)")
        finally:
            self._is_running = False
        return frames

    # ------------------------------------------------------------------ #
    # Внутренние фазы
    # ------------------------------------------------------------------ #
    def _run_fixed(self, dt: float) -> None:
        scene = self._scene_ref
        if scene is not None:
            self._guard("fixed_update", scene.fixed_update, dt)
        for hook in list(self._fixed_hooks):
            self._guard("fixed hook", hook, dt)

    def _run_update(self, dt: float) -> None:
        scene = self._scene_ref
        if scene is not None:
            self._guard("update", scene.update, dt)
        for hook in list(self._update_hooks):
            self._guard("update hook", hook, dt)

    def _run_late(self, dt: float) -> None:
        scene = self._scene_ref
        if scene is not None:
            self._guard("late_update", scene.late_update, dt)
        for hook in list(self._late_hooks):
            self._guard("late hook", hook, dt)

    def _run_render(self, dt: float, alpha: float) -> None:
        for hook in list(self._render_hooks):
            self._guard("render hook", hook, dt, alpha)

    def _guard(self, phase: str, func: Callable[..., None], *args: float) -> None:
        """Вызвать фазу, изолировав ошибку при ``continue_on_error``."""
        try:
            func(*args)
        except Exception:  # noqa: BLE001 - политика задаётся флагом
            if not self._continue_on_error:
                raise
            _log.exception("Ошибка в фазе '%s' (кадр %d)", phase, self.time.frame)

    def __repr__(self) -> str:
        if self._paused:
            state = "paused"
        elif self._is_running:
            state = "running"
        else:
            state = "idle"
        return (
            f"<GameLoop {state} frame={self.time.frame} "
            f"fixed={self.time.fixed_delta:.4f}s fps={self.stats.fps:.1f}>"
        )
