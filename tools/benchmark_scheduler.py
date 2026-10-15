r"""Замер стоимости планировщика: 1000 задач, 60 кадров по 1/60 с.

Запуск::

    .venv\Scripts\python.exe tools\benchmark_scheduler.py

Девять сценариев: таймеры и корутины (1000), твины float / Vector3 /
``transform.position`` в стресс-размере 1000 и в реалистичном 100.
Метрики: среднее и p95 времени ``update_game`` за кадр (мс); сценарий повторяется
``REPEATS`` раз, сообщается медиана средних и p95 по всем кадрам. Результат
печатается и пишется в ``_work/a3b_scheduler_bench.md``.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine.core.math3d import Vector3  # noqa: E402
from engine.core.scene import Scene  # noqa: E402
from engine.core.scheduling import Scheduler, Wait  # noqa: E402

N = 1000
FRAMES = 60
DT = 1.0 / 60.0
REPEATS = 7
# 1-3: цель; 4-6: стресс-тест на 1000 твинов (не цель); 7-9: реалистичные 100 твинов
GOAL_MS: dict[int, float | None] = {
    1: 1.0, 2: 1.0, 3: 1.0, 4: None, 5: None, 6: None, 7: 0.5, 8: 0.5, 9: 1.0,
}


def _noop() -> None:
    return None


class _Box:
    """Цель твина с обычными атрибутами."""

    __slots__ = ("f", "v")

    def __init__(self) -> None:
        self.f = 0.0
        self.v = Vector3(0.0, 0.0, 0.0)


def _after(sched: Scheduler):
    for _ in range(N):
        sched.after(1.0, _noop)
    return None


def _every(sched: Scheduler):
    for _ in range(N):
        sched.every(0.1, _noop)
    return None


def _coroutines(sched: Scheduler):
    def routine():
        yield Wait(0.5)

    for _ in range(N):
        sched.start_coroutine(routine())
    return None


def _tween_float(sched: Scheduler, n: int = N):
    boxes = [_Box() for _ in range(n)]
    for box in boxes:
        sched.tween(box, "f", 1.0, 1.0)
    return boxes


def _tween_vector(sched: Scheduler, n: int = N):
    boxes = [_Box() for _ in range(n)]
    end = Vector3(1.0, 2.0, 3.0)
    for box in boxes:
        sched.tween(box, "v", end, 1.0)
    return boxes


def _tween_position(sched: Scheduler, n: int = N):
    scene = Scene("bench")
    objects = [scene.create_game_object(f"o{i}") for i in range(n)]
    end = Vector3(1.0, 2.0, 3.0)
    for obj in objects:
        sched.tween(obj.transform, "position", end, 1.0)
    return (scene, objects)


SCENARIOS: list[tuple[str, Callable[[Scheduler], object]]] = [
    ("1000 × after(1.0, noop)", _after),
    ("1000 × every(0.1, noop)", _every),
    ("1000 корутин wait(0.5)", _coroutines),
    ("1000 твинов float", _tween_float),
    ("1000 твинов Vector3 (атрибут)", _tween_vector),
    ("1000 твинов transform.position", _tween_position),
    ("100 твинов float", lambda s: _tween_float(s, 100)),
    ("100 твинов Vector3 (атрибут)", lambda s: _tween_vector(s, 100)),
    ("100 твинов transform.position", lambda s: _tween_position(s, 100)),
]


def _run_once(setup: Callable[[Scheduler], object]) -> list[float]:
    sched = Scheduler()
    keep = setup(sched)  # noqa: F841 - удерживаем цели до конца замера
    times: list[float] = []
    for _ in range(FRAMES):
        start = time.perf_counter()
        sched.update_game(DT)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def main() -> None:
    rows: list[tuple[int, str, float, float]] = []
    for index, (title, setup) in enumerate(SCENARIOS, start=1):
        _run_once(setup)  # прогрев
        means: list[float] = []
        pooled: list[float] = []
        for _ in range(REPEATS):
            times = _run_once(setup)
            means.append(statistics.fmean(times))
            pooled.extend(times)
        rows.append((index, title, statistics.median(means), _p95(pooled)))
        print(f"{index}. {title}: mean {rows[-1][2]:.3f} ms, p95 {rows[-1][3]:.3f} ms")

    lines = [
        "# Планировщик: стоимость 1000 задач (A3b)",
        "",
        f"{N} задач, {FRAMES} кадров по 1/60 с, `Scheduler.update_game`; "
        f"медиана средних из {REPEATS} прогонов, p95 по всем кадрам.",
        "",
        "| # | Сценарий | Среднее, мс/кадр | p95, мс | Цель, мс | Итог |",
        "|---|---|---|---|---|---|",
    ]
    for index, title, mean, p95 in rows:
        goal = GOAL_MS[index]
        if goal is None:
            goal_text, verdict = "-", "стресс-тест"
        else:
            goal_text = f"≤{goal:g}"
            verdict = "ok" if mean <= goal else "выше цели"
        lines.append(
            f"| {index} | {title} | {mean:.3f} | {p95:.3f} | {goal_text} | {verdict} |"
        )
    text = "\n".join(lines) + "\n"
    out = ROOT / "_work" / "a3b_scheduler_bench.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\nзаписано: {out}")


if __name__ == "__main__":
    main()
