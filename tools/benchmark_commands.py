"""Замер команд отмены (A12): правка/undo/redo на сцене в N объектов.

Запуск (одним процессом, без фоновой нагрузки):

    QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe tools/benchmark_commands.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_snapshot import build_scene  # noqa: E402

from engine.core.commands import (  # noqa: E402
    CommandStack,
    DeleteObjectCommand,
    ReparentCommand,
    SetFieldCommand,
)
from engine.core.math3d import Vector3  # noqa: E402

SIZES = (1000, 10000, 20000)


def timed(function, repeats: int = 20) -> float:
    """Медиана времени (мс)."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def measure(count: int) -> str:
    """Строка таблицы для сцены в ``count`` объектов."""
    scene = build_scene(count)
    stack = CommandStack(scene)
    target = scene.objects[count // 2]
    counter = itertools.count(1)

    def edit() -> None:
        i = next(counter)
        old = target.transform.position
        stack.execute(
            SetFieldCommand(
                target,
                target.transform,
                "position",
                old,
                Vector3(float(i), 0.0, 0.0),
                stamp=float(i) * 10.0,
            )
        )

    edit_ms = timed(edit)
    undo_ms = timed(stack.undo)
    redo_ms = timed(stack.redo)

    victims = iter([o for o in scene.objects if o.parent is None][:25])

    def delete() -> None:
        stack.execute(DeleteObjectCommand(next(victims)))

    del_ms = timed(delete, 20)
    undo_del = timed(stack.undo, 20)

    roots = [o for o in scene.objects if o.parent is None][-30:]
    dest = roots[0]
    movers = iter(roots[1:])

    def reparent() -> None:
        stack.execute(ReparentCommand(next(movers), dest))

    rep_ms = timed(reparent, 20)
    return (
        f"| {count} | {edit_ms:.3f} | {undo_ms:.3f} | {redo_ms:.3f} "
        f"| {del_ms:.2f} | {undo_del:.2f} | {rep_ms:.3f} |"
    )


def main() -> None:
    print(
        "| объектов | правка поля, мс | undo, мс | redo, мс | удалить поддерево, мс "
        "| undo удаления, мс | reparent, мс |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for count in SIZES:
        print(measure(count))


if __name__ == "__main__":
    main()
