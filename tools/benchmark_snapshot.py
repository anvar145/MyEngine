"""Замер стоимости снимков сцены (A12 п.0): ``scene_to_dict``/``scene_from_dict``.

Запуск (одним процессом, без фоновой нагрузки):

    QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe tools/benchmark_snapshot.py
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core.components import MeshRenderer  # noqa: E402
from engine.core.event_bus import EventBus  # noqa: E402
from engine.core.math3d import Vector3  # noqa: E402
from engine.core.physics_components import ColliderComponent  # noqa: E402
from engine.core.scene import Scene  # noqa: E402
from engine.scene_io import scene_from_dict, scene_to_dict  # noqa: E402

SIZES = (100, 1000, 5000, 10000, 20000)


def build_scene(count: int) -> Scene:
    """Сцена из ``count`` объектов: группы по 10 с вложенностью, меш и коллайдер."""
    scene = Scene("Bench", event_bus=EventBus())
    parents = []
    for i in range(count):
        parent = parents[i // 10 - 1] if i >= 10 and i % 10 else None
        obj = scene.create_game_object(
            f"Object {i}", parent, position=Vector3(i % 50, (i // 50) % 7, i // 350)
        )
        obj.add_component(MeshRenderer(mesh="cube"))
        if i % 3 == 0:
            obj.add_component(ColliderComponent())
        parents.append(obj)
    return scene


def best_of(function, repeats: int) -> float:
    """Минимум времени (мс) из нескольких прогонов."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000.0)
    return min(times)


def main() -> None:
    print(
        "| объектов | to_dict, мс | JSON, МБ | пик памяти to_dict, МБ "
        "| from_dict, мс |"
    )
    print("|---:|---:|---:|---:|---:|")
    for count in SIZES:
        scene = build_scene(count)
        repeats = 5 if count <= 5000 else 3
        to_ms = best_of(lambda s=scene: scene_to_dict(s), repeats)
        data = scene_to_dict(scene)
        size = len(json.dumps(data)) / 1e6
        tracemalloc.start()
        scene_to_dict(scene)
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        from_ms = best_of(lambda d=data: scene_from_dict(d), repeats)
        print(f"| {count} | {to_ms:.1f} | {size:.2f} | {peak:.1f} | {from_ms:.1f} |")


if __name__ == "__main__":
    main()
