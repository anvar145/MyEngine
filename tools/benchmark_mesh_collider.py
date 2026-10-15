"""Замеры mesh-коллайдеров (A11): построение BVH, raycast, shapecast персонажа.

Запуск (одним процессом, без фоновой нагрузки):

    .venv/Scripts/python.exe tools/benchmark_mesh_collider.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine" / "tests"))

from engine.physics.bvh import Bvh  # noqa: E402
from engine.physics.character import CharacterController  # noqa: E402
from engine.physics.collider import Collider  # noqa: E402
from engine.physics.shape import BoxShape, CapsuleShape  # noqa: E402
from engine.physics.world import PhysicsWorld  # noqa: E402

DT = 1.0 / 60.0


def bumpy_grid(cells: int, size: float, amplitude: float) -> tuple:
    """Холмистая сетка cells² клеток (2·cells² треугольников)."""
    xs = np.linspace(-size, size, cells + 1)
    gx, gz = np.meshgrid(xs, xs, indexing="ij")
    gy = amplitude * np.sin(gx * 0.7) * np.cos(gz * 0.5)
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    index = np.arange((cells + 1) ** 2).reshape(cells + 1, cells + 1)
    a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    c, d = index[1:, :-1].ravel(), index[1:, 1:].ravel()
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    return verts, tris


def best_of(function, repeats: int = 7) -> float:
    """Минимум времени (с) из нескольких прогонов."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


def bench_build() -> None:
    for cells in (50, 71, 100):
        verts, tris = bumpy_grid(cells, 50.0, 1.0)
        spent = best_of(lambda v=verts, t=tris: Bvh(v, t), 3)
        print(f"build BVH {len(tris):>6} tri: {spent * 1000:8.1f} ms")


def bench_raycast() -> None:
    verts, tris = bumpy_grid(71, 50.0, 1.0)
    world = PhysicsWorld()
    world.add_mesh(Bvh(verts, tris))
    rng = np.random.default_rng(1)
    origins = np.column_stack(
        [rng.uniform(-45, 45, 1000), np.full(1000, 10.0), rng.uniform(-45, 45, 1000)]
    )
    down = (0.0, -1.0, 0.0)

    def run() -> int:
        return sum(world.raycast(o, down, 50.0) is not None for o in origins)

    hits = run()
    spent = best_of(run)
    print(f"raycast 1000 лучей, {len(tris)} tri: {spent * 1000:7.1f} ms "
          f"({spent * 1e3:.0f} мкс/луч), попаданий {hits}")


def bench_packet() -> None:
    """Пакет из 100 лучей: цикл ``raycast`` против ``raycast_batch``."""
    verts, tris = bumpy_grid(71, 50.0, 1.0)
    world = PhysicsWorld()
    world.add_mesh(Bvh(verts, tris))
    rng = np.random.default_rng(2)
    origins = np.column_stack(
        [rng.uniform(-45, 45, 100), np.full(100, 10.0), rng.uniform(-45, 45, 100)]
    )
    dirs = np.tile((0.0, -1.0, 0.0), (100, 1))
    pairs = list(zip(origins, dirs, strict=True))
    loop = best_of(lambda: [world.raycast(o, d, 50.0) for o, d in pairs])
    batch = best_of(lambda: world.raycast_batch(origins, dirs, 50.0))
    print(f"пакет 100 лучей: цикл {loop * 1000:.2f} ms, "
          f"raycast_batch {batch * 1000:.2f} ms")


def bench_character() -> None:
    verts, tris = bumpy_grid(71, 50.0, 0.3)
    world = PhysicsWorld()
    world.add_mesh(Bvh(verts, tris))
    char = CharacterController(world, (0.0, 3.0, 0.0))
    for _ in range(90):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    samples = []
    for i in range(300):
        start = time.perf_counter()
        char.move((4.5 * DT, 0.0, 1.5 * DT * math.sin(i * 0.05)))
        samples.append(time.perf_counter() - start)
        world.step()
    arr = np.array(samples) * 1000.0
    print(f"character.move на mesh {len(tris)} tri: "
          f"p50 {np.median(arr):.3f} ms, p95 {np.percentile(arr, 95):.3f} ms")
    start = time.perf_counter()
    shape = CapsuleShape(0.4, 0.5)
    for _ in range(1000):
        world.shapecast(shape, char.position, (0, 0, 0, 1), (0, -1, 0), 0.5,
                        ignore=char.handle)
    spent = (time.perf_counter() - start) / 1000
    print(f"shapecast капсулы (0.5 м вниз, без персонажа): {spent * 1e6:.0f} мкс")


def bench_accuracy() -> None:
    """Попадание луча по рампе: box по габариту против mesh."""
    rise, x0, x1 = 1.2, -2.0, 2.0
    verts = np.array([[x0, 0, -1], [x1, rise, -1], [x1, rise, 1], [x0, 0, 1],
                      [x1, 0, -1], [x1, 0, 1]], float)
    tris = np.array([[0, 1, 2], [0, 2, 3], [4, 1, 0], [5, 3, 2], [5, 2, 1],
                     [4, 5, 1], [0, 3, 5], [0, 5, 4]])
    mesh_world = PhysicsWorld()
    mesh_world.add_mesh(Bvh(verts, tris))
    box_world = PhysicsWorld()
    box_world.add_static(
        Collider(BoxShape((2.0, rise / 2, 1.0))), (0.0, rise / 2, 0.0)
    )
    errors = {"box": [], "mesh": []}
    for x in np.linspace(-1.9, 1.9, 39):
        truth = rise * (x - x0) / (x1 - x0)
        for name, world in (("box", box_world), ("mesh", mesh_world)):
            hit = world.raycast((x, 5.0, 0.0), (0.0, -1.0, 0.0), 10.0)
            errors[name].append(abs(float(hit.point[1]) - truth) if hit else np.nan)
    for name, values in errors.items():
        mean, worst = np.nanmean(values) * 100, np.nanmax(values) * 100
        print(f"точность луча по рампе ({name}): ошибка высоты "
              f"средняя {mean:.1f} см, макс {worst:.1f} см")


if __name__ == "__main__":
    bench_build()
    bench_raycast()
    bench_packet()
    bench_character()
    bench_accuracy()
