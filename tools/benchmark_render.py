"""Замер производительности рендеринга: инстансинг против поштучной отрисовки.

Запуск::

    .venv\\Scripts\\python.exe tools\\benchmark_render.py

Скрипт работает без окна (offscreen-контекст) и показывает, сколько стоит кадр
при разном числе объектов и насколько GPU-инстансинг экономит вызовы отрисовки
— ключевой показатель для слабого железа.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

import numpy as np  # noqa: E402

from engine.core.math3d import Vector3, compose_matrix  # noqa: E402
from engine.core.resources import ResourceManager, cube_mesh  # noqa: E402
from engine.rendering import (  # noqa: E402
    Camera,
    GpuMesh,
    QualityTier,
    Renderer,
    RenderSettings,
    RenderTarget,
)
from engine.rendering.qt_context import QtOffscreenContext  # noqa: E402

WIDTH, HEIGHT = 1280, 720
FRAMES = 20
COUNTS = (500, 2000, 5000)


def build_matrices(count: int) -> np.ndarray:
    """Собрать матрицы объектов, разложенных сеткой по плоскости XZ."""
    side = int(count**0.5) + 1
    return np.stack(
        [
            compose_matrix(
                position=Vector3(
                    (index % side - side / 2) * 1.5,
                    0.0,
                    (index // side - side / 2) * 1.5,
                )
            )
            for index in range(count)
        ]
    )


def measure_frame(
    renderer: Renderer, camera: Camera, target: RenderTarget, draw
) -> float:
    """Измерить среднее время кадра в миллисекундах.

    Args:
        renderer: Рендерер.
        camera: Камера кадра.
        target: Цель вывода.
        draw: Функция, выполняющая постановку объектов в очередь.

    Returns:
        Среднее время кадра в миллисекундах.
    """
    # Прогрев: сборка программ, загрузка меша, создание буферов экземпляров.
    renderer.begin_frame(camera, (WIDTH, HEIGHT), target=target)
    draw()
    renderer.end_frame(target=target)

    started = time.perf_counter()
    for _ in range(FRAMES):
        renderer.begin_frame(camera, (WIDTH, HEIGHT), target=target)
        draw()
        renderer.end_frame(target=target)
    return (time.perf_counter() - started) / FRAMES * 1000.0


def main() -> int:
    """Прогнать замеры и напечатать таблицу."""
    settings = RenderSettings.for_quality(QualityTier.LOW, samples=0)
    with QtOffscreenContext((WIDTH, HEIGHT), settings=settings) as _context:
        resources = ResourceManager(roots=[ROOT / "assets"])
        renderer = Renderer(resources=resources, shader_root=ROOT / "assets")
        renderer.initialize()
        print(f"Драйвер: {renderer.info.describe()}")
        print(f"Кадр {WIDTH}x{HEIGHT}, усреднение по {FRAMES} кадрам\n")
        print(
            f"Инстансинг доступен: {renderer.capabilities.supports_instancing}; "
            f"программы: {', '.join(sorted(renderer.programs))}"
        )

        mesh = GpuMesh.from_mesh_asset(cube_mesh(1.0))
        material = renderer.create_material("bench", base_color=Vector3(0.8, 0.6, 0.3))
        target = RenderTarget("bench", WIDTH, HEIGHT)

        camera = Camera(fov=45.0, far=500.0)
        camera.position = Vector3(40.0, 34.0, 40.0)
        camera.look_at(Vector3(0.0, 0.0, 0.0))

        print(f"{'объектов':>9} | {'инстансинг, мс':>15} | {'поштучно, мс':>13} | "
              f"{'ускорение':>10} | {'draws (inst)':>12} | {'inst':>6} | "
              f"{'draws (loop)':>12}")
        print("-" * 104)
        for count in COUNTS:
            matrices = build_matrices(count)

            def draw_instanced(data: np.ndarray = matrices) -> None:
                renderer.draw_instanced(mesh, material, data)

            def draw_loop(data: np.ndarray = matrices) -> None:
                for matrix in data:
                    renderer.submit(mesh, material, matrix)

            renderer.cull_by_frustum = False
            instanced_ms = measure_frame(renderer, camera, target, draw_instanced)
            renderer.begin_frame(camera, (WIDTH, HEIGHT), target=target)
            renderer.draw_instanced(mesh, material, matrices)
            stats = renderer.end_frame(target=target)
            # Статистика — изменяемый объект: снимаем значения сразу, иначе
            # следующий прогон перезапишет их.
            instanced_draws, instanced_count = stats.draw_calls, stats.instances

            loop_ms = measure_frame(renderer, camera, target, draw_loop)
            renderer.begin_frame(camera, (WIDTH, HEIGHT), target=target)
            draw_loop()
            stats = renderer.end_frame(target=target)
            loop_draws = stats.draw_calls

            speedup = loop_ms / instanced_ms if instanced_ms > 0 else 0.0
            print(
                f"{count:>9} | {instanced_ms:>15.2f} | {loop_ms:>13.2f} | "
                f"{speedup:>9.1f}x | {instanced_draws:>12} | {instanced_count:>6} | "
                f"{loop_draws:>12}"
            )

        # Отсечение: половина объектов уводится за камеру.
        renderer.cull_by_frustum = True
        matrices = build_matrices(COUNTS[-1])
        offset = compose_matrix(position=Vector3(0, 0, -600))
        half = len(matrices) // 2
        behind = np.stack([offset @ matrix for matrix in matrices[:half]])

        def draw_mixed() -> None:
            renderer.draw_instanced(mesh, material, behind)
            renderer.submit(mesh, material, compose_matrix())

        renderer.begin_frame(camera, (WIDTH, HEIGHT), target=target)
        draw_mixed()
        stats = renderer.end_frame(target=target)
        print(
            f"\nОтсечение: из {len(matrices)} объектов отброшено {stats.culled}, "
            f"нарисовано вызовов {stats.draw_calls}"
        )

        target.dispose()
        mesh.dispose()
        renderer.dispose()
        resources.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
