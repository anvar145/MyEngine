"""Витринная сцена: плоскость, куб, сфера, сетка и россыпь инстансов.

Запуск::

    .venv\\Scripts\\python.exe scripts\\render_showcase.py --out _work/showcase.png

Скрипт рендерит «натюрморт» без окна и сохраняет PNG — удобно для проверки
света, глубины, прозрачности и инстансинга одним кадром.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Платформа Qt "offscreen" не создаёт GL-контекст: нужна оконная.
os.environ.setdefault("QT_QPA_PLATFORM", "windows")

import numpy as np  # noqa: E402

from engine.core.math3d import Quaternion, Vector3, compose_matrix  # noqa: E402
from engine.core.resources import (  # noqa: E402
    ResourceManager,
    cube_mesh,
    line_grid_mesh,
    plane_mesh,
    uv_sphere_mesh,
)
from engine.rendering import (  # noqa: E402
    Camera,
    DirectionalLight,
    GpuMesh,
    QualityTier,
    Renderer,
    RenderSettings,
    RenderTarget,
    save_png,
)
from engine.rendering.qt_context import QtOffscreenContext  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Витринный рендер сцены MyEngine")
    parser.add_argument(
        "--out", default="_work/showcase.png", help="куда сохранить PNG"
    )
    parser.add_argument("--width", type=int, default=1120, help="ширина кадра")
    parser.add_argument("--height", type=int, default=700, help="высота кадра")
    parser.add_argument("--instances", type=int, default=96, help="число кубиков")
    parser.add_argument(
        "--quality", default="medium", choices=[tier.value for tier in QualityTier]
    )
    return parser.parse_args(argv)


def scatter_matrices(count: int, radius: float = 6.5) -> np.ndarray:
    """Разложить кубики двумя кольцами вокруг центра сцены.

    Args:
        count: Общее число кубиков.
        radius: Радиус внешнего кольца.

    Returns:
        Массив матриц ``(count, 4, 4)``.
    """
    matrices: list[np.ndarray] = []
    for index in range(count):
        ring = index % 2
        ring_radius = radius * (0.55 if ring == 0 else 1.0)
        angle = (index / max(1, count)) * math.tau * (3.0 if ring == 0 else 1.0)
        size = 0.28 if ring == 0 else 0.42
        matrices.append(
            compose_matrix(
                position=Vector3(
                    math.cos(angle) * ring_radius,
                    size * 0.5,
                    math.sin(angle) * ring_radius,
                ),
                rotation=Quaternion.from_axis_angle(Vector3.up(), math.degrees(angle)),
                scale=Vector3(size, size, size),
            )
        )
    return np.stack(matrices)


def main(argv: list[str] | None = None) -> int:
    """Отрисовать витринную сцену и сохранить её в PNG."""
    args = parse_args(argv)
    quality = QualityTier(args.quality)
    settings = RenderSettings.for_quality(quality, samples=0)

    with QtOffscreenContext((args.width, args.height), settings=settings):
        resources = ResourceManager(roots=[ROOT / "assets"], name="Showcase")
        renderer = Renderer(
            resources=resources, settings=settings, shader_root=ROOT / "assets"
        )
        renderer.initialize()
        renderer.light = DirectionalLight(
            direction=Vector3(-0.45, -1.0, -0.35), ambient=0.28, intensity=1.05
        )
        print(f"Драйвер : {renderer.info.describe()}")
        print(f"Качество: {quality.value}, кадр {args.width}x{args.height}")

        ground = GpuMesh.from_mesh_asset(plane_mesh(24.0, subdivisions=4))
        cube = GpuMesh.from_mesh_asset(cube_mesh(1.6))
        sphere = GpuMesh.from_mesh_asset(uv_sphere_mesh(1.1, segments=48, rings=32))
        pebble = GpuMesh.from_mesh_asset(cube_mesh(1.0))
        grid = GpuMesh.from_mesh_asset(line_grid_mesh(24.0, divisions=24))

        ground_material = renderer.create_material(
            "ground", base_color=Vector3(0.24, 0.26, 0.30), roughness=0.9
        )
        cube_material = renderer.create_material(
            "cube", base_color=Vector3(0.88, 0.55, 0.20), roughness=0.4, metallic=0.05
        )
        sphere_material = renderer.create_material(
            "sphere", base_color=Vector3(0.35, 0.62, 0.85), roughness=0.15, metallic=0.4
        )
        pebble_material = renderer.create_material(
            "pebble", base_color=Vector3(0.55, 0.58, 0.62), roughness=0.7
        )
        glass_material = renderer.create_material(
            "glass",
            base_color=Vector3(0.95, 0.98, 1.0),
            transparent=True,
            alpha=0.3,
            roughness=0.05,
        )
        grid_material = renderer.create_material(
            "grid", program="unlit", base_color=Vector3(0.30, 0.33, 0.40)
        )

        camera = Camera(fov=48.0, far=300.0)
        camera.position = Vector3(13.0, 9.5, 15.0)
        camera.look_at(Vector3(0.0, 0.8, 0.0))

        target = RenderTarget("showcase", args.width, args.height)
        renderer.begin_frame(camera, (args.width, args.height), target=target)

        from OpenGL import GL

        renderer.submit(ground, ground_material, compose_matrix())
        renderer.submit(grid, grid_material, compose_matrix(), mode=GL.GL_LINES)
        renderer.submit(
            cube, cube_material, compose_matrix(position=Vector3(-3.2, 0.8, 0.4))
        )
        renderer.submit(
            sphere, sphere_material, compose_matrix(position=Vector3(3.0, 1.1, -0.4))
        )
        renderer.submit(
            cube,
            glass_material,
            compose_matrix(
                position=Vector3(0.0, 2.4, 0.0), scale=Vector3(1.1, 1.1, 1.1)
            ),
        )
        instances = scatter_matrices(args.instances)
        drawn = renderer.draw_instanced(pebble, pebble_material, instances)

        stats = renderer.end_frame(target=target)
        print(f"Инстансов: {drawn}, статистика: {stats.describe()}")

        output = save_png(ROOT / args.out, target.read_pixels(), flip_vertical=True)
        print(f"Сохранено: {output} ({output.stat().st_size} байт)")

        for mesh in (ground, cube, sphere, pebble, grid):
            mesh.dispose()
        target.dispose()
        renderer.dispose()
        resources.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
