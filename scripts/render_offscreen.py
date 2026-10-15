"""Отрисовать сцену в PNG без окна (headless-рендер).

Запуск::

    .venv\\Scripts\\python.exe scripts\\render_offscreen.py --mesh sphere ^
        --out _work/render.png
    .venv\\Scripts\\python.exe scripts\\render_offscreen.py --instances 400 ^
        --quality high

Скрипт используется для проверки рендера, golden-тестов и генерации превью:
он создаёт offscreen-контекст OpenGL, рисует сцену в framebuffer и сохраняет
результат в PNG (кодировщик свой, без Pillow).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Платформа Qt "offscreen" не умеет создавать GL-контекст, поэтому для
# headless-рендера нужна оконная платформа (окно при этом не показывается).
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
    parser = argparse.ArgumentParser(description="Headless-рендер сцены MyEngine в PNG")
    parser.add_argument(
        "--mesh", default="cube", help="cube | sphere | plane | grid | путь.obj"
    )
    parser.add_argument("--out", default="_work/render.png", help="куда сохранить PNG")
    parser.add_argument("--width", type=int, default=960, help="ширина кадра")
    parser.add_argument("--height", type=int, default=640, help="высота кадра")
    parser.add_argument(
        "--instances", type=int, default=0, help="число кубов (инстансинг)"
    )
    parser.add_argument(
        "--quality", default="medium", choices=[tier.value for tier in QualityTier]
    )
    parser.add_argument("--no-grid", action="store_true", help="не рисовать сетку")
    return parser.parse_args(argv)


def load_mesh(name: str, resources: ResourceManager) -> GpuMesh:
    """Получить меш по имени: процедурный или загруженный из файла."""
    match name.lower():
        case "cube":
            return GpuMesh.from_mesh_asset(cube_mesh(1.0))
        case "sphere":
            return GpuMesh.from_mesh_asset(uv_sphere_mesh(0.5, segments=32, rings=20))
        case "plane":
            return GpuMesh.from_mesh_asset(plane_mesh(6.0, subdivisions=1))
        case _:
            asset = resources.load(name)
            return GpuMesh.from_mesh_asset(asset)


def main(argv: list[str] | None = None) -> int:
    """Отрисовать кадр и сохранить его в файл."""
    args = parse_args(argv)
    quality = QualityTier(args.quality)
    settings = RenderSettings.for_quality(quality, samples=0)
    resources = ResourceManager(roots=[ROOT / "assets"], name="OffscreenDemo")

    print(
        f"Рендер: меш={args.mesh}, {args.width}x{args.height}, "
        f"качество={quality.value}"
    )
    with QtOffscreenContext((args.width, args.height), settings=settings) as context:
        width, height = context.surface.size()
        renderer = Renderer(
            resources=resources, settings=settings, shader_root=ROOT / "assets"
        )
        renderer.initialize()
        print(f"Драйвер : {renderer.info.describe()}")
        print(f"Лимиты  : {renderer.capabilities.describe()}")

        mesh = load_mesh(args.mesh, resources)
        material = renderer.create_material(
            "demo", base_color=Vector3(0.85, 0.55, 0.22), roughness=0.35, metallic=0.1
        )
        grid_mesh = GpuMesh.from_mesh_asset(line_grid_mesh(10.0, divisions=10))
        grid_material = renderer.create_material(
            "grid", program="unlit", base_color=Vector3(0.28, 0.30, 0.36)
        )

        camera = Camera(fov=45.0, far=500.0)
        if args.instances > 0:
            # Для россыпи объектов отводим камеру пропорционально их количеству.
            span = (int(args.instances**0.5) + 1) * 1.4
            camera.position = Vector3(span * 0.9, span * 1.05, span * 0.9)
        else:
            camera.position = Vector3(3.2, 2.4, 3.6)
        camera.look_at(Vector3(0.0, 0.0, 0.0))

        target = RenderTarget("capture", width, height)
        renderer.begin_frame(camera, (width, height), target=target)

        from OpenGL import GL

        if not args.no_grid:
            renderer.submit(
                grid_mesh, grid_material, compose_matrix(), mode=GL.GL_LINES
            )
        renderer.submit(mesh, material, compose_matrix())

        if args.instances > 0:
            # Разбрасываем объекты «лесом»: случайные смещение, поворот и размер,
            # чтобы кадр показывал объём, а не плоскую плиту.
            rng = np.random.default_rng(42)
            side = int(args.instances**0.5) + 1
            span = side * 1.4
            matrices = []
            for index in range(args.instances):
                size = float(rng.uniform(0.6, 1.4))
                position = Vector3(
                    (index % side - side / 2) * 1.4 + float(rng.uniform(-0.3, 0.3)),
                    size * 0.5,
                    (index // side - side / 2) * 1.4 + float(rng.uniform(-0.3, 0.3)),
                )
                rotation = Quaternion.from_axis_angle(
                    Vector3.up(), float(rng.uniform(0.0, 360.0))
                )
                matrices.append(
                    compose_matrix(
                        position=position,
                        rotation=rotation,
                        scale=Vector3(size, size, size),
                    )
                )
            drawn = renderer.draw_instanced(mesh, material, matrices)
            print(
                f"Инстансинг: нарисовано {drawn} экземпляров "
                f"(поле {span:.0f}x{span:.0f})"
            )

        stats = renderer.end_frame(target=target)
        print(f"Статистика: {stats.describe()}")

        pixels = target.read_pixels()
        output = save_png(ROOT / args.out, pixels, flip_vertical=True)
        print(f"Сохранено: {output} ({output.stat().st_size} байт)")

        renderer.dispose()
        mesh.dispose()
        grid_mesh.dispose()
        target.dispose()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
