"""Замер освещения: сколько стоит свет на интегрированной графике.

Скрипт отвечает на вопрос из требований — «не снижает ли добавление света FPS».
Сравниваются кадры с разным числом источников и с отключённым динамическим
освещением (запечённый путь), а также стоимость прохода карты теней.

Запуск::

    .venv\\Scripts\\python.exe tools\\benchmark_lighting.py
    .venv\\Scripts\\python.exe tools\\benchmark_lighting.py --objects 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from engine.core.math3d import Vector3, compose_matrix  # noqa: E402
from engine.core.resources import ResourceManager, cube_mesh, plane_mesh  # noqa: E402
from engine.rendering import (  # noqa: E402
    AmbientLight,
    Camera,
    DirectionalLight,
    GpuMesh,
    PointLight,
    Renderer,
    RenderSettings,
    RenderTarget,
    SpotLight,
)
from engine.rendering.qt_context import (  # noqa: E402
    QtOffscreenContext,
    apply_surface_format,
)

FRAME_SIZE = (1280, 720)
FRAMES = 20
WARMUP = 5


def build_lights(count: int) -> list:
    """Собрать набор света: солнце, фон и ``count`` местных источников.

    Args:
        count: Сколько точечных источников и прожекторов добавить (поровну).

    Returns:
        Список источников.
    """
    lights: list = [
        DirectionalLight(direction=Vector3(-0.5, -1.0, -0.3), cast_shadow=True),
        AmbientLight(intensity=0.18),
    ]
    for index in range(count):
        angle = index * (2.0 * np.pi / max(1, count))
        radius = 8.0
        position = Vector3(
            float(np.cos(angle) * radius), 3.0, float(np.sin(angle) * radius)
        )
        if index % 2 == 0:
            lights.append(
                PointLight(position=position, range=12.0, intensity=1.5)
            )
        else:
            lights.append(
                SpotLight(
                    position=position,
                    direction=Vector3(-position.x, -3.0, -position.z),
                    range=14.0,
                    intensity=1.5,
                )
            )
    return lights


def build_items(renderer: Renderer, objects: int) -> list[tuple]:
    """Разложить объекты сеткой по полу.

    Args:
        renderer: Рендерер (нужен для создания материалов).
        objects: Сколько объектов разместить.

    Returns:
        Список ``(меш, материал, матрица)``.
    """
    floor = GpuMesh.from_mesh_asset(plane_mesh(60.0))
    cube = GpuMesh.from_mesh_asset(cube_mesh(1.0))
    stone = renderer.create_material("floor", base_color=Vector3(0.5, 0.5, 0.55))
    items = [(floor, stone, compose_matrix(position=Vector3(0, -0.05, 0)))]
    side = int(np.ceil(np.sqrt(objects)))
    palette = [
        renderer.create_material(
            f"obj{index}", base_color=Vector3(0.4 + 0.1 * index, 0.5, 0.8 - 0.1 * index)
        )
        for index in range(8)
    ]
    for index in range(objects):
        row, column = divmod(index, side)
        items.append(
            (
                cube,
                palette[index % len(palette)],
                compose_matrix(
                    position=Vector3(
                        (column - side / 2.0) * 2.5,
                        0.5,
                        (row - side / 2.0) * 2.5,
                    )
                ),
            )
        )
    return items


def main(argv: list[str] | None = None) -> int:
    """Прогнать замеры и напечатать таблицу."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=int, default=200, help="число объектов")
    parser.add_argument("--instanced", action="store_true", help="рисовать пачкой")
    arguments = parser.parse_args(argv)

    apply_surface_format(RenderSettings())
    print(
        f"Кадр {FRAME_SIZE[0]}x{FRAME_SIZE[1]}, объектов {arguments.objects}, "
        f"усреднение по {FRAMES} кадрам"
    )
    with QtOffscreenContext(FRAME_SIZE):
        resources = ResourceManager(roots=[ROOT / "assets"], name="LightBench")
        renderer = Renderer(resources=resources, shader_root=ROOT / "assets")
        renderer.initialize()
        print(f"Драйвер: {renderer.info.describe()}\n")

        camera = Camera(fov=55.0, far=200.0)
        camera.position = Vector3(26.0, 22.0, 26.0)
        camera.look_at(Vector3(0.0, 0.0, 0.0))
        target = RenderTarget("bench", *FRAME_SIZE)
        items = build_items(renderer, arguments.objects)
        use_instancing = arguments.instanced
        if use_instancing:
            # Пачка одной геометрией: тогда кадр не упирается в число вызовов
            # отрисовки, и стоимость освещения видна как есть.
            mesh = GpuMesh.from_mesh_asset(cube_mesh(1.0))
            material = renderer.create_material(
                "inst", base_color=Vector3(0.7, 0.6, 0.4)
            )
            matrices = [matrix for _mesh, _material, matrix in items[1:]]
            items = []

        def draw_frame() -> None:
            """Отрисовать один кадр текущим способом."""
            renderer.begin_frame(camera, FRAME_SIZE, target=target)
            if use_instancing:
                renderer.draw_instanced(mesh, material, matrices)
            else:
                for mesh_item, material_item, matrix in items:
                    renderer.submit(mesh_item, material_item, matrix)
            renderer.end_frame(target=target)

        def timed() -> float:
            """Среднее время кадра в миллисекундах."""
            for _ in range(WARMUP):
                draw_frame()
            started = time.perf_counter()
            for _ in range(FRAMES):
                draw_frame()
            return (time.perf_counter() - started) / FRAMES * 1000.0

        mode = "инстансинг" if use_instancing else "поштучно"
        print(f"способ отрисовки: {mode}\n")
        print(f"{'источников':>11} | {'мс/кадр':>8} | {'FPS':>6} | прирост")
        print("-" * 46)

        baseline: float | None = None
        for count in (0, 1, 2, 4, 8):
            renderer.lights.set_lights(build_lights(count))
            renderer.lights.dynamic = True
            renderer.shadows_enabled = False
            elapsed = timed()
            if baseline is None:
                baseline = elapsed
                delta = "—"
            else:
                delta = f"{elapsed - baseline:+.2f} мс"
            total = count + 2
            print(
                f"{total:>11} | {elapsed:>8.2f} | {1000.0 / elapsed:>6.1f} | {delta}"
            )

        # Тени: тот же набор, но с проходом карты теней.
        renderer.lights.set_lights(build_lights(4))
        renderer.lights.dynamic = True
        renderer.shadows_enabled = True
        elapsed = timed()
        shadow = renderer.shadow_map
        size = shadow.size if shadow is not None else 0
        drawn = shadow.drawn_objects if shadow is not None else 0
        culled = shadow.culled_objects if shadow is not None else 0
        print(
            f"\nтени {size}x{size}: {elapsed:.2f} мс "
            f"({1000.0 / elapsed:.1f} FPS), объектов в карте {drawn}, "
            f"отсечено {culled}"
        )

        # Запечённый путь: динамическое освещение выключено целиком.
        renderer.lights.dynamic = False
        renderer.shadows_enabled = False
        elapsed = timed()
        assert baseline is not None
        print(
            f"запечённый свет: {elapsed:.2f} мс "
            f"({1000.0 / elapsed:.1f} FPS), экономия {baseline - elapsed:+.2f} мс"
        )

        # Сколько объектов реально получили свет и сколько раз грузились массивы.
        renderer.lights.dynamic = True
        renderer.lights.set_lights(build_lights(8))
        renderer.lights.clear_cache()
        probes = items[1:6] if items else []
        for _mesh, _material, matrix in probes:
            center = Vector3(*(matrix[:3, 3].tolist()))
            renderer.lights.select(center, 0.9)
        print(f"\nОтсечение и кэш: {renderer.lights.stats.describe()}")
        if not use_instancing:
            print(
                "Подсказка: при поштучной отрисовке кадр упирается в число "
                "вызовов драйвера, а не в свет — запустите с --instanced, "
                "чтобы увидеть стоимость самого освещения."
            )

        renderer.dispose()
        resources.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
