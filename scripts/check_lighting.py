"""Проверка системы освещения: реальный рендер и отчёт с картинками.

Скрипт собирает сцену с несколькими источниками света, проверяет затухание,
отсечение по радиусу и запечённый режим, строит карту теней направленного
источника и сохраняет отчёт в ``_work/lighting_check.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from engine.core.math3d import Vector3, compose_matrix  # noqa: E402
from engine.core.resources import cube_mesh, plane_mesh, uv_sphere_mesh  # noqa: E402
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
from engine.rendering.screenshot import save_png  # noqa: E402

WIDTH, HEIGHT = 640, 400


def build_scene(renderer: Renderer) -> list[tuple]:
    """Собрать геометрию сцены: пол и несколько тел.

    Тела расставлены с промежутками: так тени от них ложатся на пол отдельно и
    их видно в кадре.
    """
    floor = GpuMesh.from_mesh_asset(plane_mesh(24.0))
    cube = GpuMesh.from_mesh_asset(cube_mesh(1.5))
    sphere = GpuMesh.from_mesh_asset(uv_sphere_mesh(0.75, segments=32, rings=20))
    stone = renderer.create_material(
        "stone", base_color=Vector3(0.55, 0.55, 0.58), roughness=0.85
    )
    warm = renderer.create_material(
        "warm", base_color=Vector3(0.90, 0.55, 0.22), roughness=0.4
    )
    cool = renderer.create_material(
        "cool", base_color=Vector3(0.35, 0.60, 0.92), roughness=0.3
    )
    items = [(floor, stone, compose_matrix(position=Vector3(0.0, -0.05, 0.0)))]
    for index in range(3):
        items.append(
            (
                cube,
                warm if index % 2 == 0 else cool,
                compose_matrix(position=Vector3(-5.0 + index * 5.0, 0.75, 0.0)),
            )
        )
    items.append(
        (
            sphere,
            warm,
            compose_matrix(position=Vector3(0.0, 0.75, -5.0)),
        )
    )
    return items


def render(renderer, camera, target, items, label: str) -> np.ndarray:
    """Отрисовать кадр и вернуть пиксели.

    Args:
        renderer: Рендерер.
        camera: Камера кадра.
        target: Цель вывода.
        items: Объекты ``(меш, материал, матрица)``.
        label: Подпись для вывода в консоль.

    Returns:
        Пиксели кадра ``(H, W, 4)``.
    """
    renderer.begin_frame(camera, (WIDTH, HEIGHT), target=target)
    for mesh, material, matrix in items:
        renderer.submit(mesh, material, matrix)
    stats = renderer.end_frame(target=target)
    print(f"  {label}: {stats.describe()}")
    return target.read_pixels()


def shot(
    renderer: Renderer,
    camera: Camera,
    target: RenderTarget,
    items: list[tuple],
    label: str,
    path: Path,
) -> np.ndarray:
    """Отрисовать кадр и сохранить его в PNG.

    Args:
        renderer: Рендерер.
        camera: Камера кадра.
        target: Цель вывода.
        items: Объекты ``(меш, материал, матрица)``.
        label: Подпись для вывода в консоль.
        path: Куда сохранить кадр.

    Returns:
        Пиксели кадра.
    """
    pixels = render(renderer, camera, target, items, label)
    save_png(path, pixels, flip_vertical=False)
    return pixels


def main() -> int:
    """Проверить освещение и записать отчёт."""
    out = ROOT / "_work"
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Проверка системы освещения", ""]

    apply_surface_format(RenderSettings())
    with QtOffscreenContext((WIDTH, HEIGHT)):
        renderer = Renderer(shader_root=ROOT / "assets")
        renderer.initialize()
        lines.append(f"Драйвер: `{renderer.info.describe()}`")
        lines.append("")

        items = build_scene(renderer)
        camera = Camera(name="View", fov=50.0, near=0.1, far=200.0)
        camera.position = Vector3(13.0, 10.0, 15.0)
        camera.look_at(Vector3(0.0, 0.4, -1.0))
        target = RenderTarget("lighting", WIDTH, HEIGHT)

        # --- 1. Солнце + фоновая подсветка ------------------------------- #
        # Солнце наклонено так, чтобы тени уходили вбок и читались в кадре.
        sun = DirectionalLight(
            direction=Vector3(-0.55, -0.7, -0.3), intensity=1.0, cast_shadow=True
        )
        sun.shadow_bias = 0.004
        sun.normal_bias = 0.05
        ambient = AmbientLight(intensity=0.18)
        renderer.lights.cast_shadows = False  # сначала без теней
        renderer.lights.set_lights([sun, ambient])
        shot(renderer, camera, target, items, "солнце + фон", out / "light_sun.png")
        lines += [
            "## 1. Направленный свет + фоновая подсветка",
            "",
            f"Источников: {len(renderer.lights.lights)}, "
            f"фон `{renderer.lights.ambient_color}`",
            "",
            "![Солнце](light_sun.png)",
            "",
        ]

        # --- 2. Локальные источники: точечный и прожектор ---------------- #
        point = PointLight(
            position=Vector3(-5.0, 3.5, 1.5),
            range=9.0,
            intensity=1.8,
            color=Vector3(1.0, 0.4, 0.25),
        )
        spot = SpotLight(
            position=Vector3(5.0, 5.0, 3.0),
            direction=Vector3(-0.2, -1.0, -0.2),
            range=16.0,
            inner_angle=16.0,
            outer_angle=26.0,
            intensity=2.4,
            color=Vector3(0.5, 0.75, 1.0),
        )
        renderer.lights.set_lights([sun, ambient, point, spot])
        shot(
            renderer,
            camera,
            target,
            items,
            "плюс точечный и прожектор",
            out / "light_local.png",
        )
        lines += [
            "## 2. Точечный источник и прожектор",
            "",
            "Слева — тёплый точечный свет, справа — холодный прожектор с конусом:",
            "",
            "![Локальный свет](light_local.png)",
            "",
        ]

        # --- 3. Отсечение и наборы света для объектов -------------------- #
        probe_near = Vector3(-5.0, 0.5, 1.5)
        probe_far = Vector3(60.0, 0.5, 60.0)
        renderer.lights.clear_cache()
        near_lights = renderer.lights.select(probe_near, 0.9)
        far_lights = renderer.lights.select(probe_far, 0.9)
        lines += [
            "## 3. Отсечение источников по радиусу",
            "",
            f"Объект рядом с точечным светом получает {len(near_lights)} источника "
            f"(`{', '.join(light.type_name for light in near_lights)}`)",
            f"Объект в 85 единицах — {len(far_lights)} "
            f"(`{', '.join(light.type_name for light in far_lights)}`)",
            f"Счётчики: `{renderer.lights.stats.describe()}`",
            "",
        ]

        # --- 4. Карта теней --------------------------------------------- #
        renderer.lights.cast_shadows = True
        renderer.shadows_enabled = True
        renderer.lights.set_lights([sun, ambient])
        shot(
            renderer,
            camera,
            target,
            items,
            "с картой теней",
            out / "light_shadows.png",
        )
        shadow = renderer.shadow_map
        shadow_info = (
            f"{shadow.size}×{shadow.size}, объектов в карте {shadow.drawn_objects}"
            if shadow is not None
            else "не построена"
        )
        lines += [
            "## 4. Тени направленного источника",
            "",
            f"Карта теней: {shadow_info}",
            "Тела отбрасывают тень на пол (фильтрация 2×2 со смещением "
            "по нормали):",
            "",
            "![Тени](light_shadows.png)",
            "",
        ]

        # --- 5. Запечённый свет (оптимизация под слабые устройства) ------ #
        renderer.lights.dynamic = False
        shot(
            renderer,
            camera,
            target,
            items,
            "запечённый свет",
            out / "light_baked.png",
        )
        renderer.lights.dynamic = True

        # Запекание в вершинные цвета: считаем сами и проверяем диапазон.
        floor_mesh = plane_mesh(4.0)
        renderer.lights.set_lights([sun, ambient])
        colors = renderer.lights.bake_to_vertex_colors(floor_mesh)
        lines += [
            "## 5. Запечённый свет",
            "",
            "Динамическое освещение выключено: кадр не считает источники вовсе.",
            "",
            "![Запечённый свет](light_baked.png)",
            "",
            f"Запекание в вершинные цвета: массив `{colors.shape}`, "
            f"яркость `{colors[:, :3].min():.3f}..{colors[:, :3].max():.3f}`",
            "",
        ]

        # --- 6. Гизмо источников света ---------------------------------- #
        from editor.gizmos import LightGizmo, LightGizmoStyle
        from engine.rendering.lighting import LIGHT_TYPES

        gizmo_target = RenderTarget("gizmos", WIDTH, HEIGHT)
        gizmo_view = Camera(name="EditorView", fov=50.0, near=0.1, far=200.0)
        gizmo_view.position = Vector3(11.0, 9.0, 11.0)
        gizmo_view.look_at(Vector3(1.0, 1.5, -1.0))

        # Источники для показа гизмо: солнце, точечный, прожектор и фон.
        # Расставлены с промежутками, чтобы иконки и зоны не накладывались.
        rig = [
            (
                DirectionalLight(
                    direction=Vector3(-0.4, -1.0, -0.3), name="Gizmo Sun"
                ),
                False,
            ),
            (
                PointLight(
                    position=Vector3(-3.5, 2.5, -2.5), range=3.0, name="Gizmo Lamp"
                ),
                True,
            ),
            (
                SpotLight(
                    position=Vector3(3.5, 4.0, -2.0),
                    direction=Vector3(-0.4, -1.0, -0.2),
                    range=6.0,
                    outer_angle=32.0,
                    name="Gizmo Spot",
                ),
                False,
            ),
            (
                # У фонового света нет позиции: гизмо рисует его в начале
                # координат, поэтому в кадре он оказывается у центра сцены.
                AmbientLight(name="Gizmo Fill"),
                False,
            ),
        ]
        style = LightGizmoStyle(direction_length=1.2)

        gizmo_target.bind(renderer.state)
        renderer.begin_frame(gizmo_view, (WIDTH, HEIGHT), target=gizmo_target)
        for mesh, material, matrix in items:
            renderer.submit(mesh, material, matrix)
        for light, selected in rig:
            gizmo = LightGizmo(light, selected=selected, style=style)
            icon = renderer.create_material(
                f"gizmo_icon_{light.name}",
                program="unlit",
                base_color=gizmo.color,
            )
            wire = renderer.create_material(
                f"gizmo_wire_{light.name}",
                program="unlit",
                base_color=gizmo.range_color,
                transparent=True,
                alpha=0.6,
            )
            for mesh, material, matrix, mode in gizmo.draw_items(
                icon,
                wire,
                viewer_position=gizmo_view.position,
                # Зона влияния — только у выделенного источника, как в редакторе.
                show_range=selected,
            ):
                renderer.submit(mesh, material, matrix, mode=mode)
        gizmo_stats = renderer.end_frame(target=gizmo_target)
        gizmo_target.unbind(renderer.state)
        save_png(
            out / "light_gizmos.png",
            gizmo_target.read_pixels(),
            flip_vertical=False,
        )
        lines += [
            "## 6. Гизмо источников света",
            "",
            "Иконки четырёх типов источников; у выделенного точечного света "
            "показана сфера влияния (три кольца радиуса действия), у солнца и "
            "прожектора — лучи направления:",
            "",
            f"Типы: `{', '.join(LIGHT_TYPES)}`. Статистика: "
            f"`{gizmo_stats.describe()}`",
            "",
            "![Гизмо источников](light_gizmos.png)",
            "",
        ]

        # --- 7. Влияние на кадр ----------------------------------------- #
        lines += [
            "## 7. Производительность",
            "",
            "Замеры делает `tools/benchmark_lighting.py`: он показывает, что при "
            "инстансной отрисовке переход от 1 к 8 местным источникам не меняет "
            "время кадра (разница в пределах шума), а карта теней 512×512 "
            "добавляет около полутора миллисекунд. При поштучной отрисовке кадр "
            "упирается в число вызовов драйвера, а не в свет.",
            "",
        ]

        renderer.dispose()

    report = out / "lighting_check.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Отчёт: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
