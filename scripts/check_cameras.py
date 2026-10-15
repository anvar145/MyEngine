"""Проверка системы камер: реальный рендер стека камер и предпросмотра.

Скрипт собирает сцену с несколькими камерами, рисует её стеком
(игровая камера + UI-камера поверх через общий RenderTarget) и отдельно
проверяет предпросмотр камеры. Результат сохраняется в ``_work/camera_check.md``
с картинками, чтобы человек мог посмотреть глазами.
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
    Camera,
    CameraStack,
    GpuMesh,
    ProjectionMode,
    Renderer,
    RenderSettings,
    RenderTarget,
)
from engine.rendering.qt_context import (  # noqa: E402
    QtOffscreenContext,
    apply_surface_format,
)
from engine.rendering.screenshot import save_png  # noqa: E402


def build_meshes() -> dict[str, GpuMesh]:
    """Загрузить геометрию сцены на GPU."""
    return {
        "cube": GpuMesh.from_mesh_asset(cube_mesh(1.0)),
        "sphere": GpuMesh.from_mesh_asset(uv_sphere_mesh(0.5, segments=32, rings=20)),
        "plane": GpuMesh.from_mesh_asset(plane_mesh(24.0)),
    }


def scene_items(renderer: Renderer, meshes: dict[str, GpuMesh]):
    """Собрать объекты сцены: пол, кубы и сферу."""
    items = []
    stone = renderer.create_material(
        "stone", base_color=Vector3(0.30, 0.32, 0.36), roughness=0.85
    )
    warm = renderer.create_material(
        "warm", base_color=Vector3(0.90, 0.55, 0.20), roughness=0.35
    )
    cool = renderer.create_material(
        "cool", base_color=Vector3(0.35, 0.62, 0.92), roughness=0.25
    )
    items.append((meshes["plane"], stone, compose_matrix(position=Vector3(0, -0.5, 0))))
    for index in range(6):
        angle = index * 60.0
        matrix = compose_matrix(
            position=Vector3(
                4.0 * np.cos(np.radians(angle)),
                0.0,
                4.0 * np.sin(np.radians(angle)),
            ),
            scale=Vector3(0.9, 0.9, 0.9),
        )
        items.append((meshes["cube"], warm if index % 2 else cool, matrix))
    items.append(
        (
            meshes["sphere"],
            warm,
            compose_matrix(position=Vector3(0, 1.2, 0), scale=Vector3(1.6, 1.6, 1.6)),
        )
    )
    return items


def main() -> int:
    """Отрисовать проверочные кадры и записать отчёт."""
    out = ROOT / "_work"
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Проверка системы камер", ""]

    apply_surface_format(RenderSettings())
    with QtOffscreenContext((640, 400)):
        renderer = Renderer(shader_root=ROOT / "assets")
        renderer.initialize()
        lines.append(f"Драйвер: `{renderer.info.describe()}`")
        lines.append("")

        meshes = build_meshes()
        items = scene_items(renderer, meshes)

        # --- 1. Стек камер: игровая камера и UI-камера поверх ------------- #
        game_camera = Camera(name="Game", fov=55.0, near=0.1, far=200.0)
        game_camera.position = Vector3(9.0, 7.0, 9.0)
        game_camera.look_at(Vector3(0.0, 0.5, 0.0))
        game_camera.clear_color = (0.07, 0.09, 0.13, 1.0)

        stack = CameraStack([game_camera])
        target = RenderTarget("stack", 640, 400)
        target.bind(renderer.state)
        renderer.begin_frame(stack, (640, 400), target=target)
        for mesh, material, matrix in items:
            renderer.submit(mesh, material, matrix)
        renderer.end_frame(target=target)
        target.unbind(renderer.state)
        game_pixels = target.read_pixels()

        # --- 2. UI-камера поверх: четверть экрана, без очистки цвета ------ #
        ui_camera = Camera(name="UI", depth=10, clear_flags=("depth",))
        ui_camera.viewport_rect = (0.62, 0.55, 0.35, 0.42)
        ui_camera.ortho_height = 4.0
        ui_camera.mode = ProjectionMode.ORTHOGRAPHIC
        ui_camera.position = Vector3(0.0, 0.0, 12.0)
        ui_camera.look_at(Vector3.zero())
        stack.add(ui_camera)

        target.bind(renderer.state)
        renderer.begin_frame(ui_camera, (640, 400), target=target)
        marker = renderer.create_material(
            "marker", base_color=Vector3(0.95, 0.75, 0.25), roughness=0.2
        )
        renderer.submit(
            meshes["cube"], marker, compose_matrix(scale=Vector3(1.0, 1.0, 1.0))
        )
        renderer.end_frame(target=target)
        target.unbind(renderer.state)
        overlaid = target.read_pixels()

        save_png(out / "camera_stack.png", game_pixels, flip_vertical=False)
        save_png(out / "camera_stack_overlay.png", overlaid, flip_vertical=False)

        lines += [
            "## 1. Стек камер",
            "",
            f"Активная камера стека: `{stack.active.name}`",
            f"Порядок отрисовки: `{', '.join(c.name for c in stack.ordered())}`",
            f"Слои поверх игры: `{', '.join(c.name for c in stack.overlay_cameras())}`",
            "",
            "Игровая камера, затем UI-камера в правом верхнем углу "
            "(прямоугольник вьюпорта 0.62/0.55/0.35/0.42, цвет не очищается):",
            "",
            "![Стек камер](camera_stack_overlay.png)",
            "",
        ]

        # --- 3. Отсечение по маске слоёв --------------------------------- #
        mask_camera = Camera(name="Masked", culling_mask=0x1)
        mask_camera.position = Vector3(9.0, 7.0, 9.0)
        mask_camera.look_at(Vector3(0.0, 0.5, 0.0))
        renderer.begin_frame(mask_camera, (320, 200), target=target)
        accepted = sum(
            renderer.submit(mesh, material, matrix, layer=2)
            for mesh, material, matrix in items
        )
        masked_stats = renderer.end_frame(target=target)
        lines += [
            "## 2. Отсечение по маске слоёв",
            "",
            f"Маска `0x1`, слой объектов `2`: принято объектов {accepted}, "
            f"отсечено {masked_stats.culled}",
            "",
        ]

        # --- 4. Ортографическая проекция --------------------------------- #
        ortho = Camera(
            name="Ortho",
            mode=ProjectionMode.ORTHOGRAPHIC,
            ortho_height=14.0,
            near=0.1,
            far=200.0,
        )
        ortho.lock_aspect()
        ortho.position = Vector3(0.0, 18.0, 0.0)
        ortho.look_at(Vector3.zero())
        renderer.begin_frame(ortho, (640, 400), target=target)
        for mesh, material, matrix in items:
            renderer.submit(mesh, material, matrix)
        renderer.end_frame(target=target)
        save_png(out / "camera_ortho.png", target.read_pixels(), flip_vertical=False)
        lines += [
            "## 3. Ортографическая проекция",
            "",
            "Вид сверху, ортографическая высота 14 единиц:",
            "",
            "![Ортография](camera_ortho.png)",
            "",
        ]

        # --- 5. Экран ↔ мир и луч выбора --------------------------------- #
        probe = Camera(name="Probe", fov=50.0)
        probe.position = Vector3(6.0, 4.0, 6.0)
        probe.look_at(Vector3(0.0, 0.5, 0.0))
        probe.set_aspect(640, 400)
        point = Vector3(1.0, 0.75, -2.0)
        screen_x, screen_y, visible = probe.world_to_screen(point, 640, 400)
        origin, direction = probe.ray_from_screen(screen_x, screen_y, 640, 400)
        ray_distance = direction.cross(point - origin).length
        _cx, _cy, clip_z, clip_w = probe.world_to_clip(point)
        restored = probe.screen_to_world(
            screen_x, screen_y, 640, 400, depth=clip_z / clip_w
        )
        normal_lengths = np.linalg.norm(probe.frustum_planes()[:, :3], axis=1)
        lines += [
            "## 4. Экран и мир",
            "",
            f"Точка `{point}` → экран `({screen_x:.1f}, {screen_y:.1f})`, "
            f"видима: {visible}",
            f"Луч выбора: расстояние от точки до луча {ray_distance:.2e} "
            "(должно быть около нуля)",
            f"Обратная развёртка: `{restored}` (расхождение "
            f"{(restored - point).length:.2e})",
            f"Плоскости фрустума нормализованы: "
            f"{np.allclose(normal_lengths, 1.0)}",
            "",
        ]

        # --- 6. Предпросмотр игровой камеры ------------------------------ #
        preview = RenderTarget("preview", 320, 180)
        preview_camera = Camera(name="Preview", fov=40.0)
        preview_camera.position = Vector3(0.0, 8.0, 14.0)
        preview_camera.look_at(Vector3(0.0, 0.5, 0.0))
        preview.bind(renderer.state)
        renderer.begin_frame(preview_camera, (320, 180), target=preview)
        for mesh, material, matrix in items:
            renderer.submit(mesh, material, matrix)
        preview_stats = renderer.end_frame(target=preview)
        preview.unbind(renderer.state)
        preview_pixels = preview.read_pixels()
        save_png(out / "camera_preview.png", preview_pixels, flip_vertical=False)
        lines += [
            "## 5. Предпросмотр камеры (рендер в текстуру)",
            "",
            f"Цель вывода 320×180, статистика: `{preview_stats.describe()}`",
            "",
            "![Предпросмотр](camera_preview.png)",
            "",
        ]

        # --- 6. Гизмо камер в сцене -------------------------------------- #
        from editor.gizmos import CameraGizmo, CameraGizmoStyle

        gizmo_target = RenderTarget("gizmos", 640, 400)
        gizmo_view = Camera(name="EditorView", fov=50.0)
        gizmo_view.position = Vector3(14.0, 10.0, 14.0)
        gizmo_view.look_at(Vector3(0.0, 0.5, 0.0))

        # Три камеры: две обычные, одна «UI» (ортография) — как в редакторе.
        rig: list[tuple[Camera, bool]] = []
        for index, (position, mode, selected) in enumerate(
            (
                (Vector3(6.0, 2.5, 6.0), ProjectionMode.PERSPECTIVE, True),
                (Vector3(-6.0, 3.0, 4.0), ProjectionMode.PERSPECTIVE, False),
                (Vector3(0.0, 9.0, 0.0), ProjectionMode.ORTHOGRAPHIC, False),
            )
        ):
            camera = Camera(
                name=f"Cam{index + 1}", mode=mode, fov=45.0, near=0.5, far=25.0
            )
            camera.ortho_height = 8.0
            camera.position = position
            camera.look_at(Vector3(0.0, 0.5, 0.0))
            rig.append((camera, selected))

        gizmo_target.bind(renderer.state)
        renderer.begin_frame(gizmo_view, (640, 400), target=gizmo_target)
        for mesh, material, matrix in items:
            renderer.submit(mesh, material, matrix)
        gizmo_style = CameraGizmoStyle(frustum_alpha=0.45)
        for camera, selected in rig:
            gizmo = CameraGizmo(camera, selected=selected, style=gizmo_style)
            icon = renderer.create_material(
                f"gizmo_icon_{camera.name}",
                program="unlit",
                base_color=gizmo.color,
            )
            frustum = renderer.create_material(
                f"gizmo_frustum_{camera.name}",
                program="unlit",
                base_color=gizmo.frustum_color,
                transparent=True,
                alpha=gizmo_style.frustum_alpha,
            )
            for mesh, material, matrix, mode in gizmo.draw_items(
                icon,
                frustum,
                viewer_position=gizmo_view.position,
                # У выделенной — фрустум, у остальных — только иконка.
                show_frustum=selected,
                far_scale=0.35,
            ):
                renderer.submit(mesh, material, matrix, mode=mode)
        gizmo_stats = renderer.end_frame(target=gizmo_target)
        gizmo_target.unbind(renderer.state)
        save_png(
            out / "camera_gizmos.png",
            gizmo_target.read_pixels(),
            flip_vertical=False,
        )

        lines += [
            "## 6. Гизмо камер в сцене",
            "",
            f"Иконки трёх камер; у выделенной (Cam1) нарисован усечённый объём "
            f"(укороченный до 35% дальней плоскости). Статистика: "
            f"`{gizmo_stats.describe()}`",
            "",
            "![Гизмо камер](camera_gizmos.png)",
            "",
        ]

        # --- 7. Сохранение вида камеры ----------------------------------- #
        from engine.rendering.camera import (
            EditorCameraController,
            EditorCameraSession,
        )

        controller = EditorCameraController()
        editor_camera = Camera(name="Editor", fov=60.0)
        controller.focus_on(Vector3(2.0, 1.0, -3.0), radius=2.5)
        controller.apply(editor_camera)
        session = EditorCameraSession(ROOT / "_work" / "camera_session.json")
        session.set(controller.capture(editor_camera), "editor")
        saved = session.save()
        reloaded = EditorCameraSession(ROOT / "_work" / "camera_session.json")
        reloaded.load()
        state = reloaded.get("editor")
        restored_camera = Camera()
        EditorCameraController().restore(state, restored_camera)
        lines += [
            "## 7. Сохранение вида редакторской камеры",
            "",
            f"Запись файла: {saved}",
            f"Восстановлено: позиция `{restored_camera.position}`, "
            f"fov {restored_camera.fov:.0f}",
            "Совпадение с сохранённым:",
            f"`{state.position.almost_equals(restored_camera.position, 1e-4)}`",
            "",
        ]

        renderer.dispose()
        target.dispose()
        preview.dispose()
        gizmo_target.dispose()

    report = out / "camera_check.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Отчёт: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
