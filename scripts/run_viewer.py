"""Интерактивный просмотрщик: окно Qt, 3D-модель и орбитальная камера.

Запуск::

    .venv\\Scripts\\python.exe scripts\\run_viewer.py
    .venv\\Scripts\\python.exe scripts\\run_viewer.py --mesh sphere --instances 225

Управление:
    ЛКМ          — орбита вокруг объекта
    ПКМ / СКМ    — панорама
    колесо       — зум
    R            — сбросить вид
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    InstanceGroup,
    QualityTier,
    Renderer,
    RenderSettings,
)
from engine.rendering.qt_context import apply_surface_format  # noqa: E402
from engine.rendering.qt_viewport import ViewportWidget  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Просмотрщик сцен MyEngine")
    parser.add_argument(
        "--mesh", default="cube", help="cube | sphere | plane | путь.obj"
    )
    parser.add_argument(
        "--instances", type=int, default=0, help="число кубов (инстансинг)"
    )
    parser.add_argument(
        "--quality", default="medium", choices=[tier.value for tier in QualityTier]
    )
    parser.add_argument(
        "--spin", type=float, default=45.0, help="скорость вращения, °/с"
    )
    parser.add_argument("--fps", type=int, default=60, help="частота обновления окна")
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="закрыть окно после N кадров (0 — работать до закрытия вручную)",
    )
    return parser.parse_args(argv)


def build_mesh(name: str, resources: ResourceManager) -> GpuMesh:
    """Получить меш по имени: процедурный или загруженный из файла."""
    match name.lower():
        case "cube":
            return GpuMesh.from_mesh_asset(cube_mesh(1.0))
        case "sphere":
            return GpuMesh.from_mesh_asset(uv_sphere_mesh(0.5, segments=48, rings=32))
        case "plane":
            return GpuMesh.from_mesh_asset(plane_mesh(6.0))
        case _:
            asset = resources.load(name)
            return GpuMesh.from_mesh_asset(asset)


def main(argv: list[str] | None = None) -> int:
    """Открыть окно просмотра и запустить цикл кадров."""
    from OpenGL import GL
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    args = parse_args(argv)
    quality = QualityTier(args.quality)
    settings = RenderSettings.for_quality(quality)
    apply_surface_format(settings)

    application = QApplication(sys.argv)
    resources = ResourceManager(roots=[ROOT / "assets"], name="Viewer")
    state: dict[str, object] = {"time": 0.0, "meshes": [], "groups": []}

    def on_ready(renderer: Renderer) -> None:
        """Собрать сцену, когда GL-контекст создан."""
        mesh = build_mesh(args.mesh, resources)
        material = renderer.create_material(
            "main", base_color=Vector3(0.86, 0.56, 0.24), roughness=0.35, metallic=0.1
        )
        grid_mesh = GpuMesh.from_mesh_asset(line_grid_mesh(10.0, divisions=10))
        grid_material = renderer.create_material(
            "grid", program="unlit", base_color=Vector3(0.26, 0.28, 0.34)
        )
        state["meshes"] = [mesh, grid_mesh]

        if args.instances > 0:
            side = int(args.instances**0.5) + 1
            group = InstanceGroup(mesh=mesh, material=material)
            group.matrices = [
                compose_matrix(
                    position=Vector3(
                        (index % side - side / 2) * 1.5,
                        0.0,
                        (index // side - side / 2) * 1.5,
                    )
                )
                for index in range(args.instances)
            ]
            state["groups"] = [group]
            widget.set_instance_groups([group])

        def provider() -> list[tuple[GpuMesh, object, object, int | None]]:
            """Содержимое кадра: сетка линиями и вращающийся объект."""
            angle = float(state["time"]) * args.spin
            rotation = Quaternion.from_axis_angle(Vector3.up(), angle)
            return [
                (grid_mesh, grid_material, compose_matrix(), GL.GL_LINES),
                (mesh, material, compose_matrix(rotation=rotation), None),
            ]

        widget.set_draw_provider(provider)
        widget.controller.distance = 7.5
        widget.controller.pitch = 24.0
        widget.controller.apply(widget.camera)
        print(f"Драйвер: {renderer.info.describe()}")
        print(f"Лимиты : {renderer.capabilities.describe()}")

    widget = ViewportWidget(
        settings=settings, camera=Camera(fov=50.0, far=300.0),
        shader_root=ROOT / "assets", on_ready=on_ready,
    )
    widget.setWindowTitle("MyEngine — просмотрщик")
    widget.resize(1280, 720)
    widget.setFocusPolicy(Qt.StrongFocus)

    rendered_frames = {"count": 0}

    def on_frame(fps: float) -> None:
        """Обновить заголовок и, если задан лимит, закрыть окно."""
        rendered_frames["count"] += 1
        widget.setWindowTitle(f"MyEngine — просмотрщик | {fps:.0f} FPS")
        if args.frames and rendered_frames["count"] >= args.frames:
            application.quit()

    widget.frame_rendered.connect(on_frame)
    widget.show()

    def tick() -> None:
        """Продвигать анимацию (таймер Qt вызывает перерисовку)."""
        state["time"] = float(state["time"]) + 1.0 / max(1, args.fps)

    from PySide6.QtCore import QTimer

    animation_timer = QTimer()
    animation_timer.timeout.connect(tick)
    animation_timer.start(max(1, int(1000 / max(1, args.fps))))
    widget.start_animation(args.fps)

    started = time.perf_counter()
    application.aboutToQuit.connect(resources.clear)
    exit_code = application.exec()
    print(f"Окно закрыто за {time.perf_counter() - started:.1f} с")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
