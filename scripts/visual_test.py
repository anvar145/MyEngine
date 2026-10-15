"""Визуальный тест редактора «от лица пользователя» с замерами производительности.

Запускает настоящее окно :class:`MainWindow`, проходит по сценарию (создание
камеры и света, правка в инспекторе, превью камеры, гизмо, вкладки сцен,
зум, выбор скрипта) и делает скриншоты в ``_work/visual/``. Параллельно
измеряет FPS кадра и время отклика интерфейса, затем гоняет стресс-тесты.

Запуск::

    .venv\\Scripts\\python.exe scripts/visual_test.py [--stress-only]
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from itertools import pairwise
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from editor.app import create_application  # noqa: E402
from editor.context import EditorContext  # noqa: E402
from editor.main_window import MainWindow  # noqa: E402
from engine.core.math3d import Vector3  # noqa: E402

OUT = REPO_ROOT / "_work" / "visual"
RESULTS: dict[str, object] = {}


def pump(window: MainWindow, frames: int = 6) -> None:
    """Прокрутить цикл событий, пока вьюпорт не отрисует ``frames`` кадров."""
    gl = window.viewport.viewport
    loop = QEventLoop()
    left = {"n": frames}

    def on_frame(_fps: float) -> None:
        left["n"] -= 1
        if left["n"] <= 0:
            loop.quit()

    gl.frame_rendered.connect(on_frame)
    QTimer.singleShot(5000, loop.quit)
    gl.update()
    loop.exec()
    gl.frame_rendered.disconnect(on_frame)


def shot(window: MainWindow, name: str) -> None:
    """Сохранить скриншот окна."""
    pump(window, 3)
    gl = window.viewport.viewport
    gl.stop_animation()
    window.grab().toImage().save(str(OUT / f"{name}.png"), "PNG")
    gl.start_animation(60)
    print(f"  скриншот {name}.png")


def measure_fps(window: MainWindow, seconds: float = 2.0) -> dict[str, float]:
    """Измерить FPS и время кадра на свободном рендере."""
    gl = window.viewport.viewport
    stamps: list[float] = []
    loop = QEventLoop()

    def on_frame(_fps: float) -> None:
        stamps.append(time.perf_counter())

    gl.frame_rendered.connect(on_frame)
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()
    gl.frame_rendered.disconnect(on_frame)
    if len(stamps) < 3:
        return {"fps": 0.0, "mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    deltas = [(b - a) * 1000 for a, b in pairwise(stamps)]
    deltas.sort()
    return {
        "fps": round(1000 / statistics.mean(deltas), 1),
        "mean_ms": round(statistics.mean(deltas), 2),
        "p95_ms": round(deltas[int(len(deltas) * 0.95) - 1], 2),
        "max_ms": round(deltas[-1], 2),
    }


def latency(label: str, action, repeat: int = 10) -> float:
    """Измерить время синхронного отклика действия в мс (медиана)."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        action()
        times.append((time.perf_counter() - t0) * 1000)
    median = round(statistics.median(times), 2)
    RESULTS.setdefault("latency_ms", {})[label] = median  # type: ignore[index]
    print(f"  отклик {label}: {median} мс")
    return median


def scenario(window: MainWindow) -> None:
    """Пользовательский сценарий со скриншотами."""
    ctx = window.context
    window.viewport.viewport.start_animation(60)
    pump(window, 10)
    shot(window, "01_start")
    RESULTS["fps_start"] = measure_fps(window)

    # Создание камеры и света через меню «Создать».
    window.action_create_cube.trigger()
    window.action_create_camera.trigger()
    camera = ctx.selection.primary
    window.inspector.refresh()
    shot(window, "02_camera_created_preview")
    for action in window.light_actions.values():
        action.trigger()
    shot(window, "03_lights_created")

    # Превью камеры
    ctx.selection.select(camera)
    pump(window, 20)
    shot(window, "04_camera_selected_preview")
    RESULTS["fps_preview"] = measure_fps(window)

    # Гизмо
    for key in (Qt.Key.Key_W, Qt.Key.Key_E, Qt.Key.Key_R):
        QTest.keyClick(window.viewport.viewport, key)
        shot(window, f"05_gizmo_{key.name}")
    QTest.keyClick(window.viewport.viewport, Qt.Key.Key_W)

    # Вкладки
    window.new_scene()
    shot(window, "06_second_tab")
    window.context.activate_document(0)
    shot(window, "07_back_to_first_tab")

    # Зум к порогу
    controller = window.viewport.viewport.controller
    camera_view = window.viewport.camera
    start = camera_view.position
    for _ in range(60):
        controller.zoom(1.0)
        controller.apply(camera_view)
    pump(window, 3)
    shot(window, "08_zoom_in_hard")
    RESULTS["zoom_travelled_60_steps"] = round((camera_view.position - start).length, 2)
    RESULTS["zoom_distance_after_60"] = round(window.viewport.camera_distance(), 3)

    # Отклик
    objs = list(ctx.scene)
    i = {"n": 0}

    def select_next() -> None:
        ctx.selection.select(objs[i["n"] % len(objs)])
        i["n"] += 1
        window.inspector.refresh()

    latency("select_to_inspector", select_next)
    t = {"n": 0}

    def switch_tab() -> None:
        t["n"] ^= 1
        ctx.activate_document(t["n"])

    latency("tab_switch", switch_tab)
    latency("create_cube", lambda: window.create_primitive("cube"), repeat=15)
    latency("modify_notify", lambda: ctx.notify_modified(), repeat=30)


def _fill_objects(window: MainWindow, total: int, target: int) -> int:
    """Добавить кубы сеткой, пока их не станет ``target``."""
    from engine.core.components import MeshRenderer

    ctx = window.context
    grid = int(target**0.5)
    while total < target:
        ix, iz = total % grid, total // grid
        obj = ctx.scene.create_game_object(
            f"c{total}", position=Vector3(ix * 1.5 - grid * 0.75, 0.5, iz * 1.5)
        )
        obj.add_component(MeshRenderer(mesh="cube"))
        total += 1
    ctx.notify_modified()
    window.viewport.reset_camera()
    pump(window, 5)
    return total


def _set_lights(window: MainWindow, count: int) -> None:
    """Оставить в сцене ровно ``count`` точечных/прожекторных источников."""
    ctx = window.context
    for obj in [o for o in ctx.scene if o.name.startswith("L")]:
        ctx.scene.destroy_game_object(obj)
    for n in range(count):
        light = window.create_light("point" if n % 2 == 0 else "spot")
        if light is not None:
            light.name = f"L{n}"
            light.transform.position = Vector3((n % 4) * 8 - 12, 3, (n // 4) * 8)
    ctx.notify_modified()
    pump(window, 5)


def stress(window: MainWindow) -> None:
    """Стресс-тесты: много объектов и источников света."""
    ctx = window.context
    window.new_scene()
    total = _fill_objects(window, 0, 1000)

    light_rows: list[dict[str, object]] = []
    for count in (1, 8, 16):
        _set_lights(window, count)
        res = measure_fps(window, 2.0)
        res["lights"] = count
        res["objects"] = total
        light_rows.append(res)
        print(f"  1000 объектов + света {count}: {res}")
    shot(window, "stress_lights_16")
    RESULTS["stress_lights"] = light_rows

    objs = list(ctx.scene)
    k = {"n": 0}

    def select_next() -> None:
        ctx.selection.select(objs[k["n"] % len(objs)])
        k["n"] += 1
        window.inspector.refresh()

    latency("select_to_inspector@1000", select_next)
    latency("create_cube@1000", lambda: window.create_primitive("cube"), repeat=5)
    latency("modify_notify@1000", lambda: ctx.notify_modified(), repeat=10)

    rows: list[dict[str, object]] = []
    for target in (1000, 5000, 10000):
        total = _fill_objects(window, total, target)
        res = measure_fps(window, 2.0)
        res["objects"] = target
        res["lights"] = 16
        rows.append(res)
        print(f"  объектов {target} + 16 света: {res}")
        shot(window, f"stress_objects_{target}")
    RESULTS["stress_objects"] = rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stress_only = "--stress-only" in sys.argv
    _application, _args = create_application(["--quality", "low"])
    context = EditorContext(project_root=REPO_ROOT / "projects" / "TestProject")
    window = MainWindow(context)
    window.resize(1480, 900)
    window.show()
    QTest.qWaitForWindowExposed(window)
    pump(window, 5)
    if not stress_only:
        print("== Сценарий")
        scenario(window)
    print("== Стресс")
    stress(window)
    window.viewport.viewport.stop_animation()
    (OUT / "results.json").write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(RESULTS, ensure_ascii=False, indent=2))
    sys.stdout.flush()
    _ = QPoint
    os._exit(0)  # без диалога о несохранённой сцене


if __name__ == "__main__":
    raise SystemExit(main())
