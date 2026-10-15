r"""Замер стоимости UI: 50 виджетов (панели, тексты, кнопки), 1280×720, 60 кадров.

Запуск::

    .venv\Scripts\python.exe tools\benchmark_ui.py

Результат печатается и пишется в ``_work/ui_perf.md``. Сравниваются пустая сцена
и сцена с UI: статичная (пакеты не пересобираются) и динамичная (текст счётчика
меняется каждый кадр → пересборка раскладки и пакетов).
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("QT_QPA_PLATFORM", "windows")

# Как в плеере (engine/player/__main__.py): без glGetError после каждого вызова.
# Флаг --checked включает проверки обратно (для сравнения).
CHECKED = "--checked" in sys.argv
if not CHECKED:
    import OpenGL

    OpenGL.ERROR_CHECKING = False

WIDTH, HEIGHT, FRAMES, WARMUP = 1280, 720, 60, 10
OUT = ROOT / "_work" / "ui_perf.md"


def main() -> int:
    from OpenGL import GL

    from engine.core.resources import ResourceManager
    from engine.core.scene import Scene
    from engine.core.ui_components import UIButton, UIPanel, UIText
    from engine.rendering import Camera, Renderer, RenderTarget
    from engine.rendering.qt_context import QtOffscreenContext
    from engine.rendering.settings import RenderSettings
    from engine.rendering.text import TextBatch
    from engine.rendering.ui import UIBatch, UIFiller, UIPass, UIScreen

    context = QtOffscreenContext((WIDTH, HEIGHT), settings=RenderSettings())
    if not context.make_current():
        print("OpenGL-контекст недоступен")
        return 1
    assets = ROOT / "assets"
    renderer = Renderer(
        resources=ResourceManager(roots=[assets], name="UIBench"),
        shader_root=assets,
    )
    renderer.initialize()
    camera = Camera(fov=50.0, far=100.0)
    target = RenderTarget("bench", WIDTH, HEIGHT)

    scene = Scene("bench")
    counter = None
    for index in range(50):
        col, row = index % 10, index // 10
        position = (10.0 + col * 125.0, 10.0 + row * 70.0)
        game_object = scene.create_game_object(f"W{index}")
        kind = index % 3
        if kind == 0:
            game_object.add_component(
                UIPanel, position=position, size=(115.0, 60.0), border_width=1.0
            )
        elif kind == 1:
            text = game_object.add_component(
                UIText, position=position, size=(115.0, 60.0)
            )
            text.text.text = f"Текст {index}"
            if counter is None:
                counter = text
        else:
            game_object.add_component(
                UIButton, text=f"Кнопка {index}", position=position, size=(115.0, 60.0)
            )

    screen = UIScreen()
    batch, text_batch = UIBatch(), TextBatch()
    renderer.add_pass(UIPass(screen, batch, text_batch))
    filler = UIFiller()
    state = {"calls": 0, "quads": 0}

    def run(with_ui: bool, dynamic: bool = False) -> tuple[list[float], list[float]]:
        frame_ms: list[float] = []
        fill_ms: list[float] = []
        for frame in range(WARMUP + FRAMES):
            start = time.perf_counter()
            if with_ui:
                if dynamic and counter is not None:
                    counter.text.text = f"Счёт: {frame}"
                filler.fill(scene, screen, batch, text_batch, (assets,))
            else:
                batch.clear()
                text_batch.clear()
            filled = time.perf_counter()
            renderer.begin_frame(camera, (WIDTH, HEIGHT), target=target)
            stats = renderer.end_frame(target=target)
            GL.glFinish()
            done = time.perf_counter()
            if frame >= WARMUP:
                frame_ms.append((done - start) * 1000.0)
                fill_ms.append((filled - start) * 1000.0)
                state["calls"] = stats.draw_calls
        return frame_ms, fill_ms

    empty, _ = run(False)
    static, static_fill = run(True)
    ui_calls = state["calls"]
    dynamic, dynamic_fill = run(True, dynamic=True)
    delta = statistics.mean(static) - statistics.mean(empty)
    delta_dyn = statistics.mean(dynamic) - statistics.mean(empty)
    median_delta = statistics.median(static) - statistics.median(empty)
    verdict = "✅" if median_delta <= 1.5 and ui_calls <= 3 else "❌"
    lines = [
        "# Перф UI",
        "",
        f"Устройство: {renderer.info.describe()}",
        f"Сцена: 50 виджетов (панели, тексты, кнопки), {WIDTH}×{HEIGHT}, "
        f"{FRAMES} кадров (+{WARMUP} прогрев), glFinish в кадре. "
        "Проверки ошибок PyOpenGL: "
        + ("включены." if CHECKED else "выключены, как в плеере."),
        "",
        "| Сцена | мс/кадр (среднее) | мс/кадр (медиана) |",
        "|---|---|---|",
        f"| пустая | {statistics.mean(empty):.3f} | {statistics.median(empty):.3f} |",
        f"| 50 виджетов, статика | {statistics.mean(static):.3f} "
        f"| {statistics.median(static):.3f} |",
        f"| 50 виджетов, текст меняется | {statistics.mean(dynamic):.3f} "
        f"| {statistics.median(dynamic):.3f} |",
        "",
        f"- Прирост (статика): **{median_delta:.3f} мс/кадр по медиане** "
        f"({delta:.3f} по среднему; на iGPU среднее шумит выбросами); цель ≤ 1.5 мс.",
        f"- Прирост (динамика, пересборка каждый кадр): **{delta_dyn:.3f} мс/кадр**.",
        f"- `UIFiller.fill` статика: {statistics.mean(static_fill):.3f} мс; "
        f"динамика: {statistics.mean(dynamic_fill):.3f} мс.",
        f"- Draw calls UI: **{ui_calls}** (цель ≤ 3).",
        f"- Итог: {verdict}",
        "",
    ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), "utf-8")
    print("\n".join(lines))
    batch.dispose()
    text_batch.dispose()
    filler.dispose()
    target.dispose()
    renderer.dispose()
    context.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
