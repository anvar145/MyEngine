r"""Замер стоимости текста: 100 текстов одного шрифта, 1280×720, 60 кадров.

Запуск::

    .venv\Scripts\python.exe tools\benchmark_text.py

Результат печатается и пишется в ``_work/text_perf.md``. Сравниваются пустая
сцена и сцена со 100 текстами (сборка пакета каждый кадр + рисование + glFinish).
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

WIDTH, HEIGHT, FRAMES, TEXTS, WARMUP = 1280, 720, 60, 100, 10
OUT = ROOT / "_work" / "text_perf.md"


def main() -> int:
    from OpenGL import GL

    from engine.core.components import TextRenderer
    from engine.core.math3d import Vector3
    from engine.core.resources import ResourceManager
    from engine.core.scene import Scene
    from engine.rendering import Camera, Renderer, RenderTarget
    from engine.rendering.qt_context import QtOffscreenContext
    from engine.rendering.settings import RenderSettings
    from engine.rendering.text import TextBatch, TextPass, fill_text_batch

    context = QtOffscreenContext((WIDTH, HEIGHT), settings=RenderSettings())
    if not context.make_current():
        print("OpenGL-контекст недоступен")
        return 1
    assets = ROOT / "assets"
    renderer = Renderer(
        resources=ResourceManager(roots=[assets], name="TextBench"),
        shader_root=assets,
    )
    renderer.initialize()
    camera = Camera(fov=50.0, far=100.0)
    camera.position = Vector3(0.0, 0.0, 12.0)
    camera.look_at(Vector3(0.0, 0.0, 0.0))
    target = RenderTarget("bench", WIDTH, HEIGHT)

    scene = Scene("bench")
    for index in range(TEXTS):
        game_object = scene.create_game_object(
            f"T{index}",
            position=Vector3((index % 10) * 1.6 - 7.2, (index // 10) * 0.9 - 4.0, 0.0),
        )
        game_object.add_component(
            TextRenderer(text=f"Текст {index}: Hello", size_px=32, units_per_pixel=0.01)
        )

    batch = TextBatch()
    renderer.add_pass(TextPass(batch))

    def run(
        with_text: bool, move: bool = False
    ) -> tuple[list[float], list[float], int, int]:
        frame_ms: list[float] = []
        fill_ms: list[float] = []
        calls = quads = 0
        for frame in range(WARMUP + FRAMES):
            if move:
                camera.position = Vector3(frame * 0.01, 0.0, 12.0)
                camera.look_at(Vector3(0.0, 0.0, 0.0))
            start = time.perf_counter()
            if with_text:
                fill_text_batch(batch, scene.objects, (), scene)
            else:
                batch.clear()
            filled = time.perf_counter()
            renderer.begin_frame(camera, (WIDTH, HEIGHT), target=target)
            stats = renderer.end_frame(target=target)
            GL.glFinish()
            done = time.perf_counter()
            if frame >= WARMUP:
                frame_ms.append((done - start) * 1000.0)
                fill_ms.append((filled - start) * 1000.0)
                calls, quads = batch.last_draw_calls, batch.last_quads
        del stats
        return frame_ms, fill_ms, calls, quads

    empty, _, _, _ = run(False)
    text, fill, calls, quads = run(True)
    empty_move, _, _, _ = run(False, True)
    moving, _, _, _ = run(True, True)
    move_delta = statistics.mean(moving) - statistics.mean(empty_move)
    mean_empty, mean_text = statistics.mean(empty), statistics.mean(text)
    delta = mean_text - mean_empty
    median_delta = statistics.median(text) - statistics.median(empty)
    verdict = "✅" if delta <= 1.0 and calls <= 2 else "❌"
    lines = [
        "# Перф текста",
        "",
        f"Устройство: {renderer.info.describe()}",
        f"Сцена: {TEXTS} текстов одного шрифта (32 px), {quads} квадов, "
        f"{WIDTH}×{HEIGHT}, {FRAMES} кадров (+{WARMUP} прогрев), glFinish в кадре.",
        "",
        "| Сцена | мс/кадр (среднее) | мс/кадр (медиана) |",
        "|---|---|---|",
        f"| пустая | {mean_empty:.3f} | {statistics.median(empty):.3f} |",
        f"| {TEXTS} текстов | {mean_text:.3f} | {statistics.median(text):.3f} |",
        f"| пустая, камера движется | {statistics.mean(empty_move):.3f} "
        f"| {statistics.median(empty_move):.3f} |",
        f"| {TEXTS} текстов, камера движется | {statistics.mean(moving):.3f} "
        f"| {statistics.median(moving):.3f} |",
        "",
        f"- Прирост при движущейся камере (пересборка вершин каждый кадр): "
        f"**{move_delta:.3f} мс/кадр**.",
        f"- Прирост: **{delta:.3f} мс/кадр** (по медиане {median_delta:.3f}); "
        "цель ≤ 1 мс.",
        f"- `fill_text_batch` в статичной сцене: {statistics.mean(fill):.3f} мс.",
        f"- Draw calls текста: **{calls}** (цель ≤ 2).",
        f"- Итог: {verdict}",
        "",
    ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), "utf-8")
    print("\n".join(lines))
    batch.dispose()
    target.dispose()
    renderer.dispose()
    context.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
