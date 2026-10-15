r"""Скриншоты пост-обработки A10a: tonemap, FXAA, fade (``_work/a10a_*.png``).

Рисует сцену PlayModeTest плеером (``--headless --hdr on``) с разными секциями
``post`` и склеивает пары кадров. Запуск:
    .venv\Scripts\python.exe scripts\render_post_compare.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "PlayModeTest"
SCENE = PROJECT / "Assets" / "Scenes" / "main.scene.json"
WORK = ROOT / "_work"


def render(post: dict | None, png: Path, extra: tuple[str, ...] = ()) -> None:
    data = json.loads(SCENE.read_text("utf-8"))
    if post is not None:
        data["post"] = post
    scene = WORK / f"{png.stem}.scene.json"
    scene.write_text(json.dumps(data), "utf-8")
    command = [
        sys.executable, "-m", "engine.player", "--project", str(PROJECT),
        "--scene", str(scene), "--headless", "--frames", "10", "--hdr", "on",
        "--render-scale", "1.0", "--no-dynamic-res", "--screenshot", str(png), *extra,
    ]  # fmt: skip
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    done = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, errors="replace", env=env
    )
    scene.unlink(missing_ok=True)
    if done.returncode != 0:
        raise SystemExit(done.stdout + done.stderr)


def side_by_side(left: Path, right: Path, target: Path, zoom: bool = False) -> None:
    from PySide6.QtCore import QPoint, QRect, Qt
    from PySide6.QtGui import QImage, QPainter

    first, second = QImage(str(left)), QImage(str(right))
    if zoom:  # увеличенный фрагмент без сглаживания: видно ступеньки краёв
        box = QRect(900, 255, 120, 68)
        mode = Qt.AspectRatioMode.IgnoreAspectRatio
        fast = Qt.TransformationMode.FastTransformation
        first = first.copy(box).scaled(640, 360, mode, fast)
        second = second.copy(box).scaled(640, 360, mode, fast)
    result = QImage(
        first.width() + second.width(), first.height(), QImage.Format.Format_RGB32
    )
    painter = QPainter(result)
    painter.drawImage(QPoint(0, 0), first)
    painter.drawImage(QPoint(first.width(), 0), second)
    painter.end()
    result.save(str(target), "PNG")
    print(f"Сохранено: {target}")


def main() -> int:
    WORK.mkdir(exist_ok=True)
    names = ("linear", "aces", "fxaa_off", "fxaa_on", "fade_off", "fade_on")
    paths = {name: WORK / f"a10a_{name}.png" for name in names}
    bright = {"exposure": 2.5, "fxaa": False}
    render({"tonemap": "linear", **bright}, paths["linear"])
    render({"tonemap": "aces", **bright}, paths["aces"])
    render({"tonemap": "aces", "exposure": 1.0, "fxaa": False}, paths["fxaa_off"])
    render({"tonemap": "aces", "exposure": 1.0, "fxaa": True}, paths["fxaa_on"])
    render({"tonemap": "aces", "exposure": 1.0, "fxaa": True}, paths["fade_off"])
    render(
        {"tonemap": "aces", "exposure": 1.0, "fxaa": True}, paths["fade_on"],
        ("--fade", "0.5"),
    )
    side_by_side(paths["linear"], paths["aces"], WORK / "a10a_tonemap.png")
    side_by_side(paths["fxaa_off"], paths["fxaa_on"], WORK / "a10a_fxaa.png", zoom=True)
    side_by_side(paths["fade_off"], paths["fade_on"], WORK / "a10a_fade.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
