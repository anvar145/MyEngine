"""Рисует сцену PlayModeTest без неба и с небом и склеивает кадры в один PNG.

Результат: ``_work/sky_before_after.png`` (слева — небо выключено, справа — включено).

Запуск:
    .venv\\Scripts\\python.exe scripts\\render_sky_compare.py
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


def render(scene: Path, png: Path) -> None:
    command = [
        sys.executable, "-m", "engine.player", "--project", str(PROJECT),
        "--scene", str(scene), "--headless", "--frames", "10",
        "--screenshot", str(png),
    ]  # fmt: skip
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    done = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, errors="replace", env=env
    )
    if done.returncode != 0:
        raise SystemExit(done.stdout + done.stderr)


def main() -> int:
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage, QPainter

    data = json.loads(SCENE.read_text("utf-8"))
    WORK.mkdir(exist_ok=True)
    off_scene = WORK / "sky_off.scene.json"
    off = dict(data)
    off["sky"] = {**data.get("sky", {}), "enabled": False}
    off_scene.write_text(json.dumps(off), "utf-8")
    left, right = WORK / "sky_off.png", WORK / "sky_on.png"
    render(off_scene, left)
    render(SCENE, right)

    first, second = QImage(str(left)), QImage(str(right))
    result = QImage(
        first.width() + second.width(), first.height(), QImage.Format.Format_RGB32
    )
    painter = QPainter(result)
    painter.drawImage(QPoint(0, 0), first)
    painter.drawImage(QPoint(first.width(), 0), second)
    painter.end()
    target = WORK / "sky_before_after.png"
    result.save(str(target), "PNG")
    for path in (off_scene, left, right):
        path.unlink(missing_ok=True)
    print(f"Сохранено: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
