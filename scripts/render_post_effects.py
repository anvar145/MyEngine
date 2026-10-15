r"""Скриншоты A10b: bloom, виньетка, LUT (``_work/a10b_*.png``).

Рисует ``post_demo.scene.json`` плеером (``--headless --hdr on``) с разными
секциями ``post`` и склеивает пары «без / с эффектом». Сначала
``scripts/seed_post_demo.py``. Запуск:
    .venv\Scripts\python.exe scripts\render_post_effects.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "PlayModeTest"
SCENE = PROJECT / "Assets" / "Scenes" / "post_demo.scene.json"
WORK = ROOT / "_work"


def render(overrides: dict, png: Path) -> None:
    data = json.loads(SCENE.read_text("utf-8"))
    post = data["post"]
    post.update(
        {"bloom": False, "vignette": False, "lut": None, "fxaa": False, "exposure": 1.0}
    )
    post.update(overrides)
    scene = WORK / f"{png.stem}.scene.json"
    scene.write_text(json.dumps(data), "utf-8")
    command = [
        sys.executable, "-m", "engine.player", "--project", str(PROJECT),
        "--scene", str(scene), "--headless", "--frames", "10", "--hdr", "on",
        "--render-scale", "1.0", "--no-dynamic-res", "--screenshot", str(png),
    ]  # fmt: skip
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    done = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, errors="replace", env=env
    )
    scene.unlink(missing_ok=True)
    if done.returncode != 0:
        raise SystemExit(done.stdout + done.stderr)


def side_by_side(left: Path, right: Path, target: Path) -> None:
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage, QPainter

    first, second = QImage(str(left)), QImage(str(right))
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
    data = json.loads(SCENE.read_text("utf-8"))
    lut = data["post"]["lut"]
    plain = WORK / "a10b_plain.png"
    render({}, plain)
    pairs = {
        "bloom": {"bloom": True, "bloom_threshold": 1.0, "bloom_intensity": 1.0},
        "vignette": {"vignette": True, "vignette_intensity": 0.6},
        "lut": {"lut": lut, "lut_intensity": 1.0},
    }
    for name, overrides in pairs.items():
        effect = WORK / f"a10b_{name}_on.png"
        render(overrides, effect)
        side_by_side(plain, effect, WORK / f"a10b_{name}.png")
        effect.unlink()
    plain.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
