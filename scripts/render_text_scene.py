r"""Рисует сцену PlayModeTest с тремя текстами в ``_work/text_scene.png``.

Сцена проекта не меняется: копия с добавленными текстами пишется в ``_work/``.
Тексты: кириллица, перенос по ширине и полупрозрачный текст с обводкой.

Запуск:
    .venv\Scripts\python.exe scripts\render_text_scene.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROJECT = ROOT / "projects" / "PlayModeTest"
SCENE = PROJECT / "Assets" / "Scenes" / "main.scene.json"
WORK = ROOT / "_work"

from engine.core.components import TextRenderer  # noqa: E402


def _object(object_id: int, name: str, position, text: TextRenderer) -> dict:
    return {
        "id": object_id, "name": name, "tag": "Untagged", "layer": 1,
        "active": True, "parent": None,
        "transform": {
            "position": list(position), "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "components": [{"type": "TextRenderer", **text.to_dict()}],
    }  # fmt: skip


def main() -> int:
    data = json.loads(SCENE.read_text("utf-8"))
    for game_object in data["objects"]:  # без скриптов кадр фиксирован
        game_object["components"] = [
            c for c in game_object["components"] if "path" not in c
        ]
    from engine.core.math3d import Quaternion, Vector3

    camera = data["objects"][0]["transform"]  # камера смотрит на тексты
    camera["position"] = [0.0, 2.6, 9.0]
    q = Quaternion.from_axis_angle(Vector3.up(), 180.0)  # взгляд вдоль -Z
    camera["rotation"] = [q.x, q.y, q.z, q.w]
    inner = data["objects"][0]["components"][0]["camera"]
    inner["position"], inner["rotation"] = camera["position"], camera["rotation"]
    first = max(o["id"] for o in data["objects"]) + 1
    texts = [
        ("Text Hello", (0.0, 3.4, 4.0), TextRenderer(
            text="Привет, мир!", size_px=64, color=(1.0, 0.9, 0.3),
            align="center", vertical_align="middle")),
        ("Text Wrap", (-3.2, 2.4, 4.0), TextRenderer(
            text="Длинный текст переносится по ширине блока, а не уходит за край",
            size_px=40, wrap_width=3.0, color=(0.7, 0.9, 1.0), line_spacing=1.1)),
        ("Text Outline", (2.2, 1.0, 4.0), TextRenderer(
            text="Alpha + outline", size_px=48, alpha=0.5, color=(1.0, 1.0, 1.0),
            outline_color=(0.1, 0.0, 0.3), outline_width=3.0, align="center")),
    ]  # fmt: skip
    for index, (name, position, text) in enumerate(texts):
        data["objects"].append(_object(first + index, name, position, text))
    WORK.mkdir(exist_ok=True)
    scene = WORK / "text.scene.json"
    scene.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    png = WORK / "text_scene.png"
    command = [
        sys.executable, "-m", "engine.player", "--project", str(PROJECT),
        "--scene", str(scene), "--headless", "--frames", "10",
        "--screenshot", str(png),
    ]  # fmt: skip
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    done = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, errors="replace", env=env
    )
    print(done.stdout[-600:], done.stderr[-1500:])
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
