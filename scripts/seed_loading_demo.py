"""Добавляет в ``projects/PlayModeTest`` экран загрузки ``loading.scene.json``.

Сцена из трёх объектов: камера, панель и текст «Загрузка... 0%» (символа «…»
в атласе шрифта нет). Процент обновляет скрипт ``loading_progress.py`` по
событиям ``SceneLoadProgress``.
Прогресс-бара в UI пока нет (отдельная задача), только текст.

Проект должен быть собран ``scripts/seed_playmode_test.py``.

Запуск:
    .venv/Scripts/python.exe scripts/seed_loading_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core.math3d import Vector3  # noqa: E402
from engine.core.resources.asset_db import AssetDatabase  # noqa: E402
from engine.core.scene import Scene  # noqa: E402
from engine.core.ui_components import UIPanel, UIText  # noqa: E402
from engine.rendering.camera import CameraComponent  # noqa: E402
from engine.rendering.sky import SkySettings  # noqa: E402
from engine.scene_io import save_scene  # noqa: E402
from engine.scripting import ScriptComponent  # noqa: E402

PROJECT = ROOT / "projects" / "PlayModeTest"
ASSETS = PROJECT / "Assets"
SCENES = ASSETS / "Scenes"


def build_loading() -> Scene:
    """Сцена экрана загрузки."""
    scene = Scene("Loading")
    camera = scene.create_game_object("Camera", position=Vector3(0.0, 1.0, 5.0))
    camera.add_component(CameraComponent())
    panel = scene.create_game_object("LoadingPanel")
    panel.add_component(
        UIPanel,
        anchor="middle_center",
        pivot=(0.5, 0.5),
        size=(520.0, 140.0),
        color=(0.05, 0.07, 0.12, 0.92),
        border_color=(0.35, 0.5, 0.9, 1.0),
        border_width=3.0,
    )
    label_object = scene.create_game_object("LoadingLabel")
    element = label_object.add_component(
        UIText,
        text="Загрузка... 0%",
        anchor="middle_center",
        pivot=(0.5, 0.5),
        size=(480.0, 70.0),
    )
    element.text.size_px = 44
    element.text.align = "center"
    label_object.set_parent(panel, keep_world=False)
    script = ASSETS / "Scripts" / "loading_progress.py"
    label_object.add_component(ScriptComponent(script))
    return scene


def main() -> int:
    if not SCENES.is_dir():
        print("Нет Assets/Scenes: сначала scripts/seed_playmode_test.py")
        return 1
    database = AssetDatabase(ASSETS, library=PROJECT / "Library", read_only=True)
    database.scan()
    save_scene(build_loading(), SCENES / "loading.scene.json", database, SkySettings())
    print("Готово: loading.scene.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
