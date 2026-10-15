"""Добавляет в ``projects/PlayModeTest`` демо UI: главное меню и HUD (идемпотентно).

Что делает:
    1. Создаёт ``Assets/Scenes/main_menu.scene.json``: панель, заголовок, три кнопки
       («Играть» — ``load_scene``, «Настройки» — ``event:settings``, «Выход» —
       ``event:quit``), подсказка и объект со скриптом ``quit.py``.
    2. Дописывает HUD в ``main.scene.json`` (объекты ``HUD_*``; прежний HUD
       пересоздаётся) и кнопку возврата в меню.

Проект должен быть собран ``scripts/seed_playmode_test.py``.

Запуск:
    .venv/Scripts/python.exe scripts/seed_ui_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core.math3d import Vector3  # noqa: E402
from engine.core.resources.asset_db import AssetDatabase  # noqa: E402
from engine.core.scene import Scene  # noqa: E402
from engine.core.ui_components import UIButton, UIPanel, UIText  # noqa: E402
from engine.rendering.camera import CameraComponent  # noqa: E402
from engine.rendering.sky import SkySettings  # noqa: E402
from engine.scene_io import load_scene_with_sky, save_scene  # noqa: E402
from engine.scripting import ScriptComponent  # noqa: E402

PROJECT = ROOT / "projects" / "PlayModeTest"
ASSETS = PROJECT / "Assets"
SCENES = ASSETS / "Scenes"
HUD_PREFIX = "HUD_"


def _text(scene: Scene, name: str, text: str, size_px: int, **kwargs: object):
    game_object = scene.create_game_object(name)
    element = game_object.add_component(UIText, text=text, **kwargs)
    element.text.size_px = size_px
    return game_object, element


def build_menu() -> Scene:
    """Сцена главного меню."""
    scene = Scene("MainMenu")
    camera = scene.create_game_object("Camera", position=Vector3(0.0, 1.0, 5.0))
    camera.add_component(CameraComponent())
    panel = scene.create_game_object("MenuPanel")
    panel.add_component(
        UIPanel,
        anchor="middle_center",
        pivot=(0.5, 0.5),
        size=(420.0, 440.0),
        color=(0.08, 0.1, 0.16, 0.92),
        border_color=(0.35, 0.5, 0.9, 1.0),
        border_width=3.0,
    )
    title, label = _text(
        scene, "Title", "MyEngine", 56, anchor="top_center", pivot=(0.5, 0.0)
    )
    label.position = (0.0, 28.0)
    label.size = (380.0, 70.0)
    label.text.align = "center"
    title.set_parent(panel, keep_world=False)
    for index, (name, caption, action) in enumerate(
        (
            ("PlayButton", "Играть", "load_scene:main.scene.json"),
            ("SettingsButton", "Настройки", "event:settings"),
            ("QuitButton", "Выход", "event:quit"),
        )
    ):
        button = scene.create_game_object(name)
        button.add_component(
            UIButton,
            text=caption,
            on_click=action,
            anchor="top_center",
            pivot=(0.5, 0.0),
            position=(0.0, 130.0 + index * 90.0),
            size=(300.0, 66.0),
            size_px=30,
        )
        button.set_parent(panel, keep_world=False)
    hint, label = _text(
        scene, "Hint", "Наведите курсор и нажмите кнопку", 20,
        anchor="bottom_center", pivot=(0.5, 1.0),
    )
    label.position = (0.0, -16.0)
    label.size = (380.0, 28.0)
    label.text.align = "center"
    label.text.color = (0.7, 0.75, 0.85)
    hint.set_parent(panel, keep_world=False)
    host = scene.create_game_object("QuitHandler")
    host.add_component(ScriptComponent(ASSETS / "Scripts" / "quit.py"))
    return scene


def add_hud(scene: Scene) -> None:
    """Дописать HUD в игровую сцену (старый HUD удаляется)."""
    for game_object in list(scene.objects):
        if game_object.name.startswith(HUD_PREFIX) and not game_object.is_destroyed:
            scene.destroy_game_object(game_object, immediate=True)
    panel = scene.create_game_object(f"{HUD_PREFIX}Panel")
    panel.add_component(
        UIPanel,
        anchor="top_left",
        position=(16.0, 16.0),
        size=(280.0, 84.0),
        color=(0.05, 0.07, 0.12, 0.75),
        border_color=(0.35, 0.5, 0.9, 1.0),
        border_width=2.0,
    )
    health, label = _text(
        scene, f"{HUD_PREFIX}Health", "Здоровье: 100", 28,
        anchor="top_left", position=(14.0, 10.0), size=(250.0, 32.0),
    )
    health.set_parent(panel, keep_world=False)
    score, label = _text(
        scene, f"{HUD_PREFIX}Score", "Очки: 0", 28,
        anchor="top_left", position=(14.0, 46.0), size=(250.0, 32.0),
    )
    label.text.color = (1.0, 0.85, 0.3)
    score.set_parent(panel, keep_world=False)
    back = scene.create_game_object(f"{HUD_PREFIX}MenuButton")
    back.add_component(
        UIButton,
        text="Меню",
        on_click="load_scene:main_menu.scene.json",
        anchor="top_right",
        pivot=(1.0, 0.0),
        position=(-16.0, 16.0),
        size=(140.0, 48.0),
        size_px=24,
    )


def main() -> int:
    if not (SCENES / "main.scene.json").is_file():
        print("Нет main.scene.json: сначала scripts/seed_playmode_test.py")
        return 1
    database = AssetDatabase(ASSETS, library=PROJECT / "Library", read_only=True)
    database.scan()
    save_scene(build_menu(), SCENES / "main_menu.scene.json", database, SkySettings())
    scene, sky = load_scene_with_sky(SCENES / "main.scene.json", database)
    add_hud(scene)
    save_scene(scene, SCENES / "main.scene.json", database, sky)
    print("Готово: main_menu.scene.json и HUD в main.scene.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
