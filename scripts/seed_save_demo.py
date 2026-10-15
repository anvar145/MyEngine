"""Сцена ``save_menu.scene.json`` в PlayModeTest: 3 слота на готовом UI (A7b).

Запуск: ``PYTHONPATH=. .venv/Scripts/python.exe scripts/seed_save_demo.py``.
``main.scene.json`` не затрагивается.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core.math3d import Vector3  # noqa: E402
from engine.core.resources.asset_db import AssetDatabase  # noqa: E402
from engine.core.scene import Scene  # noqa: E402
from engine.core.ui_components import UIButton, UIPanel, UIText  # noqa: E402
from engine.rendering.camera import CameraComponent  # noqa: E402
from engine.rendering.sky import SkySettings  # noqa: E402
from engine.scene_io import save_scene  # noqa: E402
from engine.scripting import ScriptComponent  # noqa: E402

PROJECT = ROOT / "projects" / "PlayModeTest"
ASSETS = PROJECT / "Assets"
SCENES = ASSETS / "Scenes"


def _text(scene, name, text, size_px, parent, **kwargs):
    game_object = scene.create_game_object(name)
    element = game_object.add_component(UIText, text=text, **kwargs)
    element.text.size_px = size_px
    game_object.set_parent(parent, keep_world=False)
    return element


def _button(scene, name, caption, action, parent, **kwargs):
    button = scene.create_game_object(name)
    button.add_component(
        UIButton, text=caption, on_click=action, size_px=22, **kwargs
    )
    button.set_parent(parent, keep_world=False)


def build() -> Scene:
    """Сцена с тремя слотами и кнопкой «+10 золота»."""
    scene = Scene("SaveMenu")
    camera = scene.create_game_object("Camera", position=Vector3(0.0, 1.0, 5.0))
    camera.add_component(CameraComponent())
    panel = scene.create_game_object("SlotsPanel")
    panel.add_component(
        UIPanel,
        anchor="middle_center",
        pivot=(0.5, 0.5),
        size=(720.0, 440.0),
        color=(0.08, 0.1, 0.16, 0.92),
        border_color=(0.35, 0.5, 0.9, 1.0),
        border_width=3.0,
    )
    title = _text(
        scene, "Title", "Сохранения", 44, panel,
        anchor="top_center", pivot=(0.5, 0.0), position=(0.0, 18.0),
        size=(600.0, 56.0),
    )
    title.text.align = "center"
    _text(
        scene, "GoldLabel", "Золото: 0", 30, panel,
        anchor="top_left", position=(30.0, 88.0), size=(260.0, 40.0),
    ).text.color = (1.0, 0.85, 0.3)
    _button(
        scene, "GoldButton", "+10 золота", "event:gold", panel,
        anchor="top_right", pivot=(1.0, 0.0), position=(-30.0, 82.0),
        size=(200.0, 46.0),
    )
    for index, number in enumerate((1, 2, 3)):
        top = 150.0 + index * 90.0
        _text(
            scene, f"SlotLabel_{number}", f"Слот {number}: пусто", 26, panel,
            anchor="top_left", position=(30.0, top + 12.0), size=(260.0, 36.0),
        )
        for column, (caption, prefix) in enumerate(
            (("Сохранить", "save"), ("Загрузить", "load"), ("Удалить", "delete"))
        ):
            _button(
                scene,
                f"{prefix.title()}_{number}",
                caption,
                f"event:{prefix}_{number}",
                panel,
                anchor="top_left", position=(300.0 + column * 134.0, top),
                size=(126.0, 56.0),
            )
    host = scene.create_game_object("SaveSlotsHandler")
    host.add_component(ScriptComponent(ASSETS / "Scripts" / "save_slots.py"))
    return scene


def main() -> int:
    database = AssetDatabase(ASSETS, library=PROJECT / "Library", read_only=True)
    database.scan()
    save_scene(build(), SCENES / "save_menu.scene.json", database, SkySettings())
    print("Готово: save_menu.scene.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
