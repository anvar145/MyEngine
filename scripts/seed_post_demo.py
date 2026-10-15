r"""Демо A10b в ``projects/PlayModeTest``: bloom, виньетка, LUT.

Создаёт:

* ``Assets/Textures/Luts/warm.png`` — тёплая LUT 256x16 (ассет с ``.meta``);
* ``Assets/Materials/glow.mat.json`` — эмиссивный материал (emissive > 1);
* ``Assets/Scenes/post_demo.scene.json`` — сцена с секцией ``post``
  (bloom + виньетка + ссылка на LUT).

Проект должен быть собран ``scripts/seed_playmode_test.py``.

Запуск:
    .venv\Scripts\python.exe scripts\seed_post_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from engine.core.components import MeshRenderer  # noqa: E402
from engine.core.math3d import Vector3  # noqa: E402
from engine.core.resources.asset_db import AssetDatabase  # noqa: E402
from engine.core.resources.asset_ref import AssetRef  # noqa: E402
from engine.core.resources.import_manager import ImportManager  # noqa: E402
from engine.core.scene import Scene  # noqa: E402
from engine.rendering.camera import (  # noqa: E402
    CameraComponent,
    quaternion_from_forward,
)
from engine.rendering.lighting import DirectionalLight, LightComponent  # noqa: E402
from engine.rendering.lut import make_lut, save_lut_png  # noqa: E402
from engine.rendering.post import PostSettings  # noqa: E402
from engine.rendering.sky import SkySettings  # noqa: E402
from engine.scene_io import save_scene  # noqa: E402

PROJECT = ROOT / "projects" / "PlayModeTest"
ASSETS = PROJECT / "Assets"
LUT_PATH = "Textures/Luts/warm.png"
GLOW_PATH = "Materials/glow.mat.json"


def warm(colors: np.ndarray) -> np.ndarray:
    """Тёплый грейдинг: красный вверх, синий вниз, лёгкий контраст."""
    graded = colors * np.array([1.12, 1.0, 0.78], dtype=np.float32)
    return np.clip((graded - 0.5) * 1.1 + 0.5, 0.0, 1.0)


def make_assets() -> None:
    save_lut_png(ASSETS / LUT_PATH, make_lut(warm))
    glow = ASSETS / GLOW_PATH
    glow.write_text(
        json.dumps(
            {
                "name": "Glow",
                "base_color": [0.1, 0.1, 0.1, 1.0],
                "roughness": 0.6,
                "emissive": [4.0, 2.2, 0.6],
            },
            indent=2,
        ),
        "utf-8",
    )


def build_scene(database: AssetDatabase) -> Scene:
    scene = Scene("post_demo")
    camera = scene.create_game_object("Camera", position=Vector3(0.0, 2.0, 7.0))
    component = CameraComponent()
    component.camera.fov = 50.0
    component.camera.name = "Camera"
    camera.add_component(component)
    camera.transform.rotation = quaternion_from_forward(Vector3(0.0, -1.2, -7.0))
    component.sync_to_camera()

    sun = scene.create_game_object("Sun", position=Vector3(0.0, 6.0, 0.0))
    light = DirectionalLight(
        color=Vector3(1.0, 0.93, 0.8), intensity=0.9, direction=Vector3(-0.4, -1, -0.5)
    )
    sun.add_component(LightComponent, light=light).sync_to_light()

    floor = scene.create_game_object("Floor", scale=Vector3(14.0, 1.0, 14.0))
    floor.add_component(MeshRenderer("plane", color=Vector3(0.55, 0.55, 0.6)))
    glow = database.guid_of(GLOW_PATH) or ""
    for i, x in enumerate((-2.5, 0.0, 2.5)):
        obj = scene.create_game_object(f"Glow {i}", position=Vector3(x, 0.6, 0.0))
        obj.add_component(MeshRenderer("sphere", material_guid=glow))
    return scene


def main() -> int:
    if not ASSETS.is_dir():
        print("Нет Assets: сначала scripts/seed_playmode_test.py")
        return 1
    make_assets()
    database = AssetDatabase(ASSETS, library=PROJECT / "Library")
    manager = ImportManager(database)
    manager.open()
    manager.wait_idle(300)
    lut = AssetRef(guid=database.guid_of(LUT_PATH), path=LUT_PATH)
    post = PostSettings(
        tonemap="aces", bloom=True, vignette=True, lut=lut, lut_intensity=1.0
    )
    save_scene(
        build_scene(database),
        ASSETS / "Scenes" / "post_demo.scene.json",
        database,
        SkySettings(),
        post,
    )
    print("Готово: post_demo.scene.json, LUT guid", lut.guid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
