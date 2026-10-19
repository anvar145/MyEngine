"""Создаёт в ``projects/PlayModeTest`` «реалистичную» сцену ``realistic.scene.json``.

Персонаж, 25 врагов (кинематические капсулы с патрулём), 5 триггер-зон,
15 динамических пропсов (засыпают), статика (пол, стены, ящики, лестница).
Используется для замера бюджета кадра (``_work/realistic_scene.md``).

Проект должен быть собран ``scripts/seed_playmode_test.py``. Запуск
идемпотентен (файл сцены перезаписывается):

    .venv/Scripts/python.exe scripts/seed_physics_demo.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core.components import MeshRenderer  # noqa: E402
from engine.core.math3d import Quaternion, Vector3  # noqa: E402
from engine.core.physics_components import (  # noqa: E402
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
)
from engine.core.resources.asset_db import AssetDatabase  # noqa: E402
from engine.core.scene import Scene  # noqa: E402
from engine.rendering.camera import (  # noqa: E402
    CameraComponent,
    quaternion_from_forward,
)
from engine.rendering.lighting import DirectionalLight, LightComponent  # noqa: E402
from engine.rendering.sky import SkySettings  # noqa: E402
from engine.scene_io import save_scene  # noqa: E402
from engine.scripting import ScriptComponent  # noqa: E402

PROJECT = ROOT / "projects" / "PlayModeTest"
ASSETS = PROJECT / "Assets"
SCENES = ASSETS / "Scenes"
SCENE_NAME = "realistic.scene.json"

FLOOR_HALF = 15.0
STEP_HEIGHT = 0.15
SLOPE_DEGREES = 20.0

#: Ряды врагов по Z. Диаметр капсулы врага 0.8 м; зазор между рядами должен быть
#: заметно больше диаметра персонажа (0.8 м), иначе на скорости он зажимается
#: между патрулирующими врагами (B9: при шаге 1.2 м зазор был 0.4 м).
ENEMY_ROW_START = 7.2
ENEMY_ROW_PITCH = 1.8
ENEMY_DIAMETER = 0.8


def look_rotation(direction: Vector3) -> Quaternion:
    """Ориентация «смотреть вдоль direction» через yaw/pitch (``+Z`` — вперёд).

    ``quaternion_from_forward`` для направлений дальше 90° от ``+Z`` с наклоном
    вниз переворачивает «верх» камеры, поэтому здесь yaw и pitch считаются явно.
    """
    direction = direction.normalized
    yaw = math.degrees(math.atan2(direction.x, direction.z))
    pitch = math.degrees(math.asin(-direction.y))
    return Quaternion.from_axis_angle(
        Vector3(0.0, 1.0, 0.0), yaw
    ) * Quaternion.from_axis_angle(Vector3(1.0, 0.0, 0.0), pitch)


def _static_box(
    scene: Scene,
    name: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    rotation: Quaternion | None = None,
):
    """Статический ящик: меш ``cube`` с масштабом и коллайдер единичного размера."""
    box = scene.create_game_object(
        name, position=Vector3(*position), scale=Vector3(*size), rotation=rotation
    )
    box.add_component(MeshRenderer("cube", color=Vector3(*color)))
    box.add_component(ColliderComponent, shape="box", size=(1.0, 1.0, 1.0))
    return box


def _dynamic(scene, name, mesh, position, scale, color, collider, mass, rotation=None):
    obj = scene.create_game_object(
        name, position=Vector3(*position), scale=Vector3(*scale), rotation=rotation
    )
    obj.add_component(MeshRenderer(mesh, color=Vector3(*color)))
    obj.add_component(ColliderComponent, **collider)
    obj.add_component(RigidBodyComponent, mass=mass)
    return obj


def build_scene() -> Scene:
    """Собрать реалистичную сцену."""
    scene = Scene("realistic")
    sun = scene.create_game_object("Sun", position=Vector3(0.0, 8.0, 0.0))
    light = DirectionalLight(
        color=Vector3(1.0, 0.95, 0.85), intensity=1.3,
        direction=Vector3(-0.4, -1.0, -0.3), shadow_map_size=1024,
        shadow_distance=40.0,
    )
    sun.transform.rotation = quaternion_from_forward(light.unit_direction)
    sun.add_component(LightComponent, light=light).sync_to_light()

    _static_box(scene, "Floor", (0.0, -0.5, 0.0),
                (FLOOR_HALF * 2, 1.0, FLOOR_HALF * 2), (0.55, 0.55, 0.52))
    wall = FLOOR_HALF + 0.5
    for name, position, size in (
        ("Wall North", (0.0, 1.5, -wall), (FLOOR_HALF * 2 + 2, 3.0, 1.0)),
        ("Wall South", (0.0, 1.5, wall), (FLOOR_HALF * 2 + 2, 3.0, 1.0)),
        ("Wall West", (-wall, 1.5, 0.0), (1.0, 3.0, FLOOR_HALF * 2 + 2)),
        ("Wall East", (wall, 1.5, 0.0), (1.0, 3.0, FLOOR_HALF * 2 + 2)),
    ):
        _static_box(scene, name, position, size, (0.4, 0.42, 0.5))
    for index in range(6):
        height = STEP_HEIGHT * (index + 1)
        _static_box(scene, f"Stair {index + 1}",
                    (2.3 + index * 0.6, height / 2, -9.0), (0.6, height, 3.0),
                    (0.7, 0.6, 0.45))
    # Статические укрытия.
    for index in range(10):
        _static_box(scene, f"Cover {index + 1}",
                    (-12.0 + index * 2.6, 0.75, -4.0), (1.5, 1.5, 1.0),
                    (0.5, 0.5, 0.6))

    # 15 динамических пропсов: падают и засыпают.
    for index in range(15):
        row, column = divmod(index, 5)
        _dynamic(
            scene, f"Prop {index + 1}", "cube",
            (4.0 + column * 1.4, 0.5 + row * 1.1, 6.0), (1.0, 1.0, 1.0),
            (0.85, 0.5, 0.2),
            {"shape": "box", "size": (1.0, 1.0, 1.0), "friction": 0.6}, 1.0,
        )

    # 25 врагов: кинематические капсулы с патрулём вдоль X.
    for index in range(25):
        row, column = divmod(index, 5)
        enemy = scene.create_game_object(
            f"Enemy {index + 1}",
            position=Vector3(
                -8.0 + column * 4.0, 0.9, ENEMY_ROW_START + row * ENEMY_ROW_PITCH
            ),
            scale=Vector3(0.8, 1.8, 0.8),
        )
        enemy.add_component(MeshRenderer("cylinder", color=Vector3(0.8, 0.2, 0.2)))
        enemy.add_component(ColliderComponent, shape="capsule", radius=0.4,
                            height=1.8)
        enemy.add_component(RigidBodyComponent, body_type="kinematic")
        enemy.add_component(ScriptComponent(ASSETS / "Scripts" / "patrol.py"))

    # 5 триггер-зон.
    for index in range(5):
        zone = scene.create_game_object(
            f"Trigger {index + 1}", position=Vector3(-8.0 + index * 4.0, 0.75, 9.5)
        )
        zone.add_component(ColliderComponent, shape="box", size=(3.0, 1.5, 3.0),
                           is_trigger=True)

    hero = scene.create_game_object("Player", position=Vector3(-3.0, 1.0, 5.0))
    hero.add_component(CharacterControllerComponent)
    hero.add_component(ScriptComponent(ASSETS / "Scripts" / "player_controller.py"))
    body = scene.create_game_object("Body", scale=Vector3(0.8, 1.8, 0.8))
    body.add_component(MeshRenderer("cylinder", color=Vector3(0.95, 0.6, 0.15)))
    body.set_parent(hero, keep_world=False)

    camera = scene.create_game_object("Camera")
    camera.set_parent(hero, keep_world=False)
    camera.transform.position = Vector3(0.0, 4.0, 6.0)
    camera.transform.rotation = look_rotation(Vector3(0.0, -4.0, -6.0))
    component = CameraComponent()
    component.camera.fov = 55.0
    component.camera.name = "Camera"
    camera.add_component(component)
    component.sync_to_camera()
    return scene


def main() -> int:
    if not (SCENES / "main.scene.json").is_file():
        print("Нет main.scene.json: сначала scripts/seed_playmode_test.py")
        return 1
    database = AssetDatabase(ASSETS, library=PROJECT / "Library", read_only=True)
    database.scan()
    save_scene(build_scene(), SCENES / SCENE_NAME, database, SkySettings())
    print(f"Готово: {SCENES / SCENE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
