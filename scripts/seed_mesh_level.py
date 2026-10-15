"""Создаёт в ``projects/PlayModeTest`` демо mesh-коллайдера ``mesh_level.scene.json``.

Уровень (пол с проёмом-ступенькой, четыре стены, рампа и колонна) генерируется
в ``Assets/Models/level.glb``, импортируется и подключается и как видимый
``MeshRenderer``, и как ``MeshColliderComponent``. По нему ходит персонаж
(``player_controller.py``), поверх стоят тестовые ящики-ориентиры.

Проект должен быть собран ``scripts/seed_playmode_test.py``. Запуск идемпотентен:

    .venv/Scripts/python.exe scripts/seed_mesh_level.py
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core.components import MeshRenderer  # noqa: E402
from engine.core.math3d import Quaternion, Vector3  # noqa: E402
from engine.core.physics_components import (  # noqa: E402
    CharacterControllerComponent,
    MeshColliderComponent,
)
from engine.core.resources.asset_db import AssetDatabase  # noqa: E402
from engine.core.resources.asset_ref import AssetRef  # noqa: E402
from engine.core.resources.import_manager import ImportManager  # noqa: E402
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
MODEL = "Models/level.glb"
SCENE_NAME = "mesh_level.scene.json"

HALF = 12.0
WALL_HEIGHT = 3.0
WALL_THICKNESS = 0.5


def _box(
    lo: tuple[float, float, float], hi: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Ящик [lo, hi]: 24 вершины (по 4 на грань), 12 треугольников, нормали наружу."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    faces = (
        ((1, 0, 0), ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))),
        ((-1, 0, 0), ((x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (x0, y0, z0))),
        ((0, 1, 0), ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0))),
        ((0, -1, 0), ((x0, y0, z1), (x0, y0, z0), (x1, y0, z0), (x1, y0, z1))),
        ((0, 0, 1), ((x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (x0, y0, z1))),
        ((0, 0, -1), ((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0))),
    )
    vertices, normals, indices = [], [], []
    for normal, corners in faces:
        base = len(vertices)
        vertices.extend(corners)
        normals.extend([normal] * 4)
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    return (
        np.array(vertices, np.float32),
        np.array(normals, np.float32),
        np.array(indices, np.uint32),
    )


def _wedge(
    x0: float, x1: float, z0: float, z1: float, rise: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Рампа: поднимается вдоль +X от 0 до ``rise`` (плоские нормали на грани)."""
    a, b = (x0, 0.0, z0), (x1, 0.0, z0)
    c, d = (x0, 0.0, z1), (x1, 0.0, z1)
    e, f = (x1, rise, z0), (x1, rise, z1)
    slope_n = np.array((-rise, x1 - x0, 0.0))
    slope_n = slope_n / np.linalg.norm(slope_n)
    tris = (
        ((a, e, f), tuple(slope_n)),
        ((a, f, c), tuple(slope_n)),
        ((b, d, f), (1, 0, 0)),
        ((b, f, e), (1, 0, 0)),
        ((a, b, e), (0, 0, -1)),
        ((c, f, d), (0, 0, 1)),
        ((a, c, d), (0, -1, 0)),
        ((a, d, b), (0, -1, 0)),
    )
    vertices, normals = [], []
    for corners, normal in tris:
        # Треугольники выше записаны по часовой; разворачиваем в CCW наружу.
        vertices.extend((corners[0], corners[2], corners[1]))
        normals.extend([normal] * 3)
    return (
        np.array(vertices, np.float32),
        np.array(normals, np.float32),
        np.arange(len(vertices), dtype=np.uint32),
    )


def build_level_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Склеить геометрию уровня в один меш."""
    parts = [
        _box((-HALF, -0.5, -HALF), (HALF, 0.0, HALF)),
        _box((-HALF - WALL_THICKNESS, 0.0, -HALF - WALL_THICKNESS),
             (HALF + WALL_THICKNESS, WALL_HEIGHT, -HALF)),
        _box((-HALF - WALL_THICKNESS, 0.0, HALF),
             (HALF + WALL_THICKNESS, WALL_HEIGHT, HALF + WALL_THICKNESS)),
        _box((-HALF - WALL_THICKNESS, 0.0, -HALF), (-HALF, WALL_HEIGHT, HALF)),
        _box((HALF, 0.0, -HALF), (HALF + WALL_THICKNESS, WALL_HEIGHT, HALF)),
        _box((-1.0, 0.0, -6.0), (1.0, 3.0, -4.0)),
        _box((4.0, 0.0, 2.0), (8.0, 0.3, 6.0)),
        _wedge(-8.0, -2.0, 2.0, 5.0, 1.2),
    ]
    vertices, normals, indices, offset = [], [], [], 0
    for v, n, i in parts:
        vertices.append(v)
        normals.append(n)
        indices.append(i + offset)
        offset += len(v)
    return np.vstack(vertices), np.vstack(normals), np.concatenate(indices)


def write_glb(path: Path) -> int:
    """Записать минимальный ``.glb`` с одним мешем; вернуть число треугольников."""
    vertices, normals, indices = build_level_mesh()
    blob = bytearray()
    views = []
    for data in (vertices, normals, indices):
        raw = data.tobytes()
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(raw)})
        blob += raw + b"\x00" * (-len(raw) % 4)
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Level"}],
        "meshes": [{
            "name": "Level",
            "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2}],
        }],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(vertices),
             "type": "VEC3", "min": vertices.min(0).tolist(),
             "max": vertices.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(normals),
             "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": len(indices),
             "type": "SCALAR"},
        ],
    }
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(blob)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total))
        handle.write(struct.pack("<I4s", len(text), b"JSON") + text)
        handle.write(struct.pack("<I4s", len(blob), b"BIN\x00") + bytes(blob))
    return len(indices) // 3


def look_rotation(direction: Vector3) -> Quaternion:
    """Ориентация «смотреть вдоль direction» через yaw/pitch (``+Z`` — вперёд)."""
    direction = direction.normalized
    yaw = math.degrees(math.atan2(direction.x, direction.z))
    pitch = math.degrees(math.asin(-direction.y))
    return Quaternion.from_axis_angle(
        Vector3(0.0, 1.0, 0.0), yaw
    ) * Quaternion.from_axis_angle(Vector3(1.0, 0.0, 0.0), pitch)


def build_scene(database: AssetDatabase) -> Scene:
    """Собрать сцену: уровень из ``level.glb`` + персонаж и камера."""
    scene = Scene("mesh_level")
    guid = database.guid_of(MODEL) or ""
    ref = AssetRef(guid, MODEL)

    sun = scene.create_game_object("Sun", position=Vector3(0.0, 8.0, 0.0))
    light = DirectionalLight(
        color=Vector3(1.0, 0.95, 0.85),
        intensity=1.3,
        direction=Vector3(-0.4, -1.0, -0.3),
        shadow_map_size=1024,
        shadow_distance=40.0,
    )
    sun.transform.rotation = quaternion_from_forward(light.unit_direction)
    sun.add_component(LightComponent, light=light).sync_to_light()

    level = scene.create_game_object("Level")
    level.add_component(MeshRenderer(mesh_ref=ref, color=Vector3(0.7, 0.68, 0.62)))
    level.add_component(MeshColliderComponent(mesh=ref))

    hero = scene.create_game_object("Player", position=Vector3(-3.0, 1.0, 8.0))
    hero.add_component(CharacterControllerComponent)
    hero.add_component(ScriptComponent(ASSETS / "Scripts" / "player_controller.py"))
    body = scene.create_game_object("Body", scale=Vector3(0.8, 1.8, 0.8))
    body.add_component(MeshRenderer("cylinder", color=Vector3(0.95, 0.6, 0.15)))
    body.set_parent(hero, keep_world=False)

    camera = scene.create_game_object("Camera")
    camera.set_parent(hero, keep_world=False)
    camera.transform.position = Vector3(0.0, 5.0, 8.0)
    camera.transform.rotation = look_rotation(Vector3(0.0, -5.0, -8.0))
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
    triangles = write_glb(ASSETS / MODEL)
    database = AssetDatabase(ASSETS, library=PROJECT / "Library")
    manager = ImportManager(database)
    manager.open()
    manager.wait_idle(300)
    save_scene(build_scene(database), SCENES / SCENE_NAME, database, SkySettings())
    print(f"Готово: {SCENES / SCENE_NAME} ({triangles} треугольников)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
