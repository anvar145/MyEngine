"""Собирает тестовый проект ``projects/PlayModeTest`` (идемпотентно).

Что делает:
    1. Создаёт структуру проекта и копирует сэмплы из ``_work/import_samples``.
    2. Генерирует недостающие фикстуры (BC1 ``.dds``, ``.mat.json``,
       ``*.atlas.json``, шейдер) — они помечены в отчёте как синтетические.
    3. Запускает импорт и ждёт его завершения.
    4. Собирает сцену ``Assets/Scenes/main.scene.json`` с секцией ``sky``.

Скрипты ``spinner.py``/``orbit_camera.py``/``collector.py`` лежат в проекте как
обычные файлы. Если сэмплов нет, скрипт завершается с кодом 0 и ничего не копирует.

Запуск:
    .venv\\Scripts\\python.exe scripts\\seed_playmode_test.py
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core.components import MeshRenderer, SpriteRenderer  # noqa: E402
from engine.core.math3d import Quaternion, Vector3  # noqa: E402
from engine.core.resources.asset_db import AssetDatabase  # noqa: E402
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

SAMPLES = ROOT / "_work" / "import_samples"
PROJECT = ROOT / "projects" / "PlayModeTest"
ASSETS = PROJECT / "Assets"

#: (источник относительно SAMPLES, назначение относительно Assets)
COPIES: tuple[tuple[str, str], ...] = (
    ("models/obj/bench.obj", "Models/bench.obj"),
    ("models/obj/bench.mtl", "Models/bench.mtl"),
    ("models/glb/chair.glb", "Models/chair.glb"),
    ("models/fbx/chair.fbx", "Models/chair.fbx"),
    ("sprites/bear_SE.png", "Sprites/bear_SE.png"),
    ("sprites/bathtub_SE.png", "Sprites/bathtub_SE.png"),
    ("audio/wav/synth1.wav", "Audio/synth1.wav"),
    ("audio/ogg/impactBell_heavy_000.ogg", "Audio/impactBell_heavy_000.ogg"),
)
TEXTURE_FOLDERS = ("Bricks075A", "PaintedMetal013")
TEXTURE_KINDS = ("Color", "NormalGL", "Roughness", "AmbientOcclusion", "Metalness")


def copy_samples() -> int:
    """Скопировать сэмплы в проект (существующие файлы не трогаем)."""
    copied = 0
    jobs = [(SAMPLES / src, ASSETS / dst) for src, dst in COPIES]
    for folder in TEXTURE_FOLDERS:
        for source in (SAMPLES / "textures" / folder).glob("*.jpg"):
            if any(f"_{kind}." in source.name for kind in TEXTURE_KINDS):
                copied += convert_jpg(source, ASSETS / "Textures" / folder)
    for source, target in jobs:
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
    return copied


def convert_jpg(source: Path, folder: Path) -> int:
    """Перекодировать JPEG в PNG средствами Qt (Pillow в окружении не стоит)."""
    from PySide6.QtGui import QImage

    target = folder / (source.stem + ".png")
    if target.exists():
        return 0
    image = QImage(str(source))
    if image.isNull():
        print(f"Не удалось прочитать {source.name}")
        return 0
    folder.mkdir(parents=True, exist_ok=True)
    return int(image.save(str(target), "PNG"))


def _rgb565(r: int, g: int, b: int) -> int:
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def write_checker_dds(path: Path, size: int = 64) -> None:
    """Синтетическая BC1-текстура 64x64: шахматка из двух цветов, без мипов."""
    light, dark = _rgb565(235, 200, 90), _rgb565(120, 60, 30)
    blocks = bytearray()
    for by in range(size // 4):
        for bx in range(size // 4):
            first, second = (light, dark) if (bx + by) % 2 == 0 else (dark, light)
            # c0 > c1 — режим из 4 цветов; индексы 0 — весь блок цвета c0.
            if first < second:
                first, second = second, first
                indices = 0x55555555  # индекс 1 → c1
            else:
                indices = 0
            blocks += struct.pack("<HHI", first, second, indices)
    header = bytearray(128)
    header[0:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<I", header, 8, 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000)
    struct.pack_into("<II", header, 12, size, size)
    struct.pack_into("<I", header, 20, len(blocks))
    struct.pack_into("<I", header, 28, 1)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, 0x4)
    header[84:88] = b"DXT1"
    struct.pack_into("<I", header, 108, 0x1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + bytes(blocks))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")


def make_fixtures() -> list[str]:
    """Сгенерировать синтетические фикстуры; вернуть их список для отчёта."""
    made: list[str] = []
    dds = ASSETS / "Textures" / "checker.dds"
    if not dds.exists():
        write_checker_dds(dds)
        made.append("Textures/checker.dds")
    bricks = "../Textures/Bricks075A/Bricks075A_1K-JPG_"
    metal = "../Textures/PaintedMetal013/PaintedMetal013_1K-JPG_"
    materials = {
        "Materials/brick.mat.json": {
            "name": "Brick",
            "roughness": 0.9,
            "uv_scale": [4, 4],
            "textures": {
                "base_color": bricks + "Color.png",
                "normal": bricks + "NormalGL.png",
                "roughness": bricks + "Roughness.png",
                "ao": bricks + "AmbientOcclusion.png",
            },
        },
        "Materials/metal.mat.json": {
            "name": "PaintedMetal",
            "textures": {
                "base_color": metal + "Color.png",
                "normal": metal + "NormalGL.png",
                "roughness": metal + "Roughness.png",
                "metallic": metal + "Metalness.png",
                "ao": metal + "AmbientOcclusion.png",
            },
        },
        "Materials/checker.mat.json": {
            "name": "Checker",
            "roughness": 0.6,
            "textures": {"base_color": "../Textures/checker.dds"},
        },
        "Sprites/furniture.atlas.json": {
            "mode": "pack",
            "sources": ["bear_SE.png", "bathtub_SE.png"],
            "padding": 2,
        },
    }
    for relative, data in materials.items():
        target = ASSETS / relative
        if not target.exists():
            write_json(target, data)
            made.append(relative)
    shader = ASSETS / "Shaders" / "tint.frag"
    if not shader.exists():
        shader.parent.mkdir(parents=True, exist_ok=True)
        shader.write_text(
            "#version 330 core\n"
            "in vec2 v_uv;\n"
            "out vec4 frag_color;\n"
            "uniform vec4 u_tint;\n"
            "void main() { frag_color = u_tint * vec4(v_uv, 0.5, 1.0); }\n",
            encoding="utf-8",
        )
        made.append("Shaders/tint.frag")
    return made


def run_import() -> tuple[AssetDatabase, ImportManager]:
    """Просканировать проект и дождаться импорта всех ассетов."""
    database = AssetDatabase(ASSETS, library=PROJECT / "Library")
    manager = ImportManager(database)
    manager.open()
    manager.wait_idle(300)
    return database, manager


def look_rotation(forward: Vector3) -> Quaternion:
    """Ориентация «рысканье, затем тангаж» (без крена горизонта)."""
    direction = forward.normalized
    yaw = math.degrees(math.atan2(direction.x, direction.z))
    pitch = math.degrees(math.asin(-direction.y))
    return Quaternion.from_axis_angle(
        Vector3(0.0, 1.0, 0.0), yaw
    ) * Quaternion.from_axis_angle(Vector3(1.0, 0.0, 0.0), pitch)


def guid(database: AssetDatabase, relative: str) -> str:
    return database.guid_of(relative) or ""


def build_scene(database: AssetDatabase) -> Scene:
    """Собрать сцену из импортированных ассетов."""
    scene = Scene("main")

    camera = scene.create_game_object("Camera", position=Vector3(4.0, 3.0, 6.0))
    camera.transform.rotation = look_rotation(Vector3(-4.0, -2.0, -6.0))
    component = CameraComponent()
    component.camera.fov = 50.0
    component.camera.name = "Camera"
    camera.add_component(component)
    component.sync_to_camera()
    camera.add_component(ScriptComponent(ASSETS / "Scripts" / "orbit_camera.py"))

    sun = scene.create_game_object("Sun", position=Vector3(0.0, 6.0, 0.0))
    light = DirectionalLight(
        color=Vector3(1.0, 0.93, 0.8),
        intensity=1.3,
        direction=Vector3(-0.45, -1.0, -0.35),
        shadow_map_size=512,
    )
    sun.transform.rotation = quaternion_from_forward(light.unit_direction)
    sun.add_component(LightComponent, light=light).sync_to_light()

    floor = scene.create_game_object("Floor", scale=Vector3(14.0, 1.0, 14.0))
    floor.add_component(
        MeshRenderer("plane", material_guid=guid(database, "Materials/brick.mat.json"))
    )

    def model(name: str, mesh: str, x: float, **kwargs: str) -> None:
        obj = scene.create_game_object(name, position=Vector3(x, 0.0, 0.0))
        obj.add_component(MeshRenderer(mesh, **kwargs))

    model("Bench (obj)", "Models/bench.obj", -3.0)
    model("Chair (glb)", "Models/chair.glb", -1.0)
    model("Chair (fbx)", "Models/chair.fbx", 1.0)

    cube = scene.create_game_object("Checker cube", position=Vector3(3.0, 0.5, 0.0))
    cube.add_component(
        MeshRenderer("cube", material_guid=guid(database, "Materials/checker.mat.json"))
    )
    cube.add_component(ScriptComponent(ASSETS / "Scripts" / "spinner.py"))

    sphere = scene.create_game_object("Metal sphere", position=Vector3(0.0, 0.5, -2.0))
    sphere.add_component(
        MeshRenderer("sphere", material_guid=guid(database, "Materials/metal.mat.json"))
    )
    sphere.add_component(ScriptComponent(ASSETS / "Scripts" / "collector.py"))

    sprite = scene.create_game_object("Bear sprite", position=Vector3(0.0, 2.0, -3.0))
    sprite.add_component(
        SpriteRenderer(
            sprite="bear_SE",
            atlas_guid=guid(database, "Sprites/furniture.atlas.json"),
        )
    )
    return scene


def main() -> int:
    if not SAMPLES.is_dir():
        print(f"Сэмплов нет ({SAMPLES}) — нечего делать")
        return 0
    for folder in ("Scenes", "Scripts", "Models", "Textures", "Materials", "Audio"):
        (ASSETS / folder).mkdir(parents=True, exist_ok=True)
    (ASSETS / "Sprites").mkdir(exist_ok=True)
    (PROJECT / "Settings").mkdir(exist_ok=True)
    write_json(
        PROJECT / "PlayModeTest.project.json",
        {"name": "PlayModeTest", "scene": "Assets/Scenes/main.scene.json"},
    )
    copied = copy_samples()
    made = make_fixtures()
    database, manager = run_import()
    failures = manager.failures()
    scene = build_scene(database)
    save_scene(
        scene, ASSETS / "Scenes" / "main.scene.json", database, SkySettings()
    )
    manager.close()
    print(f"Скопировано: {copied}; синтетических фикстур: {len(made)} {made}")
    print(f"Ошибок импорта: {len(failures)}")
    for key, message in failures.items():
        print(f"  {key}: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
