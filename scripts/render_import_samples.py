"""Импортирует и рисует по одному сэмплу каждого формата из ``_work/import_samples``.

Для каждого формата создаётся временный проект ``_work/import_check/<ext>``:
сэмпл копируется в ``Assets/``, проходит импорт, вешается на объект сцены и
рисуется плеером (``python -m engine.player --headless --screenshot``).
Результат — ``_work/import_check_<ext>.png`` и ``_work/import_check_report.json``.

Форматы без картинки (звук, шейдер) проверяются только импортом. JPEG требует
Pillow (основная зависимость): без него формат помечается ``skipped``.

Запуск:
    .venv\\Scripts\\python.exe scripts\\render_import_samples.py [--only obj glb]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import seed_playmode_test as seed  # noqa: E402

from engine.core.components import MeshRenderer, SpriteRenderer  # noqa: E402
from engine.core.math3d import Vector3  # noqa: E402
from engine.core.resources.asset_db import AssetDatabase  # noqa: E402
from engine.core.resources.import_manager import ImportManager  # noqa: E402
from engine.core.scene import Scene  # noqa: E402
from engine.rendering.camera import CameraComponent  # noqa: E402
from engine.rendering.lighting import DirectionalLight, LightComponent  # noqa: E402
from engine.rendering.sky import SkySettings  # noqa: E402
from engine.scene_io import save_scene  # noqa: E402

SAMPLES = ROOT / "_work" / "import_samples"
WORK = ROOT / "_work"
CHECK = WORK / "import_check"
#: Доля изменившихся пикселей, начиная с которой объект считается нарисованным.
MIN_DIFF = 0.004
SCENE = "Assets/Scenes/check.scene.json"


@dataclass
class Case:
    """Один проверяемый формат."""

    ext: str
    files: list[tuple[str, str]]  # (источник от SAMPLES | "@dds", назначение)
    kind: str  # mesh | texture | sprite | import-only
    target: str = ""  # путь ассета, который вешается на объект
    needs_pillow: bool = False


CASES = [
    Case("obj", [("models/obj/bench.obj", "m/bench.obj"),
                 ("models/obj/bench.mtl", "m/bench.mtl")], "mesh", "m/bench.obj"),
    Case("glb", [("models/glb/chair.glb", "m/chair.glb")], "mesh", "m/chair.glb"),
    Case("gltf", [("models/gltf/BoxTextured/BoxTextured.gltf", "m/BoxTextured.gltf"),
                  ("models/gltf/BoxTextured/BoxTextured0.bin", "m/BoxTextured0.bin"),
                  ("models/gltf/BoxTextured/CesiumLogoFlat.png",
                   "m/CesiumLogoFlat.png")], "mesh", "m/BoxTextured.gltf"),
    Case("fbx", [("models/fbx/chair.fbx", "m/chair.fbx")], "mesh", "m/chair.fbx"),
    Case("warrior_obj", [("warrior_sample/Warrior.obj", "m/Warrior.obj"),
                         ("warrior_sample/Warrior.mtl", "m/Warrior.mtl")],
         "mesh", "m/Warrior.obj"),
    Case("warrior_glb", [("warrior_sample/Warrior.glb", "m/Warrior.glb")],
         "mesh", "m/Warrior.glb"),
    Case("warrior_fbx", [("warrior_sample/Warrior.fbx", "m/Warrior.fbx")],
         "mesh", "m/Warrior.fbx"),
    Case("png", [("sprites/bear_SE.png", "t/color.png")],
         "texture", "t/color.png"),
    Case("jpg", [("textures/Bricks075A/Bricks075A_1K-JPG_Color.jpg", "t/color.jpg")],
         "texture", "t/color.jpg", needs_pillow=True),
    Case("dds", [("@dds", "t/checker.dds")], "texture", "t/checker.dds"),
    Case("wav", [("audio/wav/synth1.wav", "a/synth1.wav")], "import-only"),
    Case("ogg", [("audio/ogg/impactBell_heavy_000.ogg", "a/bell.ogg")],
         "import-only"),
    Case("mp3", [("audio/mp3/funkymenuloop.mp3", "a/loop.mp3")], "import-only"),
    Case("frag", [("@frag", "s/tint.frag")], "import-only"),
    Case("atlas", [("sprites/bear_SE.png", "s/bear_SE.png"),
                   ("sprites/bathtub_SE.png", "s/bathtub_SE.png"),
                   ("@atlas", "s/furniture.atlas.json")], "sprite",
         "s/furniture.atlas.json"),
]


@dataclass
class Result:
    """Итог по одному формату."""

    ext: str
    status: str = "ok"  # ok | failed | skipped
    imported: bool = False
    rendered: bool | None = None
    diff_ratio: float | None = None
    png: str = ""
    errors: list[str] = field(default_factory=list)


def has_pillow() -> bool:
    return importlib.util.find_spec("PIL") is not None


def place_files(case: Case, assets: Path) -> None:
    """Скопировать/сгенерировать файлы кейса в ``Assets``."""
    for source, target in case.files:
        destination = assets / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source == "@dds":
            seed.write_checker_dds(destination)
        elif source == "@frag":
            destination.write_text(
                "#version 330 core\nout vec4 c;\nvoid main(){c=vec4(1.0);}\n",
                encoding="utf-8",
            )
        elif source == "@atlas":
            seed.write_json(
                destination,
                {
                    "mode": "pack",
                    "sources": ["bear_SE.png", "bathtub_SE.png"],
                    "padding": 2,
                },
            )
        else:
            shutil.copy2(SAMPLES / source, destination)


def import_assets(project: Path) -> tuple[AssetDatabase, list[str]]:
    """Импортировать проект; вернуть базу и список ошибок."""
    database = AssetDatabase(project / "Assets", library=project / "Library")
    manager = ImportManager(database)
    manager.open()
    manager.wait_idle(300)
    errors = [f"{key}: {message}" for key, message in manager.failures().items()]
    manager.close()
    return database, errors


def build_scene(case: Case, project: Path, database: AssetDatabase) -> Scene:
    """Сцена из камеры, солнца и (если есть) проверяемого объекта."""
    scene = Scene("check")
    camera = scene.create_game_object("Camera", position=Vector3(1.6, 1.2, 2.4))
    camera.transform.rotation = seed.look_rotation(Vector3(-1.6, -0.8, -2.4))
    component = CameraComponent()
    component.camera.fov = 50.0
    camera.add_component(component)
    component.sync_to_camera()

    sun = scene.create_game_object("Sun", position=Vector3(0.0, 5.0, 0.0))
    light = DirectionalLight(
        color=Vector3(1.0, 0.95, 0.85),
        intensity=1.3,
        direction=Vector3(-0.4, -1.0, -0.3),
    )
    sun.add_component(LightComponent, light=light).sync_to_light()

    if case.kind == "mesh":
        obj = scene.create_game_object("Subject", position=Vector3(0.0, 0.0, 0.0))
        obj.add_component(MeshRenderer(case.target))
    elif case.kind == "texture":
        guid = write_texture_material(case, project, database)
        obj = scene.create_game_object("Subject", position=Vector3(0.0, 0.5, 0.0))
        obj.add_component(MeshRenderer("cube", material_guid=guid))
    elif case.kind == "sprite":
        obj = scene.create_game_object("Subject", position=Vector3(0.0, 0.8, 0.0))
        obj.add_component(
            SpriteRenderer(
                sprite="bear_SE", atlas_guid=database.guid_of(case.target) or ""
            )
        )
    return scene


def write_texture_material(
    case: Case, project: Path, database: AssetDatabase
) -> str:
    """Материал ``.mat.json`` с проверяемой текстурой; вернуть его GUID."""
    relative = f"t/check_{case.ext}.mat.json"
    seed.write_json(
        project / "Assets" / relative,
        {
            "name": f"check_{case.ext}",
            "textures": {"base_color": Path(case.target).name},
        },
    )
    database2, _ = import_assets(project)
    return database2.guid_of(relative) or ""


def render(project: Path, png: Path, frames: int = 4) -> str:
    """Нарисовать сцену плеером; вернуть вывод (stderr) при ошибке."""
    command = [
        sys.executable,
        "-m",
        "engine.player",
        "--project",
        str(project),
        "--scene",
        SCENE,
        "--headless",
        "--frames",
        str(frames),
        "--screenshot",
        str(png),
    ]
    # headless-плееру нужна настоящая платформа Qt, не offscreen.
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    done = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=240,
        errors="replace", env=env,
    )
    return "" if done.returncode == 0 else (done.stdout + done.stderr)[-600:]


def diff_ratio(first: Path, second: Path) -> float:
    """Доля пикселей, заметно отличающихся между двумя PNG (через Qt)."""
    from PySide6.QtGui import QImage

    a = QImage(str(first)).convertToFormat(QImage.Format.Format_RGB32)
    b = QImage(str(second)).convertToFormat(QImage.Format.Format_RGB32)
    if a.size() != b.size() or a.isNull():
        return 0.0
    changed = 0
    for y in range(0, a.height(), 2):
        for x in range(0, a.width(), 2):
            p, q = a.pixel(x, y), b.pixel(x, y)
            delta = abs((p >> 16 & 255) - (q >> 16 & 255))
            delta += abs((p >> 8 & 255) - (q >> 8 & 255))
            delta += abs((p & 255) - (q & 255))
            changed += delta > 24
    return changed / ((a.height() // 2) * (a.width() // 2))


def prepare_baseline() -> Path:
    """Кадр без объекта — эталон фона для сравнения."""
    project = CHECK / "_baseline"
    (project / "Assets" / "Scenes").mkdir(parents=True, exist_ok=True)
    database, _ = import_assets(project)
    empty = Case("baseline", [], "none")
    scene = build_scene(empty, project, database)
    save_scene(scene, project / SCENE, database, SkySettings())
    png = WORK / "import_check_baseline.png"
    error = render(project, png)
    if error:
        raise SystemExit(f"Не удалось нарисовать эталон: {error}")
    return png


def run_case(case: Case, baseline: Path) -> Result:
    result = Result(case.ext)
    if case.needs_pillow and not has_pillow():
        result.status = "skipped"
        result.errors.append("нет Pillow (pip install pillow) — JPEG не декодируется")
        return result
    project = CHECK / case.ext
    place_files(case, project / "Assets")
    database, errors = import_assets(project)
    result.errors += errors
    result.imported = not errors
    if errors:
        result.status = "failed"
        return result
    if case.kind in ("import-only",):
        return result
    scene = build_scene(case, project, database)
    save_scene(scene, project / SCENE, database, SkySettings())
    png = WORK / f"import_check_{case.ext}.png"
    result.png = str(png.relative_to(ROOT)).replace("\\", "/")
    error = render(project, png)
    if error:
        result.status = "failed"
        result.rendered = False
        result.errors.append(error)
        return result
    result.diff_ratio = round(diff_ratio(baseline, png), 4)
    result.rendered = result.diff_ratio >= MIN_DIFF
    if not result.rendered:
        result.status = "failed"
        result.errors.append("объект не виден на кадре (кадр совпал с эталоном)")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="проверить только эти форматы")
    args = parser.parse_args(argv)
    if not SAMPLES.is_dir():
        print(f"Сэмплов нет ({SAMPLES}) — нечего проверять")
        return 0
    shutil.rmtree(CHECK, ignore_errors=True)
    baseline = prepare_baseline()
    results = [
        run_case(case, baseline)
        for case in CASES
        if not args.only or case.ext in args.only
    ]
    report = {
        "pillow": has_pillow(),
        "results": {r.ext: r.__dict__ for r in results},
    }
    (WORK / "import_check_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for item in results:
        shown = "" if item.diff_ratio is None else f" diff={item.diff_ratio}"
        print(f"{item.ext:6} {item.status:8} imported={item.imported}{shown}")
        for message in item.errors:
            print(f"       {message[:200]}")
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
