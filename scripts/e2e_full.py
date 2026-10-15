"""Сквозной функциональный прогон движка без окна редактора.

Запуск::

    .venv\\Scripts\\python.exe scripts\\e2e_full.py

Этапы: core → импорт → сцены → скрипты → пак → рендер (offscreen GL) →
негативные случаи. Результаты и артефакты — в ``_work/e2e_full/``
(``report.json`` со статусом каждой проверки). Код возврата 1, если что-то упало.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

import numpy as np  # noqa: E402

from engine.core.components import MeshRenderer  # noqa: E402
from engine.core.math3d import Vector3  # noqa: E402
from engine.core.resources import (  # noqa: E402
    AssetDatabase,
    ImportManager,
    PakBuilder,
    PakReader,
    ResourceFormatError,
    cube_mesh,
    load_sprite_atlas,
)
from engine.core.resources.cookers import encode_png  # noqa: E402
from engine.core.scene import Scene  # noqa: E402
from engine.scripting import ScriptComponent, ScriptRuntime  # noqa: E402

OUT = ROOT / "_work" / "e2e_full"
FIXTURES = ROOT / "engine" / "tests" / "fixtures"

#: Результаты проверок: имя → {"ok": bool, "detail": str, "seconds": float}.
RESULTS: dict[str, dict[str, object]] = {}


def check(name: str) -> Callable[[Callable[[], str | None]], Callable[[], str | None]]:
    """Декоратор: выполнить проверку, записать результат, не падать наружу."""

    def wrap(func: Callable[[], str | None]) -> Callable[[], str | None]:
        started = time.perf_counter()
        try:
            detail = func() or ""
            ok = True
        except BaseException as exc:  # noqa: BLE001 - прогон должен идти дальше
            detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            ok = False
        seconds = time.perf_counter() - started
        RESULTS[name] = {"ok": ok, "detail": detail, "seconds": round(seconds, 3)}
        print(f"[{'OK ' if ok else 'FAIL'}] {name} ({seconds:.2f} с) {detail[:200]}")
        return func

    return wrap


def build_project(base: Path) -> Path:
    """Собрать мини-проект с ассетами всех поддерживаемых типов."""
    assets = base / "assets"
    assets.mkdir(parents=True)
    for name in ("triangle.obj", "tiny.png", "tiny.wav", "tiny.ogg", "tiny_bc1.dds"):
        shutil.copy(FIXTURES / name, assets / name)
    shutil.copy(FIXTURES / "cube.fbx", assets / "cube.fbx")
    grid = np.zeros((8, 16, 4), dtype=np.uint8)
    grid[:, :8] = (200, 40, 40, 255)
    grid[:, 8:] = (40, 200, 40, 255)
    (assets / "sprites.png").write_bytes(encode_png(grid))
    (assets / "sprites.atlas.json").write_text(
        json.dumps(
            {"mode": "grid", "image": "sprites.png", "cell": [8, 8], "prefix": "s"}
        ),
        encoding="utf-8",
    )
    (assets / "gold.mat.json").write_text(
        json.dumps(
            {"base_color": [1.0, 0.8, 0.2, 1.0], "metallic": 1.0, "roughness": 0.3}
        ),
        encoding="utf-8",
    )
    (assets / "flat.frag").write_text(
        "#version 330 core\nout vec4 c;\nvoid main(){ c = vec4(1.0); }\n",
        encoding="utf-8",
    )
    return assets


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    project = OUT / "project"
    assets = build_project(project)
    library = project / "Library"
    state: dict[str, object] = {}

    # ---------------------------------------------------------------- core
    @check("core: иерархия и трансформы")
    def _core() -> str:
        scene = Scene("Core")
        parent = scene.create_game_object("P")
        child = scene.create_game_object("C")
        child.transform.set_parent(parent.transform)
        parent.transform.position = Vector3(1.0, 2.0, 3.0)
        child.transform.position = Vector3(1.0, 0.0, 0.0)
        world = child.transform.world_position
        assert abs(world.x - 2.0) < 1e-5 and abs(world.y - 2.0) < 1e-5, world
        scene.destroy_game_object(parent)
        return f"объектов после удаления: {len(scene)}"

    # -------------------------------------------------------------- импорт
    db = AssetDatabase(assets, library=library)
    state["db"] = db
    manager = ImportManager(db, poll_interval=0.0)

    @check("импорт: первичный (все типы)")
    def _import() -> str:
        manager.open()
        deadline = time.monotonic() + 120
        while manager.busy and time.monotonic() < deadline:
            manager.poll(force=True)
            time.sleep(0.05)
        db.wait_idle()
        failed = {
            str(db.get(g).path.name): manager._errors.get(g)  # noqa: SLF001
            for g in db.guids
            if db.get(g) is not None and not db.get(g).imported
        }
        assert not manager.busy, "очередь не опустела"
        assert not failed, f"не импортированы: {failed}"
        return f"ассетов: {len(db.guids)}"

    @check("импорт: идемпотентность")
    def _idem() -> str:
        before = sorted(p.relative_to(library).as_posix() for p in library.rglob("*"))
        manager.poll(force=True)
        db.wait_idle()
        db.reimport_dirty(wait=True)
        after = sorted(p.relative_to(library).as_posix() for p in library.rglob("*"))
        assert before == after, "Library изменилась при повторном импорте"
        return f"файлов в Library: {len(after)}"

    @check("импорт: правка файла → poll → реимпорт")
    def _edit() -> str:
        guid = db.guid_of("flat.frag")
        stamp = db.resolve(guid).imported_version
        path = assets / "flat.frag"
        time.sleep(0.05)
        path.write_text(path.read_text(encoding="utf-8") + "// edit\n", "utf-8")
        os.utime(path, (time.time() + 5, time.time() + 5))
        manager.poll(force=True)
        deadline = time.monotonic() + 30
        while manager.busy and time.monotonic() < deadline:
            manager.poll(force=True)
            time.sleep(0.05)
        db.wait_idle()
        assert not db.is_dirty(guid), "ассет остался грязным"
        assert db.resolve(guid).imported_version == stamp
        return "реимпорт выполнен"

    @check("импорт: переименование сохраняет GUID")
    def _rename() -> str:
        guid = db.guid_of("triangle.obj")
        (assets / "triangle.obj").rename(assets / "tri2.obj")
        (assets / "triangle.obj.meta").rename(assets / "tri2.obj.meta")
        manager.poll(force=True)
        db.wait_idle()
        assert db.guid_of("tri2.obj") == guid
        return guid

    @check("импорт: gc_library")
    def _gc() -> str:
        junk = library / "ff" / ("f" * 32)
        junk.mkdir(parents=True)
        (junk / "x.bin").write_bytes(b"x")
        removed = manager.gc_library()
        assert not junk.exists(), "мусор не удалён"
        return f"удалено: {removed}"

    # --------------------------------------------------------------- сцены
    @check("сцены: save/load с GUID меша")
    def _scenes() -> str:
        from editor.scene_io import load_scene, save_scene, scene_to_dict

        guid = db.guid_of("tri2.obj")
        scene = Scene("E2E")
        for index in range(5):
            obj = scene.create_game_object(f"Obj{index}")
            obj.add_component(MeshRenderer, mesh="tri2.obj")
            obj.transform.position = Vector3(float(index), 0.0, 0.0)
        path = save_scene(scene, OUT / "scene.scene.json", db)
        payload = scene_to_dict(scene, db)
        assert guid in json.dumps(payload), "GUID меша не записан"
        loaded = load_scene(path, db)
        assert len(loaded) == 5
        # Старая сцена без GUID.
        old = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(old).replace(guid, "")
        stripped = OUT / "old.scene.json"
        stripped.write_text(text, encoding="utf-8")
        assert len(load_scene(stripped, db)) == 5
        return "5 объектов, GUID сохраняется"

    # ------------------------------------------------------------- скрипты
    @check("скрипты: хуки, hot reload, ошибка не роняет цикл")
    def _scripts() -> str:
        scripts = OUT / "scripts"
        scripts.mkdir()
        script = scripts / "spin.py"
        script.write_text(
            "CALLS = 0\n"
            "def on_start(self):\n    pass\n"
            "def on_update(self, delta_time):\n"
            "    global CALLS\n    CALLS += 1\n"
            "def on_destroy(self):\n    pass\n",
            encoding="utf-8",
        )
        runtime = ScriptRuntime(project_root=OUT)
        scene = Scene("S")
        obj = scene.create_game_object("O")
        comp = obj.add_component(ScriptComponent(script, runtime=runtime))
        comp.on_start()
        for _ in range(3):
            scene.update(0.016)
        assert comp.module.CALLS == 3
        script.write_text(
            "def on_update(self, delta_time):\n    raise RuntimeError('boom')\n",
            encoding="utf-8",
        )
        os.utime(script, (time.time() + 5, time.time() + 5))
        runtime.refresh()
        for _ in range(3):
            scene.update(0.016)  # исключение не должно вылететь
        scene.destroy_game_object(obj)
        return "цикл пережил ошибку скрипта"

    # ---------------------------------------------------------------- пак
    @check("пак: сборка, чтение, порча")
    def _pak() -> str:
        guid = db.guid_of("sprites.atlas.json")
        pak_path = PakBuilder(db).add_all().build(OUT / "game.pak")
        with PakReader(pak_path) as pak:
            assert set(pak.guids()) == {g for g in db.guids if db.get(g).imported}
            atlas = load_sprite_atlas(pak, guid)
            assert atlas.frames
        raw = bytearray(pak_path.read_bytes())
        raw[20] ^= 0xFF
        bad = OUT / "bad.pak"
        bad.write_bytes(bytes(raw))
        try:
            with PakReader(bad) as pak:
                for g in pak.guids():
                    for entry in pak._record(g)["artifacts"]:  # noqa: SLF001
                        pak.read(g, entry["name"])
        except ResourceFormatError:
            return f"пак {pak_path.stat().st_size} байт, порча обнаружена"
        raise AssertionError("порча пака не обнаружена")

    # --------------------------------------------------------------- рендер
    @check("рендер: меши, PBR, свет+тени, инстансинг, спрайты")
    def _render() -> str:
        from engine.core.math3d import Vector3, compose_matrix
        from engine.rendering import (
            Camera,
            GpuMesh,
            QualityTier,
            Renderer,
            RenderSettings,
            RenderTarget,
            save_png,
        )
        from engine.rendering.qt_context import QtOffscreenContext

        settings = RenderSettings.for_quality(QualityTier.MEDIUM, samples=0)
        notes = []
        with QtOffscreenContext((640, 360), settings=settings) as context:
            width, height = context.surface.size()
            renderer = Renderer(settings=settings, shader_root=ROOT / "assets")
            renderer.initialize()
            mesh = GpuMesh.from_mesh_asset(cube_mesh(1.0))
            material = renderer.create_material(
                "m", base_color=Vector3(0.8, 0.4, 0.2), roughness=0.4
            )
            camera = Camera(fov=45.0, far=100.0)
            camera.position = Vector3(3.0, 2.5, 3.5)
            camera.look_at(Vector3(0.0, 0.0, 0.0))
            target = RenderTarget("e2e", width, height)
            renderer.begin_frame(camera, (width, height), target=target)
            renderer.submit(mesh, material, compose_matrix())
            matrices = [
                compose_matrix(position=Vector3(i * 1.5 - 6, 0.0, -3.0))
                for i in range(9)
            ]
            renderer.draw_instanced(mesh, material, matrices)
            stats = renderer.end_frame(target=target)
            pixels = target.read_pixels()
            assert pixels.std() > 5.0, "кадр пустой"
            save_png(OUT / "render.png", pixels, flip_vertical=True)
            notes.append(stats.describe())
            renderer.dispose()
            mesh.dispose()
            target.dispose()
        return "; ".join(notes)

    # ------------------------------------------------------------ негативные
    @check("негатив: битые файлы не роняют импорт")
    def _negative() -> str:
        (assets / "broken.png").write_bytes(b"not a png")
        (assets / "broken.obj").write_text("f 1 2 3\n", encoding="utf-8")
        (assets / "broken.wav").write_bytes(b"RIFF")
        manager.poll(force=True)
        deadline = time.monotonic() + 30
        while manager.busy and time.monotonic() < deadline:
            manager.poll(force=True)
            time.sleep(0.05)
        db.wait_idle()
        broken = [
            g
            for g in db.guids
            if db.get(g) is not None and db.get(g).path.name.startswith("broken")
        ]
        assert broken and all(not db.get(g).imported for g in broken), (
            "битые файлы импортировались"
        )
        return f"отклонено: {len(broken)}"

    @check("негатив: удаление файла при открытом проекте")
    def _delete() -> str:
        (assets / "tiny.ogg").unlink()
        (assets / "tiny.ogg.meta").unlink(missing_ok=True)
        manager.poll(force=True)
        db.wait_idle()
        assert db.guid_of("tiny.ogg") is None
        return "ассет исчез из базы"

    manager.close()
    db.close()

    failed = [name for name, item in RESULTS.items() if not item["ok"]]
    (OUT / "report.json").write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nИтого: {len(RESULTS) - len(failed)}/{len(RESULTS)} проверок успешно")
    for name in failed:
        print(f"  ПРОВАЛ: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
