"""Тесты пайплайна импорта: GUID, .meta, реестр, JobSystem и AssetDatabase."""

from __future__ import annotations

import json
import shutil
import struct
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from engine.core.event_bus import EventBus
from engine.core.jobs import JobSystem
from engine.core.resources import (
    Artifact,
    AssetDatabase,
    AssetImported,
    AssetImportError,
    AssetImportFailed,
    AssetMeta,
    AssetPreview,
    ImportContext,
    ImporterNotFoundError,
    ImporterRegistry,
    InvalidGuidError,
    MetaFormatError,
    ResourceManager,
    ResourceNotFoundError,
    default_importers,
    is_valid_guid,
    meta_path_for,
    new_guid,
    parse_guid,
    read_meta,
    write_meta,
)
from engine.core.resources.asset_db import RECORD_FILE
from engine.core.resources.cookers import read_cooked_mesh, read_cooked_texture
from engine.core.resources.importers import DdsImporter, parse_dds_header

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """Проект с одним ассетом каждого формата в ``Assets/``."""
    assets = tmp_path / "Assets"
    assets.mkdir()
    for name in ("triangle.obj", "tiny.png", "tiny.wav", "tiny.ogg", "tiny_bc1.dds"):
        shutil.copyfile(FIXTURES / name, assets / name)
    return assets


@pytest.fixture()
def db(project: Path):
    with AssetDatabase(project) as database:
        yield database


class TestGuid:
    def test_new_guid_is_valid_and_unique(self) -> None:
        first, second = new_guid(), new_guid()
        assert is_valid_guid(first)
        assert first != second

    def test_parse_normalizes_uuid_form(self) -> None:
        guid = new_guid()
        dashed = f"{guid[:8]}-{guid[8:12]}-{guid[12:16]}-{guid[16:20]}-{guid[20:]}"
        assert parse_guid(f"  {dashed.upper()} ") == guid

    @pytest.mark.parametrize("bad", ["", "abc", "z" * 32, "a" * 31])
    def test_parse_rejects_garbage(self, bad: str) -> None:
        with pytest.raises(InvalidGuidError):
            parse_guid(bad)

    def test_is_valid_guid_rejects_non_strings(self) -> None:
        assert not is_valid_guid(123)


class TestMeta:
    def test_roundtrip_and_stable_format(self, tmp_path: Path) -> None:
        asset = tmp_path / "hero.obj"
        asset.write_text("v 0 0 0\n")
        meta = AssetMeta(new_guid(), "obj", 3, {"scale": 2.0})
        assert write_meta(asset, meta) is True
        assert meta_path_for(asset).name == "hero.obj.meta"
        assert read_meta(asset) == meta
        assert list(json.loads(meta_path_for(asset).read_text())) == [
            "guid",
            "importer",
            "version",
            "settings",
        ]

    def test_unchanged_meta_is_not_rewritten(self, tmp_path: Path) -> None:
        asset = tmp_path / "a.obj"
        meta = AssetMeta(new_guid(), "obj")
        write_meta(asset, meta)
        assert write_meta(asset, meta) is False

    def test_missing_meta_is_none(self, tmp_path: Path) -> None:
        assert read_meta(tmp_path / "nothing.obj") is None

    @pytest.mark.parametrize(
        "content",
        [
            "не json",
            "[]",
            '{"guid": "bad", "importer": "obj"}',
            '{"guid": "%s", "importer": ""}' % ("a" * 32),
            '{"guid": "%s", "importer": "obj", "version": -1}' % ("a" * 32),
            '{"guid": "%s", "importer": "obj", "settings": []}' % ("a" * 32),
        ],
    )
    def test_corrupted_meta_raises(self, tmp_path: Path, content: str) -> None:
        asset = tmp_path / "a.obj"
        meta_path_for(asset).write_text(content, encoding="utf-8")
        with pytest.raises(MetaFormatError):
            read_meta(asset)


class FakeImporter:
    name = "fake"
    extensions = (".fake",)
    version = 1

    def __init__(self) -> None:
        self.calls = 0
        self.threads: set[int] = set()

    def default_settings(self) -> dict[str, Any]:
        return {"mode": "a"}

    def inspect(self, path: Path) -> AssetPreview:
        return AssetPreview("fake")

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        self.calls += 1
        self.threads.add(threading.get_ident())
        (ctx.output_dir / "out.bin").write_bytes(
            ctx.source.read_bytes() + str(ctx.setting("mode")).encode()
        )
        for line in ctx.source.read_text().split():
            if line.startswith("dep:"):
                ctx.add_dependency(ctx.source.parent / line[4:])
        return [ctx.artifact("fake", "out.bin")]


class TestImporterRegistry:
    def test_defaults_cover_phase_one_formats(self) -> None:
        registry = ImporterRegistry(default_importers())
        assert registry.extensions == (
            ".atlas.json",
            ".bmp",
            ".comp",
            ".cue.json",
            ".dds",
            ".fbx",
            ".frag",
            ".geom",
            ".gif",
            ".glb",
            ".glsl",
            ".gltf",
            ".jpeg",
            ".jpg",
            ".mat.json",
            ".music.json",
            ".obj",
            ".ogg",
            ".png",
            ".prefab.json",
            ".vert",
            ".wav",
        )
        assert registry.importer_for(Path("X.PNG")).name == "image"

    def test_duplicate_extension_or_name_rejected(self) -> None:
        registry = ImporterRegistry([FakeImporter()])
        with pytest.raises(ValueError, match="уже зарегистрирован"):
            registry.register(FakeImporter())
        other = FakeImporter()
        other.name = "other"
        with pytest.raises(ValueError, match="уже обрабатывает"):
            registry.register(other)

    def test_invalid_declaration_rejected(self) -> None:
        bad = FakeImporter()
        bad.extensions = ("fake",)
        with pytest.raises(ValueError, match="начинаться"):
            ImporterRegistry([bad])
        bad = FakeImporter()
        bad.version = 0
        with pytest.raises(ValueError, match="version"):
            ImporterRegistry([bad])

    def test_unregister(self) -> None:
        registry = ImporterRegistry([FakeImporter()])
        assert registry.unregister("fake") is True
        assert registry.importer_for(Path("a.fake")) is None
        assert registry.unregister("fake") is False

    def test_plugins_are_loaded_and_isolated(self, monkeypatch) -> None:
        class EntryPoint:
            def __init__(self, name: str, target) -> None:
                self.name, self._target = name, target

            def load(self):
                return self._target()

        def broken():
            raise RuntimeError("плагин сломан")

        monkeypatch.setattr(
            "importlib.metadata.entry_points",
            lambda group: [
                EntryPoint("bad", broken),
                EntryPoint("ok", lambda: lambda reg: reg.register(FakeImporter())),
            ],
        )
        registry = ImporterRegistry()
        assert registry.load_plugins() == 1
        assert registry.get("fake") is not None


class TestJobSystem:
    def test_callback_runs_in_pumping_thread(self) -> None:
        main = threading.get_ident()
        seen: list[tuple[int, int]] = []
        with JobSystem(workers=2) as jobs:
            job = jobs.submit(
                lambda: threading.get_ident(),
                on_done=lambda j: seen.append((j.result(), threading.get_ident())),
            )
            assert jobs.wait_idle(5.0)
        worker, callback = seen[0]
        assert worker != main
        assert callback == main
        assert job.done()

    def test_error_is_reported_without_raising_in_pump(self) -> None:
        def boom() -> None:
            raise ValueError("x")

        errors: list[BaseException | None] = []
        with JobSystem(workers=1) as jobs:
            jobs.submit(boom, on_done=lambda j: errors.append(j.error))
            jobs.wait_idle(5.0)
        assert isinstance(errors[0], ValueError)

    def test_failing_callback_does_not_block_queue(self) -> None:
        done: list[int] = []

        def bad(_job) -> None:
            raise RuntimeError("callback")

        with JobSystem(workers=1) as jobs:
            jobs.submit(lambda: 1, on_done=bad)
            jobs.submit(lambda: 2, on_done=lambda j: done.append(j.result()))
            assert jobs.wait_idle(5.0)
            assert jobs.pending == 0
        assert done == [2]

    def test_wait_idle_timeout(self) -> None:
        gate = threading.Event()
        jobs = JobSystem(workers=1)
        jobs.submit(gate.wait)
        assert jobs.wait_idle(0.1) is False
        gate.set()
        assert jobs.wait_idle(5.0) is True
        jobs.shutdown()

    def test_submit_after_shutdown_fails(self) -> None:
        jobs = JobSystem()
        jobs.shutdown()
        with pytest.raises(RuntimeError):
            jobs.submit(lambda: None)


class TestAssetDatabase:
    def test_scan_registers_all_and_creates_meta(self, db: AssetDatabase, project: Path) -> None:
        guids = db.scan()
        assert len(guids) == len(set(guids)) == 5
        for path in project.iterdir():
            if path.suffix != ".meta":
                assert meta_path_for(path).is_file()
        handle = db.resolve(db.guid_of(project / "triangle.obj"))
        assert handle.importer == "obj"
        assert not handle.imported
        assert handle.directory == db.library / handle.guid[:2] / handle.guid

    def test_guid_survives_new_database_instance(self, project: Path) -> None:
        with AssetDatabase(project) as first:
            guid = first.register(project / "triangle.obj")
        with AssetDatabase(project) as second:
            assert second.register(project / "triangle.obj") == guid

    def test_register_unknown_and_missing(self, db: AssetDatabase, project: Path) -> None:
        (project / "readme.txt").write_text("x")
        with pytest.raises(ImporterNotFoundError):
            db.register(project / "readme.txt")
        with pytest.raises(ResourceNotFoundError):
            db.register(project / "nope.obj")

    def test_scan_skips_unknown_corrupt_meta_and_library(
        self, db: AssetDatabase, project: Path
    ) -> None:
        (project / "readme.txt").write_text("x")
        meta_path_for(project / "tiny.png").write_text("broken", encoding="utf-8")
        db.library.mkdir(parents=True, exist_ok=True)
        guids = db.scan()
        assert len(guids) == 4
        assert meta_path_for(project / "tiny.png").read_text() == "broken"

    def test_duplicated_meta_gets_fresh_guid(self, db: AssetDatabase, project: Path) -> None:
        original = db.register(project / "tiny.png")
        shutil.copyfile(project / "tiny.png", project / "copy.png")
        shutil.copyfile(
            meta_path_for(project / "tiny.png"), meta_path_for(project / "copy.png")
        )
        copied = db.register(project / "copy.png")
        assert copied != original
        assert read_meta(project / "copy.png").guid == copied

    def test_resolve_unknown_raises(self, db: AssetDatabase) -> None:
        with pytest.raises(ResourceNotFoundError):
            db.resolve(new_guid())
        assert db.get(new_guid()) is None

    def test_import_all_formats_creates_artifacts(
        self, db: AssetDatabase, project: Path
    ) -> None:
        db.scan()
        assert len(db.dirty()) == 5
        db.reimport_dirty(wait=True)
        assert db.dirty() == []

        obj = db.resolve(db.guid_of(project / "triangle.obj"))
        mesh = obj.artifact("mesh")
        assert mesh.info["triangles"] == 1
        cooked = read_cooked_mesh(obj.artifact_path("mesh"))
        assert cooked.positions().shape == (3, 3)
        assert cooked.lods[0].indices.tolist() == [0, 1, 2]

        png = db.resolve(db.guid_of(project / "tiny.png"))
        assert png.artifact("texture").info["width"] == 2
        cooked_png = read_cooked_texture(png.artifact_path("texture").read_bytes())
        assert (cooked_png.width, cooked_png.height) == (2, 2)
        assert cooked_png.pixel_format == "rgba8" and len(cooked_png.mips) == 2

        dds = db.resolve(db.guid_of(project / "tiny_bc1.dds"))
        assert dds.artifact("texture").info["pixel_format"] == "bc1"

        for name, codec in (("tiny.wav", "wav"), ("tiny.ogg", "ogg")):
            clip = db.resolve(db.guid_of(project / name)).artifact("audio_clip_bin")
            assert clip.info["format"] == codec
            assert clip.info["duration"] == pytest.approx(0.1 if codec == "wav" else 1.0)

    def test_reimport_is_idempotent(self, db: AssetDatabase, project: Path) -> None:
        db.scan()
        db.reimport_dirty(wait=True)

        def snapshot() -> dict[str, bytes]:
            files = {
                str(p.relative_to(db.library)): p.read_bytes()
                for p in db.library.rglob("*")
                if p.is_file()
            }
            files.update(
                {p.name: p.read_bytes() for p in project.glob("*.meta")}
            )
            return files

        before = snapshot()
        guids = set(db.guids)
        assert db.dirty() == []
        assert db.reimport_dirty(wait=True) == []
        db.scan()
        assert set(db.guids) == guids
        db.reimport(next(iter(guids)), wait=True)
        after = snapshot()
        assert set(after) == set(before)  # ни дублей, ни мусора
        assert not list(db.library.rglob("*.tmp"))

    def test_library_survives_restart_without_reimport(self, project: Path) -> None:
        with AssetDatabase(project) as first:
            first.scan()
            first.reimport_dirty(wait=True)
        with AssetDatabase(project) as second:
            second.scan()
            assert second.dirty() == []
            assert second.resolve(second.guids[0]).imported

    def test_source_change_makes_dirty(self, db: AssetDatabase, project: Path) -> None:
        guid = db.register(project / "triangle.obj")
        db.reimport(guid, wait=True)
        assert not db.is_dirty(guid)
        (project / "triangle.obj").write_text("v 0 0 0\nv 2 0 0\nv 0 2 0\nf 1 2 3\n")
        assert db.is_dirty(guid)
        db.reimport(guid, wait=True)
        assert not db.is_dirty(guid)
        assert db.resolve(guid).artifact("mesh").info["bounds_max"] == [2.0, 2.0, 0.0]

    def test_touch_without_content_change_is_clean(
        self, db: AssetDatabase, project: Path
    ) -> None:
        guid = db.register(project / "tiny.png")
        db.reimport(guid, wait=True)
        path = project / "tiny.png"
        path.write_bytes(path.read_bytes())
        assert not db.is_dirty(guid)

    def test_settings_change_makes_dirty(self, db: AssetDatabase, project: Path) -> None:
        guid = db.register(project / "triangle.obj")
        db.reimport(guid, wait=True)
        db.set_settings(guid, {"scale": 3.0})
        assert db.is_dirty(guid)
        assert read_meta(project / "triangle.obj").settings == {"scale": 3.0}
        db.reimport(guid, wait=True)
        assert db.resolve(guid).artifact("mesh").info["bounds_max"][0] == 3.0
        assert not db.is_dirty(guid)

    def test_importer_version_bump_makes_dirty(self, project: Path) -> None:
        (project / "a.fake").write_text("hello")
        importer = FakeImporter()
        registry = ImporterRegistry([importer])
        with AssetDatabase(project, importers=registry) as database:
            guid = database.register(project / "a.fake")
            database.reimport(guid, wait=True)
            assert read_meta(project / "a.fake").version == 1
            importer.version = 2
            assert database.is_dirty(guid)
            database.reimport(guid, wait=True)
            assert read_meta(project / "a.fake").version == 2
            assert not database.is_dirty(guid)

    def test_deleted_artifact_is_detected(self, db: AssetDatabase, project: Path) -> None:
        guid = db.register(project / "triangle.obj")
        db.reimport(guid, wait=True)
        db.resolve(guid).artifact_path("mesh").unlink()
        with AssetDatabase(project) as fresh:
            assert fresh.is_dirty(fresh.register(project / "triangle.obj"))

    def test_import_runs_in_worker_and_events_in_main(self, project: Path) -> None:
        (project / "a.fake").write_text("hello")
        importer = FakeImporter()
        bus = EventBus()
        events: list[tuple[str, int]] = []
        bus.subscribe(
            AssetImported, lambda e: events.append((e.guid, threading.get_ident()))
        )
        with AssetDatabase(
            project, importers=ImporterRegistry([importer]), event_bus=bus
        ) as database:
            guid = database.register(project / "a.fake")
            job = database.reimport(guid)
            assert events == []  # событие не приходит до pump
            database.wait_idle(5.0)
            assert job.done()
        assert events == [(guid, threading.get_ident())]
        assert threading.get_ident() not in importer.threads

    def test_concurrent_reimport_shares_job(self, project: Path) -> None:
        gate = threading.Event()

        class Slow(FakeImporter):
            def import_asset(self, ctx: ImportContext) -> list[Artifact]:
                gate.wait(5.0)
                return super().import_asset(ctx)

        slow = Slow()
        (project / "a.fake").write_text("hello")
        with AssetDatabase(project, importers=ImporterRegistry([slow])) as database:
            guid = database.register(project / "a.fake")
            first = database.reimport(guid)
            assert database.reimport(guid) is first
            gate.set()
            database.wait_idle(5.0)
        assert slow.calls == 1

    def test_failed_import_publishes_event_and_cleans_up(
        self, db: AssetDatabase, project: Path
    ) -> None:
        bus = EventBus()
        db.events = bus
        failures: list[AssetImportFailed] = []
        bus.subscribe(AssetImportFailed, failures.append)
        (project / "bad.obj").write_text("f 1 2 3\n")
        guid = db.register(project / "bad.obj")
        db.reimport(guid, wait=True)
        assert len(failures) == 1 and failures[0].guid == guid
        handle = db.resolve(guid)
        assert handle.error and not handle.imported
        assert not handle.directory.exists()
        assert not list(db.library.rglob("*.tmp"))
        assert db.is_dirty(guid)

    def test_failed_reimport_keeps_previous_artifacts(
        self, db: AssetDatabase, project: Path
    ) -> None:
        guid = db.register(project / "triangle.obj")
        db.reimport(guid, wait=True)
        (project / "triangle.obj").write_text("f 1 2 3\n")
        db.reimport(guid, wait=True)
        handle = db.resolve(guid)
        assert handle.error is not None
        assert handle.imported
        assert handle.artifact_path("mesh").is_file()

    def test_dependency_graph(self, project: Path) -> None:
        (project / "mat.fake").write_text("dep:tex.fake dep:tex.fake")
        (project / "tex.fake").write_text("pixels")
        (project / "scene.fake").write_text("dep:mat.fake")
        with AssetDatabase(
            project, importers=ImporterRegistry([FakeImporter()])
        ) as database:
            scene = database.register(project / "scene.fake")
            database.reimport(scene, wait=True)
            mat = database.guid_of(project / "mat.fake")
            assert mat is not None  # зарегистрирована как зависимость
            database.reimport(mat, wait=True)
            tex = database.guid_of(project / "tex.fake")
            assert database.dependencies(scene) == [mat]
            assert database.dependencies(mat) == [tex]
            assert database.dependents(tex) == [mat]
            assert database.dependency_closure(scene) == [mat, tex]
            record = json.loads(
                (database.resolve(mat).directory / RECORD_FILE).read_text()
            )
            assert record["dependencies"] == [tex]

    def test_manager_reloads_loaded_asset_after_reimport(self, project: Path) -> None:
        manager = ResourceManager()
        with AssetDatabase(project, manager=manager) as database:
            path = project / "triangle.obj"
            guid = database.register(path)
            old = manager.load(path)
            assert old.vertex_count == 3
            path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nf 2 4 3\n")
            database.reimport(guid, wait=True)
            assert manager.get(path).vertex_count == 4


class TestImporters:
    def test_obj_inspect(self) -> None:
        preview = default_importers()[0].inspect(FIXTURES / "triangle.obj")
        assert preview.kind == "mesh"
        assert dict(preview.info) == {"vertices": 3, "faces": 1}

    def test_unsupported_dds_is_rejected(self, tmp_path: Path) -> None:
        header = bytearray((FIXTURES / "tiny_bc1.dds").read_bytes())
        header[84:88] = b"DXT3"
        path = tmp_path / "bc2.dds"
        path.write_bytes(bytes(header))
        with pytest.raises(AssetImportError, match="BC1/BC3/BC5"):
            parse_dds_header(path)

    def test_truncated_dds_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "short.dds"
        path.write_bytes((FIXTURES / "tiny_bc1.dds").read_bytes()[:-1])
        with pytest.raises(AssetImportError, match="нужно"):
            DdsImporter().inspect(path)

    def test_dds_dx10_bc5(self, tmp_path: Path) -> None:
        header = bytearray(148)
        header[0:4] = b"DDS "
        struct.pack_into("<II", header, 12, 4, 4)
        struct.pack_into("<I", header, 28, 1)
        header[84:88] = b"DX10"
        struct.pack_into("<IIII", header, 128, 83, 3, 0, 1)
        path = tmp_path / "bc5.dds"
        path.write_bytes(bytes(header) + bytes(16))
        info = parse_dds_header(path)
        assert info["pixel_format"] == "bc5"
        assert info["data_offset"] == 148

    def test_broken_png_import_fails_cleanly(self, db: AssetDatabase, project: Path) -> None:
        (project / "broken.png").write_bytes(b"not a png at all")
        guid = db.register(project / "broken.png")
        db.reimport(guid, wait=True)
        assert db.resolve(guid).error is not None
        assert not db.resolve(guid).imported


def test_importer_artifact_must_be_written(tmp_path: Path) -> None:
    ctx = ImportContext(new_guid(), tmp_path / "s", {}, tmp_path)
    with pytest.raises(AssetImportError, match="не был записан"):
        ctx.artifact("x", "missing.bin")
    with pytest.raises(AssetImportError, match="вне каталога"):
        ctx.artifact("x", "../escape.bin")


def test_resolve_is_dict_lookup(db: AssetDatabase, project: Path) -> None:
    """``resolve`` на каждый кадр: не должен зависеть от размера графа."""
    guid = db.register(project / "tiny.png")
    start = time.perf_counter()
    for _ in range(100_000):
        db.resolve(guid)
    assert time.perf_counter() - start < 1.0


def test_readonly_assets_dir_does_not_crash(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Каталог без прав записи: ассеты регистрируются, .meta не пишется."""
    import engine.core.resources.asset_db as asset_db_module

    def denied(*_args: object, **_kwargs: object) -> bool:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(asset_db_module, "write_meta", denied)
    library = project.parent / "LibraryRO"
    with AssetDatabase(project, library=library) as database:
        with caplog.at_level("WARNING"):
            guids = database.scan()
        assert guids, "ассеты должны регистрироваться без .meta"
        assert not list(project.rglob("*.meta"))
        warnings = [r for r in caplog.records if "Не удалось записать .meta" in r.message]
        assert 1 <= len(warnings) <= len({p.parent for p in project.rglob("*.*")})


def test_read_only_database_writes_no_meta(project: Path, tmp_path: Path) -> None:
    """``read_only=True``: ассеты видны и импортируются, ``.meta`` не создаются."""
    with AssetDatabase(
        project, library=tmp_path / "LibraryRO2", read_only=True
    ) as database:
        guids = database.scan()
        database.reimport_dirty(wait=True)
        assert guids and all(database.resolve(g).imported for g in guids)
    assert not list(project.rglob("*.meta"))
