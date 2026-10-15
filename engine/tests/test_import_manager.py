"""Тесты менеджера импортов: очередь, слежение за диском, настройки, Library."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from engine.core.event_bus import EventBus
from engine.core.resources import (
    Artifact,
    AssetDatabase,
    AssetImportError,
    AssetMoved,
    AssetPreview,
    AssetRemoved,
    ImportBatchFinished,
    ImportBatchProgress,
    ImportBatchStarted,
    ImportContext,
    ImporterRegistry,
    ImportManager,
    ImportState,
    default_importers,
    meta_path_for,
    settings_schema,
    validate_settings,
)

FIXTURES = Path(__file__).parent / "fixtures"


class FakeImporter:
    """Импортёр ``.fake``: копирует байты; ``fail`` в тексте — ошибка."""

    name = "fake"
    extensions = (".fake",)
    version = 1

    def default_settings(self) -> dict[str, Any]:
        return {"mode": "a", "level": 1, "flag": False}

    def inspect(self, path: Path) -> AssetPreview:
        return AssetPreview("fake")

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        text = ctx.source.read_text()
        if "fail" in text:
            raise AssetImportError("Намеренная ошибка")
        (ctx.output_dir / "out.bin").write_bytes(
            text.encode() + str(ctx.setting("mode")).encode()
        )
        return [ctx.artifact("fake", "out.bin")]


def _touch(path: Path, text: str) -> None:
    """Записать файл и сдвинуть mtime, чтобы stat гарантированно отличался."""
    previous = path.stat().st_mtime_ns if path.exists() else None
    path.write_text(text)
    if previous is not None:
        os.utime(path, ns=(previous + 5_000_000, previous + 5_000_000))


@pytest.fixture()
def assets(tmp_path: Path) -> Path:
    root = tmp_path / "Assets"
    root.mkdir()
    (root / "a.fake").write_text("alpha")
    (root / "b.fake").write_text("beta")
    return root


@pytest.fixture()
def manager(assets: Path):
    registry = ImporterRegistry([FakeImporter(), *default_importers()])
    db = AssetDatabase(assets, importers=registry, event_bus=EventBus())
    with ImportManager(db, poll_interval=0.0) as mgr:
        yield mgr
    db.close()


class TestSchema:
    def test_derived_from_defaults(self) -> None:
        specs = {spec.name: spec for spec in settings_schema(FakeImporter())}
        assert specs["mode"].kind == "str"
        assert specs["level"].kind == "int"
        assert specs["flag"].kind == "bool"

    def test_validate_keeps_only_non_defaults(self) -> None:
        cleaned = validate_settings(FakeImporter(), {"mode": "a", "level": 3})
        assert cleaned == {"level": 3}

    @pytest.mark.parametrize(
        "bad",
        [{"nope": 1}, {"level": "x"}, {"flag": 1}, {"mode": 5}, {"level": True}],
    )
    def test_validate_rejects(self, bad: dict[str, Any]) -> None:
        with pytest.raises(AssetImportError):
            validate_settings(FakeImporter(), bad)

    def test_custom_schema_ranges(self) -> None:
        from engine.core.resources import SettingSpec

        importer = FakeImporter()
        importer.settings_schema = lambda: [  # type: ignore[attr-defined]
            SettingSpec("level", "int", 1, minimum=0, maximum=4),
            SettingSpec("mode", "choice", "a", choices=("a", "b")),
        ]
        assert validate_settings(importer, {"level": 2, "mode": "b"}) == {
            "level": 2,
            "mode": "b",
        }
        with pytest.raises(AssetImportError):
            validate_settings(importer, {"level": 9})
        with pytest.raises(AssetImportError):
            validate_settings(importer, {"mode": "z"})


class TestOpen:
    def test_open_imports_everything(self, manager: ImportManager) -> None:
        assert manager.open() == 2
        assert manager.wait_idle(10)
        assert not manager.db.dirty()
        assert set(manager.states().values()) == {ImportState.OK}

    def test_second_open_is_clean(self, manager: ImportManager) -> None:
        manager.open()
        manager.wait_idle(10)
        again = ImportManager(
            AssetDatabase(
                manager.db.root, importers=manager.db.importers, event_bus=EventBus()
            )
        )
        try:
            assert again.open() == 0
            assert not again.busy
        finally:
            again.close()
            again.db.close()

    def test_batch_events(self, manager: ImportManager) -> None:
        log: list[Any] = []
        for kind in (ImportBatchStarted, ImportBatchProgress, ImportBatchFinished):
            manager.events.subscribe(kind, log.append)
        manager.open()
        manager.wait_idle(10)
        assert isinstance(log[0], ImportBatchStarted)
        progress = [e for e in log if isinstance(e, ImportBatchProgress)]
        assert [e.done for e in progress] == [1, 2]
        assert isinstance(log[-1], ImportBatchFinished)
        assert (log[-1].ok, log[-1].failed, log[-1].total) == (2, 0, 2)
        assert manager.progress() == (0, 0)


class TestPoll:
    def test_new_file_is_imported(self, manager: ImportManager, assets: Path) -> None:
        manager.open()
        manager.wait_idle(10)
        (assets / "c.fake").write_text("gamma")
        assert manager.poll() == 1
        manager.wait_idle(10)
        guid = manager.db.guid_of(assets / "c.fake")
        assert guid is not None
        assert manager.state(guid) is ImportState.OK
        assert manager.db.resolve(guid).imported

    def test_changed_file_is_reimported_once(
        self, manager: ImportManager, assets: Path
    ) -> None:
        manager.open()
        manager.wait_idle(10)
        guid = manager.db.guid_of(assets / "a.fake")
        _touch(assets / "a.fake", "alpha-2")
        assert manager.poll() == 1
        manager.wait_idle(10)
        out = manager.db.resolve(guid).artifact_path("fake")
        assert out.read_bytes() == b"alpha-2a"
        assert manager.poll() == 0
        library_dirs = list(manager.db.library.rglob("out.bin"))
        assert len(library_dirs) == 2

    def test_deleted_file_is_unregistered(
        self, manager: ImportManager, assets: Path
    ) -> None:
        manager.open()
        manager.wait_idle(10)
        guid = manager.db.guid_of(assets / "b.fake")
        removed: list[AssetRemoved] = []
        manager.events.subscribe(AssetRemoved, removed.append)
        (assets / "b.fake").unlink()
        assert manager.poll() == 1
        assert guid not in manager.db
        assert removed and removed[0].guid == guid
        assert manager.state(guid) is ImportState.IDLE
        assert meta_path_for(assets / "b.fake").exists()

    def test_moved_file_keeps_guid(self, manager: ImportManager, assets: Path) -> None:
        manager.open()
        manager.wait_idle(10)
        guid = manager.db.guid_of(assets / "a.fake")
        sub = assets / "sub"
        sub.mkdir()
        shutil.move(assets / "a.fake", sub / "a.fake")
        shutil.move(meta_path_for(assets / "a.fake"), meta_path_for(sub / "a.fake"))
        moved: list[AssetMoved] = []
        manager.events.subscribe(AssetMoved, moved.append)
        assert manager.poll() == 1
        assert manager.db.guid_of(sub / "a.fake") == guid
        assert manager.db.resolve(guid).path == sub / "a.fake"
        assert moved and moved[0].guid == guid
        manager.wait_idle(10)
        assert not manager.busy
        assert len(manager.db) == 2

    def test_interval_throttles_polling(self, assets: Path) -> None:
        now = [100.0]
        db = AssetDatabase(
            assets, importers=ImporterRegistry([FakeImporter()]), event_bus=EventBus()
        )
        with ImportManager(db, poll_interval=5.0, clock=lambda: now[0]) as mgr:
            mgr.open()
            mgr.wait_idle(10)
            (assets / "c.fake").write_text("gamma")
            assert mgr.poll() == 0
            now[0] += 6
            assert mgr.poll() == 1
            mgr.wait_idle(10)
        db.close()


class TestFailures:
    def test_failure_does_not_stop_queue_and_is_not_retried(
        self, manager: ImportManager, assets: Path
    ) -> None:
        (assets / "a.fake").write_text("fail")
        manager.open()
        manager.wait_idle(10)
        bad = manager.db.guid_of(assets / "a.fake")
        good = manager.db.guid_of(assets / "b.fake")
        assert manager.state(bad) is ImportState.FAILED
        assert manager.state(good) is ImportState.OK
        assert "Намеренная" in manager.failures()[bad]
        assert manager.poll() == 0  # без правки файла повтора нет
        _touch(assets / "a.fake", "fixed")
        assert manager.poll() == 1
        manager.wait_idle(10)
        assert manager.state(bad) is ImportState.OK
        assert bad not in manager.failures()


class TestOperations:
    def test_reimport_all_forced(self, manager: ImportManager) -> None:
        manager.open()
        manager.wait_idle(10)
        assert manager.reimport_all(force=False) == 0
        assert manager.reimport_all() == 2
        assert manager.wait_idle(10)

    def test_reimport_by_path_registers_new_file(
        self, manager: ImportManager, assets: Path
    ) -> None:
        (assets / "d.fake").write_text("delta")
        guid = manager.reimport(assets / "d.fake")
        manager.wait_idle(10)
        assert manager.db.resolve(guid).imported

    def test_reimport_folder(self, manager: ImportManager, assets: Path) -> None:
        (assets / "sub").mkdir()
        (assets / "sub" / "x.fake").write_text("x")
        manager.open()
        manager.wait_idle(10)
        assert manager.reimport_folder(assets / "sub") == 1
        manager.wait_idle(10)

    def test_priority_goes_first(self, assets: Path) -> None:
        for name in "cde":
            (assets / f"{name}.fake").write_text(name)
        db = AssetDatabase(
            assets, importers=ImporterRegistry([FakeImporter()]), event_bus=EventBus()
        )
        with ImportManager(db, max_parallel=1) as mgr:
            mgr.open()
            queued = mgr.queued()
            last = queued[-1]
            mgr.reimport(last, priority=True)
            assert mgr.queued()[0] == last
            mgr.wait_idle(10)
        db.close()

    def test_cancel_removes_queued(self, assets: Path) -> None:
        for name in "cde":
            (assets / f"{name}.fake").write_text(name)
        db = AssetDatabase(
            assets, importers=ImporterRegistry([FakeImporter()]), event_bus=EventBus()
        )
        with ImportManager(db, max_parallel=1) as mgr:
            mgr.open()
            waiting = len(mgr.queued())
            assert waiting >= 1
            assert mgr.cancel() == waiting
            assert not mgr.queued()
            mgr.wait_idle(10)
            total_done, total = mgr.progress()
            assert (total_done, total) == (0, 0)
        db.close()

    def test_set_settings_validates_and_reimports(
        self, manager: ImportManager, assets: Path
    ) -> None:
        manager.open()
        manager.wait_idle(10)
        guid = manager.db.guid_of(assets / "a.fake")
        with pytest.raises(AssetImportError):
            manager.set_settings(guid, {"level": "много"})
        saved = manager.set_settings(guid, {"mode": "b", "level": 1})
        assert saved == {"mode": "b"}
        manager.wait_idle(10)
        out = manager.db.resolve(guid).artifact_path("fake")
        assert out.read_bytes() == b"alphab"
        assert manager.db.settings_of(guid)["mode"] == "b"

    def test_real_obj_setting_changes_artifact(self, tmp_path: Path) -> None:
        root = tmp_path / "Assets"
        root.mkdir()
        shutil.copyfile(FIXTURES / "triangle.obj", root / "t.obj")
        db = AssetDatabase(root, event_bus=EventBus())
        with ImportManager(db) as mgr:
            mgr.open()
            mgr.wait_idle(20)
            guid = db.guid_of(root / "t.obj")
            assert mgr.state(guid) is ImportState.OK
            mgr.set_settings(guid, {"scale": 2.0})
            mgr.wait_idle(20)
            assert db.settings_of(guid)["scale"] == 2.0
            assert not db.is_dirty(guid)
        db.close()


class TestLibrary:
    def test_gc_removes_only_orphans(self, manager: ImportManager) -> None:
        manager.open()
        manager.wait_idle(10)
        library = manager.db.library
        orphan = library / "ab" / ("c" * 32)
        orphan.mkdir(parents=True)
        (orphan / "x").write_text("x")
        stray_tmp = library / "ab" / ("d" * 32 + ".tmp")
        stray_tmp.mkdir()
        foreign = library / "ShaderCache"
        foreign.mkdir()
        (foreign / "keep.bin").write_text("keep")
        assert set(manager.orphans()) == {orphan, stray_tmp}
        assert manager.gc_library() == 2
        assert not orphan.exists() and not stray_tmp.exists()
        assert (foreign / "keep.bin").exists()
        assert not manager.db.dirty()

    def test_clear_and_rebuild(self, manager: ImportManager) -> None:
        manager.open()
        manager.wait_idle(10)
        assert manager.clear_library() == 2
        assert not list(manager.db.library.rglob("out.bin"))
        assert len(manager.db.dirty()) == 2
        assert manager.rebuild_library() == 2
        manager.wait_idle(10)
        assert not manager.db.dirty()
        assert len(list(manager.db.library.rglob("out.bin"))) == 2
