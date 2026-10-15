"""Интеграция менеджера импортов в редактор: контекст, панели, форма настроек."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from editor.context import EditorContext
from editor.panels import AssetsPanel, ImportPanel, InspectorPanel
from engine.core.resources import ImportState

OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Проект с одним корректным OBJ и одной битой текстурой."""
    assets = tmp_path / "proj" / "assets"
    assets.mkdir(parents=True)
    (assets / "tri.obj").write_text(OBJ, encoding="utf-8")
    (assets / "broken.png").write_bytes(b"not a png")
    return tmp_path / "proj"


@pytest.fixture
def context(qapp, project: Path):
    ctx = EditorContext(project_root=project, language="en")
    yield ctx
    ctx.stop_imports()


def _settle(context: EditorContext) -> None:
    assert context.imports is not None
    assert context.imports.wait_idle(20)


def test_imports_are_off_until_started(context: EditorContext, project: Path) -> None:
    assert context.imports is None
    assert not (project / "Library").exists()
    assert not (project / "assets" / "tri.obj.meta").exists()


def test_start_imports_builds_library(context: EditorContext, project: Path) -> None:
    events: list[int] = []
    context.imports_changed.connect(lambda: events.append(1))
    manager = context.start_imports()
    _settle(context)
    guid = manager.db.guid_of(project / "assets" / "tri.obj")
    assert guid is not None
    assert manager.state(guid) is ImportState.OK
    assert (project / "assets" / "tri.obj.meta").is_file()
    assert (project / "Library").is_dir()
    assert events


def test_failed_asset_is_reported(context: EditorContext, project: Path) -> None:
    manager = context.start_imports()
    _settle(context)
    guid = manager.db.guid_of(project / "assets" / "broken.png")
    assert guid is not None
    assert manager.state(guid) is ImportState.FAILED
    assert guid in manager.failures()


def test_stop_imports_is_idempotent(context: EditorContext) -> None:
    context.start_imports()
    context.stop_imports()
    context.stop_imports()
    assert context.imports is None


def test_project_switch_restarts_manager(
    context: EditorContext, project: Path, tmp_path: Path
) -> None:
    context.start_imports()
    other = tmp_path / "other"
    (other / "Assets").mkdir(parents=True)
    (other / "Assets" / "a.obj").write_text(OBJ, encoding="utf-8")
    restarted: list[int] = []
    context.imports_restarted.connect(lambda: restarted.append(1))
    assert context.set_project_root(other)
    _settle(context)
    assert restarted
    assert context.imports is not None
    assert context.imports.db.root == other / "Assets"
    assert context.imports.db.guid_of(other / "Assets" / "a.obj") is not None


def test_import_panel_lists_assets_and_filters(
    qapp, context: EditorContext, project: Path
) -> None:
    panel = ImportPanel(context)
    context.start_imports()
    _settle(context)
    panel.refresh()
    paths = {panel.table.item(r, 0).text() for r in range(panel.table.rowCount())}
    assert paths == {"tri.obj", "broken.png"}

    panel.filter_combo.setCurrentIndex(1)  # только ошибки
    assert panel.table.rowCount() == 1
    assert panel.table.item(0, 0).text() == "broken.png"
    panel.table.selectRow(0)
    assert panel.error_view.toPlainText()


def test_import_panel_reimport_buttons(
    qapp, context: EditorContext, project: Path
) -> None:
    panel = ImportPanel(context)
    manager = context.start_imports()
    _settle(context)
    panel.refresh()
    panel.reimport_all()
    assert manager.busy or manager.progress()[1] >= 0
    _settle(context)
    panel.table.selectRow(0)
    panel.reimport_selected()
    _settle(context)


def test_import_panel_without_manager(qapp, context: EditorContext) -> None:
    panel = ImportPanel(context)
    panel.refresh()
    assert panel.table.rowCount() == 0
    panel.reimport_all()  # не падает


def test_inspector_shows_import_settings(
    qapp, context: EditorContext, project: Path
) -> None:
    inspector = InspectorPanel(context)
    manager = context.start_imports()
    _settle(context)
    path = project / "assets" / "tri.obj"
    inspector.show_asset(path)
    section = inspector.asset_section
    assert section is not None and section.guid is not None
    assert {spec.name for spec in section.specs} >= {"scale", "flip_uv"}

    section._fields["scale"].setValue(2.0)
    assert section.apply()
    _settle(context)
    meta = json.loads(path.with_name("tri.obj.meta").read_text(encoding="utf-8"))
    assert meta["settings"] == {"scale": 2.0}
    assert manager.state(section.guid) is ImportState.OK

    section.reset()
    assert section.values()["scale"] == 1.0

    inspector.show_asset(None)
    assert inspector.asset_section is None


def test_inspector_rejects_invalid_settings(
    qapp, context: EditorContext, project: Path
) -> None:
    inspector = InspectorPanel(context)
    context.start_imports()
    _settle(context)
    inspector.show_asset(project / "assets" / "tri.obj")
    section = inspector.asset_section
    assert section is not None
    section._fields["lod_ratios"].setText("not json")
    assert section.apply() is False
    assert "Invalid" in section.info.text()


def test_selecting_object_hides_asset_form(
    qapp, context: EditorContext, project: Path
) -> None:
    inspector = InspectorPanel(context)
    context.start_imports()
    _settle(context)
    inspector.show_asset(project / "assets" / "tri.obj")
    assert inspector.asset_section is not None
    cube = context.create_object("Cube", mesh="cube")
    context.selection.select(cube)
    assert inspector.asset_section is None


def test_inspector_asset_without_manager(
    qapp, context: EditorContext, project: Path
) -> None:
    inspector = InspectorPanel(context)
    inspector.show_asset(project / "assets" / "tri.obj")
    assert inspector.asset_section is not None
    assert inspector.asset_section.guid is None


def test_assets_panel_copies_dropped_files(
    qapp, context: EditorContext, project: Path, tmp_path: Path
) -> None:
    panel = AssetsPanel(context)
    manager = context.start_imports()
    _settle(context)
    outside = tmp_path / "outside"
    outside.mkdir()
    source = outside / "new.obj"
    source.write_text(OBJ, encoding="utf-8")

    first = panel.import_files([source])
    second = panel.import_files([source])
    assert first == [project / "assets" / "new.obj"]
    assert second == [project / "assets" / "new (1).obj"]
    _settle(context)
    assert manager.db.guid_of(first[0]) is not None
    assert manager.db.guid_of(second[0]) is not None
    assert panel.import_files([outside]) == []  # каталоги пропускаются


def test_assets_panel_marks_failed_files(
    qapp, context: EditorContext, project: Path
) -> None:
    from PySide6.QtCore import Qt

    panel = AssetsPanel(context)
    context.start_imports()
    _settle(context)
    index = panel.fs_model.index(str(project / "assets" / "broken.png"))
    tip = panel.fs_model.data(index, Qt.ItemDataRole.ToolTipRole)
    assert tip
    brush = panel.fs_model.data(index, Qt.ItemDataRole.ForegroundRole)
    assert brush is not None


def test_meta_files_are_hidden(qapp, context: EditorContext, project: Path) -> None:
    panel = AssetsPanel(context)
    context.start_imports()
    _settle(context)
    index = panel.fs_model.index(str(project / "assets" / "tri.obj.meta"))
    assert index.isValid()
    assert panel.proxy.mapFromSource(index).isValid() is False
