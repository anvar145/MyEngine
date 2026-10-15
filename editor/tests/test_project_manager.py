"""Тесты окна менеджера проектов."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDialog

from editor.context import EditorContext
from editor.project import create_project
from editor.project_manager import (
    KEY_AUTO_OPEN,
    KEY_LAST_PROJECT,
    ProjectManagerDialog,
    choose_project,
    resolve_launch_project,
)


@pytest.fixture()
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    create_project(root, "Alpha")
    create_project(root, "Beta")
    return root


def _dialog(_app, workspace: Path, settings: QSettings, lang: str = "ru"):
    dialog = ProjectManagerDialog(workspace, lang, settings)
    return dialog


def test_lists_projects(qapp, workspace, settings) -> None:
    dialog = _dialog(qapp, workspace, settings)
    assert dialog.list.count() == 2
    assert dialog.selected_path() is not None
    assert dialog.open_button.isEnabled()


def test_empty_workspace_shows_hint(qapp, tmp_path, settings) -> None:
    dialog = _dialog(qapp, tmp_path, settings)
    assert dialog.selected_path() is None
    assert not dialog.open_button.isEnabled()


def test_filter_and_sort(qapp, workspace, settings) -> None:
    dialog = _dialog(qapp, workspace, settings)
    dialog.search.setText("bet")
    assert dialog.list.count() == 1
    assert dialog.selected_path().name == "Beta"
    dialog.search.setText("")
    dialog.sort_box.setCurrentIndex(2)  # по имени
    assert dialog.list.item(0).text().startswith("Alpha")


def test_open_selected_returns_choice(qapp, workspace, settings) -> None:
    dialog = _dialog(qapp, workspace, settings)
    dialog.search.setText("beta")
    dialog.open_selected()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.choice.project == workspace / "projects" / "Beta"
    assert settings.value(KEY_LAST_PROJECT) == "Beta"


def test_create_validates_name(qapp, workspace, settings) -> None:
    dialog = _dialog(qapp, workspace, settings)
    for bad in ("", "a/b", "alpha"):
        dialog.name_edit.setText(bad)
        assert not dialog.create_button.isEnabled()
    assert dialog.error_label.text()  # для «alpha» — причина отказа
    dialog.name_edit.setText("Gamma")
    assert dialog.create_button.isEnabled()


def test_create_new_project(qapp, workspace, settings) -> None:
    dialog = _dialog(qapp, workspace, settings)
    dialog.name_edit.setText("Gamma")
    dialog.create_new()
    project = workspace / "projects" / "Gamma"
    assert (project / "Assets" / "Scripts").is_dir()
    assert dialog.choice.project == project


def test_language_switch(qapp, workspace, settings) -> None:
    dialog = _dialog(qapp, workspace, settings, "ru")
    ru_title = dialog.windowTitle()
    dialog.language_box.setCurrentIndex(dialog.language_box.findData("en"))
    assert dialog.language == "en"
    assert dialog.windowTitle() != ru_title


def test_close_returns_none(qapp, workspace, settings, monkeypatch) -> None:
    monkeypatch.setattr(ProjectManagerDialog, "exec", lambda self: self.reject() or 0)
    assert choose_project(workspace, "ru", settings) is None


def test_auto_open_skips_window(workspace, settings) -> None:
    settings.setValue(KEY_AUTO_OPEN, True)
    settings.setValue(KEY_LAST_PROJECT, "Alpha")
    choice = choose_project(workspace, "ru", settings)
    assert choice is not None and choice.project.name == "Alpha"


def test_explicit_project_skips_launcher(workspace) -> None:
    args = argparse.Namespace(
        project=str(workspace), no_launcher=False, screenshot="", language="ru"
    )
    assert resolve_launch_project(args, workspace)
    assert args.project == str(workspace)
    args = argparse.Namespace(
        project=None, no_launcher=True, screenshot="", language="ru"
    )
    assert resolve_launch_project(args, workspace)
    assert args.project == str(workspace)


def test_project_has_no_nested_projects_dir(qapp, workspace) -> None:
    project = workspace / "projects" / "Alpha"
    EditorContext(project_root=project, workspace_root=workspace)
    assert not (project / "projects").exists()


def test_template_is_copied(qapp, workspace, settings) -> None:
    dialog = _dialog(qapp, workspace, settings)
    dialog.show_page(1)
    assert dialog.pages.currentIndex() == 1
    dialog.template_buttons["3d"].setChecked(True)
    assert dialog.selected_template() == "3d"
    dialog.name_edit.setText("Scene3D")
    dialog.create_new()
    project = workspace / "projects" / "Scene3D"
    assert (project / "Assets" / "Scenes" / "Main.scene.json").is_file()


def test_empty_state_shown_without_projects(qapp, tmp_path, settings) -> None:
    dialog = _dialog(qapp, tmp_path, settings)
    assert dialog.list_stack.currentWidget() is dialog.empty_state
    assert not dialog.empty_create_button.isHidden()
    dialog.empty_create_button.click()
    assert dialog.pages.currentIndex() == 1


def test_nothing_found_hides_create_button(qapp, workspace, settings) -> None:
    dialog = _dialog(qapp, workspace, settings)
    dialog.search.setText("zzz")
    assert dialog.list_stack.currentWidget() is dialog.empty_state
    assert dialog.empty_create_button.isHidden()


def test_open_folder_returns_choice(qapp, workspace, settings, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    folder = tmp_path / "external"
    folder.mkdir()
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: str(folder)
    )
    dialog = _dialog(qapp, workspace, settings)
    dialog.open_folder()
    assert dialog.choice.project == folder


def test_scan_counts_scenes_and_scripts_in_assets_layout(tmp_path) -> None:
    """Сводка проекта берёт сцены и скрипты из ``Assets/``, включая вложенные."""
    from editor.panels.script_picker import list_scripts
    from editor.project import scripts_root_for
    from editor.project_manager import _scan

    project = create_project(tmp_path, "Scan", template="3d")
    scripts = scripts_root_for(project)
    (scripts / "ai").mkdir()
    (scripts / "ai" / "patrol.py").write_text("", encoding="utf-8")

    info = _scan(project)
    assert info.scripts == len(list_scripts(scripts)) >= 1
    assert info.scenes >= 1
