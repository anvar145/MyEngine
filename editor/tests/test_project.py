"""Тесты системы проектов: структура каталогов и переключение проекта."""

from __future__ import annotations

from pathlib import Path

import pytest

from editor.context import EditorContext
from editor.project import (
    PROJECT_FOLDERS,
    PROJECTS_DIRNAME,
    create_project,
    ensure_projects_dir,
    list_projects,
    projects_root,
    validate_project_name,
)


@pytest.fixture()
def project_window(qapp, tmp_path: Path):
    """Окно редактора с проектом во временном каталоге.

    Yields:
        :class:`~editor.main_window.MainWindow`.
    """
    from editor.main_window import MainWindow

    context = EditorContext(project_root=tmp_path, language="ru")
    widget = MainWindow(context)
    yield widget
    widget.close()
    widget.deleteLater()


class TestProjectFolders:
    def test_projects_directory_is_created_with_context(
        self, qapp, tmp_path: Path
    ) -> None:
        """Папка ``projects/`` появляется вместе с контекстом редактора."""
        context = EditorContext(project_root=tmp_path, language="ru")
        assert projects_root(context.project_root).is_dir()
        assert projects_root(tmp_path).name == PROJECTS_DIRNAME

    def test_repository_has_projects_directory(self, window) -> None:
        assert projects_root(window.context.project_root).is_dir()

    def test_ensure_projects_dir_is_idempotent(self, tmp_path: Path) -> None:
        first = ensure_projects_dir(tmp_path)
        assert first.is_dir()
        assert ensure_projects_dir(tmp_path) == first

    def test_standard_folders_cover_user_request(self) -> None:
        """В проекте есть скрипты, модели, спрайты и звуки."""
        assert {"Scripts", "Models", "Sprites", "Audio"} <= set(PROJECT_FOLDERS)


class TestProjectCreation:
    def test_create_project_builds_every_folder(self, tmp_path: Path) -> None:
        project = create_project(tmp_path, "Demo")
        assert project == tmp_path / PROJECTS_DIRNAME / "Demo"
        for folder in PROJECT_FOLDERS:
            assert (project / "Assets" / folder).is_dir(), f"нет папки {folder}"
        assert (project / "Library").is_dir()
        assert (project / "Settings").is_dir()

    def test_projects_are_isolated(self, tmp_path: Path) -> None:
        first = create_project(tmp_path, "First")
        second = create_project(tmp_path, "Second")
        assert first != second
        assert first.parent == second.parent == projects_root(tmp_path)

    def test_existing_project_needs_explicit_flag(self, tmp_path: Path) -> None:
        create_project(tmp_path, "Demo")
        with pytest.raises(FileExistsError):
            create_project(tmp_path, "Demo")
        assert create_project(tmp_path, "Demo", exists_ok=True).is_dir()

    def test_name_is_trimmed(self, tmp_path: Path) -> None:
        assert validate_project_name("  My Game  ") == "My Game"
        assert create_project(tmp_path, "  Space  ").name == "Space"

    @pytest.mark.parametrize(
        "name",
        ["", "   ", ".", "..", "...", "a/b", "a\\b", "bad:name", "x?y", "star*", "dot."],
    )
    def test_invalid_names_are_rejected(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(ValueError):
            create_project(tmp_path, name)

    def test_list_projects_is_sorted_and_empty_initially(self, tmp_path: Path) -> None:
        assert list_projects(tmp_path) == ()
        create_project(tmp_path, "Beta")
        create_project(tmp_path, "Alpha")
        assert [path.name for path in list_projects(tmp_path)] == ["Alpha", "Beta"]

    def test_list_projects_skips_files(self, tmp_path: Path) -> None:
        create_project(tmp_path, "Real")
        (projects_root(tmp_path) / "note.txt").write_text("x", encoding="utf-8")
        assert [path.name for path in list_projects(tmp_path)] == ["Real"]


class TestProjectSwitching:
    def test_set_project_root_emits_signal_once(self, qapp, tmp_path: Path) -> None:
        context = EditorContext(project_root=tmp_path, language="ru")
        seen: list[object] = []
        context.project_changed.connect(seen.append)
        other = tmp_path / "other"
        other.mkdir()

        assert context.set_project_root(other) is True
        assert seen == [other]
        # Повторная установка того же корня ничего не меняет.
        assert context.set_project_root(other) is False
        assert len(seen) == 1

    def test_set_project_root_resets_scene_path(self, qapp, tmp_path: Path) -> None:
        context = EditorContext(project_root=tmp_path, language="ru")
        context.scene_path = tmp_path / "scene.json"
        other = tmp_path / "other"
        other.mkdir()
        context.set_project_root(other)
        assert context.scene_path is None

    def test_switch_project_does_not_create_nested_projects_dir(
        self, project_window, tmp_path: Path
    ) -> None:
        """``projects/`` живёт рядом с рабочим корнем, а не внутри проекта."""
        project = create_project(tmp_path, "Game")
        project_window.switch_project(project)

        assert project_window.context.workspace_root == tmp_path
        assert not (project / PROJECTS_DIRNAME).exists()
        assert projects_root(tmp_path).is_dir()

    def test_switch_project_updates_panels(self, project_window, tmp_path: Path) -> None:
        project = create_project(tmp_path, "Game")
        project_window.switch_project(project)

        assert project_window.context.project_root == project
        assert project_window.assets.root == project / "Assets"
        assert project_window.script_editor.project_root() == project
        assert project_window.script_editor.scripts_root() == (
            project / "Assets" / "Scripts"
        )
        assert "Game" in project_window.windowTitle()

    def test_assets_panel_prefers_assets_subfolder(
        self, project_window, tmp_path: Path
    ) -> None:
        project = create_project(tmp_path, "WithAssets")
        project_window.switch_project(project)
        assert project_window.assets.root == project / "Assets"

    def test_project_actions_are_in_file_menu(self, project_window) -> None:
        actions = project_window.menu_file.actions()
        assert project_window.action_new_project in actions
        assert project_window.action_open_project in actions
        assert not hasattr(project_window, "menu_project")

    def test_new_project_creates_and_switches(
        self, project_window, tmp_path: Path, monkeypatch
    ) -> None:
        from PySide6.QtWidgets import QInputDialog

        monkeypatch.setattr(
            QInputDialog, "getText", staticmethod(lambda *a, **k: ("Demo", True))
        )
        created = project_window.new_project()

        assert created == tmp_path / PROJECTS_DIRNAME / "Demo"
        assert created is not None and (created / "Assets" / "Scripts").is_dir()
        assert project_window.context.project_root == created

    def test_new_project_cancelled_changes_nothing(
        self, project_window, monkeypatch
    ) -> None:
        from PySide6.QtWidgets import QInputDialog

        before = project_window.context.project_root
        monkeypatch.setattr(
            QInputDialog, "getText", staticmethod(lambda *a, **k: ("", False))
        )
        assert project_window.new_project() is None
        assert project_window.context.project_root == before

    def test_new_project_reports_bad_name(
        self, project_window, monkeypatch, caplog
    ) -> None:
        from PySide6.QtWidgets import QInputDialog

        monkeypatch.setattr(
            QInputDialog, "getText", staticmethod(lambda *a, **k: ("bad:name", True))
        )
        assert project_window.new_project() is None
        assert "недопустимые символы" in caplog.text
