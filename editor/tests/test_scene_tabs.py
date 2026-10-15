"""Вкладки сцен: открытие, переключение, сохранение и закрытие."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from editor import scene_io
from engine.core.scene import Scene


def _answer(monkeypatch, button) -> None:
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: button)
    )


def _scene_file(tmp_path: Path, name: str, objects: int) -> Path:
    scene = Scene(name)
    for index in range(objects):
        scene.create_game_object(f"obj{index}")
    return scene_io.save_scene(scene, tmp_path / f"{name}.scene.json")


class TestDocuments:
    def test_startup_has_single_clean_tab(self, window) -> None:
        assert len(window.context.documents) == 1
        assert window.context.active_document.dirty is False
        assert window.scene_tabs.bar.count() == 1

    def test_opening_file_adds_tab_and_switches(self, window, tmp_path) -> None:
        first = window.context.scene
        path = _scene_file(tmp_path, "level1", 2)
        assert window.load_scene_file(path)
        assert len(window.context.documents) == 2
        assert window.context.scene is not first
        assert window.scene_tabs.tab_titles()[1] == "level1"
        assert window.scene_tabs.bar.currentIndex() == 1

    def test_same_file_switches_instead_of_duplicating(self, window, tmp_path) -> None:
        path = _scene_file(tmp_path, "level1", 1)
        window.load_scene_file(path)
        window.context.activate_document(0)
        window.load_scene_file(path)
        assert len(window.context.documents) == 2
        assert window.context.active_index == 1

    def test_tab_click_switches_scene_and_hierarchy(self, window, tmp_path) -> None:
        path = _scene_file(tmp_path, "level1", 5)
        window.load_scene_file(path)
        window.scene_tabs.bar.setCurrentIndex(0)
        assert window.context.active_index == 0
        assert len(window.context.scene) == 3
        window.scene_tabs.bar.setCurrentIndex(1)
        assert len(window.context.scene) == 5

    def test_edit_marks_tab_dirty(self, window) -> None:
        window.create_primitive("cube")
        assert window.context.active_document.dirty is True
        assert window.scene_tabs.tab_titles()[0].endswith("*")

    def test_switching_does_not_mark_dirty(self, window, tmp_path) -> None:
        path = _scene_file(tmp_path, "level1", 1)
        window.load_scene_file(path)
        window.context.activate_document(0)
        assert all(not d.dirty for d in window.context.documents)

    def test_new_scene_opens_new_tab(self, window) -> None:
        window.new_scene()
        assert len(window.context.documents) == 2
        assert window.context.active_document.dirty is False

    def test_selection_is_cleared_on_switch(self, window, tmp_path) -> None:
        path = _scene_file(tmp_path, "level1", 1)
        window.load_scene_file(path)
        assert window.context.selection.is_empty

    def test_camera_view_is_remembered_per_tab(self, window, tmp_path) -> None:
        controller = window.viewport.viewport.controller
        path = _scene_file(tmp_path, "level1", 1)
        controller.orbit(90.0, 0.0)
        before = controller.capture(window.viewport.camera)
        window.load_scene_file(path)
        controller.orbit(-200.0, 15.0)
        window.context.activate_document(0)
        after = controller.capture(window.viewport.camera)
        assert after.yaw == before.yaw


class TestSavingAndClosing:
    def test_save_known_path_does_not_ask(self, window, tmp_path, monkeypatch) -> None:
        path = _scene_file(tmp_path, "level1", 1)
        window.load_scene_file(path)
        window.create_primitive("cube")
        from PySide6.QtWidgets import QFileDialog

        def boom(*_a, **_k):
            raise AssertionError("диалог не должен открываться")

        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(boom))
        assert window.save_scene() is True
        assert window.context.active_document.dirty is False
        assert len(scene_io.load_scene(path)) == 2

    def test_close_clean_tab_does_not_ask(self, window, tmp_path, monkeypatch) -> None:
        path = _scene_file(tmp_path, "level1", 1)
        window.load_scene_file(path)

        def boom(*_a, **_k):
            raise AssertionError("вопрос не нужен")

        monkeypatch.setattr(QMessageBox, "question", staticmethod(boom))
        assert window.close_scene_tab(1) is True
        assert len(window.context.documents) == 1

    def test_close_dirty_tab_cancel_keeps_it(self, window, monkeypatch) -> None:
        window.create_primitive("cube")
        _answer(monkeypatch, QMessageBox.StandardButton.Cancel)
        assert window.close_scene_tab(0) is False
        assert len(window.context.documents) == 1

    def test_close_dirty_tab_discard(self, window, tmp_path, monkeypatch) -> None:
        path = _scene_file(tmp_path, "level1", 1)
        window.load_scene_file(path)
        window.create_primitive("cube")
        _answer(monkeypatch, QMessageBox.StandardButton.Discard)
        assert window.close_scene_tab(1) is True
        assert len(window.context.documents) == 1
        assert len(scene_io.load_scene(path)) == 1

    def test_close_dirty_tab_save(self, window, tmp_path, monkeypatch) -> None:
        path = _scene_file(tmp_path, "level1", 1)
        window.load_scene_file(path)
        window.create_primitive("cube")
        _answer(monkeypatch, QMessageBox.StandardButton.Save)
        assert window.close_scene_tab(1) is True
        assert len(scene_io.load_scene(path)) == 2

    def test_closing_last_tab_leaves_empty_scene(self, window) -> None:
        window.context.mark_clean()
        window.close_scene_tab(0)
        assert len(window.context.documents) == 1
        assert len(window.context.scene) == 0

    def test_close_inactive_tab_keeps_active_scene(self, window, tmp_path) -> None:
        path = _scene_file(tmp_path, "level1", 4)
        window.load_scene_file(path)
        active = window.context.scene
        window.close_scene_tab(0)
        assert window.context.scene is active
        assert window.context.active_index == 0

    def test_assets_double_click_opens_scene(self, window, tmp_path) -> None:
        path = _scene_file(tmp_path, "lvl", 1)
        window.assets.scene_activated.emit(str(path))
        assert window.context.scene_path == path
