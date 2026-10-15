"""Тесты главного окна редактора: панели, команды, локализация, инспектор."""

from __future__ import annotations

import logging

import pytest

from editor.context import EditorContext
from editor.panels.assets import MODEL_EXTENSIONS, TEXTURE_EXTENSIONS
from engine.core.components import BUILTIN_MESHES, MeshRenderer
from engine.core.math3d import Vector3
from engine.core.resources import AssetRef


class TestWindowStructure:
    def test_docks_are_present(self, window) -> None:
        panels = [
            window.hierarchy,
            window.inspector,
            window.assets,
            window.console,
            window.stats_panel,
        ]
        assert all(panel is not None for panel in panels)
        # Все панели — доки: их можно перетаскивать, отстыковывать и закрывать.
        from PySide6.QtWidgets import QDockWidget

        for panel in panels:
            assert panel.features() & QDockWidget.DockWidgetFloatable
            assert panel.features() & QDockWidget.DockWidgetMovable
            assert panel.features() & QDockWidget.DockWidgetClosable

    def test_central_widget_is_viewport(self, window) -> None:
        central = window.centralWidget()
        assert window.viewport.parentWidget() is central
        assert window.scene_tabs.parentWidget() is central
        assert hasattr(window.viewport, "viewport")

    def test_menus_and_toolbar(self, window) -> None:
        titles = [
            menu.title()
            for menu in window.menuBar().findChildren(type(window.menu_file))
        ]
        assert any("Файл" in title for title in titles)
        assert window.toolbar.actions(), "панель инструментов пуста"

    def test_scene_starts_with_objects(self, window) -> None:
        assert len(window.context.scene) == 3  # Ground, Cube, Directional Light
        assert window.hierarchy.model.rowCount() == 3
        assert window.context.selection.primary is not None

    def test_statusbar_has_hint_and_language(self, window) -> None:
        assert "ЛКМ" in window.hint_label.text()
        assert window.language_label.text() == "Русский"


class TestCommands:
    def test_create_primitive(self, window) -> None:
        before = len(window.context.scene)
        cube = window.create_primitive("cube")
        assert len(window.context.scene) == before + 1
        assert cube.get_component(MeshRenderer).mesh == "cube"
        assert window.context.selection.primary is cube
        assert window.hierarchy.model.rowCount() == before + 1

    def test_create_empty_object(self, window) -> None:
        game_object = window.create_empty()
        assert game_object.get_component(MeshRenderer) is None

    def test_delete_selected(self, window) -> None:
        """Удаление применяется сразу: редактор не крутит игровой цикл.

        ``GameObject.destroy()`` в ядре отложенный, поэтому редактор обязан
        применять очередь удалений сам — иначе объект останется и в дереве, и
        во вьюпорте.
        """
        cube = window.context.selection.primary
        assert cube is not None
        rows_before = window.hierarchy.model.rowCount()
        window.delete_selected()
        assert cube not in window.context.scene.objects
        assert cube.is_destroyed
        assert window.context.selection.is_empty
        assert window.hierarchy.model.rowCount() == rows_before - 1

    def test_delete_child_updates_parent_and_tree(self, window) -> None:
        parent = window.context.selection.primary
        assert parent is not None
        child = window.context.create_object(
            "Child", mesh="cube", parent=parent, position=Vector3(1, 1, 1)
        )
        window.hierarchy.refresh()
        assert child in parent.children

        window.context.selection.select(child)
        window.delete_selected()

        assert child not in parent.children
        assert child not in window.context.scene.objects
        assert window.hierarchy.model.object_for(
            window.hierarchy.proxy.mapToSource(window.hierarchy.tree.currentIndex())
        ) is not child

    def test_hierarchy_panel_delete_keeps_scene_consistent(self, window) -> None:
        cube = window.context.selection.primary
        assert cube is not None
        window.hierarchy._delete_selected()  # команда контекстного меню дерева
        assert cube not in window.context.scene.objects
        assert window.hierarchy.model.rowCount() == len(
            window.context.scene.root_objects
        )

    def test_duplicate_selected(self, window) -> None:
        cube = window.context.selection.primary
        component = cube.get_component(MeshRenderer)
        component.mesh_ref = AssetRef(path="sphere")
        component.color = Vector3(0, 1, 0)
        window.duplicate_selected()
        clones = [obj for obj in window.context.scene.objects if "copy" in obj.name]
        assert len(clones) == 1
        clone_component = clones[0].get_component(MeshRenderer)
        assert clone_component is not None
        assert clone_component.mesh == "sphere"
        assert clone_component.color.almost_equals(Vector3(0, 1, 0), 1e-6)
        assert clones[0].transform.position.almost_equals(
            cube.transform.position + Vector3(1, 0, 0), 1e-6
        )

    def test_add_model_from_asset(self, window) -> None:
        game_object = window.add_model_from_asset("assets/primitives/cube.obj")
        component = game_object.get_component(MeshRenderer)
        assert component is not None
        assert component.mesh.endswith("cube.obj")

    def test_delete_without_selection_is_safe(self, window) -> None:
        window.context.selection.clear()
        window.delete_selected()  # не должно бросить исключение
        assert len(window.context.scene) == 3


class TestViewportMeshes:
    def test_every_builtin_mesh_has_a_factory(self) -> None:
        """Инспектор предлагает BUILTIN_MESHES — вьюпорт обязан их уметь.

        Регрессия: ``cylinder`` был объявлен встроенным, но геометрии не имел,
        поэтому каждый кадр писал в лог «Не удалось загрузить меш 'cylinder'».
        """
        from editor.viewport.editor_viewport import BUILTIN_MESH_FACTORIES

        assert set(BUILTIN_MESHES) <= set(BUILTIN_MESH_FACTORIES)

    def test_builtin_factories_build_geometry(self) -> None:
        from editor.viewport.editor_viewport import BUILTIN_MESH_FACTORIES

        for name, factory in BUILTIN_MESH_FACTORIES.items():
            asset = factory()
            assert asset.vertex_count > 0, name
            assert asset.triangle_count > 0, name


class TestLocalization:
    def test_switch_to_english_updates_ui(self, window) -> None:
        window.set_language("en")
        assert window.context.translator.language == "en"
        assert window.menu_file.title() == "&File"
        assert window.hierarchy.windowTitle() == "Hierarchy"
        assert window.inspector.windowTitle() == "Inspector"
        assert window.stats_panel.windowTitle() == "Statistics"
        assert window.console.windowTitle() == "Console"
        assert window.action_create_cube.text() == "Cube"
        assert "LMB" in window.hint_label.text()
        assert window.language_label.text() == "English"

    def test_switch_back_to_russian(self, window) -> None:
        window.set_language("en")
        window.set_language("ru")
        assert window.menu_file.title() == "&Файл"
        assert window.hierarchy.windowTitle() == "Иерархия"
        assert window.hierarchy.search.placeholderText() == "Поиск объектов…"

    def test_panel_titles_follow_language_signal(self, window) -> None:
        window.context.set_language("en")
        assert window.assets.windowTitle() == "Assets"
        assert window.assets.filter_combo.itemText(0) == "All files"
        window.context.set_language("ru")
        assert window.assets.filter_combo.itemText(0) == "Все файлы"

    def test_language_actions_are_exclusive(self, window) -> None:
        window.set_language("en")
        assert window.language_actions["en"].isChecked()
        assert not window.language_actions["ru"].isChecked()
        window.set_language("ru")
        assert window.language_actions["ru"].isChecked()


class TestInspector:
    def test_shows_transform_of_selected(self, window) -> None:
        cube = window.context.scene.find("Cube")
        window.context.selection.select(cube)
        window.inspector.refresh()
        assert window.inspector.game_object is cube
        assert window.inspector.position_editor.value().almost_equals(
            cube.transform.position, 1e-6
        )

    def test_edits_position(self, window) -> None:
        cube = window.context.selection.primary
        window.inspector.position_editor.set_value(Vector3(5, 6, 7), emit=True)
        assert cube.transform.position.almost_equals(Vector3(5, 6, 7), 1e-6)

    def test_edits_rotation_in_degrees(self, window) -> None:
        cube = window.context.selection.primary
        window.inspector.rotation_editor.set_value(Vector3(0, 90, 0), emit=True)
        forward = cube.transform.forward
        assert forward.almost_equals(Vector3(1, 0, 0), 1e-4)

    def test_edits_scale_and_name(self, window) -> None:
        cube = window.context.selection.primary
        window.inspector.scale_editor.set_value(Vector3(2, 2, 2), emit=True)
        assert cube.transform.scale.almost_equals(Vector3(2, 2, 2), 1e-6)
        window.inspector._apply_name("Renamed")
        assert cube.name == "Renamed"

    def test_edits_material(self, window) -> None:
        cube = window.context.selection.primary
        window.inspector._apply_color(Vector3(1.0, 0.0, 0.0))
        window.inspector._apply_mesh("sphere")
        component = cube.get_component(MeshRenderer)
        assert component.color.almost_equals(Vector3(1, 0, 0), 1e-6)
        assert component.mesh == "sphere"

    def test_placeholder_without_selection(self, window) -> None:
        window.context.selection.clear()
        window.inspector.refresh()
        assert window.inspector.game_object is None

    def test_add_mesh_renderer_to_empty_object(self, window) -> None:
        empty = window.create_empty()
        window.context.selection.select(empty)
        window.inspector.refresh()
        window.inspector._add_mesh_renderer()
        assert empty.get_component(MeshRenderer) is not None

    def test_toggle_active(self, window) -> None:
        cube = window.context.selection.primary
        window.inspector._apply_active(False)
        assert cube.active_self is False
        window.inspector._apply_active(True)
        assert cube.active_self is True


class TestAssetsAndConsole:
    def test_asset_filter_categories(self, window) -> None:
        panel = window.assets
        assert panel.filter_combo.count() == 5
        panel.filter_combo.setCurrentIndex(1)  # модели
        assert panel.proxy._extensions == MODEL_EXTENSIONS  # noqa: SLF001
        panel.filter_combo.setCurrentIndex(2)  # текстуры
        assert panel.proxy._extensions == TEXTURE_EXTENSIONS  # noqa: SLF001
        panel.filter_combo.setCurrentIndex(0)  # все файлы
        assert not panel.proxy._extensions  # noqa: SLF001

    def test_asset_root_points_to_assets(self, window) -> None:
        assert window.assets.root.name == "Assets"
        assert window.assets.root.is_dir()

    def test_console_receives_engine_logs(self, window) -> None:
        logging.getLogger("engine.test").warning("тестовое сообщение")
        assert any(
            "тестовое сообщение" in text for text, _level in window.console.history
        )

    def test_console_clear(self, window) -> None:
        logging.getLogger("engine.test").error("ошибка")
        window.console.clear()
        assert window.console.history == ()

    def test_stats_panel_updates(self, window) -> None:
        window._refresh_stats()  # noqa: SLF001 - проверяем связку с панелью
        assert window.stats_panel.scene_labels["stats.objects"].text() == "3"


class TestContext:
    def test_selection_additive_and_clear(self, context) -> None:
        first = context.create_object("A", mesh="cube")
        second = context.create_object("B", mesh="sphere")
        context.selection.select(first)
        context.selection.select(second, additive=True)
        assert len(context.selection) == 2
        assert context.selection.primary is first
        context.selection.select(second, additive=True)  # снятие с выделения
        assert context.selection.objects == (first,)
        context.selection.clear()
        assert context.selection.is_empty

    def test_forget_removed_object(self, context) -> None:
        game_object = context.create_object("A", mesh="cube")
        context.selection.select(game_object)
        context.delete_objects([game_object])
        assert game_object not in context.selection

    def test_scene_replacement_resets_selection(self, context) -> None:
        game_object = context.create_object("A", mesh="cube")
        context.selection.select(game_object)
        context.set_scene(type(context.scene)("New"))
        assert context.selection.is_empty
        assert len(context.scene) == 0

    def test_translate_and_repr(self, context: EditorContext) -> None:
        context.create_object("A", mesh="cube")
        assert "EditorContext" in repr(context)

    def test_duplicate_requires_same_scene(self, context) -> None:
        other = type(context.scene)("Other")
        foreign = other.create_game_object("Foreign")
        assert context.duplicate_object(foreign) is None


def test_editor_is_a_separate_application(qapp) -> None:
    """Редактор импортирует движок, но не наоборот."""
    import editor
    import engine

    assert editor.__version__
    assert engine.__version__
    assert pytest.importorskip("PySide6") is not None


class TestFitToScreen:
    def test_oversized_window_is_fitted_to_available_area(self, window) -> None:
        window.resize(9000, 9000)
        window.move(-500, -500)
        window.fit_to_available_screen()
        area = window.screen().availableGeometry()
        assert area.contains(window.frameGeometry())

    def test_window_below_screen_is_moved_inside(self, window) -> None:
        area = window.screen().availableGeometry()
        window.resize(800, 600)
        window.move(area.left(), area.bottom() - 100)
        window.fit_to_available_screen()
        assert area.contains(window.frameGeometry())

    def test_stats_panel_is_scrollable(self, window) -> None:
        scroll = window.stats_panel.scroll_area
        assert scroll.widgetResizable()
        assert scroll.widget() is not None
