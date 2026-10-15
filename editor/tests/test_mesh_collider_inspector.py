"""Mesh-коллайдер в редакторе: пункт меню, поле ссылки и источник геометрии."""

from __future__ import annotations

from editor.panels.script_fields import ReferenceField
from engine.core.physics_components import MeshColliderComponent


def _menu_action(window, label_key: str):
    """Пункт меню «Добавить компонент» по ключу перевода."""
    text = window.context.tr(label_key)
    for submenu_action in window.inspector.add_component_menu.actions():
        menu = submenu_action.menu()
        if menu is None:
            continue
        for action in menu.actions():
            if action.text() == text:
                return action
    raise AssertionError(f"нет пункта меню {label_key}")


class TestMeshColliderMenu:
    def test_menu_adds_component_with_reference_field(self, window) -> None:
        cube = next(o for o in window.context.scene.objects if o.name == "Cube")
        window.context.selection.select(cube)
        action = _menu_action(window, "inspector.physics.add.mesh_collider")
        assert action.isEnabled()
        action.trigger()
        assert cube.get_component(MeshColliderComponent) is not None
        fields = window.inspector.findChildren(ReferenceField)
        assert fields, "для mesh нет ReferenceField"

    def test_added_component_is_blocked_in_menu(self, window) -> None:
        cube = next(o for o in window.context.scene.objects if o.name == "Cube")
        window.context.selection.select(cube)
        _menu_action(window, "inspector.physics.add.mesh_collider").trigger()
        window.context.selection.select(cube)
        action = _menu_action(window, "inspector.physics.add.mesh_collider")
        assert not action.isEnabled()


class TestAssetSource:
    def test_start_imports_sets_asset_source_on_initial_scene(
        self, context
    ) -> None:
        scene = context.scene
        assert scene.asset_source is None
        context.start_imports()
        try:
            assert scene.asset_source is context.asset_db
        finally:
            context.stop_imports()


class TestInheritButton:
    def test_button_copies_mesh_from_renderer(self, window) -> None:
        from editor.panels.auto_section import AutoSection
        from engine.core.components import MeshRenderer
        from engine.core.resources.asset_ref import AssetRef

        cube = next(o for o in window.context.scene.objects if o.name == "Cube")
        collider = cube.add_component(MeshColliderComponent)
        assert collider.mesh is None
        cube.get_component(MeshRenderer).mesh_ref = AssetRef("b" * 32, "m.glb")
        window.context.selection.select(cube)
        sections = [
            s for s in window.inspector.findChildren(AutoSection)
            if s.inherit_button is not None
        ]
        assert sections
        sections[0].inherit_button.click()
        assert collider.mesh == AssetRef("b" * 32, "m.glb")
