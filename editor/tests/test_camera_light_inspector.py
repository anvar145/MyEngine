"""Камера и свет как компоненты объекта: создание, инспектор, сохранение."""

from __future__ import annotations

import pytest

from editor import scene_io
from editor.panels.camera_section import CameraSection
from editor.panels.light_section import LightSection
from engine.rendering.camera import CameraComponent
from engine.rendering.lighting import (
    DirectionalLight,
    LightComponent,
    PointLight,
    SpotLight,
)


class TestCreation:
    def test_camera_is_created_from_menu_action(self, window) -> None:
        count = len(window.context.scene)
        window.create_camera()
        assert len(window.context.scene) == count + 1
        selected = window.context.selection.primary
        assert selected.get_component(CameraComponent) is not None

    def test_old_panels_are_gone(self, window) -> None:
        assert not hasattr(window, "camera_panel")
        assert not hasattr(window, "light_panel")

    @pytest.mark.parametrize("kind", ["directional", "point", "spot", "ambient"])
    def test_light_is_created_for_each_kind(self, window, kind) -> None:
        game_object = window.create_light(kind)
        assert game_object is not None
        assert game_object.get_component(LightComponent).light.type_name == kind

    @pytest.mark.parametrize("language", ["ru", "en"])
    def test_created_object_names_do_not_depend_on_language(
        self, window, language
    ) -> None:
        window.context.set_language(language)
        camera = window.create_camera()
        spot = window.create_light("spot")
        assert camera.name.startswith("Camera")
        assert spot.name == "Spot Light"

    def test_repeated_names_get_numeric_suffix(self, window) -> None:
        first = window.create_camera()
        second = window.create_camera()
        third = window.create_camera()
        assert first.name != second.name != third.name
        assert second.name == f"{first.name} 2"
        assert third.name == f"{first.name} 3"

    def test_unknown_light_kind_is_rejected(self, window) -> None:
        assert window.create_light("laser") is None

    def test_new_sun_points_down(self, window) -> None:
        sun = window.create_light("directional")
        component = sun.get_component(LightComponent)
        component.sync_to_light()
        assert component.light.world_direction().y < -0.5

    def test_new_spot_points_down(self, window) -> None:
        spot = window.create_light("spot")
        component = spot.get_component(LightComponent)
        component.sync_to_light()
        assert component.light.world_direction().y < -0.5

    def test_new_scene_has_a_sun(self, window) -> None:
        lights = [
            obj
            for obj in window.context.scene.objects
            if obj.get_component(LightComponent) is not None
        ]
        assert len(lights) == 1
        assert isinstance(lights[0].get_component(LightComponent).light, DirectionalLight)

    def test_camera_does_not_follow_editor_camera(self, window) -> None:
        camera_object = window.create_camera()
        component = camera_object.get_component(CameraComponent)
        before = component.camera.position
        window.viewport.viewport.controller.orbit(120.0, 20.0)
        window.viewport.viewport.controller.zoom(5.0)
        window.viewport.viewport.controller.apply(window.viewport.viewport.camera)
        component.sync_to_camera()
        assert component.camera.position.almost_equals(before, 1e-6)

    def test_editor_camera_is_not_listed_among_scene_cameras(self, window) -> None:
        window.create_camera()
        cameras = window.viewport.camera_components()
        assert all(c.camera is not window.viewport.camera for c in cameras)


class TestInspectorSections:
    def test_camera_object_shows_camera_section(self, window) -> None:
        window.create_camera()
        assert hasattr(window.inspector, "camera_section")
        assert isinstance(window.inspector.camera_section, CameraSection)

    def test_light_object_shows_light_section(self, window) -> None:
        window.create_light("point")
        assert isinstance(window.inspector.light_section, LightSection)

    def test_plain_cube_has_no_camera_or_light_sections(self, window) -> None:
        cube = next(o for o in window.context.scene.objects if o.name == "Cube")
        window.context.selection.select(cube)
        # Старые секции прячутся и удаляются отложенно: учитываем только живые.
        for kind in (CameraSection, LightSection):
            shown = [w for w in window.inspector.findChildren(kind) if not w.isHidden()]
            assert shown == []

    def test_editing_fov_updates_camera(self, window) -> None:
        camera_object = window.create_camera()
        section = window.inspector.camera_section
        section._fov.setValue(75.0)  # noqa: SLF001 - поле формы
        assert camera_object.get_component(CameraComponent).camera.fov == 75.0

    def test_editing_intensity_updates_light(self, window) -> None:
        light_object = window.create_light("point")
        section = window.inspector.light_section
        section._apply_intensity(2.5)  # noqa: SLF001
        assert light_object.get_component(LightComponent).light.intensity == 2.5

    def test_section_edit_keeps_focus_in_field(self, window) -> None:
        """Правка поля не перестраивает инспектор (иначе пропадает фокус)."""
        window.create_camera()
        section = window.inspector.camera_section
        section._apply_near(0.3)  # noqa: SLF001
        assert window.inspector.camera_section is section

    def test_spot_cone_keeps_outer_larger_than_inner(self, window) -> None:
        light_object = window.create_light("spot")
        section = window.inspector.light_section
        section._apply_inner_angle(80.0)  # noqa: SLF001
        light = light_object.get_component(LightComponent).light
        assert isinstance(light, SpotLight)
        assert light.outer_angle > light.inner_angle

    def test_add_component_menu_offers_missing_components(self, window) -> None:
        cube = next(o for o in window.context.scene.objects if o.name == "Cube")
        window.context.selection.select(cube)
        window.inspector._add_camera_component()  # noqa: SLF001
        assert cube.get_component(CameraComponent) is not None
        window.inspector._add_light_component("point")  # noqa: SLF001
        assert isinstance(cube.get_component(LightComponent).light, PointLight)


class TestPersistence:
    def test_camera_and_light_survive_scene_roundtrip(self, window, tmp_path) -> None:
        camera_object = window.create_camera()
        camera_object.get_component(CameraComponent).camera.fov = 42.0
        light_object = window.create_light("spot")
        light_object.get_component(LightComponent).light.outer_angle = 33.0

        path = scene_io.save_scene(window.context.scene, tmp_path / "s.scene.json")
        restored = scene_io.load_scene(path)

        cameras = [
            o.get_component(CameraComponent)
            for o in restored.objects
            if o.get_component(CameraComponent)
        ]
        lights = [
            o.get_component(LightComponent)
            for o in restored.objects
            if o.get_component(LightComponent)
        ]
        assert [c.camera.fov for c in cameras] == [42.0]
        assert any(
            isinstance(c.light, SpotLight) and c.light.outer_angle == 33.0
            for c in lights
        )

    def test_duplicate_copies_camera_independently(self, window) -> None:
        original = window.create_camera()
        clone = window.context.duplicate_object(original)
        a = original.get_component(CameraComponent)
        b = clone.get_component(CameraComponent)
        assert b is not None and b is not a
        assert b.camera is not a.camera
        b.camera.fov = 10.0
        assert a.camera.fov != 10.0

    def test_duplicate_copies_light(self, window) -> None:
        original = window.create_light("point")
        clone = window.context.duplicate_object(original)
        a = original.get_component(LightComponent)
        b = clone.get_component(LightComponent)
        assert b is not None and b.light is not a.light
        assert b.light.type_name == "point"


class TestOverlay:
    def test_overlay_hidden_without_camera_selected(self, window) -> None:
        cube = next(o for o in window.context.scene.objects if o.name == "Cube")
        window.context.selection.select(cube)
        assert not window.viewport.update_camera_overlay()
        assert not window.viewport.camera_overlay.isVisible()

    def test_selected_camera_is_reported(self, window) -> None:
        camera_object = window.create_camera()
        assert (
            window.viewport.selected_camera()
            is camera_object.get_component(CameraComponent).camera
        )

    def test_overlay_is_child_of_scene_viewport(self, window) -> None:
        assert window.viewport.camera_overlay.parentWidget() is window.viewport.viewport
