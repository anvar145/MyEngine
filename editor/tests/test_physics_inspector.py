"""Физические компоненты в инспекторе: секции, правка, дублирование, i18n."""

from __future__ import annotations

import json
from pathlib import Path

from editor.panels.physics_sections import (
    CharacterControllerSection,
    ColliderSection,
    RigidBodySection,
)
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
)


def _select(window, *components):
    game_object = window.context.scene.create_game_object("Phys")
    for component in components:
        game_object.add_component(component)
    window.context.selection.select(game_object)
    window.inspector.refresh()
    return game_object


def test_sections_built_and_edit_components(window) -> None:
    game_object = _select(
        window, ColliderComponent, RigidBodyComponent, CharacterControllerComponent
    )
    inspector = window.inspector
    assert isinstance(inspector.rigidbody_section, RigidBodySection)
    assert isinstance(inspector.collider_section, ColliderSection)
    assert isinstance(inspector.character_section, CharacterControllerSection)
    inspector.rigidbody_section.mass.setValue(7.5)
    inspector.collider_section.shape.setCurrentIndex(
        inspector.collider_section.shape.findData("capsule")
    )
    inspector.collider_section.is_trigger.setChecked(True)
    inspector.character_section.step_offset.setValue(0.4)
    assert game_object.get_component(RigidBodyComponent).mass == 7.5
    collider = game_object.get_component(ColliderComponent)
    assert collider.shape == "capsule" and collider.is_trigger
    assert game_object.get_component(CharacterControllerComponent).step_offset == 0.4


def test_duplicate_keeps_physics_components(window) -> None:
    game_object = _select(window, ColliderComponent, RigidBodyComponent)
    game_object.get_component(RigidBodyComponent).mass = 3.0
    clone = window.context.duplicate_object(game_object)
    assert clone.get_component(RigidBodyComponent).mass == 3.0
    assert clone.get_component(ColliderComponent) is not None


def test_physics_keys_present_in_both_languages() -> None:
    root = Path(__file__).resolve().parents[1] / "i18n"
    ru = json.loads((root / "ru.json").read_text(encoding="utf-8"))
    en = json.loads((root / "en.json").read_text(encoding="utf-8"))
    ru_keys = {k for k in ru if k.startswith("inspector.physics.")}
    assert ru_keys and ru_keys == {k for k in en if k.startswith("inspector.physics.")}


def test_view_menu_collider_modes(window) -> None:
    actions = window.collider_actions
    assert all(a.text() for a in actions.values())
    assert window.viewport.collider_mode == "selected"
    assert actions["selected"].isChecked()
    window.set_collider_mode("all")
    assert window.viewport.show_colliders and actions["all"].isChecked()
    window.set_collider_mode("off")
    assert not window.viewport.show_colliders and actions["off"].isChecked()
    window.set_collider_mode("selected")


def test_collider_mode_selected_filters_and_creates_world(window) -> None:
    from engine.core.math3d import Vector3
    from engine.core.physics_components import ColliderComponent

    viewport = window.viewport
    scene = window.context.scene
    a = scene.create_game_object("A", position=Vector3(0, 0, 0))
    b = scene.create_game_object("B", position=Vector3(3, 0, 0))
    a.add_component(ColliderComponent)
    b.add_component(ColliderComponent)
    viewport.set_collider_mode("selected")
    window.context.selection.select(None)
    assert viewport._collider_filter() == frozenset()  # ничего не выбрано
    window.context.selection.select(a)
    source = viewport._physics_source()
    assert source is not None
    handles = viewport._collider_filter()
    assert handles == source.handles_for([a]) and len(handles) == 1
    viewport.set_collider_mode("all")
    assert viewport._collider_filter() is None
    viewport.set_collider_mode("off")
    assert viewport._physics_source() is None
    viewport.set_collider_mode("selected")
