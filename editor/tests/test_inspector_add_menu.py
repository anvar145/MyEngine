"""Инспектор: меню «Добавить компонент» с группами-подменю и иконками."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QPushButton

from editor.panels.inspector import COMPONENT_GROUPS
from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
)

ROOT = Path(__file__).resolve().parents[2]
I18N = ROOT / "editor" / "i18n"
ICONS = ROOT / "assets" / "icons"


def _select(window):
    go = window.context.scene.create_game_object("Hero", position=Vector3(0, 1, 0))
    window.context.selection.select(go)
    window.inspector.refresh()
    return go


def _menu(window):
    text = window.context.tr("inspector.add_component")
    buttons = [
        b
        for b in window.inspector.findChildren(QPushButton)
        if b.text() == text and not b.isHidden()
    ]
    assert len(buttons) == 1
    return buttons[0].menu()


def _submenu(window, key: str):
    title = window.context.tr(f"inspector.group.{key}")
    return next(a.menu() for a in _menu(window).actions() if a.text() == title)


def test_all_group_icons_exist() -> None:
    names = {g.icon for g in COMPONENT_GROUPS}
    names |= {e.icon for g in COMPONENT_GROUPS for e in g.entries}
    missing = sorted(n for n in names if not (ICONS / f"{n}.svg").is_file())
    assert not missing


def test_menu_has_four_groups_with_expected_entries(window) -> None:
    _select(window)
    titles = [a.text() for a in _menu(window).actions()]
    tr = window.context.tr
    assert titles == [tr(f"inspector.group.{g.key}") for g in COMPONENT_GROUPS]
    for group in COMPONENT_GROUPS:
        texts = [a.text() for a in _submenu(window, group.key).actions()]
        assert texts == [tr(e.label_key) for e in group.entries]


def test_present_component_is_disabled_with_tooltip(window) -> None:
    go = _select(window)
    go.add_component(ColliderComponent)
    window.inspector.refresh()
    tr = window.context.tr
    actions = {a.text(): a for a in _submenu(window, "physics").actions()}
    collider = actions[tr("inspector.physics.add.collider")]
    assert not collider.isEnabled()
    assert collider.toolTip() == tr("inspector.component.already_added")
    assert actions[tr("inspector.physics.add.rigidbody")].isEnabled()


def test_physics_conflict_is_explained(window) -> None:
    go = _select(window)
    go.add_component(RigidBodyComponent)
    window.inspector.refresh()
    tr = window.context.tr
    actions = {a.text(): a for a in _submenu(window, "physics").actions()}
    character = actions[tr("inspector.physics.add.character")]
    assert not character.isEnabled()
    assert character.toolTip() == tr("inspector.component.physics_conflict")


def test_triggering_entry_adds_component(window) -> None:
    go = _select(window)
    tr = window.context.tr
    action = next(
        a
        for a in _submenu(window, "physics").actions()
        if a.text() == tr("inspector.physics.add.character")
    )
    action.trigger()
    assert go.get_component(CharacterControllerComponent) is not None


@pytest.mark.parametrize("language", ["ru", "en"])
def test_menu_keys_exist(language: str) -> None:
    table = json.loads((I18N / f"{language}.json").read_text(encoding="utf-8"))
    keys = [f"inspector.group.{g.key}" for g in COMPONENT_GROUPS]
    keys += [e.label_key for g in COMPONENT_GROUPS for e in g.entries]
    keys += [
        "inspector.component.already_added",
        "inspector.component.ui_present",
        "inspector.component.light_present",
        "inspector.component.physics_conflict",
    ]
    assert [k for k in keys if not table.get(k)] == []
