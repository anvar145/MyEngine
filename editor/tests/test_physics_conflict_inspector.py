"""Инспектор: подсветка конфликта CharacterController + RigidBody.

Движок не берёт в мир объект, у которого активны оба компонента (``ERROR`` в
консоли). В редакторе об этом должна сказать сама секция: красный текст у обоих
компонентов, редактирование при этом не блокируется. Конфликт считается по
*активным* компонентам: выключенный компонент не конфликтует (паттерн
«смерть/рэгдолл», ``docs/CORE.md`` §7.2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from editor.panels.physics_sections import (
    CharacterControllerSection,
    ColliderSection,
    RigidBodySection,
)
from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
)

KEY = "inspector.physics.conflict.cc_rb"
I18N = Path(__file__).resolve().parents[1] / "i18n"


def _object(window, *components):
    go = window.context.scene.create_game_object("Hero", position=Vector3(0, 1, 0))
    added = [go.add_component(kind) for kind in components]
    window.context.selection.select(go)
    window.inspector.refresh()
    return go, added


def _shown(section) -> bool:
    """Подпись конфликта показана (``isHidden`` не зависит от видимости окна)."""
    return not section.conflict_label.isHidden()


def test_cc_and_rb_both_show_the_conflict(window) -> None:
    _object(window, CharacterControllerComponent, RigidBodyComponent)
    assert _shown(window.inspector.character_section)
    assert _shown(window.inspector.rigidbody_section)


def test_alone_components_show_no_conflict(window) -> None:
    _object(window, CharacterControllerComponent)
    assert not _shown(window.inspector.character_section)
    _object(window, RigidBodyComponent)
    assert not _shown(window.inspector.rigidbody_section)


def test_conflict_text_is_translated_and_red(window) -> None:
    _object(window, CharacterControllerComponent, RigidBodyComponent)
    for section in (
        window.inspector.character_section, window.inspector.rigidbody_section
    ):
        label = section.conflict_label
        assert label.text() == window.context.tr(KEY)
        assert KEY not in label.text()
        assert label.text().startswith("Конфликт")
        assert label.property("role") == "error"  # красный текст темы


def test_removing_one_component_clears_the_conflict(window) -> None:
    go, (_cc, rb) = _object(window, CharacterControllerComponent, RigidBodyComponent)
    assert _shown(window.inspector.character_section)

    go.remove_component(rb)
    window.inspector.refresh()

    assert not _shown(window.inspector.character_section)
    assert window.inspector.rigidbody_section is None  # не остаётся старой секции


def test_removing_the_character_clears_rigidbody_warning(window) -> None:
    go, (cc, _rb) = _object(window, CharacterControllerComponent, RigidBodyComponent)
    assert _shown(window.inspector.rigidbody_section)

    go.remove_component(cc)
    window.inspector.refresh()

    assert not _shown(window.inspector.rigidbody_section)
    assert window.inspector.character_section is None


@pytest.mark.parametrize("disabled", ["character", "rigidbody"])
def test_disabled_component_does_not_conflict(window, disabled: str) -> None:
    _go, (cc, rb) = _object(
        window, CharacterControllerComponent, RigidBodyComponent
    )
    (cc if disabled == "character" else rb).enabled = False
    window.inspector.refresh()

    assert not _shown(window.inspector.character_section)
    assert not _shown(window.inspector.rigidbody_section)


def test_inactive_object_does_not_conflict(window) -> None:
    go, _ = _object(window, CharacterControllerComponent, RigidBodyComponent)
    go.active_self = False
    window.inspector.refresh()

    assert not _shown(window.inspector.character_section)
    assert not _shown(window.inspector.rigidbody_section)


def test_conflict_does_not_block_editing(window) -> None:
    _go, (cc, rb) = _object(window, CharacterControllerComponent, RigidBodyComponent)
    character = window.inspector.character_section
    body = window.inspector.rigidbody_section
    assert _shown(character)
    assert character.isEnabled() and body.isEnabled()

    character.move_speed.setValue(7.0)
    body.mass.setValue(3.0)

    assert cc.move_speed == 7.0
    assert rb.mass == 3.0


def test_collider_section_has_no_conflict_label(window) -> None:
    _object(
        window, CharacterControllerComponent, RigidBodyComponent, ColliderComponent
    )
    collider = window.inspector.collider_section
    assert isinstance(collider, ColliderSection)
    assert not hasattr(collider, "conflict_label")


def test_other_section_updates_when_a_component_is_toggled(window) -> None:
    """Выключили RigidBody - подпись у контроллера гаснет без пересборки."""
    _go, (_cc, rb) = _object(window, CharacterControllerComponent, RigidBodyComponent)
    character = window.inspector.character_section
    assert _shown(character)

    rb.enabled = False
    window.inspector.rigidbody_section.changed.emit()
    assert not _shown(character)

    rb.enabled = True
    window.inspector.rigidbody_section.changed.emit()
    assert _shown(character)


def test_sections_are_the_expected_types(window) -> None:
    _object(window, CharacterControllerComponent, RigidBodyComponent)
    assert isinstance(window.inspector.character_section, CharacterControllerSection)
    assert isinstance(window.inspector.rigidbody_section, RigidBodySection)


@pytest.mark.parametrize("both_active", [True, False])
def test_warning_matches_what_the_engine_does(window, both_active: bool) -> None:
    """Подсказка редактора совпадает с решением адаптера физики (``_conflicts``)."""
    go, (_cc, rb) = _object(window, CharacterControllerComponent, RigidBodyComponent)
    rb.enabled = both_active
    window.inspector.refresh()

    window.context.scene.physics.sync_structure()
    engine_conflict = go.id in window.context.scene.physics._conflicts

    assert engine_conflict is both_active
    assert _shown(window.inspector.character_section) is engine_conflict


@pytest.mark.parametrize("language", ["ru", "en"])
def test_conflict_key_exists_in_both_languages(language: str) -> None:
    table = json.loads((I18N / f"{language}.json").read_text(encoding="utf-8"))
    assert table.get(KEY), f"нет {KEY} в {language}"
