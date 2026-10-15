"""Инспектор: заголовок физической секции - «Включён» и «Удалить».

У секций ``RigidBody``, ``Collider`` и ``CharacterController`` в строке заголовка
есть галочка (``component.enabled``) и кнопка удаления компонента. Для
контроллера выключение - «заморозка» (двойник остаётся в мире, не двигается).
Удаление необратимо (undo в редакторе нет), поэтому идёт только после
подтверждения. Серая полоса у заголовка выключенного компонента - динамическое
свойство ``frozen`` (стиль в ``dark.qss.tpl``). Подробности - ``docs/EDITOR.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox, QPushButton

from editor.panels.physics_sections import PHYSICS_COMPONENTS
from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
)

I18N = Path(__file__).resolve().parents[1] / "i18n"
KEYS = (
    "inspector.physics.enabled",
    "inspector.physics.enabled_hint",
    "inspector.physics.character.enabled_hint",
    "inspector.physics.remove",
    "inspector.physics.remove_title",
    "inspector.physics.remove_hint",
    "inspector.physics.remove_confirm",
    "inspector.physics.removed",
)
SECTIONS = [
    pytest.param(RigidBodyComponent, "rigidbody_section", id="rigidbody"),
    pytest.param(ColliderComponent, "collider_section", id="collider"),
    pytest.param(CharacterControllerComponent, "character_section", id="character"),
]


def _object(window, *components):
    go = window.context.scene.create_game_object("Hero", position=Vector3(0, 1, 0))
    added = [go.add_component(kind) for kind in components]
    window.context.selection.select(go)
    window.inspector.refresh()
    return go, added


def _answer(monkeypatch, button: QMessageBox.StandardButton) -> list[tuple]:
    """Подменяет ``QMessageBox.question``; возвращает журнал вызовов."""
    calls: list[tuple] = []

    def question(*args, **kwargs):
        calls.append(args)
        return button

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return calls


def _dirty(window) -> bool:
    return window.context.active_document.dirty


# --- заголовок -----------------------------------------------------------------

@pytest.mark.parametrize(("kind", "attribute"), SECTIONS)
def test_header_has_enabled_check_and_remove_button(window, kind, attribute) -> None:
    _object(window, kind)
    section = getattr(window.inspector, attribute)
    tr = window.context.tr

    assert section.enabled_check.isChecked()
    assert section.enabled_check.text() == tr("inspector.physics.enabled")
    assert section.remove_button.text() == tr("inspector.physics.remove")
    assert section.remove_button.toolTip() == tr("inspector.physics.remove_hint")
    assert not section.header.property("frozen")


def test_character_check_tooltip_explains_freezing(window) -> None:
    _object(window, CharacterControllerComponent)
    check = window.inspector.character_section.enabled_check
    assert check.toolTip() == window.context.tr(
        "inspector.physics.character.enabled_hint"
    )


@pytest.mark.parametrize(("kind", "attribute"), SECTIONS)
def test_check_toggles_the_component(window, kind, attribute) -> None:
    _go, (component,) = _object(window, kind)
    section = getattr(window.inspector, attribute)

    section.enabled_check.click()
    assert component.enabled is False
    assert not section.enabled_check.isChecked()

    section.enabled_check.click()
    assert component.enabled is True


@pytest.mark.parametrize(("kind", "attribute"), SECTIONS)
def test_toggle_emits_changed_and_marks_the_scene_modified(
    window, kind, attribute
) -> None:
    _object(window, kind)
    section = getattr(window.inspector, attribute)
    seen: list[int] = []
    section.changed.connect(lambda: seen.append(1))
    window.context.active_document.dirty = False

    section.enabled_check.click()

    assert seen == [1]
    assert _dirty(window)


@pytest.mark.parametrize(("kind", "attribute"), SECTIONS)
def test_disabled_component_has_the_grey_stripe(window, kind, attribute) -> None:
    _, (component,) = _object(window, kind)
    component.enabled = False
    window.inspector.refresh()
    section = getattr(window.inspector, attribute)

    assert not section.enabled_check.isChecked()
    assert section.header.property("frozen") is True
    assert section.title.property("frozen") is True

    section.enabled_check.click()  # включили: полоса уходит
    assert section.header.property("frozen") is False
    assert section.title.property("frozen") is False

    section.enabled_check.click()
    assert section.header.property("frozen") is True


def test_toggle_updates_the_conflict_label_of_the_other_section(window) -> None:
    _object(window, CharacterControllerComponent, RigidBodyComponent)
    character = window.inspector.character_section
    body = window.inspector.rigidbody_section
    assert not character.conflict_label.isHidden()

    body.enabled_check.click()  # RigidBody выключен: конфликта нет
    assert character.conflict_label.isHidden()

    body.enabled_check.click()
    assert not character.conflict_label.isHidden()


# --- заморозка из редактора ---------------------------------------------------------

def test_check_freezes_the_character_twin_without_recreating_it(window) -> None:
    go, _ = _object(window, CharacterControllerComponent)
    physics = window.context.scene.physics
    physics.sync_structure()
    handle = physics.handle_of(go)
    assert handle is not None

    window.inspector.character_section.enabled_check.click()
    physics.sync_structure()

    assert physics.handle_of(go) == handle
    assert physics._bindings[go.id].frozen is True
    window.inspector.character_section.enabled_check.click()
    physics.sync_structure()
    assert physics.handle_of(go) == handle
    assert physics._bindings[go.id].frozen is False


# --- удаление ------------------------------------------------------------------------

@pytest.mark.parametrize(("kind", "attribute"), SECTIONS)
def test_remove_asks_and_declining_keeps_the_component(
    window, monkeypatch, kind, attribute
) -> None:
    go, (component,) = _object(window, kind)
    calls = _answer(monkeypatch, QMessageBox.StandardButton.No)
    window.context.active_document.dirty = False

    getattr(window.inspector, attribute).remove_button.click()

    assert len(calls) == 1
    assert go.get_component(kind) is component
    assert getattr(window.inspector, attribute) is not None
    assert not _dirty(window)


def test_confirmation_text_warns_that_it_cannot_be_undone(
    window, monkeypatch
) -> None:
    _object(window, CharacterControllerComponent)
    calls = _answer(monkeypatch, QMessageBox.StandardButton.No)

    window.inspector.character_section.remove_button.click()

    assert window.context.tr("inspector.physics.remove_confirm") in calls[0]
    assert "нельзя отменить" in window.context.tr("inspector.physics.remove_confirm")


@pytest.mark.parametrize(("kind", "attribute"), SECTIONS)
def test_confirmed_remove_deletes_the_component_and_its_section(
    window, monkeypatch, kind, attribute
) -> None:
    go, _ = _object(window, kind)
    _answer(monkeypatch, QMessageBox.StandardButton.Yes)
    window.context.active_document.dirty = False

    getattr(window.inspector, attribute).remove_button.click()

    assert go.get_component(kind) is None
    assert getattr(window.inspector, attribute) is None
    assert go in window.context.scene  # объект остаётся
    assert _dirty(window)


@pytest.mark.parametrize(("kind", "attribute"), SECTIONS)
def test_default_dialog_answer_does_not_remove(window, kind, attribute) -> None:
    """Фикстура ``_no_modal_questions`` отвечает «Discard»: это не «Да»."""
    go, (component,) = _object(window, kind)

    getattr(window.inspector, attribute).remove_button.click()

    assert go.get_component(kind) is component


def test_removed_character_leaves_the_world(window, monkeypatch) -> None:
    go, _ = _object(window, CharacterControllerComponent)
    physics = window.context.scene.physics
    physics.sync_structure()
    assert physics.handle_of(go) is not None
    _answer(monkeypatch, QMessageBox.StandardButton.Yes)

    window.inspector.character_section.remove_button.click()
    physics.sync_structure()

    assert physics.handle_of(go) is None


def test_removing_a_frozen_character_removes_its_twin(window, monkeypatch) -> None:
    go, (component,) = _object(window, CharacterControllerComponent)
    component.enabled = False
    window.inspector.refresh()
    physics = window.context.scene.physics
    physics.sync_structure()
    assert physics.handle_of(go) is not None
    _answer(monkeypatch, QMessageBox.StandardButton.Yes)

    window.inspector.character_section.remove_button.click()
    physics.sync_structure()

    assert physics.handle_of(go) is None


# --- добавление через интерфейс ----------------------------------------------------------

def _add_menu(window):
    text = window.context.tr("inspector.add_component")
    # прежние кнопки скрыты и ждут deleteLater: берём только видимую
    buttons = [
        b
        for b in window.inspector.findChildren(QPushButton)
        if b.text() == text and not b.isHidden()
    ]
    assert len(buttons) == 1
    return buttons[0].menu()


def test_all_three_components_are_added_from_the_ui(window) -> None:
    """Кнопка удаления безопасна: компонент всегда можно добавить обратно."""
    tr = window.context.tr

    for key, kind in PHYSICS_COMPONENTS.items():
        # RigidBody и CharacterController несовместимы: объект на каждый тип
        go, _ = _object(window)
        menu = _add_menu(window)
        submenu = next(
            a.menu() for a in menu.actions() if a.text() == tr("inspector.group.physics")
        )
        action = next(
            a for a in submenu.actions() if a.text() == tr(f"inspector.physics.add.{key}")
        )
        action.trigger()
        assert go.get_component(kind) is not None


# --- перевод ------------------------------------------------------------------------------

@pytest.mark.parametrize("language", ["ru", "en"])
def test_header_keys_exist_in_both_languages(language: str) -> None:
    table = json.loads((I18N / f"{language}.json").read_text(encoding="utf-8"))
    for key in KEYS:
        assert table.get(key), f"нет {key} в {language}"
