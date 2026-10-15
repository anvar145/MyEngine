"""UI-компоненты в редакторе: меню «Создать», секция инспектора, дублирование."""

from __future__ import annotations

from editor.panels.text_section import TextSection
from editor.panels.ui_sections import UISection
from engine.core.ui_components import UIButton, UIImage, UIPanel, UIText


def test_create_each_ui_widget(window) -> None:
    for kind in (UIPanel, UIButton, UIText, UIImage):
        game_object = window.create_ui(kind)
        assert game_object.get_component(kind) is not None
        assert window.context.selection.primary is game_object


def test_ui_menu_actions_exist_and_are_titled(window) -> None:
    assert set(window.ui_actions) == {"ui_panel", "ui_button", "ui_text", "ui_image"}
    assert all(action.text() for action in window.ui_actions.values())
    window.ui_actions["ui_button"].trigger()
    assert window.context.selection.primary.get_component(UIButton) is not None


def test_inspector_edits_button(window) -> None:
    game_object = window.create_ui(UIButton)
    button = game_object.get_component(UIButton)
    section = window.inspector.ui_section
    assert isinstance(section, UISection) and section.component is button
    section.pos_x.setValue(40.0)
    section.size_w.setValue(250.0)
    section.opacity.setValue(0.5)
    section.label.setText("Играть")
    section.on_click.setText("load_scene:main.scene.json")
    section.on_click.editingFinished.emit()
    section.normal_alpha.setValue(0.25)
    section.anchor.setCurrentIndex(section.anchor.findData("bottom_right"))
    assert button.position[0] == 40.0 and button.size[0] == 250.0
    assert button.opacity == 0.5 and button.text == "Играть"
    assert button.on_click == "load_scene:main.scene.json"
    assert button.normal_color[3] == 0.25 and button.anchor == "bottom_right"


def test_invalid_click_action_is_rejected_by_component(window) -> None:
    game_object = window.create_ui(UIButton)
    button = game_object.get_component(UIButton)
    section = window.inspector.ui_section
    section.on_click.setText("bogus")
    section.on_click.editingFinished.emit()
    assert button.on_click == "" and section.on_click.text() == ""


def test_ui_text_gets_text_section(window) -> None:
    window.create_ui(UIText)
    assert isinstance(window.inspector.ui_text_section, TextSection)


def test_duplicate_keeps_ui_component(window) -> None:
    game_object = window.create_ui(UIPanel)
    panel = game_object.get_component(UIPanel)
    panel.border_width = 7.0
    clone = window.context.duplicate_object(game_object)
    copy = clone.get_component(UIPanel)
    assert copy is not None and copy is not panel and copy.border_width == 7.0
