"""Данные UI-компонентов: значения по умолчанию, валидация, to_dict/from_dict."""

from __future__ import annotations

import pytest

from engine.core.scene import Scene
from engine.core.ui_components import UIButton, UIImage, UIPanel, UIText


def test_defaults() -> None:
    panel = UIPanel()
    assert panel.anchor == "top_left"
    assert panel.size == (100.0, 100.0)
    assert panel.visible and panel.enabled and panel.opacity == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("anchor", "nowhere"),
        ("pivot", (1.5, 0.0)),
        ("opacity", 2.0),
        ("opacity", -0.1),
        ("size", (-1.0, 10.0)),
        ("size", (1.0,)),
        ("position", ("a", "b")),
        ("color", (2.0, 0.0, 0.0, 1.0)),
    ],
)
def test_invalid_values_rejected(field: str, value: object) -> None:
    panel = UIPanel()
    with pytest.raises(ValueError):
        setattr(panel, field, value)


def test_button_validation() -> None:
    with pytest.raises(ValueError):
        UIButton(on_click="bogus:thing")
    with pytest.raises(ValueError):
        UIButton(size_px=0)
    assert UIButton(on_click="event:quit").on_click == "event:quit"


def test_enabled_uses_component_property() -> None:
    scene = Scene()
    button = scene.create_game_object("B").add_component(UIButton)
    button.enabled = False
    assert not button.enabled
    assert button.to_dict()["enabled"] is False


def test_roundtrip_non_default_fields() -> None:
    image = UIImage(
        texture_path="a.png",
        uv_rect=(0.0, 0.0, 0.5, 0.5),
        color=(1.0, 0.5, 0.25, 0.8),
        anchor="bottom_right",
        pivot=(1.0, 1.0),
        position=(-10.0, -10.0),
        sorting_layer=3,
        clip=True,
        opacity=0.5,
    )
    clone = UIImage.from_dict(image.to_dict())
    assert clone.to_dict() == image.to_dict()
    assert clone.anchor == "bottom_right" and clone.sorting_layer == 3


def test_button_state_not_serialized() -> None:
    button = UIButton(text="Go", on_click="load_scene:main.scene.json")
    button.state = "hover"
    data = button.to_dict()
    assert "state" not in data
    clone = UIButton.from_dict(data)
    assert clone.state == "normal" and clone.text == "Go"


def test_ui_text_roundtrip() -> None:
    label = UIText("Привет", size=(200.0, 50.0), anchor="top_center")
    label.text.align = "center"
    clone = UIText.from_dict(label.to_dict())
    assert clone.text.text == "Привет" and clone.text.align == "center"
    assert clone.size == (200.0, 50.0) and clone.anchor == "top_center"


def test_revision_bumps_on_change() -> None:
    from engine.core.ui_components import UIElement

    panel = UIPanel()
    before = UIElement.revision
    panel.opacity = 0.3
    assert UIElement.revision > before
