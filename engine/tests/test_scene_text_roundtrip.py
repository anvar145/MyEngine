"""``TextRenderer``: значения по умолчанию, валидация и сохранение в сцену."""

from __future__ import annotations

import json

import pytest

from engine.core.components import TextRenderer
from engine.core.scene import Scene
from engine.scene_io import scene_from_dict, scene_to_dict


def test_defaults() -> None:
    text = TextRenderer()
    assert text.text == "Text" and text.size_px == 32
    assert text.color == (1.0, 1.0, 1.0) and text.alpha == 1.0
    assert text.align == "left" and text.vertical_align == "top"
    assert text.billboard is True and text.units_per_pixel == 0.01
    assert text.visible is True and text.sorting_layer == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size_px": 0},
        {"units_per_pixel": 0.0},
        {"alpha": 1.5},
        {"alpha": -0.1},
        {"align": "justify"},
        {"vertical_align": "baseline"},
        {"wrap_width": -1.0},
        {"line_spacing": 0.0},
        {"outline_width": -0.5},
    ],
)
def test_invalid_values_raise(kwargs) -> None:
    with pytest.raises(ValueError):
        TextRenderer(**kwargs)


def _non_default() -> TextRenderer:
    return TextRenderer(
        text="Привет,\nмир!",
        font_path="Assets/Fonts/My.ttf",
        size_px=48,
        color=(0.2, 0.4, 0.6),
        alpha=0.5,
        align="center",
        vertical_align="bottom",
        wrap_width=3.5,
        line_spacing=1.5,
        bold=True,
        italic=True,
        outline_color=(1.0, 0.0, 0.5),
        outline_width=2.0,
        billboard=False,
        units_per_pixel=0.02,
        visible=False,
        sorting_layer=7,
    )


def test_scene_text_roundtrip() -> None:
    scene = Scene("s")
    scene.create_game_object("T").add_component(_non_default())
    data = json.loads(json.dumps(scene_to_dict(scene)))
    entry = data["objects"][0]["components"][0]
    assert entry["type"] == "TextRenderer"

    restored = scene_from_dict(data).objects[0].get_component(TextRenderer)
    expected = _non_default()
    for name in TextRenderer.__slots__:
        assert getattr(restored, name) == getattr(expected, name), name


def test_missing_fields_use_defaults() -> None:
    scene = Scene("s")
    scene.create_game_object("T").add_component(TextRenderer())
    data = scene_to_dict(scene)
    data["objects"][0]["components"][0] = {"type": "TextRenderer", "text": "Hi"}
    restored = scene_from_dict(data).objects[0].get_component(TextRenderer)
    assert restored.text == "Hi" and restored.size_px == 32 and restored.billboard
