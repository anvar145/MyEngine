"""Сохранение и загрузка сцен с UI-компонентами."""

from __future__ import annotations

import json

from engine.core.scene import Scene
from engine.core.ui_components import UIButton, UIImage, UIPanel, UIText
from engine.scene_io import load_scene, save_scene, scene_from_dict, scene_to_dict


def _build() -> Scene:
    scene = Scene("UI")
    panel = scene.create_game_object("Panel")
    panel.add_component(
        UIPanel,
        position=(10.0, 20.0),
        size=(300.0, 200.0),
        anchor="middle_center",
        pivot=(0.5, 0.5),
        color=(0.1, 0.2, 0.3, 0.4),
        border_color=(1.0, 0.0, 0.0, 1.0),
        border_width=4.0,
        sorting_layer=2,
        clip=True,
        opacity=0.75,
    )
    image = scene.create_game_object("Image")
    image.add_component(
        UIImage,
        texture_path="Assets/icon.png",
        uv_rect=(0.0, 0.0, 0.5, 1.0),
        color=(1.0, 0.5, 0.5, 1.0),
        anchor="bottom_right",
    )
    image.set_parent(panel, keep_world=False)
    label = scene.create_game_object("Label")
    text = label.add_component(UIText, text="Привет", anchor="top_center")
    text.text.align = "center"
    text.text.size_px = 40
    text.text.color = (0.9, 0.8, 0.1)
    label.set_parent(panel, keep_world=False)
    button = scene.create_game_object("Button")
    button.add_component(
        UIButton,
        text="Играть",
        on_click="load_scene:main.scene.json",
        normal_color=(0.5, 0.5, 0.5, 1.0),
        size_px=30,
        enabled=False,
        visible=False,
    )
    return scene


def _ui_dump(scene: Scene) -> list[tuple[str, str | None, dict]]:
    result = []
    for game_object in scene.objects:
        parent = game_object.parent
        for component in game_object.components:
            to_dict = getattr(component, "to_dict", None)
            if to_dict is not None and type(component).__name__.startswith("UI"):
                result.append(
                    (
                        game_object.name,
                        None if parent is None else parent.name,
                        {"type": type(component).__name__, **to_dict()},
                    )
                )
    return result


def test_roundtrip_through_dict_keeps_all_fields() -> None:
    scene = _build()
    restored = scene_from_dict(json.loads(json.dumps(scene_to_dict(scene))))
    assert _ui_dump(restored) == _ui_dump(scene)
    assert len(_ui_dump(restored)) == 4


def test_roundtrip_through_file(tmp_path) -> None:
    scene = _build()
    path = tmp_path / "ui.scene.json"
    save_scene(scene, path)
    restored = load_scene(path)
    assert _ui_dump(restored) == _ui_dump(scene)
    button = restored.find("Button").get_component(UIButton)
    assert button is not None
    assert not button.enabled and not button.visible
    assert button.on_click == "load_scene:main.scene.json"
    assert button.state == "normal"


def test_non_default_values_survive() -> None:
    restored = scene_from_dict(scene_to_dict(_build()))
    panel = restored.find("Panel").get_component(UIPanel)
    assert panel is not None
    assert panel.anchor == "middle_center" and panel.pivot == (0.5, 0.5)
    assert panel.opacity == 0.75 and panel.clip and panel.sorting_layer == 2
    label = restored.find("Label").get_component(UIText)
    assert label is not None and label.text.text == "Привет"
    assert label.text.align == "center" and label.text.size_px == 40


def test_old_scene_without_ui_still_loads(tmp_path) -> None:
    scene = Scene("Old")
    scene.create_game_object("Plain")
    path = tmp_path / "old.scene.json"
    save_scene(scene, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 2
    assert all(
        component["type"] != "UIPanel"
        for entry in data["objects"]
        for component in entry["components"]
    )
    assert load_scene(path).find("Plain") is not None
