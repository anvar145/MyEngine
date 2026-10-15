"""``self.screen`` в скриптах: fade/flash через ``Scene.screen``."""

from __future__ import annotations

from engine.core.scene import Scene
from engine.scripting import ScriptComponent
from engine.scripting.script_component import API_NAMES


def test_screen_in_api_names() -> None:
    assert "screen" in API_NAMES


def test_script_screen_controls_scene_fx() -> None:
    scene = Scene()
    game_object = scene.create_game_object("o")
    component = game_object.add_component(ScriptComponent(source="pass", auto_start=False))
    component.screen.fade_to((0, 0, 0), 1.0, 0.0)
    scene.screen.update(0.1)
    assert component.screen.fade_alpha == 1.0
    component.screen.clear()
    assert component.screen.fade_alpha == 0.0
