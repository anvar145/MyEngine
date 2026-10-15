"""Ввод UI: hover/press/click, перекрытие, отключённые и скрытые кнопки."""

from __future__ import annotations

import pytest

from engine.core.input import MouseState
from engine.core.scene import Scene
from engine.core.ui_components import UIButton, UIPanel
from engine.core.ui_events import (
    SceneLoadRequested,
    UIClick,
    UIEventFired,
    UIHoverEnter,
    UIHoverExit,
    UIPress,
    UIRelease,
)
from engine.core.ui_system import UISystem
from engine.rendering.ui import UIScreen

EVENTS = (UIHoverEnter, UIHoverExit, UIPress, UIRelease, UIClick, UIEventFired)


class Rig:
    """Сцена + экран 1280×720 (окно 1280×720) + запись событий."""

    def __init__(self, window=(1280, 720)) -> None:
        self.scene = Scene()
        self.screen = UIScreen((1280, 720))
        self.screen.set_viewport(*window)
        self.mouse = MouseState()
        self.system = UISystem(self.scene, self.screen, self.mouse)
        self.log: list[object] = []
        for event_type in (*EVENTS, SceneLoadRequested):
            self.scene.events.subscribe(event_type, self.log.append)

    def button(self, name="B", **kwargs):
        kwargs.setdefault("position", (100.0, 100.0))
        kwargs.setdefault("size", (200.0, 50.0))
        return self.scene.create_game_object(name).add_component(UIButton, **kwargs)

    def move(self, x, y):
        self.mouse.move(x, y)
        self.system.update()

    def click(self, x, y):
        self.mouse.move(x, y)
        self.mouse.button("left", True)
        self.mouse.button("left", False)
        self.system.update()

    def kinds(self):
        return [type(event).__name__ for event in self.log]


@pytest.fixture
def rig() -> Rig:
    return Rig()


def test_hover_enter_and_exit(rig: Rig) -> None:
    button = rig.button()
    rig.move(150, 120)
    assert rig.kinds() == ["UIHoverEnter"] and button.state == "hover"
    rig.move(150, 120)  # повторное смещение внутри — без событий
    assert len(rig.log) == 1
    rig.move(10, 10)
    assert rig.kinds() == ["UIHoverEnter", "UIHoverExit"] and button.state == "normal"


def test_press_release_click_sequence(rig: Rig) -> None:
    button = rig.button(on_click="event:go")
    rig.move(150, 120)
    rig.log.clear()
    rig.mouse.button("left", True)
    rig.system.update()
    assert rig.kinds() == ["UIPress"] and button.state == "pressed"
    rig.mouse.button("left", False)
    rig.system.update()
    assert rig.kinds() == ["UIPress", "UIRelease", "UIClick", "UIEventFired"]
    assert rig.log[-1].event_name == "go"
    assert button.state == "hover"


def test_press_and_release_in_one_frame_is_a_click(rig: Rig) -> None:
    rig.button()
    rig.click(150, 120)
    assert rig.kinds() == ["UIHoverEnter", "UIPress", "UIRelease", "UIClick"]


def test_release_outside_is_not_a_click(rig: Rig) -> None:
    button = rig.button()
    rig.move(150, 120)
    rig.mouse.button("left", True)
    rig.system.update()
    rig.move(600, 600)
    rig.mouse.button("left", False)
    rig.system.update()
    assert "UIClick" not in rig.kinds() and "UIRelease" in rig.kinds()
    assert button.state == "normal"


def test_disabled_and_hidden_buttons_ignore_mouse(rig: Rig) -> None:
    disabled = rig.button("D", enabled=False)
    hidden = rig.button("H", position=(400.0, 100.0), visible=False)
    rig.click(150, 120)
    rig.click(450, 120)
    assert rig.log == []
    assert disabled.state == "normal" and hidden.state == "normal"


def test_topmost_button_wins_on_overlap(rig: Rig) -> None:
    rig.button("low", on_click="event:low")
    rig.button("high", on_click="event:high", sorting_layer=5)
    rig.button("later", on_click="event:later", position=(500.0, 500.0))
    rig.click(150, 120)
    assert [e.event_name for e in rig.log if isinstance(e, UIEventFired)] == ["high"]


def test_later_sibling_is_above_earlier_with_equal_layer(rig: Rig) -> None:
    rig.button("first", on_click="event:first")
    rig.button("second", on_click="event:second")
    rig.click(150, 120)
    assert [e.event_name for e in rig.log if isinstance(e, UIEventFired)] == ["second"]


def test_load_scene_action(rig: Rig) -> None:
    rig.button(on_click="load_scene:main.scene.json")
    rig.click(150, 120)
    requests = [e for e in rig.log if isinstance(e, SceneLoadRequested)]
    assert [r.path for r in requests] == ["main.scene.json"]


def test_window_scaling_maps_mouse_to_ui() -> None:
    rig = Rig(window=(2560, 1440))
    rig.button(position=(100.0, 100.0), size=(200.0, 50.0))
    rig.move(150, 120)  # в пикселях окна это вне кнопки (x1 → 300..700)
    assert rig.log == []
    rig.move(400, 250)
    assert rig.kinds() == ["UIHoverEnter"]


def test_panel_does_not_receive_events_but_button_child_does(rig: Rig) -> None:
    panel = rig.scene.create_game_object("P")
    panel.add_component(UIPanel, position=(50.0, 50.0), size=(400.0, 300.0))
    child = rig.scene.create_game_object("C")
    child.add_component(
        UIButton, position=(10.0, 10.0), size=(100.0, 40.0), on_click="event:c"
    )
    child.set_parent(panel)
    rig.click(70, 70)
    assert "UIClick" in rig.kinds()
    rig.move(600, 600)
    rig.log.clear()
    rig.click(300, 300)
    assert rig.log == []


def test_mouse_leaving_window_clears_hover(rig: Rig) -> None:
    rig.button()
    rig.move(150, 120)
    rig.mouse.leave()
    rig.system.update()
    assert rig.kinds() == ["UIHoverEnter", "UIHoverExit"]
