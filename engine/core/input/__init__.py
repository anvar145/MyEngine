"""Ввод: мышь, действия (``InputAction``), маппинг, контексты, бэкенды.

Ядро без Qt и GL. ``MouseState`` остался доступным из ``engine.core.input``.
"""

from __future__ import annotations

from engine.core.input.action import ACTION_KINDS, InputAction
from engine.core.input.backends import InputBackend, ManualInputBackend
from engine.core.input.binding import DEVICES, MOUSE_DELTAS, Binding
from engine.core.input.context import (
    SYSTEM_ACTIONS,
    UI_ACTIONS,
    InputContext,
    InputContextStack,
    cutscene_context,
    game_context,
    paused_context,
    ui_context,
)
from engine.core.input.input_map import DEFAULT_MAP_PATH, InputMap, default_input_map
from engine.core.input.keys import KEY_F2, MOUSE_BUTTON_NAMES, glfw_key_name
from engine.core.input.mouse import MOUSE_BUTTONS, MouseEvent, MouseState
from engine.core.input.state import InputState

__all__ = [
    "ACTION_KINDS",
    "DEFAULT_MAP_PATH",
    "DEVICES",
    "KEY_F2",
    "MOUSE_BUTTONS",
    "MOUSE_BUTTON_NAMES",
    "MOUSE_DELTAS",
    "SYSTEM_ACTIONS",
    "UI_ACTIONS",
    "Binding",
    "InputAction",
    "InputBackend",
    "InputContext",
    "InputContextStack",
    "InputMap",
    "InputState",
    "ManualInputBackend",
    "MouseEvent",
    "MouseState",
    "cutscene_context",
    "default_input_map",
    "game_context",
    "glfw_key_name",
    "paused_context",
    "ui_context",
]
