"""Соответствие кодов клавиш GLFW именам, которые видят скрипты и маппинг."""

from __future__ import annotations

__all__ = [
    "KEY_F2",
    "KEY_F5",
    "KEY_F9",
    "KEY_SHIFT_F5",
    "MOD_SHIFT",
    "MOUSE_BUTTON_NAMES",
    "glfw_key_name",
]

KEY_F2 = 291
KEY_F5 = 294
KEY_F9 = 298
#: Бит Shift в маске модификаторов GLFW.
MOD_SHIFT = 1
#: Хоткей «стоп» (Shift+F5, как в редакторе); в ``key_events`` вместо ``KEY_F5``.
KEY_SHIFT_F5 = 1_000_000 + KEY_F5

#: Кнопки мыши GLFW (0, 1, 2) → имена привязок.
MOUSE_BUTTON_NAMES: dict[int, str] = {0: "left", 1: "right", 2: "middle"}

_SPECIAL: dict[int, str] = {
    32: "space",
    256: "escape",
    257: "enter",
    258: "tab",
    259: "backspace",
    262: "right",
    263: "left",
    264: "down",
    265: "up",
    340: "shift",
    344: "shift",
    341: "ctrl",
    345: "ctrl",
    342: "alt",
    346: "alt",
}


def glfw_key_name(key: int) -> str | None:
    """Имя клавиши для скрипта (``"w"``, ``"space"``, ``"f2"``) или ``None``."""
    if 65 <= key <= 90:  # A..Z
        return chr(key + 32)
    if 48 <= key <= 57:  # 0..9
        return chr(key)
    if 290 <= key <= 301:  # F1..F12
        return f"f{key - 289}"
    return _SPECIAL.get(key)
