"""Состояние ввода: сырые устройства + действия + контексты.

Бэкенд (GLFW, Qt, ручной) пишет события в :class:`InputState`; игра читает
действия: ``held``/``pressed``/``released``/``axis``/``axis2``. Ядро ничего не
знает про GLFW и Qt.

Кадр: события приходят между кадрами, игра опрашивает состояние в течение
кадра, в конце кадра вызывается :meth:`InputState.end_frame` - он запоминает
текущее состояние как «предыдущее» (от него считаются ``pressed``/``released``)
и сбрасывает накопленные нажатия и дельту мыши. Нажатие и отпускание внутри
одного кадра не теряются.

Старый API (``is_key_down("w")``, ``pressed_keys``, ``press``/``release``)
работает как раньше и контекстами не фильтруется.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from pathlib import Path

from engine.core.input.action import InputAction
from engine.core.input.binding import Binding
from engine.core.input.context import InputContextStack
from engine.core.input.input_map import InputMap, default_input_map
from engine.core.math3d import Vector3

__all__ = ["InputState"]

_log = logging.getLogger("engine.input")

#: Колбэк ремаппинга: ``(действие, индекс привязки, новая привязка)``.
RebindCallback = Callable[[str, int, Binding], None]

#: Порог превращения аналогового значения в «нажато» для кнопочных действий.
BUTTON_THRESHOLD = 0.5


class _Frame:
    """Снимок сырых устройств (состояние на конец предыдущего кадра)."""

    __slots__ = (
        "gamepad_axes",
        "gamepad_buttons",
        "mouse_buttons",
        "mouse_delta",
        "pressed_keys",
    )

    def __init__(self, src: InputState | None = None) -> None:
        self.pressed_keys: set[str] = set(src.pressed_keys) if src else set()
        self.mouse_buttons: set[str] = set(src.mouse_buttons) if src else set()
        self.gamepad_buttons: set[str] = set(src.gamepad_buttons) if src else set()
        self.gamepad_axes: dict[str, float] = dict(src.gamepad_axes) if src else {}
        self.mouse_delta: tuple[float, float] = (0.0, 0.0)


def _raw(binding: Binding, src: InputState | _Frame) -> float:
    """Значение привязки с учётом ``scale`` в состоянии ``src``."""
    code = binding.code
    if binding.device == "keyboard":
        value = 1.0 if code in src.pressed_keys else 0.0
    elif binding.device == "mouse":
        if code == "delta_x":
            value = src.mouse_delta[0]
        elif code == "delta_y":
            value = src.mouse_delta[1]
        else:
            value = 1.0 if code in src.mouse_buttons else 0.0
    else:
        kind, _, name = code.partition(":")
        if kind == "button":
            value = 1.0 if name in src.gamepad_buttons else 0.0
        else:
            value = src.gamepad_axes.get(name, 0.0)
    return value * binding.scale


class InputState:
    """Состояние ввода игры (доступно скрипту как ``self.input``).

    Args:
        pressed_keys: Клавиши, считающиеся нажатыми с начала.
        input_map: Маппинг действий; по умолчанию - дефолтный (лениво).
        contexts: Стек контекстов; по умолчанию - только ``game``.

    Attributes:
        pressed_keys: Удерживаемые клавиши (имена в нижнем регистре).
        mouse_buttons: Удерживаемые кнопки мыши.
        mouse_delta: Смещение мыши за текущий кадр ``(dx, dy)``.
        gamepad_buttons: Удерживаемые кнопки геймпада (без префикса).
        gamepad_axes: Значения осей геймпада (без префикса).
        contexts: Стек контекстов ввода.
    """

    __slots__ = (
        "_map",
        "_prev",
        "_rebind",
        "_rebind_callbacks",
        "_tap_down",
        "_tap_up",
        "_warned",
        "contexts",
        "gamepad_axes",
        "gamepad_buttons",
        "mouse_buttons",
        "mouse_delta",
        "pressed_keys",
        "settings_path",
    )

    def __init__(
        self,
        pressed_keys: Iterable[str] = (),
        input_map: InputMap | None = None,
        contexts: InputContextStack | None = None,
    ) -> None:
        self.pressed_keys: set[str] = {str(key).lower() for key in pressed_keys}
        self.mouse_buttons: set[str] = set()
        self.mouse_delta: tuple[float, float] = (0.0, 0.0)
        self.gamepad_buttons: set[str] = set()
        self.gamepad_axes: dict[str, float] = {}
        self.contexts = contexts if contexts is not None else InputContextStack()
        self._map = input_map
        self._prev = _Frame()
        self._tap_down: set[str] = set()
        self._tap_up: set[str] = set()
        self._warned: set[str] = set()
        self._rebind: tuple[str, int] | None = None
        self._rebind_callbacks: list[tuple[str | None, RebindCallback]] = []
        #: Файл пользовательских привязок (``Settings/input.json``).
        self.settings_path: Path | None = None

    # ------------------------------------------------------------------ #
    # Маппинг
    # ------------------------------------------------------------------ #
    @property
    def input_map(self) -> InputMap:
        """Маппинг действий (дефолтный создаётся при первом обращении)."""
        if self._map is None:
            self._map = default_input_map()
        return self._map

    @input_map.setter
    def input_map(self, value: InputMap) -> None:
        self._map = value

    # ------------------------------------------------------------------ #
    # События от бэкендов
    # ------------------------------------------------------------------ #
    def key_down(self, key: str) -> None:
        """Клавиша нажата."""
        name = str(key).lower()
        if self._rebind is not None:
            self._capture(None if name == "escape" else Binding("keyboard", name))
            return
        if name not in self.pressed_keys:
            self.pressed_keys.add(name)
        self._tap_down.add(f"keyboard:{name}")

    def key_up(self, key: str) -> None:
        """Клавиша отпущена."""
        name = str(key).lower()
        self.pressed_keys.discard(name)
        self._tap_up.add(f"keyboard:{name}")

    def mouse_button(self, name: str, down: bool) -> None:
        """Кнопка мыши нажата или отпущена."""
        if down and self._rebind is not None:
            self._capture(Binding("mouse", name))
            return
        if down:
            self.mouse_buttons.add(name)
            self._tap_down.add(f"mouse:{name}")
        else:
            self.mouse_buttons.discard(name)
            self._tap_up.add(f"mouse:{name}")

    def mouse_move(self, dx: float, dy: float) -> None:
        """Мышь сместилась (дельта накапливается до конца кадра)."""
        self.mouse_delta = (self.mouse_delta[0] + dx, self.mouse_delta[1] + dy)

    def gamepad_button(self, name: str, down: bool) -> None:
        """Кнопка геймпада нажата или отпущена."""
        if down and self._rebind is not None:
            self._capture(Binding("gamepad", f"button:{name}"))
            return
        if down:
            self.gamepad_buttons.add(name)
            self._tap_down.add(f"gamepad:button:{name}")
        else:
            self.gamepad_buttons.discard(name)
            self._tap_up.add(f"gamepad:button:{name}")

    def gamepad_axis(self, name: str, value: float) -> None:
        """Ось геймпада приняла значение ``value`` (-1..1)."""
        self.gamepad_axes[name] = float(value)

    # ------------------------------------------------------------------ #
    # Ремаппинг
    # ------------------------------------------------------------------ #
    @property
    def rebinding(self) -> tuple[str, int] | None:
        """``(действие, индекс)``, пока ждём клавишу; иначе ``None``."""
        return self._rebind

    def begin_rebind(self, action: str, index: int = 0) -> None:
        """Ждать следующую клавишу/кнопку и записать её в привязку ``index``.

        Индекс, равный числу привязок, добавляет новую. Нажатие поглощается
        (игре не уходит), ``Escape`` отменяет.

        Raises:
            KeyError: Нет такого действия.
            IndexError: Индекс вне диапазона.
        """
        target = self.input_map.get(action)
        if target is None:
            raise KeyError(f"Нет действия {action!r}")
        if not 0 <= index <= len(target.bindings):
            raise IndexError(f"{action}: нет привязки #{index}")
        self._rebind = (action, index)

    def cancel_rebind(self) -> None:
        """Выйти из режима ожидания, ничего не меняя."""
        self._rebind = None

    def on_rebind(self, action: str | None, callback: RebindCallback) -> None:
        """Подписаться на смену привязки (``action=None`` - любого действия)."""
        self._rebind_callbacks.append((action, callback))

    def save_bindings(self, path: Path | str | None = None) -> Path:
        """Сохранить отличия от дефолтов (по умолчанию - ``settings_path``).

        Raises:
            ValueError: Путь не задан.
        """
        target = self._settings_target(path)
        self.input_map.save(target, only_changes=True)
        return target

    def load_bindings(self, path: Path | str | None = None) -> Path:
        """Загрузить привязки поверх дефолтов (``input_map`` заменяется).

        Raises:
            ValueError: Путь не задан или файл некорректен.
            OSError: Файл не читается.
        """
        target = self._settings_target(path)
        self._map = InputMap.load(target)
        return target

    def _settings_target(self, path: Path | str | None) -> Path:
        chosen = Path(path) if path is not None else self.settings_path
        if chosen is None:
            raise ValueError("Не задан путь к файлу привязок")
        return chosen

    def _capture(self, binding: Binding | None) -> None:
        """Завершить ожидание: записать ``binding`` (``None`` - отмена)."""
        assert self._rebind is not None
        action, index = self._rebind
        self._rebind = None
        if binding is None:
            return
        target = self.input_map.get(action)
        if target is None or index > len(target.bindings):
            return
        if index < len(target.bindings):
            old = target.bindings[index]
            if not old.is_analog:  # направление оси сохраняется при смене клавиши
                binding = Binding(binding.device, binding.code, old.scale, old.axis)
        self.input_map.rebind(action, index, binding)
        for wanted, callback in list(self._rebind_callbacks):
            if wanted is None or wanted == action:
                callback(action, index, binding)

    # Совместимость со старым API
    def press(self, key: str) -> None:
        """Отметить клавишу нажатой (синоним :meth:`key_down`)."""
        self.key_down(key)

    def release(self, key: str) -> None:
        """Снять отметку нажатия (синоним :meth:`key_up`)."""
        self.key_up(key)

    def clear(self) -> None:
        """Сбросить все нажатия (например, при потере фокуса окном)."""
        self.pressed_keys.clear()
        self.mouse_buttons.clear()
        self.gamepad_buttons.clear()
        self.gamepad_axes.clear()

    def end_frame(self) -> None:
        """Завершить кадр: текущее состояние становится «предыдущим»."""
        self._prev = _Frame(self)
        self._tap_down.clear()
        self._tap_up.clear()
        self.mouse_delta = (0.0, 0.0)

    # ------------------------------------------------------------------ #
    # Старый API (клавиатура)
    # ------------------------------------------------------------------ #
    def is_key_down(self, key: str) -> bool:
        """Нажата ли клавиша (имя: ``"space"``, ``"w"``, ``"shift"``)."""
        return str(key).lower() in self.pressed_keys

    def is_key_pressed(self, key: str) -> bool:
        """Синоним :meth:`is_key_down` для читаемости скриптов."""
        return self.is_key_down(key)

    def __contains__(self, key: object) -> bool:
        return str(key).lower() in self.pressed_keys

    # ------------------------------------------------------------------ #
    # Действия
    # ------------------------------------------------------------------ #
    def held(self, action: str) -> bool:
        """Удерживается ли действие (для осей - значение не нулевое)."""
        found = self._lookup(action)
        if found is None:
            return False
        return self._down(found, self)

    def pressed(self, action: str) -> bool:
        """Действие началось в этом кадре."""
        found = self._lookup(action)
        if found is None:
            return False
        if self._down(found, self) and not self._down(found, self._prev):
            return True
        return any(b.raw_id in self._tap_down for b in found.bindings)

    def released(self, action: str) -> bool:
        """Действие закончилось в этом кадре."""
        found = self._lookup(action)
        if found is None:
            return False
        if self._down(found, self._prev) and not self._down(found, self):
            return True
        return any(b.raw_id in self._tap_up for b in found.bindings)

    def axis(self, action: str) -> float:
        """Значение оси в диапазоне -1..1 (``0.0``, если действие закрыто)."""
        found = self._lookup(action)
        if found is None:
            return 0.0
        return self._axis(found, self)

    def axis2(self, action: str) -> tuple[float, float]:
        """Пара ``(x, y)`` длиной не больше 1 (для движения и взгляда)."""
        found = self._lookup(action)
        if found is None:
            return (0.0, 0.0)
        return self._axis2(found, self)

    def vec2(self, action_x: str, action_y: str) -> Vector3:
        """``Vector3(axis(action_x), axis(action_y), 0)``."""
        return Vector3(self.axis(action_x), self.axis(action_y), 0.0)

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #
    def _lookup(self, name: str) -> InputAction | None:
        """Действие, если оно есть и контексты его пропускают."""
        action = self.input_map.get(name)
        if action is None:
            if name not in self._warned:
                self._warned.add(name)
                _log.warning("Неизвестное действие ввода: %r", name)
            return None
        return action if self.contexts.allows(name) else None

    def _down(self, action: InputAction, src: InputState | _Frame) -> bool:
        if action.kind == "button":
            return any(
                abs(_raw(b, src)) >= BUTTON_THRESHOLD for b in action.bindings
            )
        if action.kind == "axis":
            return self._axis(action, src) != 0.0
        x, y = self._axis2(action, src)
        return x != 0.0 or y != 0.0

    @staticmethod
    def _axis(action: InputAction, src: InputState | _Frame) -> float:
        value = sum(_raw(b, src) for b in action.bindings)
        if abs(value) < action.deadzone:
            return 0.0
        return max(-1.0, min(1.0, value))

    @staticmethod
    def _axis2(action: InputAction, src: InputState | _Frame) -> tuple[float, float]:
        sums = [0.0, 0.0]
        for b in action.bindings:
            sums[b.axis] += _raw(b, src)
        length = math.hypot(sums[0], sums[1])
        if length < action.deadzone:
            return (0.0, 0.0)
        if length > 1.0:
            return (sums[0] / length, sums[1] / length)
        return (sums[0], sums[1])

    def __repr__(self) -> str:
        return f"<InputState нажато={sorted(self.pressed_keys)}>"
