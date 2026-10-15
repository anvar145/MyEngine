"""Привязка действия к физическому элементу устройства.

Устройства: ``keyboard`` (код - имя клавиши скрипта: ``"w"``, ``"space"``),
``mouse`` (``left``/``right``/``middle`` и дельта ``delta_x``/``delta_y``),
``gamepad`` (``button:<имя>`` или ``axis:<имя>``).

Example:
    >>> Binding("keyboard", "w", axis=1).raw_id
    'keyboard:w'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.core.input.mouse import MOUSE_BUTTONS

__all__ = ["DEVICES", "MOUSE_DELTAS", "Binding"]

#: Поддерживаемые устройства.
DEVICES: tuple[str, ...] = ("keyboard", "mouse", "gamepad")

#: Оси смещения мыши (за кадр).
MOUSE_DELTAS: tuple[str, ...] = ("delta_x", "delta_y")


@dataclass(frozen=True, slots=True)
class Binding:
    """Одна привязка.

    Attributes:
        device: ``keyboard``, ``mouse`` или ``gamepad``.
        code: Код элемента внутри устройства.
        scale: Множитель значения (``-1`` - обратное направление оси).
        axis: Компонент для действий ``axis2``: 0 - X, 1 - Y.
    """

    device: str
    code: str
    scale: float = 1.0
    axis: int = 0

    def __post_init__(self) -> None:
        if self.device not in DEVICES:
            raise ValueError(f"Неизвестное устройство ввода: {self.device!r}")
        code = str(self.code).lower() if self.device == "keyboard" else str(self.code)
        if not code:
            raise ValueError("Код привязки пуст")
        if self.device == "mouse" and code not in (*MOUSE_BUTTONS, *MOUSE_DELTAS):
            raise ValueError(f"Неизвестный код мыши: {code!r}")
        if self.device == "gamepad" and code.split(":", 1)[0] not in ("button", "axis"):
            raise ValueError(
                f"Код геймпада должен быть 'button:<имя>' или 'axis:<имя>': {code!r}"
            )
        if self.axis not in (0, 1):
            raise ValueError(f"axis должен быть 0 или 1, получено {self.axis!r}")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "scale", float(self.scale))

    @property
    def raw_id(self) -> str:
        """Идентификатор элемента: ``"<устройство>:<код>"``."""
        return f"{self.device}:{self.code}"

    @property
    def is_analog(self) -> bool:
        """Источник даёт непрерывное значение (ось геймпада, дельта мыши)."""
        return self.code in MOUSE_DELTAS or self.code.startswith("axis:")

    def to_dict(self) -> dict[str, Any]:
        """Словарь для JSON (значения по умолчанию опускаются)."""
        data: dict[str, Any] = {"device": self.device, "code": self.code}
        if self.axis:
            data["axis"] = self.axis
        if self.scale != 1.0:
            data["scale"] = self.scale
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Binding:
        """Создать из словаря JSON.

        Raises:
            ValueError: Нет обязательных полей или значения некорректны.
        """
        try:
            return cls(
                str(data["device"]),
                str(data["code"]),
                float(data.get("scale", 1.0)),
                int(data.get("axis", 0)),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Некорректная привязка: {data!r}") from exc
