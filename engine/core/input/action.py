"""Абстрактное действие ввода (Jump, Fire, Move...)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.core.input.binding import Binding

__all__ = ["ACTION_KINDS", "InputAction"]

#: ``button`` - нажато/нет; ``axis`` - число -1..1; ``axis2`` - пара (x, y).
ACTION_KINDS: tuple[str, ...] = ("button", "axis", "axis2")


@dataclass(slots=True)
class InputAction:
    """Действие и список его привязок.

    Attributes:
        name: Имя действия.
        kind: Тип из :data:`ACTION_KINDS`.
        bindings: Привязки; для ``axis2`` каждая задаёт компонент (``axis``).
        deadzone: Мёртвая зона для ``axis``/``axis2`` (0..1).
    """

    name: str
    kind: str = "button"
    bindings: list[Binding] = field(default_factory=list)
    deadzone: float = 0.15

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Имя действия пусто")
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"Неизвестный тип действия: {self.kind!r}")
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError(f"deadzone должен быть в [0, 1): {self.deadzone!r}")
        self.bindings = list(self.bindings)

    def copy(self) -> InputAction:
        """Независимая копия (Binding неизменяем)."""
        return InputAction(self.name, self.kind, list(self.bindings), self.deadzone)

    def to_dict(self) -> dict[str, Any]:
        """Словарь для JSON (без имени - оно ключ)."""
        data: dict[str, Any] = {"kind": self.kind}
        if self.kind != "button":
            data["deadzone"] = self.deadzone
        data["bindings"] = [b.to_dict() for b in self.bindings]
        return data

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> InputAction:
        """Создать из словаря JSON.

        Raises:
            ValueError: Данные некорректны.
        """
        try:
            return cls(
                name,
                str(data.get("kind", "button")),
                [Binding.from_dict(b) for b in data.get("bindings", [])],
                float(data.get("deadzone", 0.15)),
            )
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"Некорректное действие {name!r}") from exc
