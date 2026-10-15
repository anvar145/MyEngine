"""Физический материал и правила смешивания параметров пары."""

from __future__ import annotations

from dataclasses import dataclass

COMBINE_MODES = ("multiply", "min", "max", "average")


def combine(mode: str, a: float, b: float) -> float:
    """Смешивает два значения по режиму ``mode``."""
    if mode == "multiply":
        return a * b
    if mode == "min":
        return min(a, b)
    if mode == "max":
        return max(a, b)
    if mode == "average":
        return 0.5 * (a + b)
    raise ValueError(f"неизвестный режим смешивания: {mode!r}")


@dataclass(slots=True)
class PhysicsMaterial:
    """Трение и упругость поверхности."""

    friction: float = 0.5
    restitution: float = 0.0
    friction_combine: str = "multiply"
    restitution_combine: str = "max"

    def __post_init__(self) -> None:
        for mode in (self.friction_combine, self.restitution_combine):
            if mode not in COMBINE_MODES:
                raise ValueError(f"неизвестный режим смешивания: {mode!r}")
        if self.friction < 0.0:
            raise ValueError("friction не может быть отрицательным")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution должен быть в диапазоне 0..1")


def _pick(mode_a: str, mode_b: str) -> str:
    """Режим пары: приоритетнее тот, что позже в ``COMBINE_MODES``."""
    if COMBINE_MODES.index(mode_a) >= COMBINE_MODES.index(mode_b):
        return mode_a
    return mode_b


def combine_friction(a: PhysicsMaterial, b: PhysicsMaterial) -> float:
    """Трение пары."""
    mode = _pick(a.friction_combine, b.friction_combine)
    return combine(mode, a.friction, b.friction)


def combine_restitution(a: PhysicsMaterial, b: PhysicsMaterial) -> float:
    """Упругость пары."""
    mode = _pick(a.restitution_combine, b.restitution_combine)
    return combine(mode, a.restitution, b.restitution)
