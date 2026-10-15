"""Функции сглаживания для твинов: ``f(t) -> float``, ``f(0) = 0``, ``f(1) = 1``."""

from __future__ import annotations

from collections.abc import Callable

__all__ = [
    "EASES",
    "ease_in_cubic",
    "ease_in_out_cubic",
    "ease_in_out_quad",
    "ease_in_quad",
    "ease_out_cubic",
    "ease_out_quad",
    "get_ease",
    "linear",
    "smoothstep",
]


def linear(t: float) -> float:
    """Без сглаживания."""
    return t


def ease_in_quad(t: float) -> float:
    """Медленный старт (``t²``)."""
    return t * t


def ease_out_quad(t: float) -> float:
    """Медленный финиш."""
    return t * (2.0 - t)


def ease_in_out_quad(t: float) -> float:
    """Медленные старт и финиш."""
    if t < 0.5:
        return 2.0 * t * t
    return -1.0 + (4.0 - 2.0 * t) * t


def ease_in_cubic(t: float) -> float:
    """Медленный старт (``t³``)."""
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """Медленный финиш (кубический)."""
    u = 1.0 - t
    return 1.0 - u * u * u


def ease_in_out_cubic(t: float) -> float:
    """Медленные старт и финиш (кубический)."""
    if t < 0.5:
        return 4.0 * t * t * t
    u = -2.0 * t + 2.0
    return 1.0 - u * u * u / 2.0


def smoothstep(t: float) -> float:
    """``3t² - 2t³``: гладкие производные на концах."""
    return t * t * (3.0 - 2.0 * t)


EASES: dict[str, Callable[[float], float]] = {
    "linear": linear,
    "ease_in_quad": ease_in_quad,
    "ease_out_quad": ease_out_quad,
    "ease_in_out_quad": ease_in_out_quad,
    "ease_in_cubic": ease_in_cubic,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_cubic": ease_in_out_cubic,
    "smoothstep": smoothstep,
}


def get_ease(name: str | Callable[[float], float]) -> Callable[[float], float]:
    """Функция сглаживания по имени (или сама функция, если передана).

    Raises:
        ValueError: Если имя неизвестно.
    """
    if callable(name):
        return name
    try:
        return EASES[name]
    except KeyError:
        raise ValueError(
            f"неизвестный ease {name!r}; доступны: {', '.join(EASES)}"
        ) from None
