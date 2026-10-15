"""Функции сглаживания."""

from __future__ import annotations

from itertools import pairwise

import pytest

from engine.core.scheduling.easing import (
    EASES,
    ease_in_quad,
    ease_out_quad,
    get_ease,
    smoothstep,
)


@pytest.mark.parametrize("name", list(EASES))
def test_bounds(name: str) -> None:
    func = EASES[name]
    assert func(0.0) == pytest.approx(0.0)
    assert func(1.0) == pytest.approx(1.0)


def test_eight_eases_registered() -> None:
    assert len(EASES) == 8


def test_in_out_directions() -> None:
    assert ease_in_quad(0.5) < 0.5
    assert ease_out_quad(0.5) > 0.5


@pytest.mark.parametrize("name", list(EASES))
def test_monotonic(name: str) -> None:
    func = EASES[name]
    values = [func(i / 100) for i in range(101)]
    assert all(b >= a - 1e-12 for a, b in pairwise(values))


def test_smoothstep_symmetric() -> None:
    for t in (0.1, 0.25, 0.4):
        assert smoothstep(t) + smoothstep(1.0 - t) == pytest.approx(1.0)


def test_unknown_name_lists_available() -> None:
    with pytest.raises(ValueError, match="ease_in_quad"):
        get_ease("bounce")


def test_callable_passes_through() -> None:
    assert get_ease(ease_in_quad) is ease_in_quad
