"""Пикинг во вьюпорте: вложенный мелкий объект важнее огромной коробки уровня."""

from __future__ import annotations

import numpy as np
import pytest

from editor.viewport.editor_viewport import (
    _box_volume,
    _PickHit,
    _ray_box_distance,
    _ray_box_span,
    choose_pick,
)
from engine.core.math3d import Vector3


def _hit(name: str, enter: float, exit_: float, volume: float) -> _PickHit:
    return _PickHit(name, enter, exit_, volume)  # type: ignore[arg-type]


def test_nested_small_object_beats_big_level_box() -> None:
    level = _hit("level", 1.0, 30.0, 4000.0)
    player = _hit("player", 10.0, 11.0, 2.0)
    assert choose_pick([level, player]) == "player"
    assert choose_pick([player, level]) == "player"


def test_nearest_wins_for_similar_boxes() -> None:
    near = _hit("near", 5.0, 6.0, 1.0)
    far = _hit("far", 5.5, 6.5, 0.8)  # перекрывается, но не вдвое меньше
    assert choose_pick([far, near]) == "near"


def test_box_behind_the_exit_of_the_first_is_not_nested() -> None:
    wall = _hit("wall", 2.0, 3.0, 100.0)
    behind = _hit("behind", 8.0, 9.0, 1.0)
    assert choose_pick([behind, wall]) == "wall"


def test_nested_chain_and_empty() -> None:
    hits = [
        _hit("level", 0.0, 50.0, 8000.0),
        _hit("room", 5.0, 20.0, 1000.0),
        _hit("crate", 10.0, 11.0, 1.0),
    ]
    assert choose_pick(hits) == "crate"
    assert choose_pick([]) is None


def test_ray_span_and_distance_agree() -> None:
    bounds = (Vector3(-1, -1, -1), Vector3(1, 1, 1))
    matrix = np.eye(4)
    origin, direction = Vector3(0, 0, -5), Vector3(0, 0, 1)
    assert _ray_box_span(origin, direction, matrix, bounds) == (4.0, 6.0)
    assert _ray_box_distance(origin, direction, matrix, bounds) == 4.0
    assert _ray_box_span(Vector3(5, 0, -5), direction, matrix, bounds) is None


def test_box_volume_accounts_for_scale() -> None:
    bounds = (Vector3(-1, -1, -1), Vector3(1, 1, 1))
    matrix = np.diag([2.0, 3.0, 4.0, 1.0])
    assert _box_volume(matrix, bounds) == pytest.approx(8.0 * 24.0)
