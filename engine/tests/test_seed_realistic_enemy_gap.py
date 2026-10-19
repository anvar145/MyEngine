"""Сид ``realistic``: между рядами врагов проходит персонаж (B9, зажатие на скорости)."""

from __future__ import annotations

import importlib.util
import sys
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHARACTER_DIAMETER = 0.8
FLOOR_HALF = 15.0


def _load_seed():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "seed_realistic_scene", ROOT / "scripts" / "seed_realistic_scene.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gap_between_enemy_rows_fits_the_character() -> None:
    seed = _load_seed()
    gap = seed.ENEMY_ROW_PITCH - seed.ENEMY_DIAMETER
    assert gap >= CHARACTER_DIAMETER + 0.1


def test_enemies_stay_inside_the_arena() -> None:
    seed = _load_seed()
    enemies = [o for o in seed.build_scene().objects if o.name.startswith("Enemy")]
    assert len(enemies) == 25
    zs = sorted({round(o.transform.position.z, 3) for o in enemies})
    assert len(zs) == 5
    assert zs[-1] + seed.ENEMY_DIAMETER / 2 < FLOOR_HALF
    assert all(b - a >= seed.ENEMY_ROW_PITCH - 1e-6 for a, b in pairwise(zs))
