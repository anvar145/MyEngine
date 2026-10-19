"""P4: бюджет кадра профиля качества и единый выбор профиля."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.core.frame_budget import (
    BUDGETS,
    FrameBudget,
    current_budget,
    resolve_quality,
    set_current_budget,
)
from engine.rendering.settings import QualityTier


def _project(tmp_path: Path, **fields: object) -> Path:
    (tmp_path / "g.project.json").write_text(json.dumps({"name": "g", **fields}))
    return tmp_path


def test_every_tier_has_budget_and_low_is_cheapest() -> None:
    for tier in QualityTier:
        assert tier.budget == BUDGETS[tier.value]
    low, high = QualityTier.LOW.budget, QualityTier.HIGH.budget
    assert low.target_fps == 30 and low.frame_ms == pytest.approx(33.33, abs=0.01)
    assert low.animation_lod > high.animation_lod
    assert low.ai_tick_rate < high.ai_tick_rate
    assert low.physics_max_active < high.physics_max_active
    assert QualityTier.MEDIUM.budget.ai_interval == pytest.approx(0.05)


def test_unknown_tier_raises() -> None:
    with pytest.raises(ValueError):
        FrameBudget.for_tier("ultra")


def test_priority_cli_build_project_default(tmp_path: Path) -> None:
    root = _project(tmp_path, quality="low")
    assert resolve_quality(root).tier == "low"
    assert resolve_quality(root).source == "проект"
    assert resolve_quality(root, build_info={"quality": "high"}).tier == "high"
    assert resolve_quality(root, cli="medium", build_info={"quality": "high"}).tier == (
        "medium"
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    assert resolve_quality(empty).tier == "medium"
    assert resolve_quality(None).budget == BUDGETS["medium"]


def test_project_budget_overrides_apply_to_any_tier(tmp_path: Path) -> None:
    root = _project(tmp_path, budget={"ai_tick_rate": 12, "physics_max_active": 40})
    config = resolve_quality(root, cli="high")
    assert config.budget.ai_tick_rate == 12.0
    assert isinstance(config.budget.ai_tick_rate, float)
    assert config.budget.physics_max_active == 40
    assert config.budget.animation_lod == BUDGETS["high"].animation_lod


def test_bad_project_values_are_skipped(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        quality="ultra",
        budget={"ai_tick_rate": "fast", "typo": 1, "animation_lod": -1,
                "physics_max_active": True},
    )
    config = resolve_quality(root)
    assert config.tier == "medium"
    assert config.budget == BUDGETS["medium"]


def test_current_budget_roundtrip() -> None:
    before = current_budget()
    try:
        set_current_budget(BUDGETS["low"])
        assert current_budget() is BUDGETS["low"]
    finally:
        set_current_budget(before)
