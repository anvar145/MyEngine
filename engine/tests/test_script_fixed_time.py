"""ScriptTime: счётчики фиксированных шагов для скриптов (``on_fixed_update``)."""

from __future__ import annotations

from engine.scripting.script_component import ScriptTime


def test_advance_fixed_counts_steps_and_keeps_delta() -> None:
    t = ScriptTime()
    assert (t.fixed_frame, t.fixed_delta) == (0, 0.0)
    t.advance_fixed(1 / 60)
    t.advance_fixed(1 / 60)
    assert t.fixed_frame == 2
    assert t.fixed_delta == 1 / 60
    assert t.frame == 0
