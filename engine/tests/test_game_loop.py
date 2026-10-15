"""Тесты игрового цикла: фиксированный шаг, пауза, масштаб времени, защита."""

from __future__ import annotations

import pytest

from engine.core.component import Component
from engine.core.game_loop import GameLoop
from engine.core.scene import Scene


class Counter:
    """Счётчик вызовов фазы с сохранением последнего ``dt``."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_dt = 0.0

    def __call__(self, dt: float) -> None:
        self.calls += 1
        self.last_dt = dt


class TestFixedStep:
    def test_single_fixed_step_for_exact_frame(self) -> None:
        loop = GameLoop(fixed_delta=0.1, max_substeps=10)
        fixed = Counter()
        loop.add_fixed_update(fixed)

        time = loop.step(0.1)
        assert fixed.calls == 1
        assert fixed.last_dt == pytest.approx(0.1)
        assert time.fixed_frame == 1
        assert time.alpha == pytest.approx(0.0)

    def test_leftover_accumulates_and_alpha_grows(self) -> None:
        loop = GameLoop(fixed_delta=0.1, max_substeps=10)
        loop.step(0.15)
        assert loop.accumulator == pytest.approx(0.05, abs=1e-9)
        time = loop.step(0.0)
        assert time.alpha == pytest.approx(0.5, abs=1e-6)

    def test_multiple_steps_for_long_frame(self) -> None:
        loop = GameLoop(
            fixed_delta=0.05, max_substeps=10, max_frame_delta=1.0
        )
        fixed = Counter()
        loop.add_fixed_update(fixed)
        loop.step(0.25)
        assert fixed.calls == 5
        assert loop.time.fixed_frame == 5

    def test_no_fixed_step_for_small_frame(self) -> None:
        loop = GameLoop(fixed_delta=0.1)
        fixed = Counter()
        loop.add_fixed_update(fixed)
        loop.step(0.04)
        assert fixed.calls == 0
        assert loop.time.frame == 1


class TestGuards:
    def test_frame_delta_is_clamped(self) -> None:
        loop = GameLoop(fixed_delta=0.1, max_substeps=10, max_frame_delta=0.2)
        fixed = Counter()
        loop.add_fixed_update(fixed)
        time = loop.step(5.0)
        assert time.unscaled_delta == pytest.approx(0.2)
        assert fixed.calls == 2

    def test_max_substeps_drops_time(self) -> None:
        loop = GameLoop(fixed_delta=0.1, max_substeps=2, max_frame_delta=1.0)
        fixed = Counter()
        loop.add_fixed_update(fixed)
        loop.step(0.5)
        assert fixed.calls == 2
        assert loop.accumulator == pytest.approx(0.0)
        assert loop.stats.dropped_time == pytest.approx(0.3, abs=1e-6)
        assert loop.stats.overrun_frames == 1

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            GameLoop(fixed_delta=0.0)
        with pytest.raises(ValueError):
            GameLoop(max_substeps=0)


class TestTimeControl:
    def test_pause_freezes_simulation_but_keeps_render(self) -> None:
        loop = GameLoop(fixed_delta=0.1, max_substeps=10)
        fixed = Counter()
        update = Counter()
        render = Counter()
        loop.add_fixed_update(fixed)
        loop.add_update(update)
        loop.add_render(lambda dt, alpha: render(dt))

        loop.pause()
        loop.step(0.5)
        assert fixed.calls == 0
        assert update.calls == 1
        assert render.calls == 1
        assert loop.is_paused

        loop.resume()
        loop.step(0.2)
        assert fixed.calls == 2

    def test_time_scale(self) -> None:
        loop = GameLoop(fixed_delta=0.1, max_substeps=10)
        loop.time_scale = 0.5
        time = loop.step(0.2)
        assert time.delta == pytest.approx(0.1)
        assert time.unscaled_delta == pytest.approx(0.2)
        assert time.fixed_frame == 1

        loop.time_scale = 0.0
        loop.step(0.2)
        assert loop.time.delta == pytest.approx(0.0)

    def test_render_hook_receives_alpha(self) -> None:
        loop = GameLoop(fixed_delta=0.1, max_substeps=10)
        received: list[tuple[float, float]] = []
        loop.add_render(lambda dt, alpha: received.append((dt, alpha)))
        loop.step(0.15)
        assert received[0][1] == pytest.approx(0.5, abs=1e-6)

    def test_reset_clears_time_and_stats(self) -> None:
        loop = GameLoop(fixed_delta=0.1)
        loop.step(0.1)
        loop.reset()
        assert loop.time.frame == 0
        assert loop.time.elapsed == 0.0
        assert loop.stats.frames == 0
        assert loop.accumulator == 0.0

    def test_target_fps_setter(self) -> None:
        loop = GameLoop(fixed_delta=0.1)
        loop.target_fps = 30
        assert loop.target_fps == 30.0
        loop.target_fps = None
        assert loop.target_fps is None


class TestStatsAndScene:
    def test_stats_accumulate(self) -> None:
        loop = GameLoop(fixed_delta=0.1, max_substeps=10, smoothing=1.0)
        loop.step(0.1)
        loop.step(0.2)
        assert loop.stats.frames == 2
        assert loop.stats.fixed_steps == 3
        assert loop.time.elapsed == pytest.approx(0.3)
        assert loop.stats.fps > 0.0

    def test_scene_phases_are_called(self) -> None:
        scene = Scene("Test")
        game_object = scene.create_game_object("Obj")

        class Tracker(Component):
            def __init__(self) -> None:
                super().__init__()
                self.updates = 0

            def on_update(self, dt: float) -> None:
                self.updates += 1

        tracker = game_object.add_component(Tracker)
        loop = GameLoop(scene, fixed_delta=0.1, max_substeps=10)
        loop.step(0.1)
        assert tracker.updates == 1
        assert loop.scene is scene

    def test_run_with_frame_limit(self) -> None:
        loop = GameLoop(fixed_delta=1.0 / 60.0)
        assert loop.run(max_frames=3) == 3
        assert loop.time.frame == 3
        assert not loop.is_running

    def test_error_policy(self) -> None:
        strict = GameLoop(fixed_delta=0.1)

        def boom(dt: float) -> None:
            raise RuntimeError("boom")

        strict.add_update(boom)
        with pytest.raises(RuntimeError):
            strict.step(0.1)

        tolerant = GameLoop(fixed_delta=0.1, continue_on_error=True)
        survived = Counter()
        tolerant.add_update(boom)
        tolerant.add_update(survived)
        tolerant.step(0.1)
        assert survived.calls == 1

    def test_repr(self) -> None:
        loop = GameLoop(fixed_delta=0.1)
        loop.step(0.1)
        assert "GameLoop" in repr(loop)
