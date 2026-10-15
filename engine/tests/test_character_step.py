"""Верификация CharacterController: ``step_offset`` (подъём на ступеньки).

Предельная высота ступени равна ``step_offset`` (с допуском 0.05 из-за сетки
замеров) и не зависит от скорости ходьбы. Ступень не выше ``step_offset``
проходится, выше — нет; шаг не проходит сквозь стены, потолки и обрывы.
"""

from __future__ import annotations

import pytest
from character_kit import (
    DT,
    WALK,
    add_step,
    add_wall,
    make_char,
    make_world,
    settle,
    walk,
)

from engine.physics.collider import Collider
from engine.physics.shape import BoxShape

STEP_X = 3.0  # ближняя грань уступа
CLIMBED_X = STEP_X + 1.0  # персонаж явно оказался на уступе


def _walk_at_step(height: float, step_offset: float = 0.3, speed: float = WALK,
                  depth: float = 2.0):
    """Идёт на уступ; останавливается, пока персонаж ещё над уступом (x ≈ 5)."""
    world = make_world()
    add_step(world, height, x=STEP_X, depth=depth)
    char = make_char(world, step_offset=step_offset)
    settle(char)
    walk(char, (speed, 0.0, 0.0), 5.0 / speed)
    return char


def _climbed(char, height: float) -> bool:
    return bool(
        char.position[0] > CLIMBED_X
        and abs(char.position[1] - char.height / 2 - (height + 0.02)) < 0.05
        and char.grounded
    )


def _blocked(char) -> bool:
    return bool(char.position[0] < STEP_X - char.radius + 0.1)


def _max_climbable(step_offset: float, speed: float) -> float:
    """Максимальная проходимая высота; перебор (шаг 0.05) около ``step_offset``."""
    best = 0.0
    start = max(round(step_offset / 0.05) - 2, 1)
    for i in range(start, start + 6):
        height = i * 0.05
        if _climbed(_walk_at_step(height, step_offset, speed), height):
            best = height
    return best


# --- 2a-2c: граница step_offset=0.3 -------------------------------------------------

@pytest.mark.parametrize("height", [0.05, 0.1, 0.2, 0.25, 0.29, 0.30])
def test_step_up_to_offset_is_climbed(height: float) -> None:
    assert _climbed(_walk_at_step(height), height)


@pytest.mark.parametrize("height", [0.31, 0.35, 0.4, 0.6])
def test_step_above_offset_is_not_climbed(height: float) -> None:
    char = _walk_at_step(height)
    assert _blocked(char)
    assert char.position[1] - char.height / 2 < height  # на уступ не поднялся


@pytest.mark.parametrize("speed", [2.0, 3.0, 4.5, 6.0, 9.0, 12.0])
def test_step_limit_does_not_depend_on_walk_speed(speed: float) -> None:
    assert _climbed(_walk_at_step(0.30, speed=speed), 0.30)
    assert _blocked(_walk_at_step(0.35, speed=speed))


# --- 2d: другое значение step_offset ----------------------------------------------

@pytest.mark.parametrize(("step_offset", "height"), [(0.5, 0.40), (0.5, 0.50),
                                                       (0.7, 0.70)])
def test_larger_step_offset_climbs_higher_steps(step_offset: float,
                                                height: float) -> None:
    assert _climbed(_walk_at_step(height, step_offset=step_offset), height)


def test_step_offset_05_blocks_060() -> None:
    assert _blocked(_walk_at_step(0.60, step_offset=0.5))


def test_larger_step_offset_raises_climbable_height() -> None:
    """Параметр влияет: 0.4 блокируется при 0.3 и проходится при 0.5."""
    assert _blocked(_walk_at_step(0.40, step_offset=0.3))
    assert _climbed(_walk_at_step(0.40, step_offset=0.5), 0.40)


@pytest.mark.parametrize("speed", [3.0, WALK])
def test_max_climbable_height_equals_step_offset(speed: float) -> None:
    for step_offset in (0.2, 0.5):
        assert _max_climbable(step_offset, speed) == pytest.approx(step_offset,
                                                                   abs=0.051)


# --- 2e: лестница ------------------------------------------------------------------

def _staircase(n: int, rise: float, run: float = 0.5):
    world = make_world()
    for i in range(n):
        h = rise * (i + 1)
        world.add_static(Collider(BoxShape((run / 2 + 6.0, h / 2, 3.0))),
                         (STEP_X + i * run + run / 2 + 6.0 - run / 2, h / 2, 0.0))
    return world


def test_staircase_of_five_steps_is_climbed() -> None:
    world = _staircase(5, 0.2)
    char = make_char(world)
    settle(char)
    walk(char, (WALK, 0.0, 0.0), 3.0)
    assert char.grounded
    assert char.position[1] - char.height / 2 == pytest.approx(1.0 + 0.02, abs=0.05)
    assert char.position[0] > STEP_X + 5 * 0.5


def test_staircase_does_not_make_character_bounce() -> None:
    world = _staircase(5, 0.2)
    char = make_char(world)
    settle(char)
    airborne = 0
    for _ in range(180):
        char.simple_move((WALK, 0.0, 0.0))
        world.step()
        airborne += not char.grounded
    assert airborne <= 8  # допуск на мерцание grounded на ступенях


def test_staircase_speed_has_no_big_jerks() -> None:
    """Рывок вперёд только на подъёме: за кадр не больше v·dt + radius/2."""
    world = _staircase(5, 0.2)
    char = make_char(world)
    settle(char)
    travelled = 0.0
    for _ in range(120):
        before = float(char.position[0])
        char.simple_move((WALK, 0.0, 0.0))
        world.step()
        step = float(char.position[0]) - before
        assert step <= WALK * DT + char.radius * 0.5 + 1e-6
        travelled += step
    assert travelled <= WALK * 2.0 * 1.3  # на лестнице не быстрее 1.3× ровного


# --- 2f: step_offset = 0 -----------------------------------------------------------

def test_zero_step_offset_blocks_regular_step() -> None:
    assert _blocked(_walk_at_step(0.2, step_offset=0.0))


def test_zero_step_offset_still_rolls_over_tiny_curb() -> None:
    """Низкий бордюр (до ~0.1 м) преодолевается скольжением по полусфере."""
    assert _climbed(_walk_at_step(0.05, step_offset=0.0), 0.05)


def test_zero_step_offset_still_walks_on_flat_floor() -> None:
    world = make_world()
    char = make_char(world, step_offset=0.0)
    settle(char)
    assert char.grounded
    walk(char, (WALK, 0.0, 0.0), 1.0)
    assert char.position[0] == pytest.approx(WALK, abs=0.1)


# --- 2g: большой step_offset -------------------------------------------------------

@pytest.mark.parametrize("height", [0.5, 0.9, 1.0])
def test_step_offset_one_climbs_high_steps(height: float) -> None:
    assert _climbed(_walk_at_step(height, step_offset=1.0), height)


def test_step_offset_one_blocks_120() -> None:
    assert _blocked(_walk_at_step(1.20, step_offset=1.0))


def test_large_step_offset_does_not_pass_through_wall() -> None:
    world = make_world()
    face = add_wall(world)
    char = make_char(world, step_offset=1.0)
    settle(char)
    walk(char, (WALK, 0.0, 0.0), 2.0)
    assert char.position[0] + char.radius <= face + 1e-3
    assert char.position[1] - char.height / 2 < 0.1  # не залез на стену


# --- безопасность шага: стены, потолки, обрывы -----------------------------------------

@pytest.mark.parametrize("speed", [2.0, 4.5, 20.0])
@pytest.mark.parametrize("gap", [0.05, 0.3])
def test_step_never_carries_character_through_wall_behind_edge(
        speed: float, gap: float) -> None:
    """Высокая стена сразу за ребром ступени: персонаж не оказывается за ней."""
    wall_face = STEP_X + gap
    for phase in range(4):  # сдвиг старта по фазе кадра
        world = make_world()
        add_step(world, 0.2, x=STEP_X, depth=gap)
        world.add_static(Collider(BoxShape((0.5, 3.0, 5.0))),
                         (wall_face + 0.5, 3.0, 0.0))
        char = make_char(world, x=phase * speed * DT / 4.0)
        settle(char)
        for _ in range(90):
            char.simple_move((speed, 0.0, 0.0))
            world.step()
            assert char.position[0] + char.radius <= wall_face + 1e-3


def test_step_under_low_ceiling_is_not_climbed() -> None:
    """Потолок над ступенью ниже поднятой капсулы: шаг откатывается."""
    world = make_world()
    add_step(world, 0.25, x=STEP_X)
    # низ потолка на 1.9 над полом: капсула 1.8 + подъём 0.25 не помещается
    world.add_static(Collider(BoxShape((3.0, 0.5, 3.0))), (STEP_X + 1.0, 2.4, 0.0))
    char = make_char(world)
    settle(char)
    walk(char, (WALK, 0.0, 0.0), 1.2)
    top = float(char.position[1] + char.height / 2)
    assert top <= 1.9 + 1e-3 or char.position[0] < STEP_X


def test_step_up_before_cliff_rolls_back() -> None:
    """Уступ 0.25, за ним пустота глубже step_offset: на уступ не «телепортируемся»."""
    world = make_world()
    # узкая приподнятая плита и яма за ней (пол убран: плита вместо пола)
    world.add_static(Collider(BoxShape((0.2, 0.125, 3.0))), (STEP_X + 0.2, 0.125, 0.0))
    char = make_char(world)
    settle(char)
    for _ in range(120):
        char.simple_move((WALK, 0.0, 0.0))
        world.step()
    assert char.grounded
    assert char.position[1] - char.height / 2 < 0.3


def test_thin_low_fence_is_stepped_over_not_tunnelled() -> None:
    world = make_world()
    world.add_static(Collider(BoxShape((0.025, 0.125, 3.0))), (STEP_X + 0.5, 0.125, 0.0))
    char = make_char(world)
    settle(char)
    walk(char, (WALK, 0.0, 0.0), 1.5)
    assert char.position[0] > STEP_X + 1.5  # забор ниже step_offset — переступили
    assert char.grounded


# --- устойчивость ------------------------------------------------------------------

def test_standing_at_step_does_not_jitter() -> None:
    """Упираясь в непроходимую ступень, персонаж не «дрожит» по высоте."""
    world = make_world()
    add_step(world, 0.5, x=STEP_X)
    char = make_char(world)
    settle(char)
    walk(char, (WALK, 0.0, 0.0), 1.0)
    ys = []
    for _ in range(60):
        char.simple_move((WALK, 0.0, 0.0))
        world.step()
        ys.append(float(char.position[1]))
    assert max(ys) - min(ys) <= 0.02


def test_standing_still_next_to_step_does_not_move() -> None:
    world = make_world()
    add_step(world, 0.2, x=STEP_X)
    char = make_char(world, x=STEP_X - char_radius() - 0.1)
    settle(char)
    start = char.position.copy()
    settle(char, 60)
    assert abs(float(char.position[0] - start[0])) < 1e-6
    assert abs(float(char.position[1] - start[1])) < 1e-6


def char_radius() -> float:
    return 0.4
