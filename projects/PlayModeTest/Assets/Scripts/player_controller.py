"""Управление персонажем: WASD — ходьба, Space — прыжок.

Скрипт висит на объекте с ``CharacterControllerComponent``. Движение идёт
в фиксированном шаге физики (``on_fixed_update``); гравитацию, ступени и
склоны решает сам контроллер.
"""

from __future__ import annotations

WALK_SPEED = 4.5
JUMP_SPEED = 5.5

# Состояние скрипта живёт в глобальных именах модуля: у компонента нет __dict__.
was_jump_down = False


def on_start(self):
    self.log("player_controller: WASD — ходьба, Space — прыжок, F — бросок шара")


def on_fixed_update(self, delta_time):
    global was_jump_down
    keys = self.input
    dx = (1.0 if keys.is_key_down("d") else 0.0) - (
        1.0 if keys.is_key_down("a") else 0.0
    )
    dz = (1.0 if keys.is_key_down("s") else 0.0) - (
        1.0 if keys.is_key_down("w") else 0.0
    )
    length = (dx * dx + dz * dz) ** 0.5
    if length > 0.0:
        dx, dz = dx / length, dz / length
    jump_down = keys.is_key_down("space")
    if jump_down and not was_jump_down and self.physics.is_grounded():
        self.physics.jump(JUMP_SPEED)
    was_jump_down = jump_down
    self.physics.simple_move((dx * WALK_SPEED, 0.0, dz * WALK_SPEED))
