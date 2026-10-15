"""Бросок шара по клавише F: создаёт физический шар и толкает его вперёд.

Висит на объекте-персонаже. «Вперёд» — последнее направление ходьбы (по
умолчанию -Z). Одновременно живёт не больше ``MAX_BALLS`` шаров: самый старый
удаляется.
"""

from __future__ import annotations

from engine.core.components import MeshRenderer
from engine.core.math3d import Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent

THROW_SPEED = 12.0
BALL_RADIUS = 0.25
MAX_BALLS = 8

balls = []
was_down = False
direction = (0.0, 0.0, -1.0)


def _walk_direction(keys):
    dx = (1.0 if keys.is_key_down("d") else 0.0) - (
        1.0 if keys.is_key_down("a") else 0.0
    )
    dz = (1.0 if keys.is_key_down("s") else 0.0) - (
        1.0 if keys.is_key_down("w") else 0.0
    )
    length = (dx * dx + dz * dz) ** 0.5
    return None if length == 0.0 else (dx / length, 0.0, dz / length)


def on_fixed_update(self, delta_time):
    global was_down, direction
    walking = _walk_direction(self.input)
    if walking is not None:
        direction = walking
    down = self.input.is_key_down("f")
    if down and not was_down:
        _throw(self)
    was_down = down


def _throw(self):
    pos = self.transform.position
    dx, _, dz = direction
    spawn = Vector3(pos.x + dx * 0.9, pos.y + 0.6, pos.z + dz * 0.9)
    diameter = BALL_RADIUS * 2
    ball = self.scene.create_game_object(
        "ThrownBall", position=spawn, scale=Vector3(diameter, diameter, diameter)
    )
    ball.add_component(MeshRenderer("sphere", color=Vector3(0.95, 0.35, 0.2)))
    ball.add_component(ColliderComponent, shape="sphere", radius=0.5, restitution=0.4)
    ball.add_component(RigidBodyComponent, mass=0.8)
    self.physics.set_velocity((dx * THROW_SPEED, 2.5, dz * THROW_SPEED), target=ball)
    balls.append(ball)
    while len(balls) > MAX_BALLS:
        old = balls.pop(0)
        if not old.is_destroyed:
            self.scene.destroy_game_object(old)
