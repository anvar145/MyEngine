"""Общие помощники верификации CharacterController (не тесты).

Пол — статический ящик с верхней гранью на y=0. Персонаж по умолчанию
(``radius=0.4``, ``height=1.8``, ``skin_width=0.02``) стоит с центром на
``y = height/2 + skin_width = 0.92``.
"""

from __future__ import annotations

import math

import numpy as np

from engine.physics.character import CharacterController
from engine.physics.collider import Collider
from engine.physics.mathutil import quat_from_axis_angle
from engine.physics.shape import BoxShape
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0
WALK = 4.5  # скорость ходьбы из player_controller.py
HEIGHT = 1.8


def make_world(gravity: float = -9.81) -> PhysicsWorld:
    """Мир с плоским полом (верхняя грань y=0)."""
    world = PhysicsWorld(gravity=(0.0, gravity, 0.0))
    world.add_static(Collider(BoxShape((50.0, 0.5, 50.0))), (0.0, -0.5, 0.0))
    return world


def make_char(world, x: float = 0.0, y: float = 0.95, z: float = 0.0,
              **kwargs) -> CharacterController:
    """Персонаж чуть выше пола (по умолчанию ``height=1.8``)."""
    return CharacterController(world, (x, y, z), **kwargs)


def foot_y(char: CharacterController) -> float:
    """Высота нижнего полюса капсулы: ``position[1] - height / 2``."""
    return float(char.position[1] - char.height / 2.0)


def settle(char: CharacterController, steps: int = 30) -> None:
    for _ in range(steps):
        char.simple_move((0.0, 0.0, 0.0))
        char.world.step()


def walk(char: CharacterController, velocity, seconds: float) -> None:
    for _ in range(round(seconds * 60)):
        char.simple_move(velocity)
        char.world.step()


def add_wall(world, x: float = 5.0, half_thickness: float = 0.5) -> float:
    """Стена, перпендикулярная X; возвращает координату её ближней грани."""
    world.add_static(Collider(BoxShape((half_thickness, 3.0, 10.0))),
                     (x, 3.0, 0.0))
    return x - half_thickness


def add_step(world, height: float, x: float = 3.0, depth: float = 2.0) -> None:
    """Прямоугольный уступ высотой ``height`` с ближней гранью на ``x``."""
    world.add_static(Collider(BoxShape((depth, height / 2.0, 3.0))),
                     (x + depth, height / 2.0, 0.0))


def add_slope(world, degrees: float, length: float = 8.0, x0: float = 2.0) -> None:
    """Наклонная плоскость, поднимающаяся вдоль +X от точки ``x0`` на полу."""
    t = math.radians(degrees)
    rot = quat_from_axis_angle((0.0, 0.0, 1.0), t)
    centre = np.array([x0 + length * math.cos(t), length * math.sin(t), 0.0])
    centre -= 0.5 * np.array([-math.sin(t), math.cos(t), 0.0])
    world.add_static(Collider(BoxShape((length, 0.5, 5.0))), centre, rot)
