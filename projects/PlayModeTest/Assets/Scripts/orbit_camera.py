"""Двигает камеру по кругу вокруг центра сцены, глядя на него."""

from __future__ import annotations

import math

from engine.core.math3d import Quaternion, Vector3

CENTER = Vector3(0.0, 1.0, 0.0)
RADIUS = 7.2
HEIGHT = 3.0
SPEED = 0.25  # радиан в секунду
START_ANGLE = math.atan2(4.0, 6.0)  # совпадает со стартовой позицией (4, 3, 6)

_angle = START_ANGLE


def look_rotation(forward):
    """Ориентация «рысканье, затем тангаж» — без крена горизонта."""
    direction = forward.normalized
    yaw = math.degrees(math.atan2(direction.x, direction.z))
    pitch = math.degrees(math.asin(-direction.y))
    return Quaternion.from_axis_angle(
        Vector3(0.0, 1.0, 0.0), yaw
    ) * Quaternion.from_axis_angle(Vector3(1.0, 0.0, 0.0), pitch)


def on_start(self):
    global _angle
    _angle = START_ANGLE
    self.log("orbit_camera: старт")


def on_update(self, delta_time):
    global _angle
    _angle += SPEED * delta_time
    position = Vector3(
        CENTER.x + RADIUS * math.sin(_angle),
        HEIGHT,
        CENTER.z + RADIUS * math.cos(_angle),
    )
    self.transform.position = position
    forward = Vector3(
        CENTER.x - position.x, CENTER.y - position.y, CENTER.z - position.z
    )
    self.transform.rotation = look_rotation(forward)
