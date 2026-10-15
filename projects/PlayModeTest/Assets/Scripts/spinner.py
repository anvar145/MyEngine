"""Вращает объект вокруг вертикальной оси с постоянной скоростью."""

from __future__ import annotations

from engine.core.math3d import Vector3

SPEED_DEGREES = 60.0


def on_start(self):
    self.log("spinner: старт")


def on_update(self, delta_time):
    self.transform.rotate(Vector3(0.0, 1.0, 0.0), SPEED_DEGREES * delta_time)
