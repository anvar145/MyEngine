"""Засыпание тел: остров спит, когда все его тела долго почти неподвижны."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from engine.physics.body import RigidBody

LINEAR_THRESHOLD = 0.05
ANGULAR_THRESHOLD = 0.05
TIME_TO_SLEEP = 0.5


@dataclass(slots=True)
class SleepSettings:
    """Пороги засыпания мира (``PhysicsWorld.sleep_settings``).

    Attributes:
        linear_threshold: Линейная скорость, ниже которой тело «покоится», м/с.
        angular_threshold: То же для угловой скорости, рад/с.
        time_to_sleep: Сколько секунд покоя нужно острову, чтобы уснуть.
    """

    linear_threshold: float = LINEAR_THRESHOLD
    angular_threshold: float = ANGULAR_THRESHOLD
    time_to_sleep: float = TIME_TO_SLEEP


def is_resting(
    body: RigidBody,
    linear: float = LINEAR_THRESHOLD,
    angular: float = ANGULAR_THRESHOLD,
) -> bool:
    """Тело почти неподвижно (по квадрату скорости)."""
    return bool(
        body.linear_velocity @ body.linear_velocity < linear * linear
        and body.angular_velocity @ body.angular_velocity < angular * angular
    )


def update_island_sleep(
    bodies: Iterable[RigidBody],
    dt: float,
    settings: SleepSettings | None = None,
) -> bool:
    """Обновляет таймеры острова; усыпляет его целиком, когда все созрели.

    Args:
        bodies: Тела одного острова.
        dt: Длительность шага, с.
        settings: Пороги сна; ``None`` — значения по умолчанию.

    Returns:
        ``True``, если остров уснул на этом шаге.
    """
    cfg = settings if settings is not None else SleepSettings()
    group = list(bodies)
    min_timer = np.inf
    for body in group:
        if not body.can_sleep or not is_resting(
            body, cfg.linear_threshold, cfg.angular_threshold
        ):
            body.sleep_timer = 0.0
            min_timer = 0.0
        else:
            body.sleep_timer += dt
            min_timer = min(min_timer, body.sleep_timer)
    if group and min_timer >= cfg.time_to_sleep:
        for body in group:
            body.sleeping = True
            body.linear_velocity = np.zeros(3)
            body.angular_velocity = np.zeros(3)
        return True
    return False
