"""Патруль: объекты ходят туда-сюда вдоль оси X по синусоиде.

Пакетный хук ``on_fixed_update_batch``: сцена вызывает его один раз на все
объекты с этим скриптом, а не по разу на объект. Состояния сцены в модуле нет:
смещение интегрируется из производной синуса, фаза берётся из имени объекта
(разная у разных врагов); индексы и фазы кэшируются на время жизни набора.
"""

from __future__ import annotations

import numpy as np

from engine import debug

AMPLITUDE = 3.0
OMEGA = 0.8

_cache = None


def _prepare(items):
    """Индексы трансформов и фазы для набора объектов (кэш по составу)."""
    global _cache
    key = tuple(item.game_object.id for item in items)
    if _cache is None or _cache[0] != key:
        indices = np.array([item.transform.index for item in items], dtype=np.intp)
        phases = np.array(
            [float(sum(map(ord, item.game_object.name)) % 17) for item in items]
        )
        _cache = (key, indices, phases)
    return _cache[1], _cache[2]


def on_fixed_update_batch(self, items, dt):
    """Шаг на фиксированной сетке: скорость kinematic-тела постоянна по кадрам."""
    indices, phases = _prepare(items)
    now = self.time.fixed_frame * self.time.fixed_delta
    speed = AMPLITUDE * OMEGA * np.cos(OMEGA * now + phases)
    store = self.scene.transforms
    positions, _ = store.get_local_batch(indices)
    positions = positions.astype(np.float64)
    positions[:, 0] += speed * dt
    store.set_local_batch(indices, positions)
    if debug.overlays.visible:
        _draw_debug(positions, speed, OMEGA * now + phases, dt)


def _draw_debug(positions, speed, angles, dt):
    """F3: отрезок патруля (пути) и конус «зрения» по ходу движения.

    Шаг фиксированный, а кадров может быть больше: ключ заменяет фигуру врага
    прошлого шага, а ``duration`` держит её между шагами.
    """
    centers = positions[:, 0] - AMPLITUDE * np.sin(angles)
    life = dt * 4.0
    for i, (pos, center, v) in enumerate(zip(positions, centers, speed, strict=True)):
        y, z = pos[1], pos[2]
        debug.path([(center - AMPLITUDE, y, z), (center + AMPLITUDE, y, z)],
                   duration=life, key=("patrol_path", i))
        forward = (1.0 if v >= 0 else -1.0, 0.0, 0.0)
        debug.cone(pos, forward, 20.0, 2.5, duration=life, key=("patrol_sight", i))
