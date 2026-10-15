"""Пространственный звук: затухание по расстоянию и стерео-панорама.

Чистые функции на numpy без состояния: принимают как скаляры, так и массивы,
поэтому микшер считает все 3D-голоса за один проход.

Модели затухания — как distance models в OpenAL; расстояние ограничивается
отрезком ``[min_distance, max_distance]``, внутри ``min_distance`` голос
звучит на полную громкость.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FALLOFF_MODELS",
    "Attenuation",
    "attenuation",
    "pan_gains",
    "pan_of",
]

#: Допустимые модели затухания.
FALLOFF_MODELS: tuple[str, ...] = ("linear", "logarithmic", "inverse")

_MIN_SPAN = 1e-6


@dataclass(frozen=True, slots=True)
class Attenuation:
    """Параметры затухания голоса.

    Args:
        min_distance: Дистанция полной громкости, > 0.
        max_distance: Дистанция, за которой линейное затухание даёт тишину.
        falloff: ``linear``/``logarithmic``/``inverse``.
        rolloff: Крутизна логарифмической кривой.
    """

    min_distance: float = 1.0
    max_distance: float = 50.0
    falloff: str = "linear"
    rolloff: float = 1.0


def attenuation(
    distance: float | np.ndarray,
    min_distance: float = 1.0,
    max_distance: float = 50.0,
    falloff: str = "linear",
    rolloff: float = 1.0,
) -> float | np.ndarray:
    """Множитель громкости ``0…1`` по расстоянию до слушателя.

    * ``linear``: ``1 - (d - min) / (max - min)`` (за ``max`` — ``0``);
    * ``logarithmic``: ``1 / (1 + rolloff * ln(d / min))``;
    * ``inverse``: ``min / d``.

    Для ``logarithmic`` и ``inverse`` расстояние зажимается в
    ``[min, max]`` (как в OpenAL): за ``max`` громкость не падает дальше.

    Args:
        distance: Расстояние (скаляр или массив).
        min_distance: Дистанция полной громкости.
        max_distance: Предельная дистанция.
        falloff: Модель затухания.
        rolloff: Крутизна логарифмической кривой.

    Raises:
        ValueError: Неизвестная модель.
    """
    if falloff not in FALLOFF_MODELS:
        raise ValueError(f"неизвестная модель затухания: {falloff!r}")
    near = max(float(min_distance), _MIN_SPAN)
    far = max(float(max_distance), near + _MIN_SPAN)
    d = np.asarray(distance, dtype=np.float64)
    if falloff == "linear":
        result = 1.0 - (d - near) / (far - near)
    else:
        clamped = np.clip(d, near, far)
        if falloff == "logarithmic":
            result = 1.0 / (1.0 + max(0.0, float(rolloff)) * np.log(clamped / near))
        else:
            result = near / clamped
    result = np.clip(result, 0.0, 1.0)
    return float(result) if result.ndim == 0 else result


def pan_of(
    offset: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Панорама ``sin(азимут)`` ``-1…1`` для векторов «слушатель → источник».

    Args:
        offset: ``(n, 3)`` смещения источников относительно слушателя.
        right: Единичный вектор «вправо» слушателя ``(3,)``.

    Returns:
        ``(n,)``; источник в точке слушателя даёт ``0``.
    """
    length = np.linalg.norm(offset, axis=1)
    safe = np.where(length > 1e-9, length, 1.0)
    pan = (offset @ right) / safe
    pan = np.where(length > 1e-9, pan, 0.0)
    return np.clip(pan, -1.0, 1.0)


def pan_gains(pan: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Множители левого и правого каналов для панорамы ``pan`` (-1…1).

    При ``pan = 0`` оба равны ``1`` (центр не просаживается), крайняя
    сторона гасит противоположный канал.
    """
    p = np.clip(np.asarray(pan, dtype=np.float64), -1.0, 1.0)
    return np.minimum(1.0, 1.0 - p), np.minimum(1.0, 1.0 + p)
