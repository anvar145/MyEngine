"""SoundCue: набор вариантов клипа со случайным выбором (``*.cue.json``).

Формат файла::

    {
      "clips": [{"clip": {"guid": ..., "path": ...}, "gain": 1.0,
                 "pitch": 1.0, "weight": 1.0}, ...],
      "randomize_gain": 0.1,
      "randomize_pitch": 0.05,
      "avoid_repeat": true
    }

Выбор детерминирован относительно переданного генератора
(``scene.random``): при одинаковом зерне и порядке вызовов последовательность
вариантов повторяется.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from engine.audio.asset_json import AudioAssetError, load_json_asset, ref_from_json
from engine.core.resources.asset_ref import AssetRef

__all__ = ["CueClip", "CueVariant", "SoundCue", "load_cue", "parse_cue"]

_log = logging.getLogger("engine.audio.cue")

#: Роль артефакта в ``Library`` (файл ``cue.json``).
ARTIFACT = "sound_cue"


@dataclass(frozen=True, slots=True)
class CueClip:
    """Вариант клипа в кью."""

    ref: AssetRef
    gain: float = 1.0
    pitch: float = 1.0
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class CueVariant:
    """Результат выбора: клип с итоговыми множителями громкости и высоты."""

    ref: AssetRef
    gain: float
    pitch: float
    index: int


@dataclass(slots=True)
class SoundCue:
    """Кью: варианты, разброс и запрет повтора подряд.

    Args:
        clips: Варианты (минимум один).
        randomize_gain: Относительный разброс громкости ``±``.
        randomize_pitch: Относительный разброс высоты ``±``.
        avoid_repeat: Не выбирать подряд один и тот же вариант.
    """

    clips: tuple[CueClip, ...]
    randomize_gain: float = 0.0
    randomize_pitch: float = 0.0
    avoid_repeat: bool = True
    _last: int = field(default=-1, repr=False)

    def pick(self, rng: random.Random) -> CueVariant:
        """Выбрать вариант.

        Вес — вероятность выбора; при ``avoid_repeat`` и нескольких вариантах
        прошлый исключается из выборки. На каждый вызов тратится ровно три
        обращения к ``rng`` (выбор, разброс громкости, разброс высоты).
        """
        indices = list(range(len(self.clips)))
        if self.avoid_repeat and len(indices) > 1 and self._last in indices:
            indices.remove(self._last)
        weights = [self.clips[i].weight for i in indices]
        index = rng.choices(indices, weights=weights)[0]
        spread_gain = rng.uniform(-1.0, 1.0) * self.randomize_gain
        spread_pitch = rng.uniform(-1.0, 1.0) * self.randomize_pitch
        self._last = index
        clip = self.clips[index]
        return CueVariant(
            clip.ref,
            max(0.0, clip.gain * (1.0 + spread_gain)),
            max(0.01, clip.pitch * (1.0 + spread_pitch)),
            index,
        )


def _number(data: dict[str, Any], key: str, default: float, low: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioAssetError(f"{key}: ожидается число")
    if value < low:
        raise AudioAssetError(f"{key}: значение {value} меньше {low}")
    return float(value)


def parse_cue(data: dict[str, Any]) -> SoundCue:
    """Собрать :class:`SoundCue` из словаря файла.

    Raises:
        AudioAssetError: Нет вариантов, неверные типы, нулевая сумма весов.
    """
    raw = data.get("clips")
    if not isinstance(raw, list) or not raw:
        raise AudioAssetError("clips: нужен непустой список вариантов")
    clips: list[CueClip] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AudioAssetError(f"clips[{i}]: ожидается объект")
        clips.append(
            CueClip(
                ref_from_json(item.get("clip"), f"clips[{i}].clip"),
                _number(item, "gain", 1.0, 0.0),
                _number(item, "pitch", 1.0, 0.01),
                _number(item, "weight", 1.0, 0.0),
            )
        )
    if sum(c.weight for c in clips) <= 0.0:
        raise AudioAssetError("clips: сумма весов должна быть больше нуля")
    return SoundCue(
        tuple(clips),
        min(1.0, _number(data, "randomize_gain", 0.0, 0.0)),
        min(1.0, _number(data, "randomize_pitch", 0.0, 0.0)),
        bool(data.get("avoid_repeat", True)),
    )


def load_cue(vfs: Any, ref: AssetRef | str | dict | None) -> SoundCue | None:
    """Загрузить кью по ссылке; при ошибке — ``None`` и WARNING."""
    data = load_json_asset(vfs, ref, ARTIFACT)
    if data is None:
        return None
    try:
        return parse_cue(data)
    except AudioAssetError as exc:
        asset = AssetRef.coerce(ref)
        _log.warning("SoundCue %s: %s", asset.path if asset else "?", exc)
        return None
