"""Динамическое разрешение: подбор ``render_scale`` под время кадра (A10a).

Чистая логика без GL. Хост подаёт длительность кадра в миллисекундах; контроллер
сглаживает её и двигает масштаб шагами по 0.05 в границах ``floor..ceiling``.
Вниз — быстро (просадка заметна сразу), вверх — медленно (запас на 25 %, долгое
окно и пауза после смены), чтобы масштаб не дрожал.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from engine.rendering.post import RenderScaleHints

__all__ = [
    "TARGET_FRAME_MS",
    "DynamicResolution",
    "read_hdr_hint",
    "read_render_hints",
]

_log = logging.getLogger("engine.rendering.dynamic_resolution")

#: Целевое время кадра: 16.6 мс (60 fps) с запасом 10 %.
TARGET_FRAME_MS = 16.6 * 0.9

STEP = 0.05
SMOOTHING = 0.2
DOWN_THRESHOLD = 1.05  # сглаженный кадр дольше цели на 5 %
UP_THRESHOLD = 0.75  # короче цели на 25 %
DOWN_FRAMES = 4
UP_FRAMES = 90
DOWN_COOLDOWN = 12
UP_COOLDOWN = 60


class DynamicResolution:
    """Контроллер масштаба рендера.

    Attributes:
        scale: Текущий масштаб.
        enabled: Двигать ли масштаб (``False`` — только хранит значение).
    """

    def __init__(
        self,
        hints: RenderScaleHints,
        *,
        target_ms: float = TARGET_FRAME_MS,
    ) -> None:
        self.hints = hints
        self.target_ms = float(target_ms)
        self.enabled = hints.dynamic
        self.scale = self._clamp(hints.scale)
        self._avg = 0.0
        self._over = 0
        self._under = 0
        self._cooldown = 0

    def _clamp(self, value: float) -> float:
        low = min(self.hints.floor, self.hints.ceiling)
        high = max(self.hints.floor, self.hints.ceiling)
        return round(round(min(high, max(low, value)) / STEP) * STEP, 4)

    def set_scale(self, value: float) -> None:
        """Задать масштаб вручную (``--render-scale``), сбросив историю замеров."""
        self.scale = round(min(1.0, max(0.25, float(value))), 4)
        self.hints = RenderScaleHints(
            self.scale,
            min(self.hints.floor, self.scale),
            max(self.hints.ceiling, self.scale),
            self.hints.dynamic,
        )
        self._avg = 0.0
        self._over = self._under = 0

    def update(self, frame_ms: float) -> float | None:
        """Учесть длительность кадра.

        Args:
            frame_ms: Время кадра (CPU + GPU) в миллисекундах.

        Returns:
            Новый масштаб, если он изменился, иначе ``None``.
        """
        if not self.enabled:
            return None
        frame_ms = max(0.0, float(frame_ms))
        self._avg = frame_ms if self._avg == 0.0 else (
            self._avg + (frame_ms - self._avg) * SMOOTHING
        )
        if self._cooldown > 0:
            self._cooldown -= 1
            return None
        if self._avg > self.target_ms * DOWN_THRESHOLD:
            self._over += 1
            self._under = 0
        elif self._avg < self.target_ms * UP_THRESHOLD:
            self._under += 1
            self._over = 0
        else:
            self._over = self._under = 0

        if self._over >= DOWN_FRAMES:
            return self._move(-STEP, DOWN_COOLDOWN)
        if self._under >= UP_FRAMES:
            return self._move(STEP, UP_COOLDOWN)
        return None

    def _move(self, delta: float, cooldown: int) -> float | None:
        self._over = self._under = 0
        new = self._clamp(self.scale + delta)
        if abs(new - self.scale) < 1e-9:
            return None
        self.scale = new
        self._cooldown = cooldown
        # Сглаженное значение больше не соответствует новому разрешению.
        self._avg = 0.0
        return new


def _project_texts(project_path: str | Path, vfs: object | None) -> list[str]:
    """Тексты ``*.project.json``: с диска, иначе из пака (``vfs``)."""
    texts: list[str] = []
    for file in sorted(Path(project_path).glob("*.project.json")):
        try:
            texts.append(file.read_text(encoding="utf-8"))
        except OSError:
            continue
    if not texts and vfs is not None:
        for key in vfs.list_files(""):  # type: ignore[attr-defined]
            if "/" not in key and key.endswith(".project.json"):
                data = vfs.read_file(key)  # type: ignore[attr-defined]
                if data is not None:
                    texts.append(data.decode("utf-8", errors="replace"))
    return texts


def read_hdr_hint(
    project_path: str | Path, vfs: object | None = None
) -> bool | None:
    """Поле ``hdr`` из ``*.project.json`` в три состояния.

    Returns:
        ``None`` - поля нет (сцена с ``post`` может включить HDR сама);
        ``True``/``False`` - явное решение автора проекта.
    """
    for text in _project_texts(project_path, vfs):
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if isinstance(data, dict):
            value = data.get("hdr")
            return value if isinstance(value, bool) else None
    return None


def read_render_hints(
    project_path: str | Path, vfs: object | None = None
) -> tuple[float | None, bool | None, bool]:
    """Прочитать из ``*.project.json`` подсказки масштаба и флаг HDR.

    Args:
        project_path: Корень проекта.
        vfs: Источник файлов проекта (пак), если проекта нет на диске.

    Returns:
        ``(render_scale_hint, dynamic_resolution, hdr)``: отсутствующие поля —
        ``None`` (берётся профиль качества), ``hdr`` без поля — ``False``.
    """
    hint: float | None = None
    dynamic: bool | None = None
    hdr = False
    for text in _project_texts(project_path, vfs):
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        value = data.get("render_scale_hint")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            hint = min(1.0, max(0.25, float(value)))
        if isinstance(data.get("dynamic_resolution"), bool):
            dynamic = data["dynamic_resolution"]
        hdr = bool(data.get("hdr", False))
        break
    return hint, dynamic, hdr
