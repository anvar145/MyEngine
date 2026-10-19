"""Настройки рендеринга и профили качества.

Модуль не зависит от Qt и OpenGL: это чистые данные, которые используют и
редактор (для формата поверхности), и плеер, и headless-рендер.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from engine.core.frame_budget import FrameBudget

__all__ = ["QualityTier", "RenderSettings", "ShadowSettings"]


class QualityTier(str, Enum):
    """Профиль качества: определяет, какие эффекты включены."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def shadow_map_size(self) -> int:
        """Размер карты теней для профиля."""
        return {"low": 512, "medium": 1024, "high": 2048}[self.value]

    @property
    def shadow_cascades(self) -> int:
        """Число каскадов теней для профиля."""
        return {"low": 1, "medium": 2, "high": 3}[self.value]

    @property
    def msaa_samples(self) -> int:
        """Число сэмплов MSAA для профиля (0 — выключено)."""
        return {"low": 0, "medium": 2, "high": 4}[self.value]

    @property
    def enable_post_effects(self) -> bool:
        """Включены ли пост-эффекты (tone map, bloom и т. п.)."""
        return self is not QualityTier.LOW

    @property
    def budget(self) -> FrameBudget:
        """Бюджет кадра игровых подсистем профиля (анимация, ИИ, физика)."""
        return FrameBudget.for_tier(self.value)


@dataclass(slots=True)
class ShadowSettings:
    """Параметры карт теней (используются начиная с прохода теней)."""

    enabled: bool = True
    map_size: int = 1024
    cascades: int = 2
    depth_bias: float = 0.0015
    normal_bias: float = 0.02
    distance: float = 60.0


@dataclass(slots=True)
class RenderSettings:
    """Параметры контекста и кадра.

    Значения по умолчанию рассчитаны на OpenGL 3.3 Core — минимум, который
    обещает движок (интегрированная графика уровня Intel HD).

    Attributes:
        major: Старшая версия OpenGL.
        minor: Младшая версия OpenGL.
        core_profile: Использовать Core Profile (обязательно для 3.2+).
        forward_compatible: Запросить forward-compatible контекст.
        depth_bits: Разрядность буфера глубины.
        stencil_bits: Разрядность буфера трафарета.
        samples: Число сэмплов MSAA по умолчанию (0 — выключено).
        swap_interval: Обмен буферов: 0 — без синхронизации, 1 — VSync.
        srgb: Запросить sRGB-capable буфер кадра.
        quality: Профиль качества по умолчанию.
        hdr: Рисовать кадр в HDR-цель (RGBA16F) для пост-обработки; нужен
            ``post=True`` в ``Renderer.begin_frame``.
    """

    major: int = 3
    minor: int = 3
    core_profile: bool = True
    forward_compatible: bool = False
    depth_bits: int = 24
    stencil_bits: int = 8
    samples: int = 0
    swap_interval: int = 1
    srgb: bool = True
    quality: QualityTier = QualityTier.MEDIUM
    hdr: bool = False

    @classmethod
    def for_quality(cls, tier: QualityTier, **overrides: object) -> RenderSettings:
        """Собрать настройки под профиль качества.

        Args:
            tier: Профиль качества.
            **overrides: Точечные переопределения полей.

        Returns:
            Настройки рендеринга.
        """
        settings = cls(samples=tier.msaa_samples, quality=tier)
        for name, value in overrides.items():
            setattr(settings, name, value)
        return settings

    def describe(self) -> str:
        """Короткое описание для логов и статистики."""
        profile = "core" if self.core_profile else "compat"
        return (
            f"GL {self.major}.{self.minor} {profile}, depth {self.depth_bits}, "
            f"stencil {self.stencil_bits}, msaa x{self.samples}, "
            f"vsync {self.swap_interval}, quality {self.quality.value}"
        )
