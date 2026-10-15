"""Пустой бэкенд: ничего не играет (headless, ``--no-audio``, нет устройства)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from engine.audio.backend import Vec3, VoiceHandle

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.audio.spatial import Attenuation

__all__ = ["NullBackend"]


class NullBackend:
    """Все методы — no-op, ``is_playing`` всегда ``False``."""

    def __init__(self) -> None:
        self.sample_rate = 44100
        self.channels = 2

    def init(self, sample_rate: int = 44100, channels: int = 2) -> None:
        """Запомнить формат."""
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)

    def shutdown(self) -> None:
        """Ничего."""

    def play(
        self,
        pcm: bytes,
        *,
        gain: float,
        pan: float,
        loop: bool = False,
        bus: str = "sfx",
        pitch: float = 1.0,
        position: Vec3 | None = None,
        attenuation: Attenuation | None = None,
    ) -> VoiceHandle:
        """Вернуть неиграющий дескриптор."""
        return VoiceHandle(0, self)

    def set_position(self, handle: VoiceHandle, position: Vec3) -> None:
        """Ничего."""

    def set_listener(self, position: Vec3, forward: Vec3, up: Vec3) -> None:
        """Ничего."""

    def stop(self, handle: VoiceHandle) -> None:
        """Ничего."""

    def set_gain(self, handle: VoiceHandle, gain: float) -> None:
        """Ничего."""

    def set_master_gain(self, gain: float) -> None:
        """Ничего."""

    def set_bus_gain(self, bus: str, gain: float) -> None:
        """Ничего."""

    def is_playing(self, handle: VoiceHandle) -> bool:
        """Всегда ``False``."""
        return False
