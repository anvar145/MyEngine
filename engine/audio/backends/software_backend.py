"""Программный бэкенд без устройства: основа miniaudio-бэкенда и тестов.

Звук не выводится: блоки PCM забираются вызовом :meth:`advance` (или
``mixer.render``). Так тесты детерминированно «проигрывают» клип без
аудио-потока.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from engine.audio.backend import Vec3, VoiceHandle
from engine.audio.mixer import Mixer

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.audio.spatial import Attenuation

__all__ = ["SoftwareBackend"]


class SoftwareBackend:
    """Бэкенд поверх :class:`~engine.audio.mixer.Mixer` (стерео int16)."""

    def __init__(self) -> None:
        self.sample_rate = 44100
        self.channels = 2
        self.mixer = Mixer(self.sample_rate)
        self.initialized = False

    # ------------------------------------------------------- жизненный цикл
    def init(self, sample_rate: int = 44100, channels: int = 2) -> None:
        """Подготовить микшер (формат: только стерео)."""
        if channels != 2:
            raise ValueError("поддерживается только стерео (channels=2)")
        self.sample_rate = int(sample_rate)
        self.channels = 2
        self.mixer = Mixer(self.sample_rate)
        self.initialized = True

    def shutdown(self) -> None:
        """Сбросить все голоса."""
        self.initialized = False
        for voice_id in list(self.mixer._voices):  # noqa: SLF001 - свой микшер
            self.mixer.stop(voice_id)

    # ------------------------------------------------------------ голоса
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
        """Добавить голос (пустой PCM -> уже неиграющий дескриптор)."""
        voice_id = self.mixer.add(
            pcm,
            gain=gain,
            pan=pan,
            loop=loop,
            bus=bus,
            pitch=pitch,
            position=position,
            atten=attenuation,
        )
        return VoiceHandle(voice_id, self)

    def set_position(self, handle: VoiceHandle, position: Vec3) -> None:
        """Сменить позицию пространственного голоса."""
        self.mixer.set_position(handle.id, position)

    def set_listener(self, position: Vec3, forward: Vec3, up: Vec3) -> None:
        """Позиция и ориентация слушателя."""
        self.mixer.set_listener(position, forward, up)

    def stop(self, handle: VoiceHandle) -> None:
        """Остановить голос."""
        self.mixer.stop(handle.id)

    def set_gain(self, handle: VoiceHandle, gain: float) -> None:
        """Сменить громкость источника."""
        self.mixer.set_gain(handle.id, gain)

    def set_master_gain(self, gain: float) -> None:
        """Громкость мастера."""
        self.mixer.set_master(gain)

    def set_bus_gain(self, bus: str, gain: float) -> None:
        """Громкость шины (применяется ко всем голосам шины сразу)."""
        self.mixer.set_bus(bus, gain)

    def is_playing(self, handle: VoiceHandle) -> bool:
        """Голос ещё в микшере."""
        return handle.id != 0 and self.mixer.is_playing(handle.id)

    # ------------------------------------------------------------ тесты
    def advance(self, seconds: float) -> bytes:
        """Прокрутить микшер на ``seconds`` секунд и вернуть смешанный PCM."""
        return self.mixer.render(max(0, round(seconds * self.sample_rate)))
