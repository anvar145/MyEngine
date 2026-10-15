"""Протокол аудио-бэкенда и дескриптор голоса.

Бэкенд проигрывает готовый PCM (int16, interleaved) **в своём формате**
(``sample_rate``/``channels`` из ``init``): приведение клипа к этому формату
делает :meth:`engine.audio.clip.AudioClip.to_format`. Громкость голоса
итоговая: ``min(1, source * bus * master)``; смена шины или мастера
пересчитывается бэкендом сразу для всех активных голосов.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.audio.spatial import Attenuation

Vec3 = tuple[float, float, float]

__all__ = [
    "BUSES",
    "AudioBackendError",
    "IAudioBackend",
    "VoiceHandle",
    "Vec3",
]

#: Шины микшера. ``master`` — общий множитель, остальные — группы голосов.
BUSES: tuple[str, ...] = ("master", "music", "sfx", "voice", "ui")


class AudioBackendError(RuntimeError):
    """Бэкенд не удалось создать или инициализировать (нет устройства/пакета)."""


class VoiceHandle:
    """Дескриптор играющего голоса.

    Args:
        voice_id: Идентификатор голоса в бэкенде (``0`` — «нет голоса»).
        backend: Бэкенд, которому принадлежит голос.

    Attributes:
        id: Идентификатор голоса.
    """

    __slots__ = ("_backend", "id")

    def __init__(self, voice_id: int, backend: IAudioBackend | None = None) -> None:
        self.id = int(voice_id)
        self._backend = backend

    @property
    def is_playing(self) -> bool:
        """Голос ещё звучит."""
        return self._backend is not None and self._backend.is_playing(self)

    def stop(self) -> None:
        """Остановить голос (повторный вызов безопасен)."""
        if self._backend is not None:
            self._backend.stop(self)

    def set_gain(self, gain: float) -> None:
        """Сменить громкость источника голоса (до шины и мастера)."""
        if self._backend is not None:
            self._backend.set_gain(self, gain)

    def set_position(self, position: Vec3) -> None:
        """Сменить мировую позицию пространственного голоса."""
        if self._backend is not None:
            self._backend.set_position(self, position)

    def __repr__(self) -> str:
        return f"VoiceHandle({self.id})"


@runtime_checkable
class IAudioBackend(Protocol):
    """Аудио-бэкенд: устройство вывода и микшер голосов."""

    sample_rate: int
    channels: int

    def init(self, sample_rate: int = 44100, channels: int = 2) -> None:
        """Открыть устройство вывода.

        Raises:
            AudioBackendError: Устройства или библиотеки нет.
        """

    def shutdown(self) -> None:
        """Остановить вывод и освободить устройство (повторный вызов безопасен)."""

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
        """Начать воспроизведение PCM в формате бэкенда.

        Args:
            pcm: int16, interleaved, ``channels`` каналов, ``sample_rate`` Гц.
            gain: Громкость источника, ≥ 0.
            pan: Панорама -1 (лево) … +1 (право); для 3D-голоса игнорируется.
            loop: Зациклить.
            bus: Шина микшера.
            pitch: Множитель скорости/высоты, > 0.
            position: Мировая позиция; не ``None`` — пространственный голос:
                бэкенд сам считает затухание и панораму от слушателя.
            attenuation: Параметры затухания 3D-голоса (по умолчанию 1…50, linear).
        """

    def stop(self, handle: VoiceHandle) -> None:
        """Остановить голос."""

    def set_gain(self, handle: VoiceHandle, gain: float) -> None:
        """Сменить громкость источника голоса."""

    def set_position(self, handle: VoiceHandle, position: Vec3) -> None:
        """Сменить мировую позицию пространственного голоса."""

    def set_listener(self, position: Vec3, forward: Vec3, up: Vec3) -> None:
        """Мировые позиция и ориентация слушателя (вправо = ``up × forward``)."""

    def set_master_gain(self, gain: float) -> None:
        """Громкость мастера, 0…1."""

    def set_bus_gain(self, bus: str, gain: float) -> None:
        """Громкость шины, 0…1."""

    def is_playing(self, handle: VoiceHandle) -> bool:
        """Голос ещё звучит."""
