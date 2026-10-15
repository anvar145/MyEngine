"""Аудио для скриптов: ``self.audio`` (фасад над :class:`AudioSystem` сцены).

Без ``AudioSystem`` в сцене (редактор, тесты) все вызовы безопасны и
ничего не играют: ``play`` возвращает «мёртвый» дескриптор.

Example:
    >>> handle = self.audio.play(self.bell, gain=0.8, bus="sfx")
    >>> self.audio.set_bus_gain("music", 0.5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from engine.audio.backend import BUSES, VoiceHandle
from engine.audio.system import AudioSystem

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.scripting.script_component import ScriptComponent

__all__ = ["ScriptAudio", "ScriptMusic"]


def _vec3(value: Any) -> tuple[float, float, float]:
    """``Vector3``/последовательность -> ``(x, y, z)``."""
    if hasattr(value, "x"):
        return (float(value.x), float(value.y), float(value.z))
    x, y, z = value
    return (float(x), float(y), float(z))


class ScriptMusic:
    """``self.audio.music``: плейлисты, кроссфейд, интенсивность."""

    __slots__ = ("_audio",)

    def __init__(self, audio: ScriptAudio) -> None:
        self._audio = audio

    def _player(self) -> Any:
        system = self._audio._system()  # noqa: SLF001
        return None if system is None else system.music

    def play(self, playlist: Any, crossfade: float = 1.0, bus: str = "music") -> bool:
        """Запустить ``*.music.json`` (прежняя музыка уходит кроссфейдом)."""
        player = self._player()
        return False if player is None else player.play(playlist, crossfade, bus)

    def stop(self, crossfade: float = 1.0) -> None:
        """Остановить музыку с затуханием."""
        player = self._player()
        if player is not None:
            player.stop(crossfade)

    def set_intensity(self, value: float) -> None:
        """Интенсивность ``0…1`` (громкость «тревожного» стема)."""
        player = self._player()
        if player is not None:
            player.set_intensity(value)

    @property
    def current_track(self) -> Any:
        """Звучащий трек (``MusicTrack``) или ``None``."""
        player = self._player()
        return None if player is None else player.current_track


class ScriptAudio:
    """``self.audio``: разовые звуки и громкость шин."""

    __slots__ = ("_component", "_gains")

    def __init__(self, component: ScriptComponent) -> None:
        self._component = component
        self._gains: dict[str, float] = dict.fromkeys(BUSES, 1.0)

    def _system(self) -> AudioSystem | None:
        scene = self._component.scene
        return None if scene is None else scene.get_system(AudioSystem)

    def play(
        self,
        clip: Any,
        gain: float = 1.0,
        bus: str = "sfx",
        loop: bool = False,
        pitch: float = 1.0,
    ) -> VoiceHandle:
        """Проиграть клип (``Asset``/``AssetRef``/путь).

        Returns:
            :class:`VoiceHandle` (``.stop()``, ``.set_gain()``, ``.is_playing``);
            при ошибке или без звука в сцене — дескриптор, который не играет.

        Raises:
            ValueError: Неизвестная шина.
        """
        if bus not in BUSES or bus == "master":
            raise ValueError(f"неизвестная шина: {bus!r}")
        system = self._system()
        if system is None:
            return VoiceHandle(0, None)
        return system.play(clip, gain=gain, bus=bus, loop=loop, pitch=pitch)

    def play_at(
        self,
        clip: Any,
        position: Any,
        gain: float = 1.0,
        bus: str = "sfx",
        loop: bool = False,
        pitch: float = 1.0,
    ) -> VoiceHandle:
        """Проиграть клип в точке мира (затухание и панорама от слушателя).

        Args:
            clip: ``Asset``/``AssetRef``/путь.
            position: ``Vector3`` или ``(x, y, z)``.

        Raises:
            ValueError: Неизвестная шина.
        """
        self._check_bus(bus)
        system = self._system()
        if system is None:
            return VoiceHandle(0, None)
        return system.play_at(
            clip, _vec3(position), gain=gain, bus=bus, loop=loop, pitch=pitch
        )

    def play_cue(
        self,
        cue: Any,
        gain: float = 1.0,
        bus: str = "sfx",
        position: Any = None,
    ) -> VoiceHandle:
        """Проиграть случайный вариант ``SoundCue`` (``position`` — в 3D).

        Raises:
            ValueError: Неизвестная шина.
        """
        self._check_bus(bus)
        system = self._system()
        if system is None:
            return VoiceHandle(0, None)
        return system.play_cue(
            cue,
            gain=gain,
            bus=bus,
            position=None if position is None else _vec3(position),
        )

    @property
    def music(self) -> ScriptMusic:
        """Музыка сцены: ``play``/``stop``/``set_intensity``/``current_track``."""
        return ScriptMusic(self)

    @staticmethod
    def _check_bus(bus: str) -> None:
        if bus not in BUSES or bus == "master":
            raise ValueError(f"неизвестная шина: {bus!r}")

    def stop(self, handle: VoiceHandle | None) -> None:
        """Остановить голос, возвращённый :meth:`play`."""
        if handle is not None:
            handle.stop()

    def set_bus_gain(self, bus: str, gain: float) -> None:
        """Громкость шины 0…1 (``music``, ``sfx``, ``voice``, ``ui``, ``master``).

        Raises:
            ValueError: Неизвестная шина.
        """
        if bus not in BUSES:
            raise ValueError(f"неизвестная шина: {bus!r}")
        self._gains[bus] = min(1.0, max(0.0, float(gain)))
        system = self._system()
        if system is not None:
            system.set_bus_gain(bus, gain)

    def get_bus_gain(self, bus: str) -> float:
        """Громкость шины."""
        system = self._system()
        if system is not None:
            return system.get_bus_gain(bus)
        return self._gains[bus]

    @property
    def master_gain(self) -> float:
        """Громкость мастера 0…1."""
        return self.get_bus_gain("master")

    @master_gain.setter
    def master_gain(self, value: float) -> None:
        self.set_bus_gain("master", value)
