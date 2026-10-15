"""Музыкальная система: плейлисты, кроссфейд, интенсивность (``*.music.json``).

Формат файла::

    {
      "tracks": [{"clip": {"guid": ..., "path": ...}, "gain": 1.0,
                  "fade_in": 1.0, "fade_out": 1.0,
                  "stem": {"clip": {...}, "gain": 1.0}}, ...],
      "loop": true,
      "shuffle": false
    }

``stem`` — необязательный «тревожный» слой трека: играет синхронно с основным
клипом, его громкость пропорциональна :meth:`MusicPlayer.set_intensity`.

Каждый трек — отдельный слой с двумя голосами (клип и стем). Кроссфейд —
линейное нарастание/затухание слоёв; громкость слоя обновляется раз в кадр
(микшер плавно доводит коэффициент внутри блока, щелчков нет). Переход к
следующему треку начинается за ``fade_out`` секунд до конца клипа.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from engine.audio.asset_json import AudioAssetError, load_json_asset, ref_from_json
from engine.core.resources.asset_ref import AssetRef

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.audio.backend import VoiceHandle
    from engine.audio.system import AudioSystem

__all__ = [
    "MusicPlayer",
    "MusicPlaylist",
    "MusicTrack",
    "load_playlist",
    "parse_playlist",
]

_log = logging.getLogger("engine.audio.music")

#: Роль артефакта в ``Library`` (файл ``music.json``).
ARTIFACT = "music"
_END_EPS = 1e-3


@dataclass(frozen=True, slots=True)
class MusicTrack:
    """Трек плейлиста."""

    clip: AssetRef
    gain: float = 1.0
    fade_in: float = 1.0
    fade_out: float = 1.0
    stem: AssetRef | None = None
    stem_gain: float = 1.0


@dataclass(frozen=True, slots=True)
class MusicPlaylist:
    """Плейлист: треки, зацикливание, перемешивание."""

    tracks: tuple[MusicTrack, ...]
    loop: bool = True
    shuffle: bool = False


def _number(data: dict[str, Any], key: str, default: float, low: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioAssetError(f"{key}: ожидается число")
    if value < low:
        raise AudioAssetError(f"{key}: значение {value} меньше {low}")
    return float(value)


def parse_playlist(data: dict[str, Any]) -> MusicPlaylist:
    """Собрать :class:`MusicPlaylist` из словаря файла.

    Raises:
        AudioAssetError: Нет треков или неверные типы.
    """
    raw = data.get("tracks")
    if not isinstance(raw, list) or not raw:
        raise AudioAssetError("tracks: нужен непустой список треков")
    tracks: list[MusicTrack] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AudioAssetError(f"tracks[{i}]: ожидается объект")
        stem = item.get("stem")
        stem_ref: AssetRef | None = None
        stem_gain = 1.0
        if stem is not None:
            if not isinstance(stem, dict):
                raise AudioAssetError(f"tracks[{i}].stem: ожидается объект")
            stem_ref = ref_from_json(stem.get("clip"), f"tracks[{i}].stem.clip")
            stem_gain = _number(stem, "gain", 1.0, 0.0)
        tracks.append(
            MusicTrack(
                ref_from_json(item.get("clip"), f"tracks[{i}].clip"),
                _number(item, "gain", 1.0, 0.0),
                _number(item, "fade_in", 1.0, 0.0),
                _number(item, "fade_out", 1.0, 0.0),
                stem_ref,
                stem_gain,
            )
        )
    return MusicPlaylist(
        tuple(tracks), bool(data.get("loop", True)), bool(data.get("shuffle", False))
    )


def load_playlist(vfs: Any, ref: AssetRef | str | dict | None) -> MusicPlaylist | None:
    """Загрузить плейлист по ссылке; при ошибке — ``None`` и WARNING."""
    data = load_json_asset(vfs, ref, ARTIFACT)
    if data is None:
        return None
    try:
        return parse_playlist(data)
    except AudioAssetError as exc:
        asset = AssetRef.coerce(ref)
        _log.warning("Плейлист %s: %s", asset.path if asset else "?", exc)
        return None


class _Layer:
    """Играющий трек: голос клипа, голос стема, прогресс и фейд."""

    __slots__ = (
        "dying",
        "duration",
        "elapsed",
        "fade",
        "rate",
        "stem_voice",
        "track",
        "voice",
    )

    def __init__(
        self,
        track: MusicTrack,
        voice: VoiceHandle,
        stem_voice: VoiceHandle | None,
        duration: float,
        fade: float,
        rate: float,
    ) -> None:
        self.track = track
        self.voice = voice
        self.stem_voice = stem_voice
        self.duration = duration
        self.elapsed = 0.0
        self.fade = fade
        self.rate = rate  # изменение фейда в секунду (+ нарастание, - затухание)
        self.dying = False

    def stop(self) -> None:
        self.voice.stop()
        if self.stem_voice is not None:
            self.stem_voice.stop()


class MusicPlayer:
    """Проигрыватель музыки сцены (доступен как ``AudioSystem.music``).

    Args:
        system: Аудио-система, у которой берутся клипы и бэкенд.
    """

    def __init__(self, system: AudioSystem) -> None:
        self._system = system
        self._playlist: MusicPlaylist | None = None
        self._bus = "music"
        self._layers: list[_Layer] = []
        self._index = -1
        self._intensity = 0.0
        self._order: list[int] = []

    # ------------------------------------------------------------- состояние
    @property
    def current_track(self) -> MusicTrack | None:
        """Трек, который сейчас звучит (не уходящий), или ``None``."""
        layer = self._current()
        return None if layer is None else layer.track

    @property
    def current_index(self) -> int:
        """Индекс текущего трека в плейлисте (``-1`` — музыки нет)."""
        return self._index if self._current() is not None else -1

    @property
    def intensity(self) -> float:
        """Интенсивность ``0…1``."""
        return self._intensity

    @property
    def is_playing(self) -> bool:
        """Есть хотя бы один звучащий слой."""
        return bool(self._layers)

    def _current(self) -> _Layer | None:
        for layer in reversed(self._layers):
            if not layer.dying:
                return layer
        return None

    # ----------------------------------------------------------- управление
    def play(
        self,
        playlist: MusicPlaylist | AssetRef | str | dict | None,
        crossfade: float = 1.0,
        bus: str = "music",
    ) -> bool:
        """Запустить плейлист; играющая музыка уходит кроссфейдом.

        Args:
            playlist: Плейлист или ссылка на ``*.music.json``.
            crossfade: Длительность кроссфейда, секунды (``0`` — резко).
            bus: Шина микшера.

        Returns:
            ``True``, если плейлист запущен; ``False`` — не загрузился.
        """
        resolved = (
            playlist
            if isinstance(playlist, MusicPlaylist)
            else load_playlist(self._system.vfs, playlist)
        )
        if resolved is None:
            return False
        self._fade_out_all(crossfade)
        self._playlist = resolved
        self._bus = bus
        self._order = self._make_order(resolved)
        self._index = -1
        return self._start_next(crossfade)

    def stop(self, crossfade: float = 1.0) -> None:
        """Остановить музыку (затухание за ``crossfade`` секунд)."""
        self._playlist = None
        self._fade_out_all(crossfade)

    def set_intensity(self, value: float) -> None:
        """Интенсивность ``0…1``: громкость «тревожного» стема трека."""
        self._intensity = min(1.0, max(0.0, float(value)))

    # ------------------------------------------------------------- внутреннее
    def _make_order(self, playlist: MusicPlaylist) -> list[int]:
        order = list(range(len(playlist.tracks)))
        if playlist.shuffle and len(order) > 1:
            self._scene_random().shuffle(order)
        return order

    def _scene_random(self) -> Any:
        scene = self._system.scene
        if scene is not None:
            return scene.random
        import random

        return random.Random(0)

    def _fade_out_all(self, seconds: float) -> None:
        for layer in self._layers:
            if seconds <= 0.0:
                layer.fade = 0.0
            layer.dying = True
            layer.rate = -1.0 / seconds if seconds > 0.0 else -1.0
        self._reap()

    def _start_next(self, fade_in: float | None = None) -> bool:
        """Запустить следующий трек плейлиста; ``False`` — плейлист исчерпан."""
        playlist = self._playlist
        if playlist is None:
            return False
        for _ in range(len(self._order)):
            position = self._index + 1
            if position >= len(self._order):
                if not playlist.loop:
                    return False
                position = 0
                if playlist.shuffle and len(self._order) > 1:
                    last = self._order[-1]
                    self._order = self._make_order(playlist)
                    if self._order[0] == last:  # не повторять трек на стыке кругов
                        self._order.append(self._order.pop(0))
            self._index = position
            track = playlist.tracks[self._order[position]]
            if self._start_layer(track, fade_in):
                return True
            if len(self._order) == 1:
                break
        return False

    def _start_layer(self, track: MusicTrack, fade_in: float | None) -> bool:
        clip = self._system.load_clip(track.clip)
        if clip is None:
            return False
        playlist = self._playlist
        assert playlist is not None
        single_loop = playlist.loop and len(playlist.tracks) == 1
        seconds = track.fade_in if fade_in is None else fade_in
        start_fade = 0.0 if seconds > 0.0 else 1.0
        voice = self._system.play_clip(
            clip, gain=0.0, bus=self._bus, loop=single_loop
        )
        stem_voice: VoiceHandle | None = None
        if track.stem is not None:
            stem_clip = self._system.load_clip(track.stem)
            if stem_clip is not None:
                stem_voice = self._system.play_clip(
                    stem_clip, gain=0.0, bus=self._bus, loop=single_loop
                )
        layer = _Layer(
            track,
            voice,
            stem_voice,
            clip.duration,
            start_fade,
            1.0 / seconds if seconds > 0.0 else 0.0,
        )
        self._layers.append(layer)
        self._apply_gain(layer)
        return True

    def _apply_gain(self, layer: _Layer) -> None:
        layer.voice.set_gain(layer.track.gain * layer.fade)
        if layer.stem_voice is not None:
            layer.stem_voice.set_gain(
                layer.track.stem_gain * self._intensity * layer.fade
            )

    def _reap(self) -> None:
        alive: list[_Layer] = []
        for layer in self._layers:
            if layer.dying and layer.fade <= 0.0:
                layer.stop()
            else:
                alive.append(layer)
        self._layers = alive

    # ------------------------------------------------------------------ кадр
    def update(self, dt: float) -> None:
        """Продвинуть фейды и переходы между треками на ``dt`` секунд."""
        if not self._layers:
            return
        for layer in self._layers:
            layer.elapsed += dt
            if layer.rate:
                layer.fade = min(1.0, max(0.0, layer.fade + layer.rate * dt))
                if not layer.dying and layer.fade >= 1.0:
                    layer.rate = 0.0
        self._reap()
        current = self._current()
        if current is not None and self._playlist is not None:
            self._advance_if_ended(current)
        for layer in self._layers:
            self._apply_gain(layer)

    def _advance_if_ended(self, layer: _Layer) -> None:
        playlist = self._playlist
        assert playlist is not None
        if playlist.loop and len(playlist.tracks) == 1:
            return  # один трек в цикле: голос зациклен бэкендом
        ended = not layer.voice.is_playing
        approaching = layer.elapsed >= layer.duration - layer.track.fade_out - _END_EPS
        if not (ended or approaching):
            return
        fade_out = 0.0 if ended else max(layer.track.fade_out, 0.0)
        started = self._start_next()
        layer.dying = True
        layer.rate = -1.0 / fade_out if fade_out > 0.0 else -1.0
        if not started:
            self._playlist = None
        if fade_out <= 0.0:
            layer.fade = 0.0
        self._reap()

    def shutdown(self) -> None:
        """Немедленно остановить всю музыку."""
        self._playlist = None
        for layer in self._layers:
            layer.stop()
        self._layers.clear()
