"""Система сцены, запускающая аудио-источники на бэкенде."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.audio.backend import BUSES, IAudioBackend, Vec3, VoiceHandle
from engine.audio.clip import AudioClip, ClipCache, load_audio_clip
from engine.audio.components import AudioListenerComponent, AudioSourceComponent
from engine.audio.cue import SoundCue, load_cue
from engine.audio.music import MusicPlayer
from engine.audio.spatial import Attenuation
from engine.core.resources.asset_ref import AssetRef
from engine.core.system import System

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.scene import Scene

__all__ = ["AudioSystem", "load_audio_settings", "save_audio_settings"]

_log = logging.getLogger("engine.audio.system")

#: Файл настроек громкости относительно корня проекта.
SETTINGS_FILE = "Settings/audio.json"


def load_audio_settings(
    path: Path | str, reader: Callable[[Path], str | None] | None = None
) -> dict[str, float]:
    """Прочитать ``{шина: громкость}``; битый файл/неизвестные шины игнорируются.

    Args:
        path: Файл настроек.
        reader: Чтение текста не с диска (из пака): ``путь -> текст | None``.
    """
    try:
        text = reader(Path(path)) if reader is not None else None
        data = json.loads(
            text if text is not None else Path(path).read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        _log.warning("Настройки звука %s не прочитаны: %s", path, exc)
        return {}
    result: dict[str, float] = {}
    if isinstance(data, dict):
        for bus, value in data.items():
            if bus in BUSES and isinstance(value, (int, float)):
                result[bus] = min(1.0, max(0.0, float(value)))
    return result


def save_audio_settings(path: Path | str, gains: dict[str, float]) -> None:
    """Записать ``{шина: громкость}`` (атомарно)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(gains, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


class AudioSystem(System):
    """Запускает/останавливает голоса по состоянию ``AudioSourceComponent``.

    Кроме источников ведёт слушателя (позиция для 3D-звука), музыку
    (:attr:`music`) и кэш ``SoundCue``.

    Args:
        backend: Инициализированный аудио-бэкенд.
        vfs: Источник ассетов (VFS или ``AssetDatabase``); ``None`` — клипы
            из ссылок не грузятся, работает только ``play(AudioClip)``.
        cache: Кэш клипов (по умолчанию свой, 64 МБ).
        bus_gains: Общая карта ``{шина: громкость}``: плеер делит её между
            системами всех загруженных сцен (бэкенд тоже один).
    """

    priority = -100  # после игровой логики кадра

    def __init__(
        self,
        backend: IAudioBackend,
        vfs: Any = None,
        cache: ClipCache | None = None,
        bus_gains: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.vfs = vfs
        self.cache = cache if cache is not None else ClipCache()
        self._bus_gains: dict[str, float] = (
            bus_gains if bus_gains is not None else dict.fromkeys(BUSES, 1.0)
        )
        #: Музыка сцены: плейлисты, кроссфейд, интенсивность.
        self.music = MusicPlayer(self)
        self._cues: dict[tuple[str, str], SoundCue] = {}
        self._failed_cues: set[tuple[str, str]] = set()
        self._warned_no_listener = False
        self._listener_set = False

    # --------------------------------------------------------------- клипы
    def load_clip(self, ref: AssetRef | str | dict | None) -> AudioClip | None:
        """Клип по ссылке или ``None`` (причина в WARNING)."""
        if ref is None or self.vfs is None:
            return None
        return load_audio_clip(self.vfs, AssetRef.coerce(ref), self.cache)

    def play_clip(
        self,
        clip: AudioClip,
        *,
        gain: float = 1.0,
        bus: str = "sfx",
        loop: bool = False,
        pitch: float = 1.0,
        position: Vec3 | None = None,
        attenuation: Attenuation | None = None,
    ) -> VoiceHandle:
        """Проиграть клип одним голосом (``position`` — пространственный)."""
        pcm = clip.to_format(self.backend.sample_rate, self.backend.channels)
        if position is not None:
            self._ensure_listener()
        return self.backend.play(
            pcm,
            gain=gain,
            pan=0.0,
            loop=loop,
            bus=bus,
            pitch=pitch,
            position=position,
            attenuation=attenuation,
        )

    def play(
        self,
        clip: AudioClip | AssetRef | str | dict | None,
        *,
        gain: float = 1.0,
        bus: str = "sfx",
        loop: bool = False,
        pitch: float = 1.0,
    ) -> VoiceHandle:
        """Проиграть клип или ссылку; при ошибке — «мёртвый» дескриптор."""
        resolved = clip if isinstance(clip, AudioClip) else self.load_clip(clip)
        if resolved is None:
            return VoiceHandle(0, None)
        return self.play_clip(resolved, gain=gain, bus=bus, loop=loop, pitch=pitch)

    def play_at(
        self,
        clip: AudioClip | AssetRef | str | dict | None,
        position: Vec3,
        *,
        gain: float = 1.0,
        bus: str = "sfx",
        loop: bool = False,
        pitch: float = 1.0,
        attenuation: Attenuation | None = None,
    ) -> VoiceHandle:
        """Проиграть клип в точке мира: затухание и панорама от слушателя."""
        resolved = clip if isinstance(clip, AudioClip) else self.load_clip(clip)
        if resolved is None:
            return VoiceHandle(0, None)
        return self.play_clip(
            resolved,
            gain=gain,
            bus=bus,
            loop=loop,
            pitch=pitch,
            position=(float(position[0]), float(position[1]), float(position[2])),
            attenuation=attenuation if attenuation is not None else Attenuation(),
        )

    # ---------------------------------------------------------------- кью
    def load_cue(self, ref: AssetRef | str | dict | None) -> SoundCue | None:
        """Кью по ссылке (кэшируется: «не повторять» помнит прошлый выбор)."""
        asset = AssetRef.coerce(ref)
        if asset is None or self.vfs is None:
            return None
        key = (asset.guid or "", asset.path or "")
        cue = self._cues.get(key)
        if cue is None and key not in self._failed_cues:
            cue = load_cue(self.vfs, asset)
            if cue is None:
                self._failed_cues.add(key)
            else:
                self._cues[key] = cue
        return cue

    def _random(self) -> Any:
        scene = self.scene
        if scene is not None:
            return scene.random
        import random

        return random.Random(0)

    def play_cue(
        self,
        cue: SoundCue | AssetRef | str | dict | None,
        *,
        gain: float = 1.0,
        bus: str = "sfx",
        position: Vec3 | None = None,
        attenuation: Attenuation | None = None,
    ) -> VoiceHandle:
        """Выбрать вариант кью (``scene.random``) и проиграть его."""
        resolved = cue if isinstance(cue, SoundCue) else self.load_cue(cue)
        if resolved is None:
            return VoiceHandle(0, None)
        variant = resolved.pick(self._random())
        clip = self.load_clip(variant.ref)
        if clip is None:
            return VoiceHandle(0, None)
        if position is not None and attenuation is None:
            attenuation = Attenuation()
        return self.play_clip(
            clip,
            gain=gain * variant.gain,
            bus=bus,
            pitch=variant.pitch,
            position=position,
            attenuation=attenuation if position is not None else None,
        )

    # --------------------------------------------------------------- шины
    @property
    def master_gain(self) -> float:
        """Громкость мастера."""
        return self._bus_gains["master"]

    @master_gain.setter
    def master_gain(self, value: float) -> None:
        self.set_bus_gain("master", value)

    def set_bus_gain(self, bus: str, gain: float) -> None:
        """Громкость шины 0…1 (действует и на уже играющие голоса).

        Raises:
            ValueError: Неизвестная шина.
        """
        if bus not in self._bus_gains:
            raise ValueError(f"неизвестная шина: {bus!r}")
        value = min(1.0, max(0.0, float(gain)))
        self._bus_gains[bus] = value
        if bus == "master":
            self.backend.set_master_gain(value)
        else:
            self.backend.set_bus_gain(bus, value)

    def get_bus_gain(self, bus: str) -> float:
        """Текущая громкость шины."""
        return self._bus_gains[bus]

    def bus_gains(self) -> dict[str, float]:
        """Копия ``{шина: громкость}`` (формат ``Settings/audio.json``)."""
        return dict(self._bus_gains)

    def apply_settings(self, gains: dict[str, float]) -> None:
        """Применить ``{шина: громкость}`` (неизвестные шины пропускаются)."""
        for bus, value in gains.items():
            if bus in self._bus_gains:
                self.set_bus_gain(bus, value)

    # ---------------------------------------------------------- слушатель
    def find_listener(self, scene: Scene) -> tuple[Vec3, Vec3, Vec3] | None:
        """``(позиция, вперёд, вверх)`` слушателя сцены или ``None``.

        Слушатель — первый активный :class:`AudioListenerComponent`; без него —
        первая включённая камера (``CameraComponent``).
        """
        for obj in scene.query(AudioListenerComponent):
            if not obj.active_in_hierarchy:
                continue
            for listener in obj.get_components(AudioListenerComponent):
                if listener.active and listener.enabled:
                    return self._pose_of(obj)
        try:
            from engine.rendering.camera import CameraComponent
        except ImportError:  # pragma: no cover - рендеринг не установлен
            return None
        for obj in scene.query(CameraComponent):
            if not obj.active_in_hierarchy:
                continue
            for camera in obj.get_components(CameraComponent):
                if camera.enabled:
                    return self._pose_of(obj)
        return None

    @staticmethod
    def _pose_of(obj: Any) -> tuple[Vec3, Vec3, Vec3]:
        matrix = obj.transform.world_matrix
        return (
            (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])),
            (float(matrix[0, 2]), float(matrix[1, 2]), float(matrix[2, 2])),
            (float(matrix[0, 1]), float(matrix[1, 1]), float(matrix[2, 1])),
        )

    def _update_listener(self, scene: Scene) -> None:
        pose = self.find_listener(scene)
        if pose is None:
            self._listener_set = False
            return
        self._listener_set = True
        self.backend.set_listener(*pose)

    def _ensure_listener(self) -> None:
        """Предупредить один раз, если 3D-звук запущен без слушателя."""
        scene = self.scene
        if self._listener_set or self._warned_no_listener or scene is None:
            return
        if self.find_listener(scene) is None:
            self._warned_no_listener = True
            _log.warning(
                "Пространственный звук без слушателя: нет AudioListener и камеры; "
                "слушатель — в начале координат"
            )

    # ------------------------------------------------------------- кадр
    def on_removed(self, scene: Scene) -> None:
        """Остановить голоса источников сцены и музыку."""
        self.music.shutdown()
        for obj in scene.query(AudioSourceComponent):
            for source in obj.get_components(AudioSourceComponent):
                self._stop_voice(source)

    def on_update(self, scene: Scene, dt: float) -> None:
        """Синхронизировать слушателя, голоса источников и музыку."""
        self._update_listener(scene)
        for obj in scene.query(AudioSourceComponent):
            active = obj.active_in_hierarchy
            for source in obj.get_components(AudioSourceComponent):
                self._update_source(source, active and source.enabled)
        self.music.update(dt)

    def _stop_voice(self, source: AudioSourceComponent) -> None:
        voice, source._voice = source._voice, None  # noqa: SLF001
        if voice is not None:
            voice.stop()

    def _update_source(self, source: AudioSourceComponent, active: bool) -> None:
        voice = source._voice  # noqa: SLF001
        clip_ref = source.clip
        cue_ref = source.cue
        if voice is not None and not voice.is_playing:
            source._voice = voice = None  # noqa: SLF001
            if not source.loop and source.play_state == "playing":
                source.play_state = "stopped"
        if not active or (clip_ref is None and cue_ref is None):
            if voice is not None:
                self._stop_voice(source)
            return
        if not source._autoplayed:  # noqa: SLF001
            source._autoplayed = True  # noqa: SLF001
            if source.autoplay:
                source.play_state = "playing"
                source._restart = True  # noqa: SLF001
        clip_key = (
            (clip_ref.guid, clip_ref.path) if clip_ref is not None else None,
            (cue_ref.guid, cue_ref.path) if cue_ref is not None else None,
        )
        if voice is not None and source._clip_key != clip_key:  # noqa: SLF001
            self._stop_voice(source)  # клип сменили: играть новый с начала
            voice = None
            source._restart = source.play_state == "playing"  # noqa: SLF001
        if source.play_state != "playing":
            if voice is not None:
                self._stop_voice(source)
            return
        if voice is not None and not source._restart:  # noqa: SLF001
            if source._gain_applied != source.gain:  # noqa: SLF001
                source._gain_applied = source.gain  # noqa: SLF001
                voice.set_gain(source.gain * source._gain_scale)  # noqa: SLF001
            if source.spatial:
                self._follow(source, voice)
            return
        if voice is not None:
            self._stop_voice(source)
        self._start(source, clip_key)

    def _start(self, source: AudioSourceComponent, clip_key: Any) -> None:
        source._restart = False  # noqa: SLF001
        if source._failed_key == clip_key:  # noqa: SLF001
            source.play_state = "stopped"
            return
        gain_scale, pitch_scale = 1.0, 1.0
        clip_ref = source.clip
        if source.cue is not None:
            cue = self.load_cue(source.cue)
            if cue is not None:
                variant = cue.pick(self._random())
                clip_ref = variant.ref
                gain_scale, pitch_scale = variant.gain, variant.pitch
        clip = self.load_clip(clip_ref)
        if clip is None:
            source._failed_key = clip_key  # noqa: SLF001
            source.play_state = "stopped"
            return
        source._failed_key = None  # noqa: SLF001
        source._clip_key = clip_key  # noqa: SLF001
        source._gain_applied = source.gain  # noqa: SLF001
        source._gain_scale = gain_scale  # noqa: SLF001
        position: Vec3 | None = None
        attenuation: Attenuation | None = None
        if source.spatial:
            position = self._source_position(source)
            attenuation = Attenuation(
                source.min_distance,
                source.max_distance,
                source.falloff,
                source.rolloff,
            )
            source._position = position  # noqa: SLF001
        source._voice = self.play_clip(  # noqa: SLF001
            clip,
            gain=source.gain * gain_scale,
            bus=source.bus,
            loop=source.loop,
            pitch=source.pitch * pitch_scale,
            position=position,
            attenuation=attenuation,
        )

    @staticmethod
    def _source_position(source: AudioSourceComponent) -> Vec3:
        matrix = source.game_object.transform.world_matrix
        return (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]))

    @staticmethod
    def _follow(source: AudioSourceComponent, voice: VoiceHandle) -> None:
        """Передать бэкенду новую позицию 3D-источника (если сдвинулся)."""
        position = AudioSystem._source_position(source)
        if position != source._position:  # noqa: SLF001
            source._position = position  # noqa: SLF001
            voice.set_position(position)
