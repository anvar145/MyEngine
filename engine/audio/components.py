"""Аудио-компоненты сцены: источник звука и маркер слушателя.

Компоненты — только данные и запросы ``play``/``stop``; звук запускает
:class:`~engine.audio.system.AudioSystem` (если он добавлен в сцену, иначе
компонент молчит — так редактор ничего не воспроизводит).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from engine.audio.backend import BUSES
from engine.audio.spatial import FALLOFF_MODELS
from engine.core.component import Component
from engine.core.reflection import FieldKind, component_fields, declare, registry

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.audio.backend import VoiceHandle
    from engine.core.resources.asset_ref import AssetRef

__all__ = ["SOURCE_BUSES", "AudioListenerComponent", "AudioSourceComponent"]

#: Шины, доступные источнику (``master`` — общий множитель, не группа).
SOURCE_BUSES: tuple[str, ...] = tuple(bus for bus in BUSES if bus != "master")
_PLAY_STATES = ("stopped", "playing")


@component_fields
class AudioSourceComponent(Component):
    """Источник звука: клип, громкость, шина.

    Args:
        clip: Ссылка на аудио-ассет.
        gain: Громкость источника, ``0…4`` (итог ограничивается ``1``).
        pitch: Скорость/высота воспроизведения, ``0.1…4``.
        loop: Зациклить клип.
        spatial: Пространственный звук: затухание по расстоянию и панорама
            от слушателя; позиция берётся из трансформа владельца.
        min_distance: Дистанция полной громкости (3D).
        max_distance: Дистанция, за которой ``linear`` даёт тишину (3D).
        falloff: ``linear``/``logarithmic``/``inverse``.
        rolloff: Крутизна логарифмической кривой.
        cue: Ссылка на ``*.cue.json``; если задана, вместо ``clip`` при каждом
            запуске выбирается случайный вариант (с учётом весов и разброса).
        bus: Шина микшера.
        autoplay: Начать воспроизведение при появлении в сцене.
        play_state: ``playing``/``stopped`` (в сцену не пишется, но входит в
            сохранение игры; позиция внутри клипа не сохраняется).
    """

    __slots__ = (
        "_autoplayed",
        "_clip_key",
        "_failed_key",
        "_gain_applied",
        "_gain_scale",
        "_position",
        "_restart",
        "_voice",
        "autoplay",
        "bus",
        "clip",
        "cue",
        "falloff",
        "gain",
        "loop",
        "max_distance",
        "min_distance",
        "pitch",
        "play_state",
        "rolloff",
        "spatial",
    )

    __field_specs__ = (
        declare("clip", None, kind=FieldKind.ASSET_REF, asset_kind="audio",
                tooltip="Аудио-клип"),
        declare("gain", 1.0, range=(0.0, 4.0), step=0.05, tooltip="Громкость"),
        declare("pitch", 1.0, range=(0.1, 4.0), step=0.05,
                tooltip="Скорость воспроизведения"),
        declare("loop", False, tooltip="Зациклить"),
        declare("spatial", False, tooltip="Пространственный звук (3D)"),
        declare("min_distance", 1.0, range=(0.01, 1000.0), step=0.5,
                visible_if="spatial == True", tooltip="Дистанция полной громкости"),
        declare("max_distance", 50.0, range=(0.1, 10000.0), step=1.0,
                visible_if="spatial == True", tooltip="Дистанция тишины (linear)"),
        declare("falloff", "linear", kind=FieldKind.ENUM, choices=FALLOFF_MODELS,
                visible_if="spatial == True", tooltip="Модель затухания"),
        declare("rolloff", 1.0, range=(0.0, 10.0), step=0.1,
                visible_if="spatial == True and falloff == 'logarithmic'",
                tooltip="Крутизна логарифмической кривой"),
        declare("cue", None, kind=FieldKind.ASSET_REF, asset_kind="sound_cue",
                tooltip="SoundCue: случайный вариант вместо клипа"),
        declare("bus", "sfx", kind=FieldKind.ENUM, choices=SOURCE_BUSES,
                tooltip="Шина микшера"),
        declare("autoplay", False, tooltip="Играть при старте"),
        declare("play_state", "stopped", kind=FieldKind.ENUM,
                choices=_PLAY_STATES, serialize=False, save=True, readonly=True),
    )

    def __init__(
        self,
        clip: AssetRef | str | dict | None = None,
        gain: float = 1.0,
        pitch: float = 1.0,
        loop: bool = False,
        spatial: bool = False,
        bus: str = "sfx",
        autoplay: bool = False,
        play_state: str = "stopped",
        min_distance: float = 1.0,
        max_distance: float = 50.0,
        falloff: str = "linear",
        rolloff: float = 1.0,
        cue: AssetRef | str | dict | None = None,
    ) -> None:
        super().__init__()
        self.clip = clip
        self.gain = gain
        self.pitch = pitch
        self.loop = loop
        self.spatial = spatial
        self.bus = bus
        self.autoplay = autoplay
        self.play_state = play_state
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.falloff = falloff
        self.rolloff = rolloff
        self.cue = cue
        self._voice: VoiceHandle | None = None
        self._autoplayed = False
        self._restart = False
        self._clip_key: Any = None
        self._failed_key: Any = None
        self._gain_applied = 1.0
        self._gain_scale = 1.0  # множитель варианта SoundCue
        self._position: tuple[float, float, float] | None = None

    # ------------------------------------------------------------ управление
    def play(self) -> None:
        """Запустить с начала (выполнит :class:`AudioSystem` в ближайшем кадре)."""
        self.play_state = "playing"
        self._restart = True

    def stop(self) -> None:
        """Остановить воспроизведение."""
        self.play_state = "stopped"
        self._restart = False

    @property
    def is_playing(self) -> bool:
        """Голос источника звучит."""
        voice = self._voice
        return voice is not None and voice.is_playing

    # ------------------------------------------------------------ сцена
    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент (для сцены)."""
        return registry.serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioSourceComponent:
        """Восстановить компонент."""
        return registry.deserialize(cls, data)


@component_fields
class AudioListenerComponent(Component):
    """Слушатель: «уши» сцены для пространственного звука.

    Ставится на объект с камерой; позиция и ориентация берутся из мировой
    матрицы владельца. Слушателем становится первый активный компонент
    сцены, если его нет — первая включённая камера.

    Args:
        active: Участвует в выборе слушателя.
    """

    __slots__ = ("active",)

    __field_specs__ = (
        declare("active", True, tooltip="Активный слушатель"),
    )

    def __init__(self, active: bool = True) -> None:
        super().__init__()
        self.active = active

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент (для сцены)."""
        return registry.serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioListenerComponent:
        """Восстановить компонент."""
        return registry.deserialize(cls, data)

