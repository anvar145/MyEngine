"""Слот сохранения: метаданные, ошибки."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "SaveCorruptedError",
    "SaveError",
    "SaveScene",
    "SaveSlot",
    "SaveSlotMeta",
    "SaveVersionError",
]


class SaveError(Exception):
    """Базовая ошибка сохранения/загрузки."""


class SaveCorruptedError(SaveError):
    """Файл сохранения повреждён (не zip, нет записи, не совпал CRC)."""


class SaveVersionError(SaveError):
    """Версия сохранения не поддерживается (новее текущей или нет миграции)."""


@dataclass(frozen=True)
class SaveScene:
    """Сцена в слоте: путь и признак «дополнительная».

    Attributes:
        path: Путь к файлу сцены относительно корня проекта (в слотах без
            менеджера сцен и в сейвах A7a - имя сцены, как раньше ``scene_name``).
        additive: Загружена рядом с активной (``False`` - активная).
    """

    path: str
    additive: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-вид."""
        return {"path": self.path, "additive": self.additive}


@dataclass(frozen=True)
class SaveSlotMeta:
    """Метаданные слота (``meta.json``).

    Attributes:
        timestamp: Время записи, секунды Unix.
        version: Версия схемы сохранения.
        scene_name: Имя сцены на момент записи.
        playtime_sec: Время игры, секунды.
        screenshot_path: Имя файла скриншота внутри архива или ``None``.
        scenes: Загруженные сцены: активная первой, затем дополнительные в
            порядке загрузки. ``scene_name`` равен ``scenes[0].path``. Пусто в
            конструкторе = одна активная сцена ``scene_name`` (так читаются
            сейвы A7a без поля ``scenes``).
    """

    timestamp: float
    version: int
    scene_name: str
    playtime_sec: float = 0.0
    screenshot_path: str | None = None
    scenes: tuple[SaveScene, ...] = ()

    def __post_init__(self) -> None:
        if not self.scenes:
            object.__setattr__(self, "scenes", (SaveScene(self.scene_name, False),))

    def to_dict(self) -> dict[str, Any]:
        """JSON-вид."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "scene_name": self.scene_name,
            "playtime_sec": self.playtime_sec,
            "screenshot_path": self.screenshot_path,
            "scenes": [scene.to_dict() for scene in self.scenes],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SaveSlotMeta:
        """Прочитать из ``meta.json``.

        Raises:
            SaveCorruptedError: Нет обязательных полей или неверные типы.
        """
        try:
            shot = raw.get("screenshot_path")
            raw_scenes = raw.get("scenes")
            scenes: tuple[SaveScene, ...] = ()
            if raw_scenes is not None:
                scenes = tuple(
                    SaveScene(str(item["path"]), bool(item.get("additive", False)))
                    for item in raw_scenes
                )
            return cls(
                timestamp=float(raw["timestamp"]),
                version=int(raw["version"]),
                scene_name=str(raw["scene_name"]),
                playtime_sec=float(raw.get("playtime_sec", 0.0)),
                screenshot_path=None if shot is None else str(shot),
                scenes=scenes,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SaveCorruptedError(f"некорректный meta.json: {exc}") from exc


@dataclass(frozen=True)
class SaveSlot:
    """Слот: путь к файлу ``<имя>.save`` и его метаданные."""

    path: Path
    meta: SaveSlotMeta

    @property
    def name(self) -> str:
        """Имя слота (без расширения)."""
        return self.path.stem
