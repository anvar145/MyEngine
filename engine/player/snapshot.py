"""Снимок сцены для Play Mode: заморозка и восстановление.

Снимок — это сериализованная сцена (формат ``.scene.json``), поэтому он
годится и для передачи плееру, и для отладки руками.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.scene_io import scene_from_dict, scene_to_dict

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources import AssetDatabase
    from engine.core.scene import Scene
    from engine.rendering.post import PostSettings
    from engine.rendering.sky import SkySettings

__all__ = ["SceneSnapshot"]


@dataclass
class SceneSnapshot:
    """Замороженное состояние сцены.

    Attributes:
        data: Словарь сцены (результат ``scene_to_dict``, включая ``sky``).
        version: Версия формата снимка.
    """

    data: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    @property
    def scene_name(self) -> str:
        """Имя сцены."""
        return str(self.data.get("name", "Scene"))

    @property
    def objects(self) -> list[dict[str, Any]]:
        """Описания объектов сцены."""
        return list(self.data.get("objects", []))

    @staticmethod
    def freeze(
        scene: Scene,
        assets: AssetDatabase | None = None,
        sky: SkySettings | None = None,
        post: PostSettings | None = None,
    ) -> SceneSnapshot:
        """Снять снимок сцены.

        Args:
            scene: Сцена движка.
            assets: База ассетов (для ``mesh_guid``).
            sky: Настройки неба.
            post: Настройки пост-обработки (``None`` — секции нет).

        Returns:
            Снимок.
        """
        # Объекты в снимке адресуются стабильным ``uid`` (а не номером в
        # рантайме, который сцена выдаёт заново), поэтому ссылки переживают
        # заморозку и восстановление без перенумерации.
        return SceneSnapshot(scene_to_dict(scene, assets, sky, post))

    def restore(self, scene: Scene, assets: AssetDatabase | None = None) -> None:
        """Восстановить состояние в ту же сцену (объекты пересоздаются).

        Args:
            scene: Сцена, содержимое которой заменяется.
            assets: База ассетов.
        """
        scene.clear()
        scene_from_dict(self.data, assets, into=scene)

    def to_json(self) -> str:
        """Детерминированный JSON (стабильный порядок ключей)."""
        payload = {"version": self.version, "scene": self.data}
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> SceneSnapshot:
        """Разобрать JSON, созданный :meth:`to_json`."""
        payload = json.loads(text)
        return cls(dict(payload["scene"]), int(payload.get("version", 1)))

    def save(self, path: Path | str) -> Path:
        """Записать снимок в файл."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | str) -> SceneSnapshot:
        """Прочитать снимок из файла."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
