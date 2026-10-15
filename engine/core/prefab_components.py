"""Компонент-маркер экземпляра префаба."""

from __future__ import annotations

from typing import Any

from engine.core.component import Component
from engine.core.resources.asset_ref import AssetRef

__all__ = ["PrefabInstance", "marker_ref_parts"]


def marker_ref_parts(marker: dict[str, Any]) -> tuple[str, str]:
    """``(guid, path)`` префаба из записи маркера (новая и старая формы).

    Args:
        marker: Запись компонента ``PrefabInstance`` (``prefab_ref`` либо
            ``prefab_guid`` + ``prefab_path``).

    Returns:
        GUID и путь; пустая строка - значение не задано.
    """
    raw = marker.get("prefab_ref")
    if isinstance(raw, dict):
        return str(raw.get("guid") or ""), str(raw.get("path") or "")
    return str(marker.get("prefab_guid", "")), str(marker.get("prefab_path", ""))


class PrefabInstance(Component):
    """Помечает корень экземпляра префаба.

    Внутри файла префаба тот же компонент на объекте означает *вложенный*
    префаб (объект-заместитель, см. :mod:`engine.prefab`).

    Args:
        prefab_guid: GUID ассета ``*.prefab.json`` (удобство конструктора).
        prefab_path: Путь к файлу от корня проекта (удобство конструктора).
        overrides: Переопределения значений. Запись:
            ``{"target": "L1/L2", "component": "Transform", "field": "position",
            "value": [...]}``; признак ``"stale": true`` - запись не применилась
            при загрузке (нет цели или компонента), см. :mod:`engine.prefab`.
        instance_uid: Идентификатор экземпляра. Непустой у корня экземпляра,
            развёрнутого в сцене: по нему при загрузке восстанавливаются
            ``uid`` всех объектов (``""`` - маркер без раскрытия: заместитель
            во вложенном префабе или запись формата A4.2).
        prefab_ref: Ссылка на префаб; перекрывает ``prefab_guid``/``prefab_path``.

    Attributes:
        prefab_ref: Ссылка на префаб (:class:`AssetRef`) или ``None``.
        prefab_guid: Только чтение: GUID из ``prefab_ref`` (``""`` - нет).
        prefab_path: Только чтение: путь из ``prefab_ref`` (``""`` - нет).
        overrides: Список переопределений (как записан в сцене при загрузке).
        instance_uid: Идентификатор экземпляра.
    """

    __slots__ = ("instance_uid", "overrides", "prefab_ref")

    def __init__(
        self,
        prefab_guid: str = "",
        prefab_path: str = "",
        overrides: list[Any] | None = None,
        instance_uid: str = "",
        prefab_ref: AssetRef | None = None,
    ) -> None:
        super().__init__()
        guid, path = str(prefab_guid), str(prefab_path)
        self.prefab_ref: AssetRef | None = prefab_ref or (
            AssetRef(guid=guid, path=path) if guid or path else None
        )
        self.overrides: list[Any] = list(overrides) if overrides else []
        self.instance_uid = str(instance_uid)

    @property
    def prefab_guid(self) -> str:
        """GUID префаба (``""`` - нет; только чтение)."""
        return (self.prefab_ref.guid if self.prefab_ref else None) or ""

    @property
    def prefab_path(self) -> str:
        """Путь к файлу префаба (``""`` - нет; только чтение)."""
        return (self.prefab_ref.path if self.prefab_ref else None) or ""

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать компонент в словарь.

        ``instance_uid`` пишется только когда он задан: записи без него
        (A4.2, вложенные заместители) не меняются.
        """
        data: dict[str, Any] = {
            "prefab_ref": self.prefab_ref.to_dict() if self.prefab_ref else None,
        }
        if self.instance_uid:
            data["instance_uid"] = self.instance_uid
        data["overrides"] = [dict(item) for item in self.overrides]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrefabInstance:
        """Восстановить компонент из словаря, полученного из :meth:`to_dict`."""
        overrides = data.get("overrides") or []
        guid, path = marker_ref_parts(data)
        return cls(
            prefab_guid=guid,
            prefab_path=path,
            overrides=[dict(item) for item in overrides],
            instance_uid=str(data.get("instance_uid", "")),
        )

    def __repr__(self) -> str:
        return f"<PrefabInstance guid='{self.prefab_guid}'>"
