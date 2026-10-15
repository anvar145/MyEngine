"""``AssetRef``: ссылка на ассет по GUID с запасным путём.

GUID переживает переименование и перемещение файла, путь читаем человеком и
спасает ссылку, если ``.meta`` потерян. Пишутся оба поля; разрешение идёт
сначала по GUID, затем по пути.

Встроенные примитивы (``cube``, ``sphere``...) не имеют ни файла, ни GUID:
для них ``AssetRef(guid=None, path="cube")``, а :meth:`AssetRef.resolve`
возвращает ``None``.

Example:
    >>> ref = AssetRef(guid="ab" * 16, path="Models/tree.obj")
    >>> AssetRef.from_dict(ref.to_dict()) == ref
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["AssetRef"]


def _builtin_names() -> tuple[str, ...]:
    # Лениво: ``components`` не должен зависеть от пакета ресурсов при импорте.
    from engine.core.components import BUILTIN_MESHES

    return BUILTIN_MESHES


def _clean(value: Any) -> str | None:
    """Пустая строка и ``None`` - «нет значения»; путь нормализуется к ``/``."""
    if value is None:
        return None
    text = str(value).replace("\\", "/")
    return text or None


@dataclass(frozen=True, slots=True)
class AssetRef:
    """Ссылка на ассет.

    Attributes:
        guid: GUID ассета (``None`` для примитивов и «старых» ссылок по пути).
        path: Путь к исходнику (относительно ``Assets/``) либо имя примитива.

    Raises:
        ValueError: Если не задано ни ``guid``, ни ``path``.
    """

    guid: str | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        guid = _clean(self.guid)
        path = _clean(self.path)
        if guid is None and path is None:
            raise ValueError("AssetRef требует guid или path")
        object.__setattr__(self, "guid", guid)
        object.__setattr__(self, "path", path)

    # ------------------------------------------------------------------ #
    @property
    def is_primitive(self) -> bool:
        """Встроенный процедурный примитив (нет GUID, путь - его имя)."""
        return self.guid is None and self.path in _builtin_names()

    def to_dict(self) -> dict[str, str | None]:
        """JSON-вид: оба поля пишутся всегда."""
        return {"guid": self.guid, "path": self.path}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssetRef:
        """Прочитать ``{"guid":..., "path":...}``.

        Raises:
            ValueError: Если нет ни ``guid``, ни ``path``.
        """
        return cls(guid=data.get("guid"), path=data.get("path"))

    @classmethod
    def coerce(cls, value: Any) -> AssetRef | None:
        """Привести значение любого вида к ссылке (пусто -> ``None``).

        Принимает ``AssetRef``, строку-путь, ``{"guid","path"}`` и старую
        форму поля скрипта ``{"asset": путь[, "guid": ...]}``.

        Raises:
            ValueError: Неподдерживаемый тип значения.
        """
        if value is None or isinstance(value, AssetRef):
            return value
        if isinstance(value, str):
            return cls(path=value) if value else None
        if isinstance(value, dict):
            guid = value.get("guid")
            path = value.get("path") if "path" in value else value.get("asset")
            return cls(guid=guid, path=path) if guid or path else None
        raise ValueError(f"не ссылка на ассет: {value!r}")

    # ------------------------------------------------------------------ #
    def resolve(self, vfs: Any) -> str | None:
        """GUID ассета в ``vfs`` или ``None``.

        Порядок: GUID (если он есть в VFS) -> путь через базу ассетов или
        ``.meta`` рядом с файлом (рабочий каталог) -> ``None``. Примитивы
        всегда дают ``None``.

        Args:
            vfs: ``VFS``, ``AssetDatabase`` или ``PakReader``.

        Returns:
            GUID найденного ассета; ``None``, если ссылка битая или примитив.
        """
        if self.is_primitive:
            return None
        # Лениво: ``asset_ref`` - лёгкий тип данных, его импортируют компоненты.
        from engine.core.resources.vfs import as_assets

        assets = as_assets(vfs)
        if self.guid is not None and assets.get(self.guid) is not None:
            return self.guid
        if self.path is None:
            return None
        found = assets.guid_of(self.path)
        root = getattr(assets, "root", None)
        if not found and root is not None and not Path(self.path).is_absolute():
            # Путь от корня проекта (``Assets/...``).
            found = assets.guid_of(Path(root).parent / self.path)
        if found:
            return found
        return self._guid_from_meta(assets)

    def _guid_from_meta(self, assets: Any) -> str | None:
        root = getattr(assets, "root", None)
        if root is None or self.path is None:
            return None
        candidate = Path(root) / self.path
        if not candidate.is_file():
            return None
        from engine.core.resources.meta import read_meta

        try:
            meta = read_meta(candidate)
        except ValueError:
            return None
        return getattr(meta, "guid", None) or None
