"""Контракт загрузчика ресурсов и реестр загрузчиков.

Загрузчик — расширяемая точка движка: чтобы поддержать новый формат, достаточно
наследоваться от :class:`AssetLoader` и зарегистрировать загрузчик в
:class:`LoaderRegistry` (принцип открытости/закрытости — ядро не меняется).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from engine.core.resources.asset import Asset, AssetType

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources.resource_manager import ResourceManager

__all__ = ["AssetLoader", "LoadContext", "LoaderRegistry"]


@dataclass(slots=True)
class LoadContext:
    """Данные, доступные загрузчику во время разбора файла.

    Attributes:
        path: Разрешённый абсолютный путь к файлу.
        manager: Менеджер ресурсов — для загрузки зависимостей (например,
            материал подтягивает текстуры).
        options: Параметры импорта (масштаб, генерация нормалей и т. п.).
    """

    path: Path
    manager: ResourceManager
    options: Mapping[str, Any] = field(default_factory=dict)

    def option(self, name: str, default: Any = None) -> Any:
        """Прочитать параметр импорта с значением по умолчанию."""
        return self.options.get(name, default)


class AssetLoader(ABC):
    """Базовый загрузчик ресурсов.

    Наследники объявляют поддерживаемые расширения и категорию ресурса, а также
    реализуют :meth:`load`.
    """

    #: Поддерживаемые расширения в нижнем регистре, включая точку.
    extensions: ClassVar[tuple[str, ...]] = ()
    #: Категория загружаемых ресурсов.
    asset_type: ClassVar[AssetType] = AssetType.UNKNOWN

    def can_load(self, path: Path) -> bool:
        """``True``, если загрузчик умеет читать этот файл."""
        return path.suffix.lower() in self.extensions

    @abstractmethod
    def load(self, context: LoadContext) -> Asset:
        """Прочитать файл и вернуть готовый ресурс.

        Args:
            context: Путь, менеджер и параметры импорта.

        Returns:
            Загруженный ресурс.

        Raises:
            ResourceFormatError: Если файл повреждён или не соответствует формату.
        """

    def unload(self, asset: Asset) -> None:
        """Освободить данные ресурса перед удалением из кэша.

        По умолчанию делегирует в :meth:`Asset.release`.
        """
        asset.release()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} extensions={self.extensions}>"


class LoaderRegistry:
    """Реестр загрузчиков: расширение файла → загрузчик."""

    __slots__ = ("_by_extension", "_loaders")

    def __init__(self, loaders: Iterable[AssetLoader] = ()) -> None:
        self._loaders: list[AssetLoader] = []
        self._by_extension: dict[str, AssetLoader] = {}
        for loader in loaders:
            self.register(loader)

    def register(self, loader: AssetLoader) -> AssetLoader:
        """Зарегистрировать загрузчик.

        Raises:
            ValueError: Если загрузчик уже зарегистрирован или не объявил
                поддерживаемые расширения.
        """
        if loader in self._loaders:
            raise ValueError(f"{type(loader).__name__} уже зарегистрирован")
        if not loader.extensions:
            raise ValueError(
                f"{type(loader).__name__} не объявил ни одного расширения"
            )
        for extension in loader.extensions:
            key = extension.lower()
            if key in self._by_extension:
                raise ValueError(
                    f"Расширение '{key}' уже обрабатывает "
                    f"{type(self._by_extension[key]).__name__}"
                )
            self._by_extension[key] = loader
        self._loaders.append(loader)
        return loader

    def unregister(self, loader_type: type[AssetLoader]) -> bool:
        """Удалить загрузчик по типу.

        Returns:
            ``True``, если загрузчик был зарегистрирован.
        """
        for loader in list(self._loaders):
            if type(loader) is loader_type:
                self._loaders.remove(loader)
                for extension in loader.extensions:
                    self._by_extension.pop(extension.lower(), None)
                return True
        return False

    def loader_for(self, path: Path) -> AssetLoader | None:
        """Вернуть загрузчик для файла или ``None``."""
        return self._by_extension.get(path.suffix.lower())

    @property
    def extensions(self) -> tuple[str, ...]:
        """Все поддерживаемые расширения (отсортированы)."""
        return tuple(sorted(self._by_extension))

    @property
    def loaders(self) -> tuple[AssetLoader, ...]:
        """Зарегистрированные загрузчики."""
        return tuple(self._loaders)

    def __len__(self) -> int:
        return len(self._by_extension)

    def __contains__(self, path: object) -> bool:
        if isinstance(path, Path):
            return self.loader_for(path) is not None
        if isinstance(path, str):
            return path.lower() in self._by_extension
        return False

    def __repr__(self) -> str:
        return (
            f"<LoaderRegistry loaders={len(self._loaders)} "
            f"extensions={self.extensions}>"
        )
