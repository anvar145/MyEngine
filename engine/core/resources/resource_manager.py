"""Менеджер ресурсов: загрузка, кэширование и выгрузка ассетов.

Ключевые свойства:

* **Один файл — один ресурс.** Повторный :meth:`ResourceManager.load` того же
  файла (с теми же параметрами импорта) возвращает объект из кэша и лишь
  увеличивает счётчик ссылок: повторного чтения с диска и повторного
  декодирования не происходит.
* **Явная выгрузка.** Ресурс удаляется из кэша, когда счётчик ссылок равен нулю
  (:meth:`unload_unused`) или принудительно (:meth:`unload`), после чего у
  загрузчика запрашивается освобождение нативных данных.
* **Расширяемость.** Форматы добавляются регистрацией новых
  :class:`~engine.core.resources.loader.AssetLoader` — код менеджера не меняется.

Example:
    >>> resources = ResourceManager(roots=[Path("assets")])
    >>> mesh = resources.load("models/cube.obj")
    >>> same = resources.load("models/cube.obj")      # из кэша, без чтения файла
    >>> resources.release(mesh)
    >>> resources.unload_unused()
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.core.event_bus import Event, EventBus
from engine.core.resources.asset import Asset, AssetType
from engine.core.resources.errors import (
    MissingLoaderError,
    ResourceError,
    ResourceLoadError,
    ResourceNotFoundError,
)
from engine.core.resources.loader import AssetLoader, LoadContext, LoaderRegistry
from engine.core.resources.loaders import default_loaders

__all__ = [
    "AssetLoaded",
    "AssetUnloaded",
    "ResourceManager",
    "ResourceStats",
]

_log = logging.getLogger("engine.core.resources")

CacheKey = tuple[str, str | None]


@dataclass(slots=True)
class AssetLoaded(Event):
    """Событие: ресурс загружен и добавлен в кэш."""

    asset: Asset


@dataclass(slots=True)
class AssetUnloaded(Event):
    """Событие: ресурс выгружен из кэша."""

    path: Path
    asset_type: AssetType


@dataclass(slots=True)
class ResourceStats:
    """Статистика кэша ресурсов."""

    cached: int = 0
    loaded_total: int = 0
    unloaded_total: int = 0
    hits: int = 0
    misses: int = 0
    failed: int = 0
    bytes_estimated: int = 0
    by_type: dict[str, int] = field(default_factory=dict)

    @property
    def hit_ratio(self) -> float:
        """Доля обращений, обслуженных из кэша (``0..1``)."""
        total = self.hits + self.misses
        return 0.0 if total == 0 else self.hits / total


class ResourceManager:
    """Кэш ресурсов с учётом ссылок и расширяемыми загрузчиками.

    Args:
        roots: Каталоги поиска для относительных путей.
        loaders: Загрузчики; по умолчанию набор из
            :func:`engine.core.resources.loaders.default_loaders`.
        event_bus: Шина событий для публикации ``AssetLoaded``/``AssetUnloaded``.
        auto_unload: Выгружать ресурс сразу, как только счётчик ссылок стал нулём.
        name: Имя менеджера (для отладки).
    """

    __slots__ = (
        "_auto_unload",
        "_cache",
        "_cache_loaders",
        "_lock",
        "_registry",
        "_resolve_cache",
        "_roots",
        "events",
        "name",
        "stats_counters",
    )

    def __init__(
        self,
        roots: Iterable[Path | str] = (),
        *,
        loaders: Iterable[AssetLoader] | None = None,
        event_bus: EventBus | None = None,
        auto_unload: bool = False,
        name: str = "ResourceManager",
    ) -> None:
        self.name = name
        self._roots: list[Path] = [Path(root) for root in roots]
        self._registry = LoaderRegistry(
            default_loaders() if loaders is None else loaders
        )
        self.events = event_bus
        self._auto_unload = bool(auto_unload)
        self._cache: dict[CacheKey, Asset] = {}
        self._cache_loaders: dict[CacheKey, AssetLoader] = {}
        self._resolve_cache: dict[str, Path] = {}
        self._lock = threading.RLock()
        self.stats_counters = ResourceStats()

    # ------------------------------------------------------------------ #
    # Корни поиска и загрузчики
    # ------------------------------------------------------------------ #
    @property
    def roots(self) -> tuple[Path, ...]:
        """Каталоги поиска относительных путей."""
        return tuple(self._roots)

    def add_root(self, path: Path | str) -> Path:
        """Добавить каталог поиска (в конец списка приоритетов)."""
        root = Path(path)
        self._roots.append(root)
        self._resolve_cache.clear()
        return root

    @property
    def registry(self) -> LoaderRegistry:
        """Реестр загрузчиков (для регистрации новых форматов)."""
        return self._registry

    def register_loader(self, loader: AssetLoader) -> AssetLoader:
        """Зарегистрировать загрузчик формата."""
        return self._registry.register(loader)

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Расширения, которые умеет читать менеджер."""
        return self._registry.extensions

    def resolve(self, path: Path | str, *, required: bool = True) -> Path:
        """Найти файл ресурса по абсолютному или относительному пути.

        Args:
            path: Путь к файлу (абсолютный или относительно корней поиска).
            required: Бросать ли исключение, если файл не найден.

        Returns:
            Абсолютный путь к существующему файлу.

        Raises:
            ResourceNotFoundError: Если файл не найден и ``required=True``.
        """
        candidate = Path(path)
        cache_key = os.path.normcase(str(candidate))
        cached = self._resolve_cache.get(cache_key)
        if cached is not None and cached.is_file():
            return cached

        if candidate.is_absolute():
            if candidate.is_file():
                resolved = candidate
            elif not required:
                raise ResourceNotFoundError(str(candidate))
            else:
                raise ResourceNotFoundError(
                    f"Файл ресурса не найден: '{candidate}'"
                )
        else:
            resolved = None
            for root in self._roots:
                attempt = root / candidate
                if attempt.is_file():
                    resolved = attempt
                    break
            if resolved is None:
                if not required:
                    raise ResourceNotFoundError(str(candidate))
                raise ResourceNotFoundError(
                    f"Файл ресурса '{candidate}' не найден в корнях: "
                    f"{', '.join(str(root) for root in self._roots) or '<нет корней>'}"
                )

        absolute = Path(os.path.abspath(resolved))
        self._resolve_cache[cache_key] = absolute
        return absolute

    def is_loaded(
        self, path: Path | str, *, options: Mapping[str, Any] | None = None
    ) -> bool:
        """``True``, если ресурс уже лежит в кэше."""
        key = self._key_or_none(path, options)
        return key is not None and key in self._cache

    # ------------------------------------------------------------------ #
    # Загрузка и выдача
    # ------------------------------------------------------------------ #
    def load(
        self,
        path: Path | str,
        *,
        options: Mapping[str, Any] | None = None,
        acquire: bool = True,
    ) -> Asset:
        """Загрузить ресурс или вернуть его из кэша.

        Args:
            path: Путь к файлу (абсолютный или относительно корней поиска).
            options: Параметры импорта; файл с разными параметрами кэшируется
                как разные ресурсы.
            acquire: Увеличивать ли счётчик ссылок (обычно ``True``).

        Returns:
            Загруженный ресурс.

        Raises:
            ResourceNotFoundError: Файл не найден.
            MissingLoaderError: Для расширения нет загрузчика.
            ResourceLoadError: Загрузчик завершился ошибкой.
        """
        resolved = self.resolve(path)
        key = self._make_key(resolved, options)

        with self._lock:
            asset = self._cache.get(key)
            if asset is not None:
                if acquire:
                    asset._add_ref()  # noqa: SLF001
                self.stats_counters.hits += 1
                return asset

            loader = self._registry.loader_for(resolved)
            if loader is None:
                _log.error("Нет загрузчика для '%s'", resolved)
                raise MissingLoaderError(
                    f"Нет загрузчика для '{resolved.suffix}' "
                    f"(файл '{resolved.name}'). Доступные форматы: "
                    f"{', '.join(self._registry.extensions) or 'нет'}"
                )

            self.stats_counters.misses += 1
            context = LoadContext(resolved, self, dict(options or {}))
            try:
                asset = loader.load(context)
            except ResourceError as exc:
                self.stats_counters.failed += 1
                _log.error("Ресурс '%s' не загружен: %s", resolved, exc)
                raise
            except Exception as exc:  # noqa: BLE001 - оборачиваем в ошибку ресурса
                self.stats_counters.failed += 1
                _log.exception("Ошибка загрузки '%s'", resolved)
                raise ResourceLoadError(
                    f"Ошибка загрузки '{resolved}': {exc}"
                ) from exc

            asset.path = resolved
            if acquire:
                asset._add_ref()  # noqa: SLF001
            self._cache[key] = asset
            self._cache_loaders[key] = loader
            self.stats_counters.loaded_total += 1

        self._publish(AssetLoaded(asset))
        return asset

    def acquire(
        self, path: Path | str, *, options: Mapping[str, Any] | None = None
    ) -> Asset:
        """Псевдоним :meth:`load` — взять ресурс и увеличить счётчик ссылок."""
        return self.load(path, options=options, acquire=True)

    def preload(self, *paths: Path | str) -> list[Asset]:
        """Загрузить несколько ресурсов подряд (удобно для предзагрузки сцены)."""
        return [self.load(path) for path in paths]

    def get(
        self, path: Path | str, *, options: Mapping[str, Any] | None = None
    ) -> Asset | None:
        """Вернуть ресурс из кэша, не загружая его с диска."""
        key = self._key_or_none(path, options)
        if key is None:
            return None
        return self._cache.get(key)

    def release(self, asset_or_path: Asset | Path | str) -> int:
        """Уменьшить счётчик ссылок ресурса.

        Args:
            asset_or_path: Ресурс, путь к нему или абсолютный путь.

        Returns:
            Новое значение счётчика ссылок (``0``, если ресурс свободен или
            не найден).
        """
        asset = (
            asset_or_path
            if isinstance(asset_or_path, Asset)
            else self.get(asset_or_path)
        )
        if asset is None:
            return 0
        with self._lock:
            remaining = asset._remove_ref()  # noqa: SLF001
            if remaining <= 0 and self._auto_unload:
                key = self._key_of_asset(asset)
                if key is not None:
                    self._dispose(key)
        return remaining

    def unload(
        self,
        path: Path | str,
        *,
        options: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        """Выгрузить ресурс из кэша.

        Args:
            path: Путь к ресурсу.
            options: Параметры импорта, с которыми ресурс загружался.
            force: Выгрузить, даже если ресурс ещё используется.

        Returns:
            ``True``, если ресурс был выгружен.
        """
        key = self._key_or_none(path, options)
        if key is None:
            return False
        with self._lock:
            asset = self._cache.get(key)
            if asset is None:
                return False
            if asset.is_used and not force:
                _log.debug(
                    "Ресурс '%s' ещё используется (%d ссылок), выгрузка отложена",
                    asset.name,
                    asset.ref_count,
                )
                return False
            self._dispose(key)
        return True

    def unload_unused(self) -> int:
        """Выгрузить все ресурсы со счётчиком ссылок ``0``.

        Returns:
            Число выгруженных ресурсов.
        """
        with self._lock:
            keys = [key for key, asset in self._cache.items() if not asset.is_used]
            for key in keys:
                self._dispose(key)
        return len(keys)

    def reload(
        self, path: Path | str, *, options: Mapping[str, Any] | None = None
    ) -> Asset:
        """Перечитать ресурс с диска (для hot reload).

        Старый экземпляр выгружается принудительно; возвращается новый объект.
        Внешние держатели старого ресурса должны подписаться на события
        ``AssetLoaded``, чтобы заменить ссылку.
        """
        self.unload(path, options=options, force=True)
        return self.load(path, options=options)

    def clear(self) -> int:
        """Выгрузить все ресурсы.

        Returns:
            Число выгруженных ресурсов.
        """
        with self._lock:
            keys = list(self._cache)
            for key in keys:
                self._dispose(key)
            self._resolve_cache.clear()
        return len(keys)

    # ------------------------------------------------------------------ #
    # Статистика и служебное
    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> ResourceStats:
        """Текущая статистика кэша."""
        by_type: dict[str, int] = {}
        total_bytes = 0
        for asset in self._cache.values():
            by_type[asset.asset_type.value] = by_type.get(asset.asset_type.value, 0) + 1
            total_bytes += asset.estimated_bytes
        counters = self.stats_counters
        return ResourceStats(
            cached=len(self._cache),
            loaded_total=counters.loaded_total,
            unloaded_total=counters.unloaded_total,
            hits=counters.hits,
            misses=counters.misses,
            failed=counters.failed,
            bytes_estimated=total_bytes,
            by_type=by_type,
        )

    @property
    def cached_assets(self) -> tuple[Asset, ...]:
        """Ресурсы, лежащие в кэше."""
        return tuple(self._cache.values())

    def _make_key(self, resolved: Path, options: Mapping[str, Any] | None) -> CacheKey:
        """Построить ключ кэша: нормализованный путь + параметры импорта."""
        return (os.path.normcase(str(resolved)), _options_key(options))

    def _key_or_none(
        self, path: Path | str, options: Mapping[str, Any] | None
    ) -> CacheKey | None:
        """Ключ кэша без исключения, если файл не найден."""
        try:
            resolved = self.resolve(path, required=False)
        except ResourceNotFoundError:
            return None
        return self._make_key(resolved, options)

    def _key_of_asset(self, asset: Asset) -> CacheKey | None:
        """Найти ключ кэша по объекту ресурса."""
        for key, cached in self._cache.items():
            if cached is asset:
                return key
        return None

    def _dispose(self, key: CacheKey) -> None:
        """Убрать ресурс из кэша, освободив его данные."""
        asset = self._cache.pop(key, None)
        loader = self._cache_loaders.pop(key, None)
        if asset is None:
            return
        try:
            if loader is not None:
                loader.unload(asset)
            else:
                asset.release()
        except Exception:  # noqa: BLE001 - выгрузка не должна ломать кадр
            _log.exception("Ошибка выгрузки ресурса '%s'", asset.name)
        self.stats_counters.unloaded_total += 1
        self._publish(AssetUnloaded(asset.path, asset.asset_type))

    def _publish(self, event: Event) -> None:
        """Опубликовать событие ресурсов, если шина подключена."""
        if self.events is not None:
            self.events.emit(event)

    def __contains__(self, path: object) -> bool:
        if isinstance(path, (str, Path)):
            return self.is_loaded(path)
        return False

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        stats = self.stats
        return (
            f"<ResourceManager cached={stats.cached} loaded={stats.loaded_total} "
            f"hit_ratio={stats.hit_ratio:.2f}>"
        )


def _options_key(options: Mapping[str, Any] | None) -> str | None:
    """Стабильное строковое представление параметров импорта."""
    if not options:
        return None
    try:
        return repr(sorted(options.items()))
    except TypeError:  # несравнимые значения — просто repr словаря
        return repr(options)
