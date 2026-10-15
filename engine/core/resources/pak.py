"""Пакет ассетов ``.pak``: артефакты ``Library/`` одним файлом для экспорта игры.

``PakBuilder`` собирает импортированные артефакты по GUID, ``PakReader`` читает их
обратно и умеет выступать источником для ``load_*``-функций вместо
``AssetDatabase`` (методы ``resolve``/``get``): артефакты при первом обращении
распаковываются в каталог кэша в раскладке ``<xx>/<guid>/``.

Формат (little-endian)::

    заголовок: b"MEPK", версия u32, смещение индекса u64   (16 байт)
    блобы артефактов и файлов подряд (сырые или сжатые zlib)
    индекс: JSON utf-8 до конца файла (``assets`` и, с версии 2, ``files``)

Секция ``files`` (версия 2) — произвольные файлы проекта по относительному пути
(сцены, скрипты, ``Settings/*.json``, ``*.project.json``): то, что не является
ассетом ``Library/``, но нужно игре в exe без каталога ``Assets/``. Паки версии 1
читаются (``files`` пуст).

Пример:
    >>> PakBuilder(db).add_all().build(Path("game.pak"))
    >>> with PakReader(Path("game.pak")) as pak:
    ...     atlas = load_sprite_atlas(pak, guid)
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.core.resources.asset_db import AssetHandle
from engine.core.resources.errors import (
    ResourceFormatError,
    ResourceNotFoundError,
)
from engine.core.resources.guid import library_subdir, parse_guid
from engine.core.resources.importer import Artifact

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from collections.abc import Callable

    from engine.core.resources.asset_db import AssetDatabase

__all__ = ["PAK_VERSION", "PakBuilder", "PakReader", "normalize_file_path"]

#: Сигнатура файла пакета.
PAK_MAGIC = b"MEPK"

#: Версия формата (2: секция ``files``).
PAK_VERSION = 2

#: Версии, которые умеет читать :class:`PakReader`.
READABLE_VERSIONS = (1, 2)


def normalize_file_path(path: Path | str) -> str:
    """Привести путь файла к виду ключа секции ``files``.

    Разделители — ``/``, без ведущего ``/`` и ``./``; регистр сохраняется.
    """
    text = str(path).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


_HEADER = struct.Struct("<4sIQ")


class PakBuilder:
    """Сборщик пакета из импортированных ассетов.

    Args:
        db: База ассетов, из ``Library/`` которой берутся артефакты.
    """

    def __init__(self, db: AssetDatabase) -> None:
        self.db = db
        self._guids: dict[str, None] = {}
        self._files: dict[str, bytes] = {}

    def add_file(self, path: Path | str, data: bytes) -> PakBuilder:
        """Добавить файл в секцию ``files`` (повторный путь заменяет прежний).

        Args:
            path: Путь внутри пака (``Assets/Scenes/main.scene.json``).
            data: Содержимое.

        Returns:
            Сам сборщик.

        Raises:
            ValueError: Если путь пуст или содержит ``..``.
        """
        key = normalize_file_path(path)
        if not key or ".." in key.split("/"):
            raise ValueError(f"Недопустимый путь файла в паке: '{path}'")
        self._files[key] = bytes(data)
        return self

    def add_file_from_disk(self, source: Path | str, path: Path | str) -> PakBuilder:
        """Прочитать файл с диска и добавить под путём ``path``."""
        return self.add_file(path, Path(source).read_bytes())

    @property
    def files(self) -> tuple[str, ...]:
        """Пути файлов секции ``files``."""
        return tuple(self._files)

    def add(self, guid: str, *, with_dependencies: bool = True) -> PakBuilder:
        """Добавить ассет (и по умолчанию всё, от чего он зависит).

        Args:
            guid: GUID ассета.
            with_dependencies: Добавить транзитивные зависимости.

        Returns:
            Сам сборщик (для цепочек вызовов).

        Raises:
            ResourceNotFoundError: Если ассета нет или он не импортирован.
        """
        handle = self.db.resolve(guid)
        if not handle.imported:
            raise ResourceNotFoundError(f"Ассет {guid} не импортирован")
        self._guids[handle.guid] = None
        if with_dependencies:
            for dependency in self.db.dependency_closure(guid):
                found = self.db.get(dependency)
                if found is not None and found.imported:
                    self._guids[dependency] = None
        return self

    def add_all(self) -> PakBuilder:
        """Добавить все успешно импортированные ассеты базы."""
        for guid in self.db.guids:
            handle = self.db.get(guid)
            if handle is not None and handle.imported:
                self._guids[guid] = None
        return self

    def __len__(self) -> int:
        return len(self._guids)

    def build(self, path: Path | str, *, compress: bool = True) -> Path:
        """Записать пакет (атомарно: во временный файл, затем замена).

        Args:
            path: Путь к ``.pak``.
            compress: Сжимать блобы zlib, если это выгодно.

        Returns:
            Путь к готовому файлу.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".tmp")
        assets: dict[str, Any] = {}
        try:
            with temp.open("wb") as stream:
                stream.write(_HEADER.pack(PAK_MAGIC, PAK_VERSION, 0))
                for guid in self._guids:
                    assets[guid] = self._write_asset(stream, guid, compress)
                files = {
                    key: self._write_blob(stream, data, compress)
                    for key, data in sorted(self._files.items())
                }
                index_offset = stream.tell()
                document = {"version": PAK_VERSION, "assets": assets, "files": files}
                stream.write(json.dumps(document, ensure_ascii=False).encode("utf-8"))
                stream.seek(0)
                stream.write(_HEADER.pack(PAK_MAGIC, PAK_VERSION, index_offset))
            os.replace(temp, target)
        finally:
            if temp.exists():
                temp.unlink()
        return target

    @staticmethod
    def _write_blob(stream: Any, data: bytes, compress: bool) -> dict[str, Any]:
        """Записать блоб и вернуть описание (offset/size/stored/packed/crc)."""
        stored = zlib.compress(data, 6) if compress else data
        if len(stored) >= len(data):
            stored = data
        entry = {
            "offset": stream.tell(),
            "size": len(data),
            "stored": len(stored),
            "packed": stored is not data,
            "crc": zlib.crc32(data),
        }
        stream.write(stored)
        return entry

    def _write_asset(self, stream: Any, guid: str, compress: bool) -> dict[str, Any]:
        """Записать блобы артефактов ассета и вернуть его запись индекса."""
        handle = self.db.resolve(guid)
        try:
            source = handle.path.relative_to(self.db.root).as_posix()
        except ValueError:
            source = handle.path.name
        entries = []
        for artifact in handle.artifacts:
            data = handle.artifact_path(artifact).read_bytes()
            stored = zlib.compress(data, 6) if compress else data
            if len(stored) >= len(data):
                stored = data
            entries.append(
                {
                    **artifact.to_dict(),
                    "offset": stream.tell(),
                    "size": len(data),
                    "stored": len(stored),
                    "packed": stored is not data,
                    "crc": zlib.crc32(data),
                }
            )
            stream.write(stored)
        return {
            "path": source,
            "importer": handle.importer,
            "version": handle.imported_version,
            "artifacts": entries,
        }


class PakReader:
    """Чтение пакета ``.pak``.

    Args:
        path: Файл пакета.
        cache_dir: Каталог распаковки артефактов для :meth:`resolve`. По умолчанию
            создаётся временный и удаляется в :meth:`close`.
        cache_root: Где создавать временный каталог (вместо системного temp);
            можно передать функцию — она вызывается при первой распаковке.

    Raises:
        ResourceFormatError: Если файл не пакет или повреждён.
    """

    def __init__(
        self,
        path: Path | str,
        cache_dir: Path | str | None = None,
        cache_root: Path | str | Callable[[], Path | str] | None = None,
    ) -> None:
        self.path = Path(path)
        self._owns_cache = cache_dir is None
        self._cache_dir = None if cache_dir is None else Path(cache_dir)
        self._cache_root = cache_root
        self._stream = self.path.open("rb")
        self._files: dict[str, Any] = {}
        try:
            self._assets = self._read_index()
        except BaseException:
            self.close()
            raise

    @property
    def cache_dir(self) -> Path:
        """Каталог распаковки; временный создаётся при первом обращении."""
        if self._cache_dir is None:
            root = self._cache_root
            if callable(root):
                root = root()
            parent = None
            if root is not None:
                parent = Path(root)
                parent.mkdir(parents=True, exist_ok=True)
            self._cache_dir = Path(tempfile.mkdtemp(prefix="mepak_", dir=parent))
        return self._cache_dir

    def _read_index(self) -> dict[str, Any]:
        """Прочитать и проверить заголовок и индекс."""
        stream = self._stream
        header = stream.read(_HEADER.size)
        if len(header) < _HEADER.size:
            raise ResourceFormatError(f"Файл '{self.path}' слишком мал для пакета")
        magic, version, index_offset = _HEADER.unpack(header)
        if magic != PAK_MAGIC:
            raise ResourceFormatError(f"'{self.path}' не является пакетом ассетов")
        if version not in READABLE_VERSIONS:
            raise ResourceFormatError(
                f"Версия пакета {version} не поддерживается (нужна {PAK_VERSION})"
            )
        if index_offset < _HEADER.size:
            raise ResourceFormatError(f"Пакет '{self.path}' не дописан до конца")
        stream.seek(index_offset)
        try:
            document = json.loads(stream.read().decode("utf-8"))
            self._files = dict(document.get("files", {}))
            return dict(document["assets"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ResourceFormatError(f"Повреждён индекс '{self.path}': {exc}") from exc

    # ------------------------------------------------------------------ #
    # Запросы
    # ------------------------------------------------------------------ #
    def guids(self) -> tuple[str, ...]:
        """GUID всех ассетов пакета."""
        return tuple(self._assets)

    def __contains__(self, guid: object) -> bool:
        return isinstance(guid, str) and guid in self._assets

    def __len__(self) -> int:
        return len(self._assets)

    def guid_of(self, path: Path | str) -> str | None:
        """GUID по исходному пути (относительно корня ассетов базы)."""
        wanted = Path(path).as_posix()
        for guid, record in self._assets.items():
            if record["path"] == wanted:
                return guid
        return None

    def read(self, guid: str, name: str) -> bytes:
        """Прочитать артефакт целиком (с проверкой контрольной суммы).

        Raises:
            ResourceNotFoundError: Если ассета или артефакта нет.
            ResourceFormatError: Если данные повреждены.
        """
        entry = self._entry(guid, name)
        self._stream.seek(int(entry["offset"]))
        blob = self._stream.read(int(entry["stored"]))
        try:
            data = zlib.decompress(blob) if entry["packed"] else blob
        except zlib.error as exc:
            raise ResourceFormatError(f"Повреждён артефакт '{name}': {exc}") from exc
        if len(data) != entry["size"] or zlib.crc32(data) != entry["crc"]:
            raise ResourceFormatError(f"Не сходится контрольная сумма '{name}'")
        return data

    # ------------------------------------------------------------------ #
    # Секция files
    # ------------------------------------------------------------------ #
    def list_files(self, prefix: str = "") -> list[str]:
        """Пути файлов секции ``files`` с данным префиксом (по алфавиту)."""
        start = normalize_file_path(prefix)
        return sorted(key for key in self._files if key.startswith(start))

    def has_file(self, path: Path | str) -> bool:
        """Есть ли файл в секции ``files``."""
        return normalize_file_path(path) in self._files

    def read_file(self, path: Path | str) -> bytes | None:
        """Содержимое файла секции ``files`` или ``None``, если его нет.

        Raises:
            ResourceFormatError: Если данные повреждены.
        """
        key = normalize_file_path(path)
        entry = self._files.get(key)
        if entry is None:
            return None
        self._stream.seek(int(entry["offset"]))
        blob = self._stream.read(int(entry["stored"]))
        try:
            data = zlib.decompress(blob) if entry["packed"] else blob
        except zlib.error as exc:
            raise ResourceFormatError(f"Повреждён файл '{key}': {exc}") from exc
        if len(data) != entry["size"] or zlib.crc32(data) != entry["crc"]:
            raise ResourceFormatError(f"Не сходится контрольная сумма '{key}'")
        return data

    def _record(self, guid: str) -> dict[str, Any]:
        """Запись индекса по GUID."""
        try:
            return self._assets[parse_guid(guid)]
        except (KeyError, ValueError):
            raise ResourceNotFoundError(f"В пакете нет ассета {guid}") from None

    def _entry(self, guid: str, name: str) -> dict[str, Any]:
        """Запись артефакта по имени."""
        for entry in self._record(guid)["artifacts"]:
            if entry["name"] == name:
                return entry
        raise ResourceNotFoundError(f"У ассета {guid} нет артефакта '{name}'")

    # ------------------------------------------------------------------ #
    # Совместимость с AssetDatabase
    # ------------------------------------------------------------------ #
    def get(self, guid: str) -> AssetHandle | None:
        """Описание ассета или ``None``; артефакты распаковываются в кэш."""
        try:
            return self.resolve(guid)
        except ResourceNotFoundError:
            return None

    def resolve(self, guid: str) -> AssetHandle:
        """Описание ассета, пригодное для ``load_*`` (как у ``AssetDatabase``).

        Артефакты распаковываются в ``cache_dir/<xx>/<guid>/`` один раз.

        Raises:
            ResourceNotFoundError: Если ассета нет в пакете.
        """
        record = self._record(guid)
        guid = parse_guid(guid)
        directory = self.cache_dir / library_subdir(guid)
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for entry in record["artifacts"]:
            target = directory / entry["name"]
            if not target.is_file() or target.stat().st_size != entry["size"]:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self.read(guid, entry["name"]))
            artifacts.append(Artifact.from_dict(entry))
        return AssetHandle(
            guid=guid,
            path=Path(record["path"]),
            importer=str(record["importer"]),
            directory=directory,
            imported_version=int(record["version"]),
            artifacts=tuple(artifacts),
        )

    # ------------------------------------------------------------------ #
    # Жизненный цикл
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Закрыть файл и удалить временный кэш (если он свой)."""
        self._stream.close()
        if self._owns_cache and self._cache_dir is not None:
            shutil.rmtree(self._cache_dir, ignore_errors=True)

    def __enter__(self) -> PakReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
