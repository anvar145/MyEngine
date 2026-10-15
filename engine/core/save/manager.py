"""Менеджер слотов сохранения: запись, чтение, список, удаление.

Файл слота - zip ``<имя>.save`` с записями (в этом порядке):

* ``version`` - версия схемы, целое число текстом;
* ``meta.json`` - :class:`SaveSlotMeta`;
* ``state.json`` - ``GameState`` целиком;
* ``objects.json`` - данные объектов по ``uid`` (см. ``objects.py``);
* ``screenshot.png`` - необязательно.

Целостность проверяет CRC32 записей zip (как CRC блобов в ``.pak``). Запись
атомарна: во временный ``<имя>.save.tmp``, затем ``os.replace``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import zipfile
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.core.save.migrations import MIGRATIONS, Migration, migrate_chain
from engine.core.save.objects import apply_objects_to_scenes, collect_objects
from engine.core.save.slot import (
    SaveCorruptedError,
    SaveError,
    SaveScene,
    SaveSlot,
    SaveSlotMeta,
)

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.save.context import GameState
    from engine.core.scene import Scene

__all__ = ["SaveManager"]

_log = logging.getLogger("engine.core.save")

SUFFIX = ".save"
_NAME_RE = re.compile(r"^[\w\- ]+$")
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)  # детерминированный zip
_SCREENSHOT = "screenshot.png"


class SaveManager:
    """Слоты сохранения в каталоге ``root``.

    Args:
        root: Каталог сохранений (создаётся при первой записи); задаётся
            снаружи: плеер - ``%APPDATA%``, тесты - ``_work/saves``.
        version: Текущая версия схемы сохранений игры.
        migrations: Свой реестр миграций вместо общего ``MIGRATIONS``.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        version: int,
        migrations: dict[tuple[int, int], Migration] | None = None,
    ) -> None:
        self.root = Path(root)
        self.version = int(version)
        #: Реестр миграций (по умолчанию общий ``MIGRATIONS``).
        self.migrations = MIGRATIONS if migrations is None else migrations
        #: Идёт запись/загрузка (защита от вложенных сохранений).
        self.busy = False
        #: Автосохранение разрешено (``--no-autosave`` ставит ``False``).
        self.autosave_enabled = True
        #: Хеши состояния последних контрольных точек (только в памяти).
        self.hashes: dict[str, str] = {}

    # ------------------------------------------------------------ пути
    def _path(self, name: str) -> Path:
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ValueError(f"недопустимое имя слота: {name!r}")
        return self.root / f"{name}{SUFFIX}"

    # --------------------------------------------------------- чтение
    @staticmethod
    def _open(path: Path) -> zipfile.ZipFile:
        try:
            archive = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError) as exc:
            raise SaveCorruptedError(f"{path.name}: не удалось открыть: {exc}") from exc
        return archive

    @staticmethod
    def _read_json(archive: zipfile.ZipFile, entry: str) -> Any:
        try:
            return json.loads(archive.read(entry).decode("utf-8"))
        except KeyError as exc:
            raise SaveCorruptedError(f"нет записи {entry}") from exc
        except (zipfile.BadZipFile, zlib.error, ValueError) as exc:  # CRC, json
            raise SaveCorruptedError(f"запись {entry} повреждена: {exc}") from exc

    def slot(self, name: str) -> SaveSlot | None:
        """Слот по имени или ``None``, если файла нет.

        Raises:
            SaveCorruptedError: Файл есть, но повреждён.
        """
        path = self._path(name)
        if not path.is_file():
            return None
        with self._open(path) as archive:
            meta = SaveSlotMeta.from_dict(self._read_json(archive, "meta.json"))
        return SaveSlot(path, meta)

    def list_slots(self) -> list[SaveSlot]:
        """Все читаемые слоты, новые первыми; повреждённые пропускаются (WARNING)."""
        if not self.root.is_dir():
            return []
        slots: list[SaveSlot] = []
        for path in sorted(self.root.glob(f"*{SUFFIX}")):
            try:
                slot = self.slot(path.stem)
            except (SaveError, ValueError) as exc:
                _log.warning("Слот %s пропущен: %s", path.name, exc)
                continue
            if slot is not None:
                slots.append(slot)
        slots.sort(key=lambda s: s.meta.timestamp, reverse=True)
        return slots

    # ---------------------------------------------------------- запись
    def save(
        self,
        name: str,
        scene: Scene,
        global_state: GameState,
        *,
        playtime_sec: float = 0.0,
        screenshot: bytes | None = None,
        scenes: Sequence[SaveScene] | None = None,
        extra_scenes: Sequence[Scene] = (),
    ) -> SaveSlot:
        """Записать слот атомарно.

        Args:
            name: Имя слота.
            scene: Сцена, из которой берутся ``@save``-поля и ``on_save``.
            global_state: Глобальное состояние игры.
            playtime_sec: Время игры для метаданных.
            screenshot: PNG-байты миниатюры (необязательно).
            scenes: Список сцен для ``meta.scenes`` (активная первой); по
                умолчанию одна активная сцена с именем ``scene.name``.
            extra_scenes: Дополнительные сцены: их объекты пишутся в тот же
                ``objects.json`` (uid объектов разных сцен не пересекаются).

        Returns:
            Записанный слот.

        Raises:
            ValueError: Недопустимое имя слота.
            OSError: Ошибка записи; существующий слот при этом не тронут.
        """
        path = self._path(name)
        refs = tuple(scenes) if scenes else (SaveScene(scene.name, False),)
        meta = SaveSlotMeta(
            timestamp=time.time(),
            version=self.version,
            scene_name=refs[0].path,
            playtime_sec=float(playtime_sec),
            screenshot_path=_SCREENSHOT if screenshot else None,
            scenes=refs,
        )
        objects = collect_objects(scene)
        for other in extra_scenes:
            objects.update(collect_objects(other))
        entries: list[tuple[str, bytes]] = [
            ("version", str(self.version).encode("ascii")),
            ("meta.json", _dumps(meta.to_dict())),
            ("state.json", _dumps(global_state.to_dict())),
            ("objects.json", _dumps(objects)),
        ]
        if screenshot:
            entries.append((_SCREENSHOT, bytes(screenshot)))
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "wb") as raw:
                with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as archive:
                    for entry, data in entries:
                        info = zipfile.ZipInfo(entry, _FIXED_TIME)
                        info.compress_type = zipfile.ZIP_DEFLATED
                        archive.writestr(info, data)
                raw.flush()
                os.fsync(raw.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return SaveSlot(path, meta)

    # ---------------------------------------------------------- загрузка
    def _read_payload(self, path: Path) -> dict[str, Any]:
        """Прочитать слот как есть: ``{"meta", "state", "objects"}`` (JSON-вид).

        Raises:
            SaveCorruptedError: Файл повреждён.
        """
        with self._open(path) as archive:
            try:
                bad = archive.testzip()
            except (zlib.error, zipfile.BadZipFile) as exc:
                raise SaveCorruptedError(f"{path.name}: повреждён: {exc}") from exc
            if bad is not None:
                raise SaveCorruptedError(f"{path.name}: не совпал CRC записи {bad}")
            meta = self._read_json(archive, "meta.json")
            try:
                declared = int(archive.read("version").decode("ascii"))
            except (KeyError, ValueError) as exc:
                raise SaveCorruptedError(f"{path.name}: нет версии") from exc
            parsed = SaveSlotMeta.from_dict(meta)
            if declared != parsed.version:
                raise SaveCorruptedError(
                    f"{path.name}: версия записи {declared} != meta {parsed.version}"
                )
            return {
                "meta": meta,
                "state": self._read_json(archive, "state.json"),
                "objects": self._read_json(archive, "objects.json"),
            }

    @staticmethod
    def _apply_chain(
        payload: dict[str, Any], chain: list[Migration], to_version: int
    ) -> dict[str, Any]:
        for step in chain:
            payload = step(payload)
        payload["meta"] = {**payload["meta"], "version": to_version}
        return payload

    def migrate(
        self, slot: SaveSlot, from_version: int, to_version: int
    ) -> dict[str, Any]:
        """Данные слота, приведённые от ``from_version`` к ``to_version``.

        Файл не перезаписывается: новая версия ляжет на диск при следующем
        ``save``.

        Returns:
            ``{"meta", "state", "objects"}`` в виде ``to_version``.

        Raises:
            SaveVersionError: Сохранение новее игры или нет цепочки миграций.
            SaveCorruptedError: Файл повреждён.
        """
        chain = migrate_chain(from_version, to_version, self.migrations)
        return self._apply_chain(self._read_payload(slot.path), chain, to_version)

    def read(self, name: str) -> tuple[SaveSlot, dict[str, Any], dict[str, Any]]:
        """Прочитать слот, проверить CRC и привести к текущей версии.

        Returns:
            ``(слот, state, objects)`` в виде текущей схемы.

        Raises:
            FileNotFoundError: Слота нет.
            SaveCorruptedError: Файл повреждён.
            SaveVersionError: Сохранение новее игры или нет миграции.
        """
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(f"слот {name!r} не найден")
        payload = self._read_payload(path)
        meta = SaveSlotMeta.from_dict(payload["meta"])
        if meta.version != self.version:
            chain = migrate_chain(meta.version, self.version, self.migrations)
            _log.info(
                "Миграция слота %s: %d -> %d (шагов: %d)",
                name,
                meta.version,
                self.version,
                len(chain),
            )
            payload = self._apply_chain(payload, chain, self.version)
            meta = SaveSlotMeta.from_dict(payload["meta"])
        return SaveSlot(path, meta), payload["state"], payload["objects"]

    def load(
        self, name: str, scene: Scene, extra_scenes: Sequence[Scene] = ()
    ) -> SaveSlot:
        """Применить слот к сцене: ``GameState``, затем объекты.

        Сцена должна быть уже построена из шаблона. Uid, которых нет ни в
        одной из сцен, пропускаются с одним WARNING.

        Args:
            name: Имя слота.
            scene: Активная сцена (её ``GameState`` загружается).
            extra_scenes: Дополнительные сцены, получающие свои объекты.

        Returns:
            Загруженный слот.

        Raises:
            FileNotFoundError: Слота нет.
            SaveError: Файл повреждён или версия не поддерживается.
        """
        slot, state, objects = self.read(name)
        scene.state.load_dict(state)
        apply_objects_to_scenes([scene, *extra_scenes], objects)
        return slot

    def delete(self, name: str) -> bool:
        """Удалить слот; ``True``, если файл был."""
        path = self._path(name)
        if not path.is_file():
            return False
        path.unlink()
        return True


def _dumps(data: Any) -> bytes:
    text = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=1)
    return text.encode("utf-8")
