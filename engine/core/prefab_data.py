"""Данные префаба без зависимостей от рендеринга: проверка и граф вложенности.

Используется и импортёром ассетов (ядро), и :mod:`engine.prefab` (сборка
объектов). Формат файла описан в :mod:`engine.prefab`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from engine.core.prefab_components import marker_ref_parts

__all__ = [
    "MAX_PREFAB_DEPTH",
    "PREFAB_FORMAT",
    "PREFAB_INSTANCE_TYPE",
    "PREFAB_VERSION",
    "PrefabCycleError",
    "PrefabError",
    "check_prefab_graph",
    "nested_instances",
    "validate_prefab_data",
]

PREFAB_FORMAT = "myengine.prefab"
PREFAB_VERSION = 1

#: Значение ``type`` компонента-маркера вложенного префаба.
PREFAB_INSTANCE_TYPE = "PrefabInstance"

#: Максимальная глубина вложенности (внешний префаб = 1).
MAX_PREFAB_DEPTH = 8


class PrefabError(ValueError):
    """Файл префаба повреждён или не поддерживается."""


class PrefabCycleError(PrefabError):
    """Префаб содержит сам себя (цикл) или вложенность глубже допустимой."""


def _local_refs(value: Any) -> list[str]:
    """Все ссылки ``{"local": ...}``, вложенные в произвольное значение."""
    found: list[str] = []
    if isinstance(value, dict):
        if set(value) == {"local"}:
            found.append(str(value["local"]))
        else:
            for item in value.values():
                found.extend(_local_refs(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_local_refs(item))
    return found


def validate_prefab_data(data: Any) -> None:
    """Проверить словарь префаба.

    Проверяются: формат и версия, уникальные ``local_id``, существование
    ``root``, единственный объект без родителя (корень), разрешимость всех
    ссылок ``parent`` и ``{"local": ...}``, отсутствие циклов в иерархии.

    Args:
        data: Разобранный JSON файла.

    Raises:
        PrefabError: Если любое из условий нарушено.
    """
    if not isinstance(data, dict):
        raise PrefabError("префаб должен быть JSON-объектом")
    if data.get("format") != PREFAB_FORMAT:
        raise PrefabError(
            f"неизвестный формат {data.get('format')!r} "
            f"(ожидался {PREFAB_FORMAT!r})"
        )
    version = data.get("version")
    if version != PREFAB_VERSION:
        raise PrefabError(
            f"неподдерживаемая версия префаба {version!r} "
            f"(ожидалась {PREFAB_VERSION})"
        )
    objects = data.get("objects")
    if not isinstance(objects, list) or not objects:
        raise PrefabError("в префабе нет объектов")
    ids: list[str] = []
    for entry in objects:
        if not isinstance(entry, dict) or not isinstance(entry.get("local_id"), str):
            raise PrefabError("у каждого объекта должен быть строковый local_id")
        if "/" in entry["local_id"]:
            raise PrefabError(
                f"local_id {entry['local_id']!r} содержит '/' "
                "(зарезервировано под путь во вложенном префабе)"
            )
        ids.append(entry["local_id"])
    if len(set(ids)) != len(ids):
        raise PrefabError("local_id объектов не уникальны")
    root = data.get("root")
    if root not in ids:
        raise PrefabError(f"root {root!r} не найден среди объектов")

    parent_of: dict[str, str] = {}
    for entry in objects:
        local_id = entry["local_id"]
        parent = entry.get("parent")
        if parent is None:
            continue
        if not isinstance(parent, dict) or "local" not in parent:
            raise PrefabError(f"{local_id}: parent должен быть {{'local': ...}}")
        if str(parent["local"]) not in ids:
            raise PrefabError(
                f"{local_id}: parent ссылается на несуществующий "
                f"local_id {parent['local']!r}"
            )
        parent_of[local_id] = str(parent["local"])
    roots = [local_id for local_id in ids if local_id not in parent_of]
    if roots != [root]:
        raise PrefabError(
            f"единственным объектом без parent должен быть root {root!r}, "
            f"найдено: {roots}"
        )
    for local_id in ids:
        seen = {local_id}
        current = local_id
        while current in parent_of:
            current = parent_of[current]
            if current in seen:
                raise PrefabError(f"цикл в иерархии на объекте {local_id}")
            seen.add(current)
    for entry in objects:
        for ref in _local_refs(entry.get("components", [])):
            if ref not in ids:
                raise PrefabError(
                    f"{entry['local_id']}: ссылка на несуществующий local_id {ref!r}"
                )


def nested_instances(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Вложенные префабы данных: ``(local_id, prefab_guid, prefab_path)``.

    Порядок - по объектам файла; у объекта берётся первый компонент
    ``PrefabInstance``.
    """
    found: list[tuple[str, str, str]] = []
    for entry in data.get("objects", []):
        for component in entry.get("components", []):
            if component.get("type") == PREFAB_INSTANCE_TYPE:
                found.append((str(entry["local_id"]), *marker_ref_parts(component)))
                break
    return found


def check_prefab_graph(
    data: dict[str, Any],
    own_guid: str,
    resolve: Callable[[str, str], dict[str, Any]],
) -> None:
    """Проверить вложенные префабы на циклы и глубину.

    Args:
        data: Проверенный (:func:`validate_prefab_data`) словарь префаба.
        own_guid: GUID проверяемого префаба (``""`` - неизвестен).
        resolve: ``(prefab_guid, prefab_path)`` -> словарь вложенного префаба
            (уже проверенный); сам бросает :class:`PrefabError`, если не найден.

    Raises:
        PrefabCycleError: GUID повторяется в цепочке или глубина больше
            :data:`MAX_PREFAB_DEPTH`.
        PrefabError: Вложенный префаб не найден или некорректен.
    """

    def walk(current: dict[str, Any], chain: tuple[str, ...]) -> None:
        for local_id, guid, path in nested_instances(current):
            if not guid:
                raise PrefabError(f"{local_id}: у PrefabInstance нет prefab_guid")
            if guid in chain:
                names = " -> ".join((*chain, guid))
                raise PrefabCycleError(f"цикл вложенных префабов: {names}")
            if len(chain) + 1 > MAX_PREFAB_DEPTH:
                raise PrefabCycleError(
                    f"вложенность префабов глубже {MAX_PREFAB_DEPTH}: "
                    + " -> ".join((*chain, guid))
                )
            walk(resolve(guid, path), (*chain, guid))

    # Корневой префаб - уровень 1, даже если его GUID неизвестен.
    walk(data, (own_guid,))
