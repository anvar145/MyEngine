"""Префабы: формат ``*.prefab.json`` и сборка объектов из него.

Префаб - поддерево объектов, записанное отдельно от сцены::

    {
      "format": "myengine.prefab",
      "version": 1,
      "name": "Enemy",
      "root": "L0",
      "objects": [
        {"local_id": "L0", "name": "Enemy", "tag": "Untagged", "layer": 1,
         "active": true, "transform": {...}, "components": [...]},
        {"local_id": "L1", "name": "Weapon", "parent": {"local": "L0"},
         "tag": "Untagged", "layer": 1, "active": true,
         "transform": {...}, "components": [...]}
      ]
    }

Внутри файла нет ``uid``: объекты адресуются ``local_id`` (``L0``, ``L1``, ...
по порядку), ссылки внутри префаба - ``{"local": "L1"}``. Настоящие ``uid``
объекты получают при создании в сцене (детерминированно от инстанса и
``local_id``). Общий с :mod:`engine.scene_io` код записи/чтения объектов
параметризован ``ref_key`` (``"uid"`` для сцены, ``"local"`` для префаба).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.core.prefab_components import PrefabInstance, marker_ref_parts
from engine.core.prefab_data import (
    PREFAB_FORMAT,
    PREFAB_INSTANCE_TYPE,
    PREFAB_VERSION,
    PrefabCycleError,
    PrefabError,
    check_prefab_graph,
    validate_prefab_data,
)
from engine.prefab_overrides import (
    ROOT,
    ROOT_FIELDS,
    TRANSFORM,
    TRANSFORM_FIELDS,
    apply_overrides,
    diff_entry,
    get_field,
    merge_stale,
    override_key,
    set_field,
    to_local_refs,
    to_uid_refs,
    unset_field,
)
from engine.scene_io import (
    add_component_from_payload,
    build_objects,
    component_payload_pairs,
    serialize_objects,
)

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.game_object import GameObject
    from engine.core.math3d import Quaternion, Vector3
    from engine.core.resources import AssetDatabase
    from engine.core.scene import Scene

__all__ = [
    "PREFAB_FORMAT",
    "PREFAB_VERSION",
    "Prefab",
    "PrefabCycleError",
    "PrefabError",
    "PrefabLoader",
    "InstanceSnapshot",
    "build_prefab_objects",
    "capture_instances",
    "collapse_instances",
    "dump_prefab",
    "expand_scene_instances",
    "find_instance",
    "forget_override",
    "get_overrides",
    "overrides_by_target",
    "restore_instances",
    "revert_override",
    "instance_uid",
    "instantiate_prefab",
    "load_prefab_file",
    "make_prefab_loader",
    "nested_instance_uid",
    "object_uid",
    "prefab_from_dict",
    "prefab_to_dict",
    "set_default_prefab_loader",
    "validate_prefab_data",
]

#: ``(prefab_guid, prefab_path) -> Prefab``; бросает :class:`PrefabError`.
PrefabLoader = Callable[[str, str], "Prefab"]

_default_loader: PrefabLoader | None = None
_log = logging.getLogger("engine.prefab")

@dataclass
class Prefab:
    """Загруженный префаб.

    Attributes:
        name: Имя префаба.
        root: ``local_id`` корня.
        objects: Записи объектов в формате файла (не изменяются).
        guid: GUID ассета (``""``, если префаб не из базы ассетов).
    """

    name: str
    root: str
    objects: list[dict[str, Any]] = field(default_factory=list)
    guid: str = ""
    path: str = ""

    @property
    def local_ids(self) -> list[str]:
        """``local_id`` всех объектов в порядке файла."""
        return [str(entry["local_id"]) for entry in self.objects]

    def to_dict(self) -> dict[str, Any]:
        """Словарь в формате файла."""
        return {
            "format": PREFAB_FORMAT,
            "version": PREFAB_VERSION,
            "name": self.name,
            "root": self.root,
            "objects": self.objects,
        }

    def get_overrides(
        self,
        instance_root: GameObject,
        *,
        loader: PrefabLoader | None = None,
        assets: AssetDatabase | None = None,
    ) -> list[dict[str, Any]]:
        """Overrides живого экземпляра этого префаба (диф с префабом).

        Returns:
            Записи формата :mod:`engine.prefab_overrides`; устаревших (stale)
            среди них нет.
        """
        return _live_overrides(self, instance_root, loader, assets)

    def _entry(self, local_id: str) -> dict[str, Any]:
        for entry in self.objects:
            if entry["local_id"] == local_id:
                return entry
        raise PrefabError(f"в префабе нет объекта {local_id!r}")

    def apply_override_to_asset(
        self,
        target: str,
        component: str,
        field: str | None,
        value: Any = None,
        *,
        index: int = 0,
        unset: bool = False,
        add: bool = False,
    ) -> None:
        """Записать значение override в данные префаба (``self.objects``).

        Файл не пишется: вызывающий сохраняет :func:`dump_prefab`. Цель во
        вложенном префабе (``target`` с ``/``) записывается как override
        маркера заместителя - файл вложенного префаба не меняется.

        Raises:
            PrefabError: Нет цели/компонента/поля или значение ссылается на
                объект вложенного префаба.
        """
        if "/" in target:
            head, _, rest = target.partition("/")
            marker = _first_instance(self._entry(head))
            if marker is None:
                raise PrefabError(f"объект {head!r} не является вложенным префабом")
            record = _make_record(
                rest,
                component,
                index,
                field,
                _strip_prefix(value, head + "/"),
                unset=unset,
                add=add,
            )
            items = marker.setdefault("overrides", [])
            for position, existing in enumerate(items):
                if override_key(existing) == override_key(record):
                    items[position] = record
                    break
            else:
                items.append(record)
            return
        entry = self._entry(target)
        _check_plain_refs(value)
        if component == ROOT:
            if field not in ROOT_FIELDS:
                raise PrefabError(f"неизвестное поле объекта {field!r}")
            entry[field] = copy.deepcopy(value)
            return
        if component == TRANSFORM:
            if field not in TRANSFORM_FIELDS:
                raise PrefabError(f"неизвестное поле трансформа {field!r}")
            transform = dict(entry.get("transform") or {})
            transform[field] = copy.deepcopy(value)
            entry["transform"] = transform
            return
        components = entry.setdefault("components", [])
        if add:
            components.append({"type": component, **copy.deepcopy(value)})
            return
        matches = [c for c in components if c.get("type") == component]
        if index >= len(matches) or not isinstance(field, str):
            raise PrefabError(f"{target}: нет компонента {component}[{index}]")
        if unset:
            unset_field(matches[index], field)
        else:
            set_field(matches[index], field, value)

    def revert_override(
        self,
        instance_root: GameObject,
        override: dict[str, Any],
        *,
        loader: PrefabLoader | None = None,
        assets: AssetDatabase | None = None,
    ) -> None:
        """Вернуть значение override из префаба в живом экземпляре."""
        _revert(self, instance_root, override, loader, assets)


def _subtree(root: GameObject) -> list[GameObject]:
    """Поддерево в порядке обхода в глубину: родитель раньше потомков.

    Вложенный экземпляр префаба (объект с ``PrefabInstance``) схлопывается в
    один объект-заместитель: его потомки принадлежат вложенному префабу.
    """
    result: list[GameObject] = []
    stack = [root]
    while stack:
        current = stack.pop()
        result.append(current)
        if current.get_component(PrefabInstance) is None:
            stack.extend(reversed(current.children))
    return result


def prefab_to_dict(
    root: GameObject,
    assets: AssetDatabase | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Записать поддерево объекта как префаб.

    Args:
        root: Корень поддерева; получает ``local_id`` ``L0``.
        assets: База ассетов (``mesh_guid``, пути скриптов).
        name: Имя префаба (по умолчанию - имя корня).

    Returns:
        Словарь формата ``myengine.prefab``.
    """
    entries = serialize_objects(_subtree(root), assets, ref_key="local")
    entries[0].pop("parent", None)  # корень не привязан к родителю сцены
    for entry in entries:
        for component in entry["components"]:
            # вложенный маркер не несёт instance_uid (см. PrefabInstance)
            if component.get("type") == PREFAB_INSTANCE_TYPE:
                component.pop("instance_uid", None)
    return {
        "format": PREFAB_FORMAT,
        "version": PREFAB_VERSION,
        "name": name if name is not None else root.name,
        "root": "L0",
        "objects": entries,
    }


def dump_prefab(data: dict[str, Any]) -> str:
    """Детерминированный JSON префаба (как у сцен: отступ 2, ``\\n`` в конце)."""
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def prefab_from_dict(data: dict[str, Any], guid: str = "") -> Prefab:
    """Разобрать и проверить префаб.

    Raises:
        PrefabError: См. :func:`validate_prefab_data`.
    """
    validate_prefab_data(data)
    return Prefab(
        name=str(data.get("name", "Prefab")),
        root=str(data["root"]),
        objects=list(data["objects"]),
        guid=guid,
    )


def load_prefab_file(path: Path | str, guid: str = "") -> Prefab:
    """Прочитать префаб из файла.

    Raises:
        PrefabError: Некорректный JSON или содержимое.
        OSError: Файл недоступен.
    """
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PrefabError(f"некорректный JSON: {exc}") from exc
    return prefab_from_dict(data, guid)


def _to_scene_refs(value: Any, uid_for: Callable[[str], str]) -> Any:
    """``{"local": L}`` -> ``{"uid": uid_for(L)}`` во вложенном значении."""
    if isinstance(value, dict):
        if set(value) == {"local"}:
            return {"uid": uid_for(str(value["local"]))}
        return {key: _to_scene_refs(item, uid_for) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_scene_refs(item, uid_for) for item in value]
    return value


def build_prefab_objects(
    scene: Scene,
    prefab: Prefab,
    uid_for: Callable[[str], str],
    assets: AssetDatabase | None = None,
) -> dict[str, GameObject]:
    """Создать в сцене объекты префаба (без раскрытия вложенных).

    Args:
        scene: Сцена.
        prefab: Проверенный префаб.
        uid_for: ``local_id`` -> ``uid`` создаваемого объекта.
        assets: База ассетов.

    Returns:
        ``local_id`` -> созданный объект.
    """
    entries: list[dict[str, Any]] = []
    for source in prefab.objects:
        local_id = str(source["local_id"])
        entry = {key: value for key, value in source.items() if key != "local_id"}
        entry["uid"] = uid_for(local_id)
        parent = source.get("parent")
        entry["parent"] = (
            {"uid": uid_for(str(parent["local"]))} if parent is not None else None
        )
        entry["components"] = _to_scene_refs(source.get("components", []), uid_for)
        entries.append(entry)
    created = build_objects(scene, entries, assets, expand_instances=False)
    return dict(zip(prefab.local_ids, created, strict=True))


# --------------------------------------------------------------------------- #
# Загрузка по GUID/пути
# --------------------------------------------------------------------------- #
def set_default_prefab_loader(loader: PrefabLoader | None) -> None:
    """Задать загрузчик префабов по умолчанию (``None`` - сбросить).

    Используется, когда у сцены нет собственного ``prefab_loader``.
    """
    global _default_loader  # noqa: PLW0603
    _default_loader = loader


def make_prefab_loader(
    assets: AssetDatabase | None = None, root: Path | str | None = None
) -> PrefabLoader:
    """Загрузчик префабов: по GUID через базу ассетов, иначе по пути.

    Args:
        assets: База ассетов проекта.
        root: Дополнительный каталог для относительных путей.

    Returns:
        Функция ``(guid, path) -> Prefab``.
    """
    bases: list[Path] = []
    if root is not None:
        bases.append(Path(root))
    if assets is not None:
        bases.extend((assets.root.parent, assets.root))

    def load(guid: str, path: str) -> Prefab:
        source: Path | None = None
        if assets is not None and guid:
            handle = assets.get(guid)
            if handle is not None:
                source = Path(handle.path)
        if source is None and path:
            relative = Path(path)
            candidates = (
                [relative] if relative.is_absolute() else [b / relative for b in bases]
            )
            source = next((c for c in candidates if c.is_file()), None)
        if source is None or not source.is_file():
            raise PrefabError(f"префаб не найден (guid={guid!r}, путь={path!r})")
        found_guid = guid
        if assets is not None and not found_guid:
            found_guid = assets.guid_of(source) or ""
        prefab = load_prefab_file(source, found_guid)
        prefab.path = _project_path(source, bases)
        return prefab

    return load


def _project_path(source: Path, bases: list[Path]) -> str:
    """Путь файла относительно проекта (если файл внутри него)."""
    for base in bases:
        try:
            return source.relative_to(base).as_posix()
        except ValueError:
            continue
    return source.as_posix()


# --------------------------------------------------------------------------- #
# uid экземпляров
# --------------------------------------------------------------------------- #
def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def instance_uid(prefab_guid: str, spawn_serial: int) -> str:
    """Идентификатор экземпляра: ``blake2b("inst:<guid>:<serial>")``."""
    return _digest(f"inst:{prefab_guid}:{spawn_serial}")


def object_uid(inst_uid: str, local_id: str) -> str:
    """``uid`` объекта экземпляра: ``blake2b("<instance_uid>:<local_id>")``."""
    return _digest(f"{inst_uid}:{local_id}")


def nested_instance_uid(placeholder_uid: str, prefab_guid: str) -> str:
    """Идентификатор вложенного экземпляра.

    Зависит от ``uid`` объекта, на котором стоит ``PrefabInstance`` во внешнем
    префабе (а не от корня внешнего префаба).
    """
    return _digest(f"nest:{placeholder_uid}:{prefab_guid}")


# --------------------------------------------------------------------------- #
# Раскрытие и создание
# --------------------------------------------------------------------------- #
def _instance_payload(
    guid: str, path: str, overrides: list[Any], inst_uid: str = ""
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": PREFAB_INSTANCE_TYPE,
        "prefab_guid": guid,
        "prefab_path": path,
    }
    if inst_uid:
        payload["instance_uid"] = inst_uid
    payload["overrides"] = list(overrides)
    return payload


def _first_instance(entry: dict[str, Any]) -> dict[str, Any] | None:
    for component in entry.get("components", []):
        if component.get("type") == PREFAB_INSTANCE_TYPE:
            return component
    return None


def _expand(
    prefab: Prefab,
    inst_uid: str,
    loader: PrefabLoader,
    root_marker: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, dict[str, str]]:
    """Развернуть префаб (с вложенными) в записи формата сцены.

    Args:
        prefab: Префаб.
        inst_uid: Идентификатор этого экземпляра.
        loader: Загрузчик вложенных префабов.
        root_marker: Описание ``PrefabInstance`` для корня экземпляра.

    Returns:
        ``(записи, uid корня, пути)``; ``пути`` - ``uid`` -> путь объекта в
        экземпляре (``local_id``, для вложенных ``"<заместитель>/<путь>"``) в
        порядке записей. Объект-заместитель вложенного префаба заменяется
        корнем вложенного экземпляра: имя, активность, тег, слой, трансформ и
        родитель берутся из записи заместителя; прочие его компоненты
        игнорируются. Overrides маркера заместителя применяются к вложенному
        префабу до этого.
    """
    uid_of: dict[str, str] = {}
    expanded: dict[str, tuple[list[dict[str, Any]], str, dict[str, str]]] = {}
    for source in prefab.objects:
        local_id = str(source["local_id"])
        uid_of[local_id] = object_uid(inst_uid, local_id)
        marker = _first_instance(source)
        if marker is None:
            continue
        guid, path = marker_ref_parts(marker)
        nested = loader(guid, path)
        nested_marker = _instance_payload(
            guid, path, list(marker.get("overrides") or [])
        )
        sub_entries, sub_root, sub_paths = _expand(
            nested,
            nested_instance_uid(uid_of[local_id], guid),
            loader,
            nested_marker,
        )
        if nested_marker["overrides"]:
            processed = apply_overrides(
                sub_entries, sub_paths, nested_marker["overrides"]
            )
            if any(rec.get("stale") for rec in processed):
                _log.warning(
                    "Префаб '%s': часть overrides заместителя '%s' не применилась",
                    prefab.name,
                    local_id,
                )
        if len(source.get("components", [])) > 1:
            _log.warning(
                "Префаб '%s': компоненты заместителя '%s' кроме PrefabInstance "
                "игнорируются",
                prefab.name,
                local_id,
            )
        expanded[local_id] = (sub_entries, sub_root, sub_paths)
        uid_of[local_id] = sub_root

    entries: list[dict[str, Any]] = []
    raw_paths: dict[str, str] = {}
    for source in prefab.objects:
        local_id = str(source["local_id"])
        parent = source.get("parent")
        parent_ref = {"uid": uid_of[str(parent["local"])]} if parent else None
        if local_id in expanded:
            sub_entries, sub_root, sub_paths = expanded[local_id]
            for uid, sub_path in sub_paths.items():
                raw_paths[uid] = (
                    local_id if uid == sub_root else f"{local_id}/{sub_path}"
                )
            for entry in sub_entries:
                if entry["uid"] == sub_root:
                    for key in ("name", "tag", "layer", "active", "transform"):
                        if key in source:
                            entry[key] = source[key]
                    entry["parent"] = parent_ref
            entries.extend(sub_entries)
            continue
        entry = {key: value for key, value in source.items() if key != "local_id"}
        entry["uid"] = uid_of[local_id]
        entry["parent"] = parent_ref
        entry["components"] = _to_scene_refs(
            source.get("components", []), uid_of.__getitem__
        )
        raw_paths[entry["uid"]] = local_id
        entries.append(entry)

    root_uid = uid_of[prefab.root]
    for entry in entries:
        if entry["uid"] == root_uid:
            # один маркер на корне: внешний экземпляр вытесняет внутренний
            entry["components"] = [
                c for c in entry["components"] if c.get("type") != PREFAB_INSTANCE_TYPE
            ]
            entry["components"].append(dict(root_marker))
    paths = {entry["uid"]: raw_paths[entry["uid"]] for entry in entries}
    return entries, root_uid, paths


def _coerce_prefab(prefab: Prefab | str, loader: PrefabLoader) -> Prefab:
    if isinstance(prefab, Prefab):
        return prefab
    return loader(str(prefab), str(prefab))


def instantiate_prefab(
    scene: Scene,
    prefab: Prefab | str,
    parent: GameObject | None = None,
    position: Vector3 | None = None,
    rotation: Quaternion | None = None,
    *,
    loader: PrefabLoader | None = None,
    assets: AssetDatabase | None = None,
) -> GameObject:
    """Создать в сцене экземпляр префаба (см. :meth:`Scene.instantiate`).

    Raises:
        PrefabCycleError: Цикл вложенных префабов или глубина больше допустимой.
        PrefabError: Префаб или вложенный не найден / некорректен.
        ValueError: ``parent`` принадлежит другой сцене.
    """
    if parent is not None and parent.scene is not scene:
        raise ValueError("Родитель принадлежит другой сцене")
    active_loader = (
        loader or scene.prefab_loader or _default_loader or make_prefab_loader(assets)
    )
    source = _coerce_prefab(prefab, active_loader)
    check_prefab_graph(
        source.to_dict(),
        source.guid,
        lambda guid, path: active_loader(guid, path).to_dict(),
    )
    key = source.guid or source.path or source.name
    while True:
        scene._spawn_serial += 1  # noqa: SLF001
        inst = instance_uid(key, scene._spawn_serial)  # noqa: SLF001
        marker = _instance_payload(source.guid, source.path, [], inst)
        entries, root_uid, _paths = _expand(source, inst, active_loader, marker)
        if all(scene.find_by_uid(entry["uid"]) is None for entry in entries):
            break
    build_objects(scene, entries, assets, expand_instances=False)
    root = scene.find_by_uid(root_uid)
    if root is None:  # pragma: no cover - build_objects всегда создаёт корень
        raise PrefabError("корень экземпляра не создан")
    if parent is not None:
        root.set_parent(parent, keep_world=False)
    if position is not None:
        root.transform.position = position
    if rotation is not None:
        root.transform.rotation = rotation
    return root


# --------------------------------------------------------------------------- #
# Overrides: дифф при сохранении, применение при загрузке
# --------------------------------------------------------------------------- #
_CANON_UID = "0" * 16
_BASE_CACHE: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
_BASE_CACHE_SIZE = 32


def _scene_loader(
    scene: Scene | None,
    assets: AssetDatabase | None,
    loader: PrefabLoader | None = None,
) -> PrefabLoader:
    own = scene.prefab_loader if scene is not None else None
    return loader or own or _default_loader or make_prefab_loader(assets)


def _find_marker(entry: dict[str, Any]) -> dict[str, Any] | None:
    return _first_instance(entry)


def _path_entry(entry: dict[str, Any], paths: dict[str, str]) -> dict[str, Any]:
    """Запись объекта в пространстве путей (ссылки на членов - ``{"local"}``)."""
    result = {
        key: entry[key]
        for key in ("name", "tag", "layer", "active", "transform")
        if key in entry
    }
    result["components"] = to_local_refs(entry.get("components", []), paths)
    return result


def _layout(
    prefab: Prefab, inst_uid: str, loader: PrefabLoader
) -> tuple[list[dict[str, Any]], str, dict[str, str]]:
    """Раскрытие экземпляра ``inst_uid``: записи, uid корня, пути."""
    check_prefab_graph(
        prefab.to_dict(),
        prefab.guid,
        lambda guid, path: loader(guid, path).to_dict(),
    )
    marker = _instance_payload(prefab.guid, prefab.path, [], inst_uid)
    return _expand(prefab, inst_uid, loader, marker)


def _normalized_base(
    prefab: Prefab, loader: PrefabLoader, assets: AssetDatabase | None
) -> dict[str, dict[str, Any]]:
    """База сравнения: путь -> запись, как её сериализует живой объект.

    Префаб раскрывается с каноническим ``instance_uid`` и проходит через
    временную сцену (``build_objects`` + ``serialize_objects``): так значения
    по умолчанию, GUID мешей и пути скриптов записываются тем же кодом, что и у
    экземпляра, и дифф не видит ложных различий. Результат кэшируется по
    содержимому раскрытия.
    """
    entries, _root, paths = _expand(
        prefab,
        _CANON_UID,
        loader,
        _instance_payload(prefab.guid, prefab.path, []),
    )
    root_dir = str(assets.root) if assets is not None else ""
    key = _digest(json.dumps(entries, sort_keys=True, default=str) + root_dir)
    cached = _BASE_CACHE.get(key)
    if cached is not None:
        _BASE_CACHE.move_to_end(key)
        return cached

    from engine.core.event_bus import EventBus  # noqa: PLC0415
    from engine.core.scene import Scene  # noqa: PLC0415

    scratch = Scene("prefab-base", event_bus=EventBus())
    try:
        built = build_objects(scratch, entries, assets, expand_instances=False)
        serialized = serialize_objects(built, assets)
    except Exception:  # noqa: BLE001
        _log.warning(
            "Префаб '%s': не удалось собрать базу сравнения overrides",
            prefab.name,
            exc_info=True,
        )
        serialized = entries
    finally:
        scratch.clear()
    base = {
        paths[entry["uid"]]: _path_entry(entry, paths)
        for entry in serialized
        if entry["uid"] in paths
    }
    _BASE_CACHE[key] = base
    while len(_BASE_CACHE) > _BASE_CACHE_SIZE:
        _BASE_CACHE.popitem(last=False)
    return base


def _diff_against_base(
    prefab: Prefab,
    root_uid: str,
    paths: dict[str, str],
    live_by_uid: dict[str, dict[str, Any]],
    loader: PrefabLoader,
    assets: AssetDatabase | None,
) -> list[dict[str, Any]]:
    base = _normalized_base(prefab, loader, assets)
    diff: list[dict[str, Any]] = []
    for uid, path in paths.items():
        live = live_by_uid.get(uid)
        reference = base.get(path)
        if live is None or reference is None:
            continue
        diff.extend(
            diff_entry(
                path,
                reference,
                _path_entry(live, paths),
                skip_transform=uid == root_uid,
            )
        )
    return diff


def _live_overrides(
    prefab: Prefab,
    instance_root: GameObject,
    loader: PrefabLoader | None,
    assets: AssetDatabase | None,
) -> list[dict[str, Any]]:
    marker = instance_root.get_component(PrefabInstance)
    if marker is None or not marker.instance_uid:
        return []
    scene = instance_root.scene
    active = _scene_loader(scene, assets, loader)
    _entries, root_uid, paths = _layout(prefab, marker.instance_uid, active)
    members = [scene.find_by_uid(uid) for uid in paths] if scene is not None else []
    live = serialize_objects([m for m in members if m is not None], assets)
    return _diff_against_base(
        prefab, root_uid, paths, {e["uid"]: e for e in live}, active, assets
    )


def get_overrides(
    instance_root: GameObject,
    *,
    loader: PrefabLoader | None = None,
    assets: AssetDatabase | None = None,
) -> list[dict[str, Any]]:
    """Overrides живого экземпляра (корень с ``PrefabInstance``).

    Returns:
        Дифф значений экземпляра с префабом; пусто, если объект не корень
        экземпляра. Устаревшие (stale) записи не включаются.

    Raises:
        PrefabError: Префаб не найден или некорректен.
    """
    marker = instance_root.get_component(PrefabInstance)
    if marker is None or not marker.instance_uid:
        return []
    active = _scene_loader(instance_root.scene, assets, loader)
    prefab = active(marker.prefab_guid, marker.prefab_path)
    return _live_overrides(prefab, instance_root, active, assets)


def overrides_by_target(
    overrides: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Группировка записей по ``target`` (для инспектора)."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in overrides:
        grouped.setdefault(str(rec.get("target")), []).append(rec)
    return grouped


def find_instance(
    game_object: GameObject,
    *,
    loader: PrefabLoader | None = None,
    assets: AssetDatabase | None = None,
) -> tuple[GameObject, str] | None:
    """Экземпляр, которому принадлежит объект: ``(корень, путь объекта)``.

    ``None``, если объект не входит в экземпляр (добавленные в экземпляр
    объекты тоже не входят) или префаб не загружается.
    """
    node: GameObject | None = game_object
    while node is not None:
        marker = node.get_component(PrefabInstance)
        if marker is not None and marker.instance_uid:
            try:
                active = _scene_loader(game_object.scene, assets, loader)
                prefab = active(marker.prefab_guid, marker.prefab_path)
                _e, _r, paths = _layout(prefab, marker.instance_uid, active)
            except PrefabError:
                return None
            path = paths.get(game_object.uid)
            if path is not None:
                return node, path
        node = node.parent
    return None


def collapse_instances(
    entries: list[dict[str, Any]],
    scene: Scene,
    assets: AssetDatabase | None = None,
) -> list[dict[str, Any]]:
    """Свернуть экземпляры префабов в записи сцены: корень + overrides.

    Записи членов экземпляра заменяются одной записью корня
    (``uid``, ``parent``, ``transform``, маркер ``PrefabInstance`` со списком
    overrides); добавленные в экземпляр объекты остаются обычными записями.
    Устаревшие (stale) overrides маркера сохраняются. Если префаб не
    загружается, экземпляр пишется развёрнуто (без ``instance_uid``) - данные
    не теряются; запись-заглушка без объектов остаётся как есть.

    Args:
        entries: Результат ``serialize_objects`` (на месте не меняется).
        scene: Сцена (источник загрузчика префабов).
        assets: База ассетов.
    """
    loader = _scene_loader(scene, assets)
    live_by_uid = {entry["uid"]: entry for entry in entries}
    first_member: dict[str, dict[str, Any]] = {}
    members: set[str] = set()
    replaced: dict[str, dict[str, Any]] = {}
    for entry in entries:
        marker = _find_marker(entry)
        if marker is None or not marker.get("instance_uid"):
            continue
        try:
            prefab = loader(*marker_ref_parts(marker))
            inst = str(marker["instance_uid"])
            _built, root_uid, paths = _layout(prefab, inst, loader)
            if root_uid != entry["uid"]:
                raise PrefabError("uid корня не совпадает с раскрытием префаба")
            diff = _diff_against_base(
                prefab, root_uid, paths, live_by_uid, loader, assets
            )
        except PrefabError as exc:
            parent_ref = {"uid": entry["uid"]}
            if any(e.get("parent") == parent_ref for e in entries):
                _log.warning(
                    "Экземпляр '%s' записан развёрнуто: %s", entry.get("name"), exc
                )
                fixed = dict(entry)
                fixed["components"] = [
                    {k: v for k, v in c.items() if k != "instance_uid"}
                    if c is marker
                    else c
                    for c in entry["components"]
                ]
                replaced[entry["uid"]] = fixed
            continue
        new_marker = dict(marker)
        new_marker["overrides"] = merge_stale(
            diff, list(marker.get("overrides") or [])
        )
        record = {
            "uid": entry["uid"],
            "parent": entry.get("parent"),
            "transform": entry["transform"],
            "components": [new_marker],
        }
        present = [uid for uid in paths if uid in live_by_uid]
        first_member[present[0]] = record
        members.update(present)
    result: list[dict[str, Any]] = []
    for entry in entries:
        uid = entry["uid"]
        if uid in first_member:
            result.append(first_member[uid])
        elif uid not in members:
            result.append(replaced.get(uid, entry))
    return result


def expand_scene_instances(
    scene: Scene,
    entries: list[dict[str, Any]],
    assets: AssetDatabase | None = None,
) -> list[dict[str, Any]]:
    """Раскрыть записи корней экземпляров (маркер с ``instance_uid``).

    Порядок: раскрытие префаба -> положение корня из записи сцены -> overrides
    (``stale`` проставляется/снимается) -> ссылки на uid разрешает
    ``build_objects``. Если префаб не загружается, запись остаётся как есть.
    """
    loader = _scene_loader(scene, assets)
    result: list[dict[str, Any]] = []
    for entry in entries:
        marker = _find_marker(entry)
        if marker is None or not marker.get("instance_uid"):
            result.append(entry)
            continue
        guid, path = marker_ref_parts(marker)
        try:
            prefab = loader(guid, path)
            inst = str(marker["instance_uid"])
            built, root_uid, paths = _layout(prefab, inst, loader)
        except PrefabError as exc:
            _log.warning(
                "Экземпляр префаба '%s' не раскрыт: %s", entry.get("name"), exc
            )
            result.append(entry)
            continue
        if entry.get("uid") != root_uid:
            _log.warning(
                "Экземпляр '%s': uid корня %s заменён на %s",
                entry.get("name"),
                entry.get("uid"),
                root_uid,
            )
        root_entry = next(e for e in built if e["uid"] == root_uid)
        root_entry["parent"] = entry.get("parent")
        if entry.get("transform"):
            root_entry["transform"] = entry["transform"]
        for key in ("name", "tag", "layer", "active"):
            if key in entry:
                root_entry[key] = entry[key]
        processed = apply_overrides(
            built, paths, [dict(r) for r in marker.get("overrides") or []]
        )
        for component in root_entry["components"]:
            if component.get("type") == PREFAB_INSTANCE_TYPE:
                component["overrides"] = processed
        stale = [r for r in processed if r.get("stale")]
        if stale:
            _log.warning(
                "Экземпляр '%s': %d overrides не применились (нет цели или "
                "компонента в префабе)",
                root_entry.get("name"),
                len(stale),
            )
        result.extend(built)
    return result


def forget_override(instance_root: GameObject, override: dict[str, Any]) -> bool:
    """Убрать запись из сохранённого списка ``PrefabInstance.overrides``.

    Живые значения не меняются: после «Применить к префабу» записи в диффе
    больше нет, а старый список в маркере нужен только для stale-записей.
    """
    marker = instance_root.get_component(PrefabInstance)
    if marker is None:
        return False
    key = override_key(override)
    kept = [r for r in marker.overrides if override_key(r) != key]
    changed = len(kept) != len(marker.overrides)
    marker.overrides = kept
    return changed


def _make_record(
    target: str,
    component: str,
    index: int,
    field: str | None,
    value: Any,
    *,
    unset: bool,
    add: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {"target": target, "component": component}
    if index:
        record["index"] = index
    if field is not None and not add:
        record["field"] = field
    if unset:
        record["unset"] = True
    else:
        if add:
            record["add"] = True
        record["value"] = copy.deepcopy(value)
    return record


def _strip_prefix(value: Any, prefix: str) -> Any:
    """Пути ссылок относительно вложенного префаба; чужие пути - ошибка."""
    if isinstance(value, dict):
        if set(value) == {"local"}:
            path = str(value["local"])
            if not path.startswith(prefix):
                raise PrefabError(
                    f"ссылка {path!r} выходит за пределы вложенного префаба"
                )
            return {"local": path[len(prefix) :]}
        return {k: _strip_prefix(v, prefix) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_prefix(v, prefix) for v in value]
    return value


def _check_plain_refs(value: Any) -> None:
    """В данных объекта префаба нельзя ссылаться на объекты вложенных."""
    if isinstance(value, dict):
        if set(value) == {"local"} and "/" in str(value["local"]):
            raise PrefabError(
                f"ссылка {value['local']!r} указывает внутрь вложенного префаба"
            )
        for item in value.values():
            _check_plain_refs(item)
    elif isinstance(value, list):
        for item in value:
            _check_plain_refs(item)


def _revert(
    prefab: Prefab,
    instance_root: GameObject,
    rec: dict[str, Any],
    loader: PrefabLoader | None,
    assets: AssetDatabase | None,
) -> None:
    from engine.scene_io import _apply_transform  # noqa: PLC0415

    marker = instance_root.get_component(PrefabInstance)
    scene = instance_root.scene
    if marker is None or not marker.instance_uid or scene is None:
        raise PrefabError("объект не является корнем экземпляра префаба")
    active = _scene_loader(scene, assets, loader)
    _e, _r, paths = _layout(prefab, marker.instance_uid, active)
    path_to_uid = {path: uid for uid, path in paths.items()}
    target = str(rec.get("target"))
    obj = scene.find_by_uid(path_to_uid.get(target, ""))
    base = _normalized_base(prefab, active, assets).get(target)
    if obj is None or base is None:
        raise PrefabError(f"в экземпляре нет объекта {target!r}")
    kind = str(rec.get("component"))
    field = rec.get("field")
    if kind == ROOT and field in ROOT_FIELDS:
        value = base[field]
        if field == "active":
            obj.active_self = bool(value)
        else:
            setattr(obj, field, value)
        return
    if kind == TRANSFORM and field in TRANSFORM_FIELDS:
        current = obj.transform
        payload = {
            "position": current.position.to_tuple(),
            "rotation": current.rotation.to_tuple(),
            "scale": current.scale.to_tuple(),
        }
        payload[field] = base["transform"][field]
        _apply_transform(obj, payload)
        return
    index = int(rec.get("index", 0))
    pairs = [
        (c, p) for c, p in component_payload_pairs(obj, assets) if p["type"] == kind
    ]
    if index >= len(pairs):
        raise PrefabError(f"{target}: нет компонента {kind}[{index}]")
    live_component, payload = pairs[index]
    if rec.get("add"):
        obj.remove_component(live_component)
        return
    if not isinstance(field, str):
        raise PrefabError("в записи нет поля")
    base_payloads = [c for c in base["components"] if c.get("type") == kind]
    patched = copy.deepcopy(payload)
    try:
        original = get_field(base_payloads[index], field)
    except (KeyError, IndexError):
        unset_field(patched, field)
    else:
        set_field(patched, field, to_uid_refs(original, path_to_uid))
    obj.remove_component(live_component)
    scripts: list[Any] = []
    add_component_from_payload(obj, patched, assets, scripts)
    if scripts:
        refs = {o.uid: o for o in scene.objects}
        for script in scripts:
            script.resolve_refs(refs)  # type: ignore[arg-type]


def revert_override(
    instance_root: GameObject,
    override: dict[str, Any],
    *,
    loader: PrefabLoader | None = None,
    assets: AssetDatabase | None = None,
) -> None:
    """Вернуть в экземпляре значение override из префаба.

    Raises:
        PrefabError: Нет корня экземпляра, цели или компонента.
    """
    marker = instance_root.get_component(PrefabInstance)
    if marker is None:
        raise PrefabError("объект не является корнем экземпляра префаба")
    active = _scene_loader(instance_root.scene, assets, loader)
    prefab = active(marker.prefab_guid, marker.prefab_path)
    _revert(prefab, instance_root, override, active, assets)


# --------------------------------------------------------------------------- #
# Пересборка экземпляров при изменении префаба
# --------------------------------------------------------------------------- #
@dataclass
class InstanceSnapshot:
    """Снимок экземпляров одного префаба, снятый *до* изменения файла.

    Attributes:
        regions: Пары ``(объекты области, свёрнутые записи)``: корень
            экземпляра с потомками и их запись как «корень + overrides».
    """

    regions: list[tuple[list[GameObject], list[dict[str, Any]]]] = field(
        default_factory=list
    )

    def __len__(self) -> int:
        return len(self.regions)


def _descendants(root: GameObject) -> list[GameObject]:
    result: list[GameObject] = []
    stack = [root]
    while stack:
        current = stack.pop()
        result.append(current)
        stack.extend(reversed(current.children))
    return result


def capture_instances(
    scene: Scene,
    prefab_guid: str,
    *,
    loader: PrefabLoader | None = None,
    assets: AssetDatabase | None = None,
) -> InstanceSnapshot:
    """Свернуть открытые экземпляры префаба с *текущей* версией файла."""
    snapshot = InstanceSnapshot()
    covered: set[str] = set()
    for obj in scene.objects:
        marker = obj.get_component(PrefabInstance)
        if (
            marker is None
            or not marker.instance_uid
            or marker.prefab_guid != prefab_guid
            or obj.uid in covered
        ):
            continue
        region = _descendants(obj)
        covered.update(item.uid for item in region)
        entries = serialize_objects(region, assets)
        active = _scene_loader(scene, assets, loader)
        snapshot.regions.append(
            (region, collapse_instances_with(entries, scene, assets, active))
        )
    return snapshot


def collapse_instances_with(
    entries: list[dict[str, Any]],
    scene: Scene,
    assets: AssetDatabase | None,
    loader: PrefabLoader,
) -> list[dict[str, Any]]:
    """:func:`collapse_instances` с явным загрузчиком."""
    previous = scene.prefab_loader
    scene.prefab_loader = loader
    try:
        return collapse_instances(entries, scene, assets)
    finally:
        scene.prefab_loader = previous


def restore_instances(
    scene: Scene,
    snapshot: InstanceSnapshot,
    *,
    assets: AssetDatabase | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[GameObject]:
    """Пересобрать экземпляры из снимка по *новому* файлу префаба.

    Области удаляются и создаются заново с теми же ``uid`` (overrides и
    добавленные объекты сохраняются), родитель и позиция корня среди братьев
    восстанавливаются. Выполняется синхронно: ``Scene`` не
    потокобезопасна.

    Returns:
        Новые корни экземпляров.
    """
    roots: list[GameObject] = []
    total = len(snapshot.regions)
    # Позиции снимаем до первой правки и идём по возрастанию: к моменту
    # вставки региона все братья левее уже на своих местах.
    places = {
        region[0].uid: (
            None if region[0].parent is None else region[0].parent.uid,
            region[0].sibling_index(),
        )
        for region, _collapsed in snapshot.regions
    }
    ordered = sorted(
        snapshot.regions,
        key=lambda item: (places[item[0][0].uid][0] or "", places[item[0][0].uid][1]),
    )
    for number, (region, collapsed) in enumerate(ordered, start=1):
        parent_uid, position = places[region[0].uid]
        if progress is not None:
            progress(number, total)
        scene.destroy_game_object(region[0], immediate=True)
        created = build_objects(scene, collapsed, assets)
        built = {game_object.uid for game_object in created}
        for entry in collapsed:
            ref = entry.get("parent")
            uid = ref.get("uid") if isinstance(ref, dict) else None
            game_object = scene.find_by_uid(str(entry["uid"]))
            if game_object is None or uid is None or uid in built:
                continue
            parent = scene.find_by_uid(str(uid))
            if parent is not None:
                game_object.set_parent(parent, keep_world=False)
        root = scene.find_by_uid(str(collapsed[0]["uid"]))
        if root is not None:
            outer = None if parent_uid is None else scene.find_by_uid(parent_uid)
            root.set_parent(outer, keep_world=False, index=position)
            roots.append(root)
    return roots
