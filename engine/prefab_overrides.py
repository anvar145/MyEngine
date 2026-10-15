"""Переопределения (overrides) экземпляра префаба: diff и применение.

Модуль работает только со словарями и не знает про сцену. Запись::

    {"target": "L1", "component": "MeshRenderer", "field": "color",
     "value": [1, 0, 0]}

* ``target`` - путь объекта в экземпляре: ``local_id`` объекта префаба, для
  объекта вложенного префаба - ``"<заместитель>/<путь во вложенном>"``;
* ``component`` - тип компонента; ``"Root"`` - поля ``name/tag/layer/active``
  объекта, ``"Transform"`` - ``position/rotation/scale``;
* ``index`` - номер компонента среди компонентов того же типа (пишется, если > 0);
* ``field`` - имя поля, для вложенных словарей через точку (``size.x``);
  списки и векторы заменяются целиком;
* ``value`` - новое значение; ссылки на объекты экземпляра - ``{"local": путь}``,
  на объекты вне экземпляра - ``{"uid": ...}``;
* ``"unset": true`` вместо ``value`` - поля нет в экземпляре, но есть в префабе;
* ``"add": true`` - компонент добавлен в экземпляре (``value`` - его поля без
  ``type``), ``field`` не пишется;
* ``"stale": true`` - запись не применилась при загрузке (нет цели или
  компонента); она сохраняется и «оживает», когда цель снова появится.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = [
    "ROOT",
    "ROOT_FIELDS",
    "TRANSFORM",
    "TRANSFORM_FIELDS",
    "UNSET",
    "apply_overrides",
    "diff_entry",
    "diff_fields",
    "get_field",
    "merge_stale",
    "override_key",
    "set_field",
    "to_local_refs",
    "to_uid_refs",
    "unset_field",
]

ROOT = "Root"
TRANSFORM = "Transform"
ROOT_FIELDS = ("name", "tag", "layer", "active")
TRANSFORM_FIELDS = ("position", "rotation", "scale")

_UNSET = object()
UNSET = _UNSET


def _is_ref(value: Any) -> bool:
    """Словарь-ссылка: ``{"uid"}``, ``{"local"}``, ``{"object"}``, asset+guid."""
    if not isinstance(value, dict):
        return False
    keys = set(value)
    return keys in ({"uid"}, {"local"}, {"object"}, {"asset", "guid"})


def to_local_refs(value: Any, uid_to_path: dict[str, str]) -> Any:
    """``{"uid": u}`` -> ``{"local": путь}`` для объектов экземпляра."""
    if isinstance(value, dict):
        if set(value) == {"uid"}:
            path = uid_to_path.get(str(value["uid"]))
            return {"local": path} if path is not None else dict(value)
        return {k: to_local_refs(v, uid_to_path) for k, v in value.items()}
    if isinstance(value, list):
        return [to_local_refs(v, uid_to_path) for v in value]
    return value


def to_uid_refs(value: Any, path_to_uid: dict[str, str]) -> Any:
    """``{"local": путь}`` -> ``{"uid": u}``; ``KeyError``, если пути нет."""
    if isinstance(value, dict):
        if set(value) == {"local"}:
            return {"uid": path_to_uid[str(value["local"])]}
        return {k: to_uid_refs(v, path_to_uid) for k, v in value.items()}
    if isinstance(value, list):
        return [to_uid_refs(v, path_to_uid) for v in value]
    return value


def get_field(payload: dict[str, Any], field: str) -> Any:
    """Значение по пути через точку или ``_UNSET``-подобный маркер ``KeyError``."""
    current: Any = payload
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(field)
        current = current[part]
    return current


def set_field(payload: dict[str, Any], field: str, value: Any) -> None:
    """Записать значение по пути через точку (промежуточные словари создаются)."""
    parts = field.split(".")
    current = payload
    for part in parts[:-1]:
        nxt = current.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            current[part] = nxt
        current = nxt
    current[parts[-1]] = copy.deepcopy(value)


def unset_field(payload: dict[str, Any], field: str) -> None:
    """Удалить поле по пути через точку (нет поля - ничего не делает)."""
    parts = field.split(".")
    current: Any = payload
    for part in parts[:-1]:
        current = current.get(part) if isinstance(current, dict) else None
        if current is None:
            return
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def diff_fields(
    base: dict[str, Any], live: dict[str, Any], prefix: str = ""
) -> list[tuple[str, Any]]:
    """Различия двух словарей: ``(путь через точку, новое значение | _UNSET)``.

    Вложенные словари (кроме ссылок) сравниваются рекурсивно, списки - целиком.
    Ключ ``type`` пропускается.
    """
    found: list[tuple[str, Any]] = []
    for key, value in live.items():
        if key == "type":
            continue
        name = prefix + key
        if key not in base:
            found.append((name, value))
        elif base[key] != value:
            if (
                isinstance(value, dict)
                and isinstance(base[key], dict)
                and not _is_ref(value)
                and not _is_ref(base[key])
            ):
                found.extend(diff_fields(base[key], value, name + "."))
            else:
                found.append((name, value))
    for key in base:
        if key != "type" and key not in live:
            found.append((prefix + key, _UNSET))
    return found


def _record(
    target: str, component: str, index: int, field: str | None
) -> dict[str, Any]:
    rec: dict[str, Any] = {"target": target, "component": component}
    if index:
        rec["index"] = index
    if field is not None:
        rec["field"] = field
    return rec


def diff_entry(
    path: str,
    base: dict[str, Any],
    live: dict[str, Any],
    *,
    skip_transform: bool = False,
    skip_types: tuple[str, ...] = ("PrefabInstance",),
) -> list[dict[str, Any]]:
    """Overrides одного объекта: ``live`` относительно ``base``.

    Записи объектов - в формате сцены, уже в пространстве путей (ссылки
    ``{"local": путь}``).

    Args:
        path: Путь объекта (``target`` записей).
        base: Запись объекта из префаба.
        live: Запись объекта из экземпляра.
        skip_transform: Не сравнивать ``transform`` (корень внешнего экземпляра:
            его положение хранит сама сцена).
        skip_types: Типы компонентов, которые не сравниваются.
    """
    out: list[dict[str, Any]] = []
    for key in ROOT_FIELDS:
        if live.get(key) != base.get(key):
            rec = _record(path, ROOT, 0, key)
            rec["value"] = copy.deepcopy(live.get(key))
            out.append(rec)
    if not skip_transform:
        base_t = base.get("transform") or {}
        live_t = live.get("transform") or {}
        for key in TRANSFORM_FIELDS:
            if live_t.get(key) != base_t.get(key):
                rec = _record(path, TRANSFORM, 0, key)
                rec["value"] = copy.deepcopy(live_t.get(key))
                out.append(rec)

    base_by_type: dict[str, list[dict[str, Any]]] = {}
    for comp in base.get("components", []):
        base_by_type.setdefault(str(comp.get("type")), []).append(comp)
    seen: dict[str, int] = {}
    for comp in live.get("components", []):
        kind = str(comp.get("type"))
        if kind in skip_types:
            continue
        index = seen.get(kind, 0)
        seen[kind] = index + 1
        candidates = base_by_type.get(kind, [])
        if index >= len(candidates):
            rec = _record(path, kind, index, None)
            rec["add"] = True
            rec["value"] = copy.deepcopy({k: v for k, v in comp.items() if k != "type"})
            out.append(rec)
            continue
        for field, value in diff_fields(candidates[index], comp):
            rec = _record(path, kind, index, field)
            if value is _UNSET:
                rec["unset"] = True
            else:
                rec["value"] = copy.deepcopy(value)
            out.append(rec)
    return out


def override_key(rec: dict[str, Any]) -> tuple[Any, ...]:
    """Ключ записи: одна запись на (цель, компонент, номер, поле, добавление)."""
    return (
        rec.get("target"),
        rec.get("component"),
        int(rec.get("index", 0)),
        rec.get("field"),
        bool(rec.get("add")),
    )


def merge_stale(
    diff: list[dict[str, Any]], previous: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """``diff`` + устаревшие записи из ``previous``, которых нет в ``diff``."""
    keys = {override_key(rec) for rec in diff}
    merged = list(diff)
    for rec in previous:
        if rec.get("stale") and override_key(rec) not in keys:
            merged.append(copy.deepcopy(rec))
    return merged


def _apply_one(
    entry: dict[str, Any], rec: dict[str, Any], path_to_uid: dict[str, str]
) -> bool:
    """Применить запись к записи объекта; ``False`` - не применилась (stale)."""
    component = str(rec.get("component", ""))
    field = rec.get("field")
    try:
        value = to_uid_refs(rec.get("value"), path_to_uid)
    except KeyError:
        return False
    if component == ROOT:
        if field not in ROOT_FIELDS:
            return False
        entry[field] = copy.deepcopy(value)
        return True
    if component == TRANSFORM:
        if field not in TRANSFORM_FIELDS:
            return False
        transform = dict(entry.get("transform") or {})
        transform[field] = copy.deepcopy(value)
        entry["transform"] = transform
        return True
    components = entry.setdefault("components", [])
    if rec.get("add"):
        if not isinstance(value, dict):
            return False
        components.append({"type": component, **copy.deepcopy(value)})
        return True
    matches = [c for c in components if c.get("type") == component]
    index = int(rec.get("index", 0))
    if index >= len(matches) or not isinstance(field, str):
        return False
    if rec.get("unset"):
        unset_field(matches[index], field)
    else:
        set_field(matches[index], field, value)
    return True


def apply_overrides(
    entries: list[dict[str, Any]],
    paths: dict[str, str],
    overrides: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Применить overrides к записям раскрытого экземпляра (на месте).

    Args:
        entries: Записи объектов экземпляра (формат сцены).
        paths: ``uid`` -> путь объекта в экземпляре.
        overrides: Записи overrides.

    Returns:
        Копии записей: у неприменённых ``"stale": true``, у применённых
        признак снят.
    """
    by_path = {path: uid for uid, path in paths.items()}
    by_uid = {entry["uid"]: entry for entry in entries}
    result: list[dict[str, Any]] = []
    for rec in overrides:
        item = copy.deepcopy(rec)
        uid = by_path.get(str(rec.get("target", "")))
        applied = uid is not None and _apply_one(by_uid[uid], rec, by_path)
        if applied:
            item.pop("stale", None)
        else:
            item["stale"] = True
        result.append(item)
    return result
