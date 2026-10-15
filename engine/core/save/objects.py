"""Сбор и применение данных объектов сцены для сохранения.

Формат ``objects.json``::

    {uid: {"components": {"<Тип>#<n>": {"fields": {...}, "data": {...}}}}}

``fields`` - значения полей с ``save=True``, ``data`` - то, что компонент
положил в ``on_save``. Компонент адресуется типом и порядковым номером среди
компонентов того же типа на объекте.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from engine.core.component import Component
from engine.core.game_object import GameObject
from engine.core.reflection import coerce_value, registry, to_plain
from engine.core.save.context import SaveContext
from engine.core.scene import Scene

__all__ = ["apply_objects", "apply_objects_to_scenes", "collect_objects"]

_log = logging.getLogger("engine.core.save")


def _component_keys(obj: GameObject) -> list[tuple[str, Component]]:
    counts: dict[str, int] = {}
    result: list[tuple[str, Component]] = []
    for comp in obj.components:
        name = type(comp).__name__
        n = counts.get(name, 0)
        counts[name] = n + 1
        result.append((f"{name}#{n}", comp))
    return result


def _behaviour(comp: Component) -> Any:
    return getattr(comp, "behaviour", None)


def _save_fields(comp: Component) -> dict[str, Any]:
    behaviour = _behaviour(comp)
    if behaviour is not None:
        return behaviour.serialize_save_fields()
    return {
        d.name: to_plain(getattr(comp, d.name))
        for d in registry.fields(type(comp))
        if d.save
    }


def collect_objects(scene: Scene) -> dict[str, Any]:
    """Данные ``@save``-полей и ``on_save`` всех объектов сцены.

    Returns:
        ``{uid: {"components": {...}}}``; объекты без данных пропускаются,
        объекты пула (``pooled``) тоже.
    """
    result: dict[str, Any] = {}
    for obj in sorted(scene.objects, key=lambda o: o.uid):
        if obj.pooled:
            continue
        comps: dict[str, Any] = {}
        for key, comp in _component_keys(obj):
            entry: dict[str, Any] = {}
            fields = _save_fields(comp)
            if fields:
                entry["fields"] = fields
            hook = getattr(comp, "on_save", None)
            if hook is not None:
                ctx = SaveContext()
                try:
                    hook(ctx)
                except Exception:  # noqa: BLE001 - хук пользователя
                    _log.exception("Исключение в on_save %s объекта %r", key, obj.name)
                if ctx:
                    entry["data"] = ctx.to_dict()
            if entry:
                comps[key] = entry
        if comps:
            result[obj.uid] = {"components": comps}
    return result


def _apply_fields(comp: Component, values: dict[str, Any], scene: Scene) -> None:
    behaviour = _behaviour(comp)
    if behaviour is not None:
        objects = {o.uid: o for o in scene.objects}
        for name, raw in values.items():
            try:
                behaviour.apply_field_values({name: raw}, objects=objects)
            except (ValueError, TypeError) as exc:
                _log.warning("Поле сохранения %r пропущено: %s", name, exc)
        return
    cls = type(comp)
    for name, raw in values.items():
        desc = registry.for_name(cls, name)
        if desc is None or not desc.save:
            _log.warning("Поле сохранения %r не найдено у %s", name, cls.__name__)
            continue
        try:
            value = registry.clamp(cls, name, coerce_value(desc, raw))
            setattr(comp, name, value)
        except (ValueError, TypeError) as exc:
            _log.warning("Поле сохранения %r пропущено: %s", name, exc)


def apply_objects(scene: Scene, data: dict[str, Any]) -> list[str]:
    """Применить данные объектов к сцене: сначала поля, затем ``on_load``.

    Args:
        scene: Сцена, уже построенная из шаблона.
        data: Результат :func:`collect_objects` (возможно, после миграции).

    Returns:
        Uid объектов из сохранения, которых нет в сцене (пропущены; один
        WARNING на весь вызов).
    """
    missing: list[str] = []
    loads: list[tuple[Any, SaveContext]] = []
    for uid, payload in data.items():
        obj = scene.find_by_uid(uid)
        if obj is None:
            missing.append(uid)
            continue
        by_key = dict(_component_keys(obj))
        for key, entry in payload.get("components", {}).items():
            comp = by_key.get(key)
            if comp is None:
                _log.warning("Компонент %s объекта %r не найден", key, obj.name)
                continue
            if entry.get("fields"):
                _apply_fields(comp, entry["fields"], scene)
            hook = getattr(comp, "on_load", None)
            if hook is not None and entry.get("data") is not None:
                loads.append((hook, SaveContext.from_dict(entry["data"])))
    for hook, ctx in loads:
        try:
            hook(ctx)
        except Exception:  # noqa: BLE001 - хук пользователя
            _log.exception("Исключение в on_load")
    if missing:
        _log.warning(
            "В сохранении есть объекты, которых нет в сцене (%d): %s",
            len(missing),
            ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""),
        )
    return missing


def apply_objects_to_scenes(
    scenes: Sequence[Scene], data: dict[str, Any]
) -> list[str]:
    """Применить объекты слота к нескольким сценам: каждому - своя сцена.

    Args:
        scenes: Сцены (активная первой); uid ищется по очереди.
        data: Результат :func:`collect_objects` (объекты всех сцен слота).

    Returns:
        Uid, которых нет ни в одной сцене (пропущены; один WARNING на вызов).
    """
    parts: list[dict[str, Any]] = [{} for _ in scenes]
    leftover: dict[str, Any] = {}
    for uid, payload in data.items():
        for part, scene in zip(parts, scenes, strict=True):
            if scene.find_by_uid(uid) is not None:
                part[uid] = payload
                break
        else:
            leftover[uid] = payload
    # Отсутствующие уходят в первую сцену: она и напишет общий WARNING.
    parts[0].update(leftover)
    missing: list[str] = []
    for part, scene in zip(parts, scenes, strict=True):
        missing += apply_objects(scene, part)
    return missing
