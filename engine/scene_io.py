"""Сохранение и загрузка сцены в JSON.

Формат простой и читаемый — его удобно смотреть глазами и диффить в системе
контроля версий::

    {
      "format": "myengine.scene",
      "version": 2,
      "name": "Demo",
      "sky": {"enabled": true, "mode": "procedural", ...},
      "objects": [
        {
          "uid": "9f1c0a52d3b7e648", "name": "Cube", "tag": "Untagged",
          "parent": null, "active": true,
          "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1],
                        "scale": [1, 1, 1]},
          "components": [{"type": "MeshRenderer", "mesh": "cube", ...}]
        }
      ]
    }

Версии формата:

* **v2** (текущая): объект идентифицируется ``uid`` (16 hex, 64 бита), ссылки
  на объекты (``parent``, поля скриптов) записываются как ``{"uid": "..."}``.
  Целого ``id`` в файле нет: номер в рантайме выдаётся сценой заново.
* **v1**: объект с целым ``id``, ссылки ``{"object": id}`` и ``parent`` -
  числом. Читается и переводится в память как v2 (``uid`` выдаётся
  детерминированно, см. :func:`migrated_uid`); на диск v1 сам не
  переписывается - только при явном :func:`save_scene`, который перед этим
  делает копию ``<файл>.v1.bak``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import shutil
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import Any

from engine.audio.components import AudioListenerComponent, AudioSourceComponent
from engine.core.component import Component
from engine.core.components import (
    MeshRenderer,
    SpriteRenderer,
    TextRenderer,
)
from engine.core.event_bus import EventBus
from engine.core.game_object import GameObject, is_valid_uid
from engine.core.math3d import Quaternion, Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    MeshColliderComponent,
    RigidBodyComponent,
)
from engine.core.prefab_components import PrefabInstance
from engine.core.resources import AssetDatabase, AssetRef
from engine.core.save.components import AutoSavePolicy, CheckpointComponent, SafeZone
from engine.core.scene import Scene
from engine.core.ui_components import UIButton, UIImage, UIPanel, UIText
from engine.rendering.camera import CameraComponent
from engine.rendering.lighting import LightComponent
from engine.rendering.post import PostSettings
from engine.rendering.sky import SkySettings
from engine.scripting import ScriptComponent

#: UI-компоненты сцены: значение ``type`` → класс.
_UI_TYPES: dict[str, type] = {
    cls.__name__: cls for cls in (UIPanel, UIImage, UIText, UIButton)
}

#: Физические компоненты сцены: значение ``type`` → класс.
_PHYSICS_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        RigidBodyComponent,
        ColliderComponent,
        CharacterControllerComponent,
        MeshColliderComponent,
    )
}

#: Компоненты сохранений: значение ``type`` → класс.
_SAVE_TYPES: dict[str, type] = {
    cls.__name__: cls for cls in (CheckpointComponent, SafeZone, AutoSavePolicy)
}

__all__ = [
    "SCENE_FORMAT",
    "SCENE_VERSION",
    "add_component_from_payload",
    "build_objects",
    "component_payload_pairs",
    "iter_build_objects",
    "iter_scene_from_dict",
    "load_scene",
    "migrated_uid",
    "load_scene_full",
    "load_scene_with_sky",
    "read_scene_data",
    "read_post",
    "read_sky",
    "save_scene",
    "serialize_objects",
    "scene_from_dict",
    "scene_to_dict",
]

_log = logging.getLogger("engine.scene_io")

SCENE_FORMAT = "myengine.scene"
SCENE_VERSION = 2

#: Суффикс копии v1-файла, которую делает :func:`save_scene` перед записью v2.
V1_BACKUP_SUFFIX = ".v1.bak"


def migrated_uid(source_id: int) -> str:
    """``uid`` для объекта сцены v1 с целым ``id`` (детерминированно).

    ``blake2b`` стабилен между запусками (в отличие от ``hash()``), поэтому одна
    и та же v1-сцена при каждой загрузке получает одни и те же ``uid``.
    Уникальность гарантируется в пределах сцены.

    Args:
        source_id: Целый ``id`` объекта из файла v1.

    Returns:
        16 hex-символов.
    """
    digest = hashlib.blake2b(f"v1:{int(source_id)}".encode(), digest_size=8)
    return digest.hexdigest()


def _rel_path(assets: AssetDatabase, path: Path) -> str:
    """Путь ассета относительно ``Assets/`` (абсолютный вне проекта - как есть)."""
    if path.is_absolute():
        with contextlib.suppress(ValueError):
            return path.relative_to(assets.root).as_posix()
    return path.as_posix()


def _sync_ref(
    assets: AssetDatabase | None, ref: AssetRef | None, *, prefer_guid: bool
) -> AssetRef | None:
    """Согласовать GUID и путь ссылки с базой ассетов.

    При загрузке (``prefer_guid``) главнее GUID: файл могли переименовать,
    путь обновляется. При сохранении главнее путь: GUID берётся из базы.
    Примитивы, неизвестные базе ссылки и ``None`` остаются как есть.
    """
    if assets is None or ref is None or ref.is_primitive:
        return ref
    by_guid = assets.get(ref.guid) if ref.guid else None
    by_path = assets.guid_of(ref.path) if ref.path else None
    if ref.path and Path(ref.path).is_absolute():
        # Файл проекта пишется относительно ``Assets/``, даже если база его не знает.
        with contextlib.suppress(ValueError):
            ref = AssetRef(
                guid=ref.guid, path=Path(ref.path).relative_to(assets.root).as_posix()
            )
    if by_path is None and ref.path and not Path(ref.path).is_absolute():
        # Путь от корня проекта (``Assets/...``, так писали поля скриптов).
        by_path = assets.guid_of(assets.root.parent / ref.path)
        handle = assets.get(by_path) if by_path else None
        if handle is not None and by_guid is None:
            with contextlib.suppress(ValueError):
                ref = AssetRef(guid=ref.guid, path=_rel_path(assets, handle.path))
    if by_guid is not None and (prefer_guid or by_path is None):
        if by_path == ref.guid:
            return ref
        return AssetRef(guid=ref.guid, path=_rel_path(assets, by_guid.path))
    if by_path is not None:
        handle = assets.get(by_path)
        path = ref.path
        if handle is not None:
            # Путь пишется относительно ``Assets/`` (абсолютный - только вне проекта).
            path = _rel_path(assets, handle.path)
        if by_path != ref.guid or path != ref.path:
            return AssetRef(guid=by_path, path=path)
    return ref


def _script_path_to_project(
    assets: AssetDatabase | None, payload: dict[str, Any]
) -> None:
    """Записать путь скрипта относительно проекта (если скрипт внутри него)."""
    raw = payload.get("path")
    if assets is None or not raw or not Path(str(raw)).is_absolute():
        return
    with contextlib.suppress(ValueError):
        payload["path"] = Path(str(raw)).relative_to(assets.root.parent).as_posix()


def _script_path_from_project(
    assets: AssetDatabase | None, payload: dict[str, Any]
) -> dict[str, Any]:
    """Превратить относительный путь скрипта в абсолютный (от корня проекта)."""
    raw = payload.get("path")
    if assets is None or not raw or Path(str(raw)).is_absolute():
        return payload
    for base in (assets.root.parent, assets.root):
        candidate = base / str(raw)
        if candidate.is_file():
            return {**payload, "path": str(candidate)}
    return payload


def _is_file_ref(value: Any) -> bool:
    """Значение поля - ссылка на файл (новая или старая форма)."""
    return isinstance(value, dict) and (
        "asset" in value or "guid" in value or "path" in value
    )


def _script_fields_to_project(
    assets: AssetDatabase | None, payload: dict[str, Any]
) -> None:
    """Ссылки на файлы в полях скрипта: ``{"guid", "path"}`` по базе ассетов."""
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        return
    for name, value in list(fields.items()):
        if not _is_file_ref(value):
            continue
        ref = AssetRef.coerce(value)
        synced = _sync_ref(assets, ref, prefer_guid=False)
        fields[name] = synced.to_dict() if synced else None


def _script_fields_from_project(
    assets: AssetDatabase | None, payload: dict[str, Any]
) -> dict[str, Any]:
    """Согласовать ссылки на файлы с базой: приоритет у GUID.

    Файл могли переименовать или переместить - путь обновляется по GUID.
    Не найденная ссылка остаётся как есть (в инспекторе она «битая»).
    """
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        return payload
    resolved: dict[str, Any] = {}
    for name, value in fields.items():
        if not _is_file_ref(value):
            resolved[name] = value
            continue
        ref = _sync_ref(assets, AssetRef.coerce(value), prefer_guid=True)
        if ref is not None:
            resolved[name] = ref.to_dict()
    return {**payload, "fields": resolved}


def _component_payloads(
    game_object: GameObject, assets: AssetDatabase | None
) -> list[dict[str, Any]]:
    """Сериализовать компоненты объекта (неизвестные типы пропускаются)."""
    return [payload for _, payload in component_payload_pairs(game_object, assets)]


def component_payload_pairs(
    game_object: GameObject, assets: AssetDatabase | None = None
) -> list[tuple[Component, dict[str, Any]]]:
    """Компоненты объекта и их записи сцены (неизвестные типы пропускаются)."""
    pairs: list[tuple[Component, dict[str, Any]]] = []
    for component in game_object.components:
        payload = _component_payload(component, assets)
        if payload is not None:
            pairs.append((component, payload))
    return pairs


def _component_payload(
    component: Component, assets: AssetDatabase | None
) -> dict[str, Any] | None:
    """Запись одного компонента или ``None`` для неизвестного типа."""
    if isinstance(component, MeshRenderer):
        payload = component.to_dict()
        payload["type"] = "MeshRenderer"
        # База знает актуальный GUID по пути (и путь по GUID).
        synced = _sync_ref(assets, component.mesh_ref, prefer_guid=False)
        if synced is not None:
            payload["mesh_ref"] = synced.to_dict()
        return payload
    if isinstance(component, SpriteRenderer):
        payload = component.to_dict()
        payload["type"] = "SpriteRenderer"
        synced = _sync_ref(assets, component.sprite_ref, prefer_guid=False)
        payload["sprite_ref"] = synced.to_dict() if synced else None
        return payload
    if isinstance(component, ScriptComponent):
        payload = component.to_dict()
        payload["type"] = "ScriptComponent"
        _script_path_to_project(assets, payload)
        _script_fields_to_project(assets, payload)
        return payload
    if isinstance(component, AudioSourceComponent):
        payload = component.to_dict()
        payload["type"] = "AudioSourceComponent"
        synced = _sync_ref(assets, component.clip, prefer_guid=False)
        if synced is not None:
            payload["clip"] = synced.to_dict()
        return payload
    simple = (
        (CameraComponent, "CameraComponent"),
        (LightComponent, "LightComponent"),
        (TextRenderer, "TextRenderer"),
        (PrefabInstance, "PrefabInstance"),
        *((cls, name) for name, cls in _SAVE_TYPES.items()),
        (AudioListenerComponent, "AudioListenerComponent"),
    )
    for cls, name in simple:
        if isinstance(component, cls):
            payload = component.to_dict()
            payload["type"] = name
            return payload
    if isinstance(
        component, tuple(_UI_TYPES.values()) + tuple(_PHYSICS_TYPES.values())
    ):
        payload = component.to_dict()
        payload["type"] = type(component).__name__
        if isinstance(component, MeshColliderComponent):
            synced = _sync_ref(assets, component.mesh, prefer_guid=False)
            if synced is not None:
                payload["mesh"] = synced.to_dict()
        if type(component) in _PHYSICS_TYPES.values() and not component.enabled:
            # поле пишется только при False: старые сцены не меняются
            payload["enabled"] = False
        return payload
    return None


def _remap_refs(value: Any, local_of: dict[str, str]) -> Any:
    """Заменить ссылки ``{"uid": ...}`` на ``{"local": ...}`` (для префабов).

    Ссылка на объект вне набора обнуляется: префаб самодостаточен.
    """
    if isinstance(value, dict):
        if set(value) == {"uid"}:
            local = local_of.get(str(value["uid"]))
            if local is None:
                _log.warning(
                    "Ссылка на объект вне префаба (uid %s) сброшена", value["uid"]
                )
                return None
            return {"local": local}
        return {key: _remap_refs(item, local_of) for key, item in value.items()}
    if isinstance(value, list):
        return [_remap_refs(item, local_of) for item in value]
    return value


def _in_pool(game_object: GameObject) -> bool:
    """``True``, если объект или его предок - корень экземпляра пула."""
    current: GameObject | None = game_object
    while current is not None:
        if current.pooled:
            return True
        current = current.parent
    return False


def serialize_objects(
    objects: Iterable[GameObject],
    assets: AssetDatabase | None = None,
    ref_key: str = "uid",
) -> list[dict[str, Any]]:
    """Сериализовать набор объектов (общий код сцены и префаба).

    Args:
        objects: Объекты в порядке записи (родитель раньше потомков).
        assets: База ассетов (``mesh_guid``, пути скриптов от корня проекта).
        ref_key: ``"uid"`` - формат сцены: запись с ``uid``, ссылки
            ``{"uid": ...}``. ``"local"`` - формат префаба: запись с
            ``local_id`` (``L0``, ``L1``, ... по порядку), ссылки
            ``{"local": ...}``, uid в результат не попадают.

    Returns:
        Список описаний объектов, готовый к записи в JSON.

    Raises:
        ValueError: Неизвестный ``ref_key``.
    """
    if ref_key not in ("uid", "local"):
        raise ValueError(f"ref_key должен быть 'uid' или 'local', получено {ref_key!r}")
    items = list(objects)
    if ref_key == "uid":
        items = [item for item in items if not _in_pool(item)]
    local = ref_key == "local"
    local_of = {item.uid: f"L{index}" for index, item in enumerate(items)}
    entries: list[dict[str, Any]] = []
    for game_object in items:
        parent = game_object.parent
        transform = game_object.transform
        components = _component_payloads(game_object, assets)
        entry: dict[str, Any]
        if local:
            components = _remap_refs(components, local_of)
            entry = {"local_id": local_of[game_object.uid]}
        else:
            entry = {"uid": game_object.uid}
        entry["name"] = game_object.name
        entry["tag"] = game_object.tag
        entry["layer"] = game_object.layer
        entry["active"] = game_object.active_self
        if not local:
            entry["parent"] = {"uid": parent.uid} if parent is not None else None
        elif parent is not None and parent.uid in local_of:
            entry["parent"] = {"local": local_of[parent.uid]}
        entry["transform"] = {
            "position": list(transform.position.to_tuple()),
            "rotation": list(transform.rotation.to_tuple()),
            "scale": list(transform.scale.to_tuple()),
        }
        entry["components"] = components
        entries.append(entry)
    return entries


def _in_sibling_order(scene: Scene) -> list[GameObject]:
    """Объекты сцены в порядке создания, но с порядком братьев из иерархии.

    При загрузке порядок детей (и корней) задаёт порядок записей, поэтому
    перестановка братьев (reparent с индексом) должна попасть в файл. Каждая
    группа братьев занимает те же места в списке, что и в порядке создания, так
    что без перестановок результат идентичен ``scene.objects`` (файлы старых
    сцен не меняются).
    """
    ordered = list(scene.objects)
    place = {id(item): slot for slot, item in enumerate(ordered)}
    groups = [scene.root_objects]
    groups.extend(item.children for item in ordered if len(item.children) > 1)
    for group in groups:
        if len(group) < 2:
            continue
        slots = sorted(place[id(item)] for item in group)
        for slot, item in zip(slots, group, strict=True):
            ordered[slot] = item
    return ordered


def scene_to_dict(
    scene: Scene,
    assets: AssetDatabase | None = None,
    sky: SkySettings | None = None,
    post: PostSettings | None = None,
) -> dict[str, Any]:
    """Сериализовать сцену в словарь.

    Args:
        scene: Сцена движка.
        assets: База ассетов; если задана, рядом с путём меша записывается
            его GUID (``mesh_guid``).
        sky: Настройки неба; ``None`` — секция ``sky`` не записывается.
        post: Пост-обработка; ``None`` — секция ``post`` не записывается
            (сцена берёт умолчания профиля качества).

    Returns:
        Словарь, готовый к записи в JSON.
    """
    objects = serialize_objects(_in_sibling_order(scene), assets)
    if any(_has_instance_marker(entry) for entry in objects):
        from engine.prefab import collapse_instances  # noqa: PLC0415

        objects = collapse_instances(objects, scene, assets)
    result: dict[str, Any] = {
        "format": SCENE_FORMAT,
        "version": SCENE_VERSION,
        "name": scene.name,
    }
    if sky is not None:
        result["sky"] = sky.to_dict()
    if post is not None:
        result["post"] = post.to_dict(full=True)
    result["objects"] = objects
    return result


def read_sky(data: dict[str, Any]) -> SkySettings:
    """Прочитать секцию ``sky`` сцены (нет секции — значения по умолчанию).

    Args:
        data: Словарь сцены, полученный из :func:`scene_to_dict`.

    Returns:
        Настройки неба.
    """
    return SkySettings.from_dict(data.get("sky"))


def read_post(data: dict[str, Any]) -> PostSettings | None:
    """Прочитать секцию ``post`` сцены.

    Args:
        data: Словарь сцены.

    Returns:
        Настройки или ``None``, если секции нет (тогда действуют умолчания
        профиля качества).
    """
    section = data.get("post")
    if not isinstance(section, dict):
        return None
    return PostSettings.from_dict(section)


def scene_from_dict(
    data: dict[str, Any],
    assets: AssetDatabase | None = None,
    into: Scene | None = None,
) -> Scene:
    """Восстановить сцену из словаря.

    Args:
        data: Данные, полученные из :func:`scene_to_dict`.
        assets: База ассетов; по ``mesh_guid`` путь меша обновляется, если
            файл переименовали или переместили.
        into: Готовая (пустая) сцена, в которую загружаются объекты; ``None`` —
            создаётся новая.

    Returns:
        Сцена с объектами, иерархией и компонентами.

    Raises:
        ValueError: Если формат файла не поддерживается.
    """
    steps = iter_scene_from_dict(data, assets, into)
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def iter_scene_from_dict(
    data: dict[str, Any],
    assets: AssetDatabase | None = None,
    into: Scene | None = None,
) -> Generator[float, None, Scene]:
    """Порционно восстановить сцену из словаря (см. :func:`scene_from_dict`).

    Отдаёт долю выполненной работы ``0..1``, итог - в ``StopIteration.value``.

    Args:
        data: Данные, полученные из :func:`scene_to_dict`.
        assets: База ассетов; по ``mesh_guid`` путь меша обновляется, если
            файл переименовали или переместили.
        into: Готовая (пустая) сцена, в которую загружаются объекты; ``None`` —
            создаётся новая.

    Returns:
        Сцена с объектами, иерархией и компонентами.

    Raises:
        ValueError: Если формат файла не поддерживается.
    """
    if data.get("format") != SCENE_FORMAT:
        raise ValueError(
            f"Неизвестный формат сцены: {data.get('format')!r} "
            f"(ожидался {SCENE_FORMAT!r})"
        )
    version = int(data.get("version", 0))
    if version > SCENE_VERSION:
        raise ValueError(
            f"Сцена сохранена более новой версией редактора "
            f"(v{version} > v{SCENE_VERSION})"
        )

    scene = (
        into
        if into is not None
        else Scene(str(data.get("name", "Scene")), event_bus=EventBus())
    )
    if assets is not None:
        scene.asset_source = assets
    try:
        yield from iter_build_objects(scene, data.get("objects", []), assets)
    except GeneratorExit:
        # Отмена асинхронной загрузки: недостроенную сцену выгружаем.
        if into is None:
            scene.shutdown()
        raise
    return scene


def add_component_from_payload(
    game_object: GameObject,
    component: dict[str, Any],
    assets: AssetDatabase | None = None,
    scripts: list[ScriptComponent] | None = None,
) -> Component | None:
    """Добавить объекту компонент по записи сцены (``type`` + поля).

    Args:
        game_object: Объект.
        component: Запись компонента.
        assets: База ассетов.
        scripts: Если задан, созданный скрипт добавляется сюда (для
            ``resolve_refs`` после создания всех объектов).

    Returns:
        Созданный компонент или ``None`` для неизвестного типа.
    """
    kind = component.get("type")
    if kind == "MeshRenderer":
        renderer = MeshRenderer.from_dict(component)
        renderer.mesh_ref = _sync_ref(assets, renderer.mesh_ref, prefer_guid=True)
        return game_object.add_component(renderer)
    if kind == "ScriptComponent":
        resolved = _script_path_from_project(assets, component)
        resolved = _script_fields_from_project(assets, resolved)
        script = game_object.add_component(ScriptComponent.from_dict(resolved))
        if scripts is not None:
            scripts.append(script)
        return script
    if kind == "CameraComponent":
        return game_object.add_component(CameraComponent.from_dict(component))
    if kind == "LightComponent":
        return game_object.add_component(LightComponent.from_dict(component))
    if kind == "SpriteRenderer":
        return game_object.add_component(SpriteRenderer.from_dict(component))
    if kind == "TextRenderer":
        return game_object.add_component(TextRenderer.from_dict(component))
    if kind == "PrefabInstance":
        return game_object.add_component(PrefabInstance.from_dict(component))
    if kind == "AudioSourceComponent":
        source = AudioSourceComponent.from_dict(component)
        source.clip = _sync_ref(assets, source.clip, prefer_guid=True)
        return game_object.add_component(source)
    if kind == "AudioListenerComponent":
        return game_object.add_component(AudioListenerComponent.from_dict(component))
    if kind in _UI_TYPES:
        return game_object.add_component(_UI_TYPES[kind].from_dict(component))
    if kind in _SAVE_TYPES:
        return game_object.add_component(_SAVE_TYPES[kind].from_dict(component))
    if kind in _PHYSICS_TYPES:
        physics = _PHYSICS_TYPES[kind].from_dict(component)
        if isinstance(physics, MeshColliderComponent):
            physics.mesh = _sync_ref(assets, physics.mesh, prefer_guid=True)
        physics = game_object.add_component(physics)
        if component.get("enabled", True) is False:
            physics.enabled = False
        return physics
    return None


def _has_instance_marker(entry: dict[str, Any]) -> bool:
    return any(
        c.get("type") == "PrefabInstance" and c.get("instance_uid")
        for c in entry.get("components", [])
    )


def build_objects(
    scene: Scene,
    entries: Iterable[dict[str, Any]],
    assets: AssetDatabase | None = None,
    *,
    expand_instances: bool = True,
    link_outer: bool = False,
) -> list[GameObject]:
    """Создать в сцене объекты по записям формата сцены (``uid``/``id``).

    Синхронная обёртка над :func:`iter_build_objects`.
    """
    steps = iter_build_objects(
        scene,
        entries,
        assets,
        expand_instances=expand_instances,
        link_outer=link_outer,
    )
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def iter_build_objects(
    scene: Scene,
    entries: Iterable[dict[str, Any]],
    assets: AssetDatabase | None = None,
    *,
    expand_instances: bool = True,
    link_outer: bool = False,
) -> Generator[float, None, list[GameObject]]:
    """Порционная версия :func:`build_objects` (для асинхронной загрузки).

    После каждого созданного объекта (и каждого шага иерархии) генератор отдаёт
    долю выполненной работы ``0..1``; вызывающий решает, сколько шагов сделать
    за кадр. Результат - в ``StopIteration.value``.

    Общий код загрузки сцены и инстанцирования префабов (там записи
    предварительно переводятся из ``local_id`` в ``uid``).

    Args:
        scene: Сцена, в которой создаются объекты.
        entries: Записи объектов; ``parent`` и ссылки скриптов относятся
            только к этому набору.
        assets: База ассетов.
        expand_instances: Раскрыть записи корней экземпляров префабов (маркер с
            ``instance_uid``) в объекты префаба с overrides. ``False`` -
            записи уже раскрыты (инстанцирование).
        link_outer: Ссылки скриптов на ``uid`` вне набора записей искать в
            самой сцене (откат удаления: объекты вне поддерева живы).

    Returns:
        Созданные объекты в порядке записей.
    """
    entries = list(entries)
    if expand_instances and any(_has_instance_marker(e) for e in entries):
        from engine.prefab import expand_scene_instances  # noqa: PLC0415

        entries = expand_scene_instances(scene, entries, assets)
    created: list[GameObject] = []
    # Ссылки файла на объекты: v2 - по ``uid``, v1 (и смешанные) - по целому id.
    by_uid: dict[str, GameObject] = {}
    by_legacy_id: dict[int, GameObject] = {}
    parents: list[tuple[GameObject, Any]] = []
    scripts: list[ScriptComponent] = []

    total = max(1, len(entries)) * 2
    for index, entry in enumerate(entries):
        yield index / total
        game_object = scene.create_game_object(str(entry.get("name", "Object")))
        created.append(game_object)
        source_uid = _assign_uid(scene, game_object, entry)
        _apply_transform(game_object, entry.get("transform") or {})
        game_object.tag = str(entry.get("tag", "Untagged"))
        game_object.layer = int(entry.get("layer", 1))
        for component in entry.get("components", []):
            add_component_from_payload(game_object, component, assets, scripts)
        if not bool(entry.get("active", True)):
            game_object.active_self = False
        if source_uid is not None:
            by_uid.setdefault(source_uid, game_object)
        if "id" in entry:
            by_legacy_id.setdefault(int(entry["id"]), game_object)
        parents.append((game_object, entry.get("parent")))

    # Иерархия восстанавливается вторым проходом: к этому моменту все объекты
    # уже созданы, поэтому родитель гарантированно существует.
    for index, (child, parent_ref) in enumerate(parents):
        if index % 64 == 0:
            yield 0.5 + index / total
        parent = _lookup_ref(parent_ref, by_uid, by_legacy_id)
        if parent is not None:
            child.set_parent(parent, keep_world=False)

    # Ссылки скриптов на объекты: к этому моменту созданы все объекты. Ключи -
    # ``uid`` (строки) и целые id из v1; типы не пересекаются.
    refs: dict[int | str, GameObject] = {**by_legacy_id, **by_uid}
    if link_outer:
        for script in scripts:
            for source in script._pending_refs.values():  # noqa: SLF001
                if isinstance(source, str) and source not in refs:
                    outer = scene.find_by_uid(source)
                    if outer is not None:
                        refs[source] = outer
    yield 1.0
    for script in scripts:
        script.resolve_refs(refs)

    return created


def _assign_uid(
    scene: Scene, game_object: GameObject, entry: dict[str, Any]
) -> str | None:
    """Назначить объекту ``uid`` из записи файла.

    Приоритет: ``uid`` записи (v2); иначе детерминированный из целого ``id``
    (миграция v1); иначе остаётся случайный.

    Returns:
        ``uid`` как он записан в файле (на него ссылаются другие записи) или
        ``None``, если запись его не задаёт.
    """
    raw = entry.get("uid")
    if raw is None and "id" in entry:
        raw = migrated_uid(int(entry["id"]))
    if raw is None:
        return None
    if not is_valid_uid(raw):
        _log.error(
            "Объект '%s': некорректный uid %r (нужно 16 hex-символов); "
            "выдан новый",
            game_object.name,
            raw,
        )
        return None
    holder = scene.find_by_uid(raw)
    if holder is not None and holder is not game_object:
        _log.error(
            "Объект '%s': uid %s уже занят объектом '%s'; выдан новый uid",
            game_object.name,
            raw,
            holder.name,
        )
        return raw
    game_object.uid = raw
    return raw


def _lookup_ref(
    ref: Any,
    by_uid: dict[str, GameObject],
    by_legacy_id: dict[int, GameObject],
) -> GameObject | None:
    """Найти объект по ссылке файла: ``{"uid": ...}``, число (v1) или ``None``."""
    if ref is None:
        return None
    if isinstance(ref, dict):
        if "uid" in ref:
            return by_uid.get(str(ref["uid"]))
        if "object" in ref:
            return by_legacy_id.get(int(ref["object"]))
        return None
    if isinstance(ref, int) and not isinstance(ref, bool):
        return by_legacy_id.get(ref)
    return None


def save_scene(
    scene: Scene,
    path: Path | str,
    assets: AssetDatabase | None = None,
    sky: SkySettings | None = None,
    post: PostSettings | None = None,
) -> Path:
    """Записать сцену в файл.

    Args:
        scene: Сцена движка.
        path: Путь к файлу ``*.scene.json``.
        assets: База ассетов для записи GUID мешей.
        sky: Настройки неба для секции ``sky``.
        post: Настройки пост-обработки для секции ``post``.

    Returns:
        Путь к записанному файлу.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _backup_v1(target)
    payload = json.dumps(
        scene_to_dict(scene, assets, sky, post), ensure_ascii=False, indent=2
    )
    target.write_text(payload + "\n", encoding="utf-8")
    _log.info("Сцена '%s' сохранена: %s", scene.name, target)
    return target


def _backup_v1(target: Path) -> None:
    """Перед записью v2 сохранить копию существующего файла версии 1.

    Копия ``<файл>.v1.bak`` делается один раз: если она уже есть, самая
    старая версия не затирается. Нет файла, файл не v1 или нечитаем - копии нет.
    """
    if not target.is_file():
        return
    try:
        existing = json.loads(target.read_text(encoding="utf-8"))
        version = int(existing.get("version", 0)) if isinstance(existing, dict) else 0
    except (OSError, ValueError):
        return
    if version > SCENE_VERSION:
        _log.warning(
            "Целевой файл '%s' версии %d > %d, бэкап не создан",
            target,
            version,
            SCENE_VERSION,
        )
        return
    if version != 1:
        return
    backup = target.with_name(target.name + V1_BACKUP_SUFFIX)
    if backup.exists():
        return
    shutil.copy2(target, backup)
    _log.info("Сцена мигрирована v1->v2, бэкап: %s", backup)


def load_scene(path: Path | str, assets: AssetDatabase | None = None) -> Scene:
    """Прочитать сцену из файла.

    Args:
        path: Путь к файлу ``*.scene.json``.
        assets: База ассетов для восстановления путей мешей по GUID.

    Returns:
        Загруженная сцена.

    Raises:
        ValueError: Если файл повреждён или формат не поддерживается.
        OSError: Если файл недоступен.
    """
    return load_scene_with_sky(path, assets)[0]


def read_scene_data(
    path: Path | str, reader: Callable[[Path], str | None] | None = None
) -> dict[str, Any]:
    """Прочитать и разобрать файл сцены (без создания объектов).

    Подходит для рабочего потока: только ввод-вывод и ``json.loads``.

    Args:
        path: Путь к файлу ``*.scene.json``.
        reader: Чтение текста не с диска (из пака): ``путь -> текст | None``;
            ``None`` в ответе — читать с диска.

    Raises:
        ValueError: Если файл содержит некорректный JSON.
        OSError: Если файл недоступен.
    """
    source = Path(path)
    try:
        text = reader(source) if reader is not None else None
        if text is None:
            text = source.read_text(encoding="utf-8")
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Файл сцены '{source}' содержит некорректный JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Файл сцены '{source}' не содержит объект JSON")
    return data


def load_scene_with_sky(
    path: Path | str, assets: AssetDatabase | None = None
) -> tuple[Scene, SkySettings]:
    """Прочитать сцену вместе с настройками неба.

    Args:
        path: Путь к файлу ``*.scene.json``.
        assets: База ассетов для восстановления путей мешей по GUID.

    Returns:
        Пару ``(сцена, настройки неба)``.

    Raises:
        ValueError: Если файл повреждён или формат не поддерживается.
        OSError: Если файл недоступен.
    """
    source = Path(path)
    data = read_scene_data(source)
    scene = scene_from_dict(data, assets)
    _log.info("Сцена загружена: %s (%d объектов)", source, len(scene))
    return scene, read_sky(data)


def load_scene_full(
    path: Path | str,
    assets: AssetDatabase | None = None,
    reader: Callable[[Path], str | None] | None = None,
) -> tuple[Scene, SkySettings, PostSettings | None]:
    """Прочитать сцену вместе с небом и пост-обработкой.

    Args:
        path: Путь к файлу ``*.scene.json``.
        assets: База ассетов для восстановления путей мешей по GUID.
        reader: Чтение текста не с диска (из пака), см. :func:`read_scene_data`.

    Returns:
        Тройку ``(сцена, небо, пост-обработка или None без секции)``.

    Raises:
        ValueError: Если файл повреждён или формат не поддерживается.
        OSError: Если файл недоступен.
    """
    source = Path(path)
    data = read_scene_data(source, reader)
    scene = scene_from_dict(data, assets)
    _log.info("Сцена загружена: %s (%d объектов)", source, len(scene))
    return scene, read_sky(data), read_post(data)


def _apply_transform(game_object: GameObject, payload: dict[str, Any]) -> None:
    """Применить локальный трансформ из словаря."""
    position = payload.get("position") or (0.0, 0.0, 0.0)
    rotation = payload.get("rotation") or (0.0, 0.0, 0.0, 1.0)
    scale = payload.get("scale") or (1.0, 1.0, 1.0)
    transform = game_object.transform
    transform.position = Vector3.from_iterable(position)
    transform.rotation = Quaternion(*[float(value) for value in rotation])
    transform.scale = Vector3.from_iterable(scale)
