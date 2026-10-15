"""Сохранение поддерева как префаба и разъединение экземпляра (Unpack).

Логика без Qt: редактор вызывает :func:`save_as_prefab` из меню иерархии.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.core.game_object import GameObject
from engine.core.prefab_components import PrefabInstance
from engine.core.resources.errors import ResourceError
from engine.prefab import (
    PrefabError,
    dump_prefab,
    make_prefab_loader,
    prefab_from_dict,
    prefab_to_dict,
)
from engine.scripting.script_component import ScriptComponent

if TYPE_CHECKING:
    from engine.core.resources.asset_db import AssetDatabase
    from engine.core.scene import Scene

_log = logging.getLogger("engine.prefab_save")

__all__ = [
    "check_can_save_as_prefab",
    "find_instance_root",
    "save_as_prefab",
    "unpack_instance",
]


def check_can_save_as_prefab(root: GameObject) -> None:
    """Проверить, что поддерево можно сохранить как префаб.

    Нельзя сохранять объект, который сам является частью экземпляра другого
    префаба (корень или его член): правьте исходный префаб или сделайте Unpack.
    Вложенные экземпляры *внутри* поддерева допустимы - они остаются
    вложенными префабами.

    Raises:
        PrefabError: Объект нельзя сохранить; текст объяснения для пользователя.
    """
    if root.is_destroyed or root.scene is None:
        raise PrefabError("объект удалён")
    if root.pooled:
        raise PrefabError("объект принадлежит пулу и не сохраняется в сцену")
    node: GameObject | None = root
    while node is not None:
        marker = node.get_component(PrefabInstance)
        if marker is not None and marker.instance_uid:
            raise PrefabError(
                f"объект входит в экземпляр префаба '{node.name}': "
                "разъедините его (Unpack) или правьте исходный префаб"
            )
        node = node.parent


def save_as_prefab(
    scene: Scene,
    root: GameObject,
    path: Path | str,
    *,
    assets: AssetDatabase | None = None,
    project_root: Path | str | None = None,
    name: str | None = None,
) -> GameObject:
    """Записать поддерево в ``*.prefab.json`` и заменить его экземпляром.

    Исходное поддерево удаляется, на его место (тот же родитель и локальный
    трансформ) создаётся экземпляр. ``uid`` объектов экземпляра выводятся из
    ``instance_uid`` (см. A4.1), поэтому они **не совпадают** с прежними.

    Args:
        scene: Сцена объекта.
        root: Корень сохраняемого поддерева.
        path: Файл ``*.prefab.json``.
        assets: База ассетов (``mesh_guid``, пути скриптов, поиск GUID).
        project_root: Корень проекта для относительного пути в маркере.
        name: Имя префаба (по умолчанию - имя корня).

    Returns:
        Корень нового экземпляра.

    Raises:
        PrefabError: Объект нельзя сохранить или данные некорректны.
        OSError: Файл не записывается.
    """
    check_can_save_as_prefab(root)
    target = Path(path)
    data = prefab_to_dict(root, assets, name)
    prefab_from_dict(data)  # проверка до записи файла
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_prefab(data), encoding="utf-8")

    if assets is not None:
        try:
            # GUID нужен сразу: вложенный префаб без него не раскрывается
            assets.register(target)
        except (OSError, ValueError, ResourceError) as exc:
            _log.warning("Префаб '%s' не зарегистрирован: %s", target, exc)
    loader = make_prefab_loader(assets, project_root)
    prefab = loader("", str(target))
    scene.prefab_loader = loader
    parent = root.parent
    transform = root.transform
    scale = transform.scale
    instance = scene.instantiate(
        prefab,
        parent=parent,
        position=transform.position,
        rotation=transform.rotation,
    )
    instance.transform.scale = scale
    instance.name = root.name
    instance.active_self = root.active_self
    mapping = _pair_objects(root, instance)
    _rebind_external_refs(scene, root, mapping)
    scene.destroy_game_object(root, immediate=True)
    return instance



def _pair_objects(old: GameObject, new: GameObject) -> dict[str, GameObject]:
    """Соответствие ``uid`` старого объекта -> объект нового экземпляра.

    Деревья сопоставляются по порядку детей; ветка с расхождением (число детей
    или имя) пропускается - ссылки в неё не перепривязываются.
    """
    result: dict[str, GameObject] = {}
    stack = [(old, new)]
    while stack:
        left, right = stack.pop()
        if left.name != right.name or len(left.children) != len(right.children):
            continue
        result[left.uid] = right
        stack.extend(zip(left.children, right.children, strict=True))
    return result


def _subtree(root: GameObject) -> set[int]:
    """Идентификаторы (``id``) объектов поддерева."""
    ids: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        ids.add(node.id)
        stack.extend(node.children)
    return ids


def _remap_value(value: Any, uids: dict[str, str]) -> Any:
    """Заменить в значении override ссылки ``{"uid": старый}`` на новые."""
    if isinstance(value, dict):
        if set(value) == {"uid"} and value["uid"] in uids:
            return {"uid": uids[value["uid"]]}
        return {key: _remap_value(item, uids) for key, item in value.items()}
    if isinstance(value, list):
        return [_remap_value(item, uids) for item in value]
    return value


def _rebind_external_refs(
    scene: Scene, old_root: GameObject, mapping: dict[str, GameObject]
) -> int:
    """Перенаправить ссылки *извне* поддерева на объекты нового экземпляра.

    Затрагиваются поля скриптов (живые и ещё не разрешённые ссылки) и значения
    overrides других экземпляров. Ссылки изнутри поддерева не трогаются: они
    сохранены в префабе как ``local`` и сами перепривяжутся.

    Returns:
        Число перепривязанных ссылок.
    """
    if not mapping:
        return 0
    inside = _subtree(old_root)
    uids = {uid: obj.uid for uid, obj in mapping.items()}
    by_old_id = {
        obj.id: mapping[obj.uid]
        for obj in _walk(old_root)
        if obj.uid in mapping
    }
    count = 0
    for game_object in scene.objects:
        if game_object.id in inside:
            continue
        for script in game_object.get_components(ScriptComponent):
            for name, value in list(script.fields.items()):
                target = (
                    by_old_id.get(value.id) if isinstance(value, GameObject) else None
                )
                if target is not None and script.set_field(name, target):
                    count += 1
            for name, source in list(script._pending_refs.items()):
                if isinstance(source, str) and source in uids:
                    script._pending_refs[name] = uids[source]
                    count += 1
        marker = game_object.get_component(PrefabInstance)
        if marker is not None and marker.overrides:
            for record in marker.overrides:
                if "value" in record:
                    new_value = _remap_value(record["value"], uids)
                    if new_value != record["value"]:
                        record["value"] = new_value
                        count += 1
    return count


def _walk(root: GameObject):
    """Объекты поддерева (корень первым)."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


def find_instance_root(game_object: GameObject) -> GameObject | None:
    """Корень ближайшего экземпляра префаба вверх по иерархии (включая сам объект).

    Returns:
        Объект с ``PrefabInstance.instance_uid`` или ``None``.
    """
    node: GameObject | None = game_object
    while node is not None:
        marker = node.get_component(PrefabInstance)
        if marker is not None and marker.instance_uid:
            return node
        node = node.parent
    return None


def unpack_instance(game_object: GameObject) -> GameObject:
    """Разъединить экземпляр: объекты остаются в сцене как обычные.

    С корня снимается маркер ``PrefabInstance``. Значения (в том числе
    overrides) остаются в объектах, ``uid`` сохраняются, правки префаба к ним
    больше не применяются. Вложенные экземпляры внутри остаются экземплярами
    (каждый - самостоятельным). Объект может быть любым членом экземпляра:
    корень ищется вверх по иерархии.

    Args:
        game_object: Корень экземпляра или любой его член.

    Returns:
        Бывший корень экземпляра.

    Raises:
        PrefabError: Объект не входит в экземпляр.
    """
    root = find_instance_root(game_object)
    if root is None:
        raise PrefabError("объект не входит в экземпляр префаба")
    root.remove_component(PrefabInstance)
    stack = [root]
    while stack:
        node = stack.pop()
        node.pooled = False
        stack.extend(
            child
            for child in node.children
            if not (
                (marker := child.get_component(PrefabInstance)) and marker.instance_uid
            )
        )
    return root
