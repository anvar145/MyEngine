"""Действия редактора над экземплярами префабов: вернуть / применить к префабу.

«Применить к префабу» меняет файл ``*.prefab.json`` и пересобирает все
открытые экземпляры этого префаба с сохранением их overrides (синхронно, в
главном потоке: сцена не потокобезопасна).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.core.commands import CreateObjectCommand, ReplaceSubtreesCommand
from engine.core.game_object import GameObject
from engine.core.math3d import Vector3
from engine.core.prefab_components import PrefabInstance
from engine.prefab import (
    Prefab,
    PrefabError,
    capture_instances,
    dump_prefab,
    get_overrides,
    make_prefab_loader,
    restore_instances,
    revert_override,
)
from engine.prefab_save import find_instance_root, save_as_prefab, unpack_instance

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from editor.context import EditorContext

__all__ = [
    "REBUILD_WARN_THRESHOLD",
    "apply_override_to_prefab",
    "prefab_file",
    "revert_instance_override",
    "apply_all_overrides",
    "instantiate_prefab_file",
    "revert_all_overrides",
    "save_object_as_prefab",
    "unpack_object",
]

_log = logging.getLogger("editor.prefab")

#: Порог числа экземпляров, с которого пересборка предупреждает пользователя.
REBUILD_WARN_THRESHOLD = 20


def prefab_file(context: EditorContext, prefab: Prefab) -> Path:
    """Файл префаба на диске (по GUID из базы ассетов, иначе по пути проекта).

    Raises:
        PrefabError: Файл не найден.
    """
    assets = context.asset_db
    if assets is not None and prefab.guid:
        handle = assets.get(prefab.guid)
        if handle is not None and Path(handle.path).is_file():
            return Path(handle.path)
    candidates = [context.project_root / prefab.path]
    if assets is not None:
        candidates += [assets.root.parent / prefab.path, assets.root / prefab.path]
    for candidate in candidates:
        if prefab.path and candidate.is_file():
            return candidate
    raise PrefabError(f"файл префаба не найден: {prefab.path!r}")


def _record_replace(
    context: EditorContext,
    roots: list[GameObject],
    name: str,
    operation: Callable[[], None],
    *,
    after: Callable[[], list[GameObject]] | None = None,
) -> None:
    """Выполнить операцию и записать её как замену поддеревьев (один undo).

    Args:
        context: Контекст редактора.
        roots: Корни областей, которые операция может изменить.
        name: Подпись записи истории.
        operation: Сама правка; корни по ``uid`` ищутся после неё заново.
        after: Корни областей «после» (по умолчанию те же ``uid``, что «до»).
    """
    scene = context.scene
    uids = [root.uid for root in roots]
    pending = ReplaceSubtreesCommand.around(roots, name)
    operation()
    if after is not None:
        after_roots = after()
    else:
        found = (scene.find_by_uid(uid) for uid in uids)
        after_roots = [obj for obj in found if obj is not None]
    context.push(pending.finish(after_roots))


def revert_instance_override(
    context: EditorContext, root: GameObject, record: dict[str, Any]
) -> None:
    """Вернуть в экземпляре значение из префаба и сообщить об изменении сцены."""
    loader = make_prefab_loader(context.asset_db, context.project_root)
    _record_replace(
        context,
        [root],
        context.tr("undo.prefab_revert", name=root.name),
        lambda: revert_override(root, record, loader=loader, assets=context.asset_db),
    )


def apply_override_to_prefab(
    context: EditorContext,
    root: GameObject,
    record: dict[str, Any],
    *,
    progress: Callable[[int, int], None] | None = None,
    warn: Callable[[str], None] | None = None,
) -> int:
    """Записать override в файл префаба и пересобрать открытые экземпляры.

    Args:
        context: Контекст редактора.
        root: Корень экземпляра, из которого берётся значение.
        record: Запись override (из ``get_overrides``).
        progress: ``(сделано, всего)`` - вызывается на каждом экземпляре.
        warn: Получает предупреждение, если экземпляров больше порога.

    Returns:
        Число пересобранных экземпляров.

    Raises:
        PrefabError: Нет файла/цели/компонента или значение не записывается.
    """
    marker = root.get_component(PrefabInstance)
    scene = context.scene
    if marker is None or not marker.instance_uid:
        raise PrefabError("объект не является корнем экземпляра префаба")
    assets = context.asset_db
    loader = make_prefab_loader(assets, context.project_root)
    prefab = loader(marker.prefab_guid, marker.prefab_path)
    path = prefab_file(context, prefab)
    # Свернуть экземпляры по СТАРОМУ префабу до записи файла.
    snapshot = capture_instances(scene, prefab.guid, loader=loader, assets=assets)
    prefab.apply_override_to_asset(
        str(record["target"]),
        str(record["component"]),
        record.get("field"),
        record.get("value"),
        index=int(record.get("index", 0)),
        unset=bool(record.get("unset")),
        add=bool(record.get("add")),
    )
    path.write_text(dump_prefab(prefab.to_dict()), encoding="utf-8")
    if len(snapshot) > REBUILD_WARN_THRESHOLD:
        message = (
            f"Пересборка {len(snapshot)} экземпляров префаба '{prefab.name}' "
            "выполняется в главном потоке и может занять время"
        )
        _log.warning(message)
        if warn is not None:
            warn(message)
    scene.prefab_loader = make_prefab_loader(assets, context.project_root)
    selected = [o.uid for o in context.selection.objects]
    roots = [region[0] for region, _entries in snapshot.regions if region]
    _record_replace(
        context,
        roots,
        context.tr("undo.prefab_apply", name=prefab.name),
        lambda: restore_instances(scene, snapshot, assets=assets, progress=progress),
    )
    restored = [scene.find_by_uid(uid) for uid in selected]
    context.selection.select_many([o for o in restored if o is not None])
    return len(snapshot)


def save_object_as_prefab(
    context: EditorContext, game_object: GameObject, path: Path | str
) -> GameObject:
    """Сохранить объект как префаб и заменить его экземпляром.

    Выделение переносится на новый экземпляр, сцена помечается изменённой.

    Raises:
        PrefabError: Объект нельзя сохранить (входит в экземпляр и т.п.).
        OSError: Файл не записывается.
    """
    scene = context.scene
    created: list[GameObject] = []
    # Скрипты других объектов переписываются на новый экземпляр, поэтому
    # областью служит вся сцена.
    _record_replace(
        context,
        list(scene.root_objects),
        context.tr("undo.prefab_save", name=game_object.name),
        lambda: created.append(
            save_as_prefab(
                scene,
                game_object,
                path,
                assets=context.asset_db,
                project_root=context.project_root,
            )
        ),
        after=lambda: list(scene.root_objects),
    )
    instance = created[0]
    context.selection.forget(game_object)
    context.selection.select(instance)
    return instance


def instantiate_prefab_file(
    context: EditorContext,
    path: Path | str,
    *,
    parent: GameObject | None = None,
    position: Vector3 | None = None,
) -> GameObject:
    """Создать в сцене экземпляр префаба из файла, выделить его.

    Args:
        context: Контекст редактора.
        path: Файл ``*.prefab.json``.
        parent: Родитель корня экземпляра (``None`` - корень сцены).
        position: Локальная позиция корня (``None`` - из префаба).

    Raises:
        PrefabError: Файл не загружается или содержит ошибки.
        OSError: Файл не читается.
    """
    loader = make_prefab_loader(context.asset_db, context.project_root)
    prefab = loader("", str(path))
    context.scene.prefab_loader = loader
    root = context.scene.instantiate(prefab, parent=parent, position=position)
    context.push(
        CreateObjectCommand.capture(root, context.tr("undo.create", name=root.name))
    )
    context.selection.select(root)
    return root


def unpack_object(context: EditorContext, game_object: GameObject) -> GameObject:
    """Разъединить экземпляр, которому принадлежит объект (Unpack).

    Выделение переносится на бывший корень, сцена помечается изменённой.

    Raises:
        PrefabError: Объект не входит в экземпляр.
    """
    instance_root = find_instance_root(game_object)
    if instance_root is None:
        return unpack_instance(game_object)  # бросит PrefabError
    unpacked: list[GameObject] = []
    _record_replace(
        context,
        [instance_root],
        context.tr("undo.prefab_unpack", name=instance_root.name),
        lambda: unpacked.append(unpack_instance(game_object)),
    )
    root = unpacked[0]
    context.selection.select(root)
    return root


def revert_all_overrides(context: EditorContext, root: GameObject) -> int:
    """Вернуть все значения экземпляра к префабу.

    Returns:
        Число возвращённых записей.
    """
    records = get_overrides(root, assets=context.asset_db)
    if not records:
        return 0
    loader = make_prefab_loader(context.asset_db, context.project_root)
    def revert() -> None:
        for record in records:
            revert_override(root, record, loader=loader, assets=context.asset_db)

    _record_replace(
        context,
        [root],
        context.tr("undo.prefab_revert", name=root.name),
        revert,
    )
    return len(records)


def apply_all_overrides(
    context: EditorContext,
    root: GameObject,
    *,
    progress: Callable[[int, int], None] | None = None,
    warn: Callable[[str], None] | None = None,
) -> int:
    """Записать все overrides экземпляра в файл префаба.

    Экземпляры пересобираются после каждой записи, поэтому корень каждый раз
    ищется заново по ``uid``.

    Returns:
        Число применённых записей.
    """
    uid = root.uid
    applied = 0
    records = get_overrides(root, assets=context.asset_db)
    with context.transaction(context.tr("undo.prefab_apply", name=root.name)):
        for record in records:
            current = context.scene.find_by_uid(uid)
            if current is None:
                break
            apply_override_to_prefab(
                context, current, record, progress=progress, warn=warn
            )
            applied += 1
    return applied
