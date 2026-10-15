"""Команды над объектами: создание, удаление, перенос, свойства.

Создание и удаление хранят поддерево в виде записей сцены (формат v2, с
``uid``), поэтому откат возвращает объекты с теми же ``uid``, значениями
полей и ссылками скриптов. Runtime-``id`` и сами экземпляры ``GameObject``
при этом новые - см. докстринг пакета про инвалидацию ссылок.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from engine.core.commands.stack import MERGE_WINDOW, Command

if TYPE_CHECKING:
    from engine.core.game_object import GameObject
    from engine.core.scene import Scene


def iter_subtree(root: GameObject) -> Iterator[GameObject]:
    """Объект и все потомки в порядке DFS (родитель раньше потомков)."""
    stack = [root]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def find_object(scene: Scene, uid: str) -> GameObject:
    """Объект по ``uid`` или ``LookupError`` (команда применена не к той сцене)."""
    found = scene.find_by_uid(uid)
    if found is None:
        raise LookupError(f"Объект {uid} не найден в сцене")
    return found


def _find_optional(scene: Scene, uid: str | None) -> GameObject | None:
    return None if uid is None else find_object(scene, uid)


def clone_value(value: Any) -> Any:
    """Копия значения поля: неизменяемое возвращается как есть."""
    if value is None or isinstance(value, (bool, int, float, str, bytes, tuple)):
        return value
    return copy.deepcopy(value)


def _restore_subtree(
    scene: Scene,
    entries: list[dict[str, Any]],
    parent_uid: str | None,
    index: int,
) -> None:
    """Пересоздать поддерево из записей и поставить его на место."""
    from engine.scene_io import build_objects  # noqa: PLC0415

    parent = None if parent_uid is None else find_object(scene, parent_uid)
    objects = build_objects(
        scene,
        entries,
        scene.asset_source,
        expand_instances=False,
        link_outer=True,
    )
    objects[0].set_parent(parent, keep_world=False, index=index)


def _destroy_subtree(scene: Scene, root_uid: str) -> None:
    scene.destroy_game_object(find_object(scene, root_uid), immediate=True)


def _capture_subtree(
    root: GameObject,
) -> tuple[list[dict[str, Any]], str | None, int]:
    from engine.scene_io import serialize_objects  # noqa: PLC0415

    entries = serialize_objects(iter_subtree(root), root.scene.asset_source)
    parent = root.parent
    return entries, None if parent is None else parent.uid, root.sibling_index()


class CreateObjectCommand:
    """Создание объекта с поддеревом.

    Обычно объект уже создан кодом редактора: :meth:`capture` собирает команду
    по готовому результату, а в стек её кладёт ``CommandStack.push``. ``do``
    нужен для повторения (redo) - он строит поддерево заново по записям.
    """

    def __init__(
        self,
        entries: list[dict[str, Any]],
        parent_uid: str | None,
        index: int,
        name: str = "Создать объект",
    ) -> None:
        self.name = name
        self.entries = entries
        self.parent_uid = parent_uid
        self.index = index
        self.root_uid: str = entries[0]["uid"]

    @classmethod
    def capture(
        cls, game_object: GameObject, name: str = "Создать объект"
    ) -> CreateObjectCommand:
        """Собрать команду по уже созданному объекту (вместе с поддеревом)."""
        entries, parent_uid, index = _capture_subtree(game_object)
        return cls(entries, parent_uid, index, name)

    def do(self, scene: Scene) -> None:
        _restore_subtree(scene, self.entries, self.parent_uid, self.index)

    def undo(self, scene: Scene) -> None:
        _destroy_subtree(scene, self.root_uid)

    def merge_with(self, other: Command) -> bool:  # noqa: ARG002
        return False


class DeleteObjectCommand:
    """Удаление объекта с поддеревом (снимок записей + позиция среди братьев).

    Снимок делается при создании команды, до удаления.
    """

    def __init__(self, game_object: GameObject, name: str = "Удалить объект") -> None:
        self.name = name
        self.entries, self.parent_uid, self.index = _capture_subtree(game_object)
        self.root_uid: str = self.entries[0]["uid"]

    def do(self, scene: Scene) -> None:
        _destroy_subtree(scene, self.root_uid)

    def undo(self, scene: Scene) -> None:
        _restore_subtree(scene, self.entries, self.parent_uid, self.index)

    def merge_with(self, other: Command) -> bool:  # noqa: ARG002
        return False



def _local_state(game_object: GameObject) -> tuple[Any, Any, Any]:
    transform = game_object.transform
    return transform.position, transform.rotation, transform.scale


def _set_local_state(game_object: GameObject, state: tuple[Any, Any, Any]) -> None:
    transform = game_object.transform
    transform.position, transform.rotation, transform.scale = state


class ReparentCommand:
    """Перенос объекта под другого родителя / на другую позицию среди братьев.

    Индексы - позиции среди братьев **после** изъятия объекта из старого места
    (для корней - среди корней сцены). Локальный трансформ до и после хранится
    явно, поэтому откат побитово возвращает исходные значения (пересчёт
    ``keep_world`` обратно дал бы погрешность).

    Args:
        game_object: Переносимый объект.
        new_parent: Новый родитель или ``None`` (корень).
        new_index: Позиция среди новых братьев; ``None`` - в конец.
        keep_world: Сохранять мировое положение при переносе.
    """

    def __init__(
        self,
        game_object: GameObject,
        new_parent: GameObject | None,
        new_index: int | None = None,
        *,
        keep_world: bool = True,
        name: str = "Переместить в иерархии",
    ) -> None:
        self.name = name
        self.uid = game_object.uid
        old_parent = game_object.parent
        self.old_parent_uid = None if old_parent is None else old_parent.uid
        self.old_index = game_object.sibling_index()
        self.new_parent_uid = None if new_parent is None else new_parent.uid
        self.new_index = new_index
        self.keep_world = keep_world
        self._old_local = _local_state(game_object)
        self._new_local: tuple[Any, Any, Any] | None = None

    def do(self, scene: Scene) -> None:
        game_object = find_object(scene, self.uid)
        parent = _find_optional(scene, self.new_parent_uid)
        if self._new_local is None:
            game_object.set_parent(
                parent, keep_world=self.keep_world, index=self.new_index
            )
            self._new_local = _local_state(game_object)
        else:
            game_object.set_parent(parent, keep_world=False, index=self.new_index)
            _set_local_state(game_object, self._new_local)

    def undo(self, scene: Scene) -> None:
        game_object = find_object(scene, self.uid)
        parent = _find_optional(scene, self.old_parent_uid)
        game_object.set_parent(parent, keep_world=False, index=self.old_index)
        _set_local_state(game_object, self._old_local)

    def merge_with(self, other: Command) -> bool:  # noqa: ARG002
        return False


class _SetObjectAttr:
    """Базовая команда смены простого свойства объекта (с слиянием правок)."""

    attr = ""
    label = ""

    def __init__(
        self, game_object: GameObject, new: Any, *, stamp: float | None = None
    ) -> None:
        self.name = self.label
        self.uid = game_object.uid
        self.old = getattr(game_object, self.attr)
        self.new = new
        self.stamp = time.monotonic() if stamp is None else stamp

    def do(self, scene: Scene) -> None:
        setattr(find_object(scene, self.uid), self.attr, self.new)

    def undo(self, scene: Scene) -> None:
        setattr(find_object(scene, self.uid), self.attr, self.old)

    def merge_with(self, other: Command) -> bool:
        if (
            type(other) is not type(self)
            or other.uid != self.uid  # type: ignore[attr-defined]
            or other.stamp - self.stamp > MERGE_WINDOW  # type: ignore[attr-defined]
        ):
            return False
        self.new = other.new  # type: ignore[attr-defined]
        self.stamp = other.stamp  # type: ignore[attr-defined]
        return True


class RenameObjectCommand(_SetObjectAttr):
    """Переименование объекта."""

    attr = "name"
    label = "Переименовать"


class SetActiveCommand(_SetObjectAttr):
    """Включение/выключение объекта (``active_self``)."""

    attr = "active_self"
    label = "Включить/выключить объект"


class SetTagCommand(_SetObjectAttr):
    """Смена тега объекта."""

    attr = "tag"
    label = "Сменить тег"


class SetLayerCommand(_SetObjectAttr):
    """Смена слоя объекта."""

    attr = "layer"
    label = "Сменить слой"
