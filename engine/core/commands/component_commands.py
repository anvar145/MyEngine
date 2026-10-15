"""Команды над компонентами: правка поля, добавление, удаление.

Компонент адресуется парой «тип + порядковый номер среди компонентов этого типа
на объекте», объект - по ``uid``: после отката создания/удаления экземпляры
пересоздаются, ссылки на них становятся недействительными.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from engine.core.commands.object_commands import clone_value, find_object
from engine.core.commands.stack import MERGE_WINDOW, Command
from engine.core.transform import Transform

if TYPE_CHECKING:
    from engine.core.component import Component
    from engine.core.game_object import GameObject
    from engine.core.scene import Scene


class _Target:
    """Адрес компонента (или трансформа) объекта, устойчивый к пересозданию."""

    __slots__ = ("kind", "ordinal", "uid")

    def __init__(self, game_object: GameObject, component: Component | Transform):
        self.uid = game_object.uid
        if isinstance(component, Transform):
            self.kind: type | None = None
            self.ordinal = 0
        else:
            self.kind = type(component)
            self.ordinal = game_object.get_components(self.kind).index(component)

    def resolve(self, scene: Scene) -> Any:
        game_object = find_object(scene, self.uid)
        if self.kind is None:
            return game_object.transform
        return game_object.get_components(self.kind)[self.ordinal]

    def same(self, other: _Target) -> bool:
        return (
            self.uid == other.uid
            and self.kind is other.kind
            and self.ordinal == other.ordinal
        )


class SetFieldCommand:
    """Правка одного поля компонента (или ``Transform``: position/rotation/scale).

    Значения копируются при создании команды. Правки того же поля в окне
    :data:`~engine.core.commands.stack.MERGE_WINDOW` (от предыдущей правки)
    сливаются: ``old`` остаётся первым, ``new`` - последним.

    Args:
        game_object: Владелец компонента.
        component: Компонент или ``game_object.transform``.
        field: Имя атрибута.
        old: Значение до правки.
        new: Значение после правки.
        name: Название записи.
        stamp: Момент правки (``time.monotonic``); для тестов.
        via: Имя атрибута компонента с вложенным объектом (``"camera"``,
            ``"light"``): поле правится у него, а не у самого компонента.
    """

    def __init__(
        self,
        game_object: GameObject,
        component: Component | Transform,
        field: str,
        old: Any,
        new: Any,
        name: str | None = None,
        *,
        stamp: float | None = None,
        via: str | None = None,
    ) -> None:
        self.target = _Target(game_object, component)
        self.field = field
        self.via = via
        self.old = clone_value(old)
        self.new = clone_value(new)
        self.name = name or f"Изменить {field}"
        self.stamp = time.monotonic() if stamp is None else stamp

    def _owner(self, scene: Scene) -> Any:
        owner = self.target.resolve(scene)
        return owner if self.via is None else getattr(owner, self.via)

    def do(self, scene: Scene) -> None:
        setattr(self._owner(scene), self.field, clone_value(self.new))

    def undo(self, scene: Scene) -> None:
        setattr(self._owner(scene), self.field, clone_value(self.old))

    def merge_with(self, other: Command) -> bool:
        if (
            not isinstance(other, SetFieldCommand)
            or other.field != self.field
            or other.via != self.via
            or not other.target.same(self.target)
            or other.stamp - self.stamp > MERGE_WINDOW
        ):
            return False
        self.new = other.new
        self.stamp = other.stamp
        return True


def _payload_of(game_object: GameObject, component: Component) -> dict[str, Any] | None:
    from engine.scene_io import component_payload_pairs  # noqa: PLC0415

    for candidate, payload in component_payload_pairs(
        game_object, game_object.scene.asset_source
    ):
        if candidate is component:
            return payload
    return None


class AddComponentCommand:
    """Добавление компонента (по готовому результату: ``capture`` + ``push``).

    ``do`` строит компонент заново из записи сцены; для типа, который сцена не
    умеет сериализовать, - создаёт новый экземпляр класса без аргументов.
    """

    def __init__(
        self,
        game_object: GameObject,
        component: Component,
        name: str = "Добавить компонент",
    ) -> None:
        self.name = name
        self.uid = game_object.uid
        self.kind = type(component)
        self.ordinal = game_object.get_components(self.kind).index(component)
        self.position = game_object.components.index(component)
        self.payload = _payload_of(game_object, component)

    @classmethod
    def capture(
        cls,
        game_object: GameObject,
        component: Component,
        name: str = "Добавить компонент",
    ) -> AddComponentCommand:
        return cls(game_object, component, name)

    def do(self, scene: Scene) -> None:
        game_object = find_object(scene, self.uid)
        _readd(game_object, self.kind, self.payload, self.position)

    def undo(self, scene: Scene) -> None:
        game_object = find_object(scene, self.uid)
        game_object.remove_component(game_object.get_components(self.kind)[self.ordinal])

    def merge_with(self, other: Command) -> bool:  # noqa: ARG002
        return False


def _readd(
    game_object: GameObject,
    kind: type,
    payload: dict[str, Any] | None,
    position: int,
) -> None:
    from engine.scene_io import add_component_from_payload  # noqa: PLC0415

    component = None
    if payload is not None:
        component = add_component_from_payload(
            game_object, payload, game_object.scene.asset_source
        )
    if component is None:
        component = game_object.add_component(kind)
    game_object.reorder_component(component, position)


class RemoveComponentCommand:
    """Удаление компонента (запись компонента хранится для отката)."""

    def __init__(
        self,
        game_object: GameObject,
        component: Component,
        name: str = "Удалить компонент",
    ) -> None:
        self.name = name
        self.uid = game_object.uid
        self.kind = type(component)
        self.ordinal = game_object.get_components(self.kind).index(component)
        self.position = game_object.components.index(component)
        self.payload = _payload_of(game_object, component)

    def do(self, scene: Scene) -> None:
        game_object = find_object(scene, self.uid)
        game_object.remove_component(game_object.get_components(self.kind)[self.ordinal])

    def undo(self, scene: Scene) -> None:
        game_object = find_object(scene, self.uid)
        _readd(game_object, self.kind, self.payload, self.position)

    def merge_with(self, other: Command) -> bool:  # noqa: ARG002
        return False


_MISSING = object()


class _ObjectRef:
    """Ссылка на объект сцены в значении поля скрипта (устойчива к пересозданию)."""

    __slots__ = ("uid",)

    def __init__(self, uid: str) -> None:
        self.uid = uid


def _pack_field_value(value: Any) -> Any:
    from engine.core.game_object import GameObject  # noqa: PLC0415

    if value is _MISSING:
        return value
    if isinstance(value, GameObject):
        return _ObjectRef(value.uid)
    return clone_value(value)


class SetScriptFieldCommand:
    """Правка переопределения поля скрипта (``ScriptComponent.fields``).

    Значение «нет переопределения» (``_MISSING``) означает сброс к умолчанию
    из кода. Ссылки на объекты хранятся по ``uid``.

    Args:
        game_object: Владелец скрипта.
        component: ``ScriptComponent``.
        field: Имя поля скрипта.
        old: Прежнее значение переопределения или :data:`MISSING`.
        new: Новое значение или :data:`MISSING`.
        name: Название записи.
        stamp: Момент правки; для тестов.
    """

    MISSING = _MISSING

    def __init__(
        self,
        game_object: GameObject,
        component: Component,
        field: str,
        old: Any,
        new: Any,
        name: str | None = None,
        *,
        stamp: float | None = None,
    ) -> None:
        self.target = _Target(game_object, component)
        self.field = field
        self.old = _pack_field_value(old)
        self.new = _pack_field_value(new)
        self.name = name or f"Изменить {field}"
        self.stamp = time.monotonic() if stamp is None else stamp

    def _apply(self, scene: Scene, packed: Any) -> None:
        component = self.target.resolve(scene)
        if packed is _MISSING:
            component.clear_field(self.field)
            return
        if isinstance(packed, _ObjectRef):
            value = find_object(scene, packed.uid)
        else:
            value = clone_value(packed)
        component.set_field(self.field, value)

    def do(self, scene: Scene) -> None:
        self._apply(scene, self.new)

    def undo(self, scene: Scene) -> None:
        self._apply(scene, self.old)

    def merge_with(self, other: Command) -> bool:
        if (
            not isinstance(other, SetScriptFieldCommand)
            or other.field != self.field
            or not other.target.same(self.target)
            or other.stamp - self.stamp > MERGE_WINDOW
        ):
            return False
        self.new = other.new
        self.stamp = other.stamp
        return True
