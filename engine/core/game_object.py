"""Игровой объект: узел сцены с именем, трансформом и набором компонентов.

Модель гибридная — «объект + компоненты», как в привычных редакторах, но с
плоским numpy-хранилищем трансформов (см. :mod:`engine.core.transform`), чтобы
иерархия обновлялась пакетно, а не питоновским циклом по матрицам.

Example:
    >>> cube = scene.create_game_object("Cube")
    >>> cube.add_component(Spinner, speed=45.0)
    >>> child = scene.create_game_object("Child", parent=cube)
"""

from __future__ import annotations

import logging
import secrets
import weakref
from collections.abc import Iterator
from typing import TYPE_CHECKING, TypeVar

from engine.core.component import Component
from engine.core.math3d import Quaternion, Vector3
from engine.core.transform import ROOT_INDEX, Transform, TransformStore

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.scene import Scene

__all__ = ["UID_LENGTH", "GameObject", "is_valid_uid", "new_uid"]

_log = logging.getLogger("engine.core.game_object")

#: Длина ``uid`` объекта: 16 hex-символов (64 бита).
UID_LENGTH = 16
_HEX = frozenset("0123456789abcdef")


def new_uid() -> str:
    """Выдать случайный ``uid`` (16 hex-символов)."""
    return secrets.token_hex(UID_LENGTH // 2)


def is_valid_uid(value: object) -> bool:
    """``True``, если значение - строка из 16 hex-символов в нижнем регистре."""
    return (
        isinstance(value, str)
        and len(value) == UID_LENGTH
        and all(char in _HEX for char in value)
    )

ComponentT = TypeVar("ComponentT", bound=Component)

DEFAULT_LAYER = 1

#: Общее хранилище для объектов, созданных вне сцены. Одно на процесс, поэтому
#: «свободные» объекты можно связывать в иерархию и позже переносить в сцену.
_ORPHAN_STORE = TransformStore(initial_capacity=64)


class GameObject:
    """Узел сцены: трансформ, компоненты и иерархия «родитель — дети».

    Объект можно создать и вне сцены (тогда он живёт во временном хранилище),
    а затем добавить в сцену методом :meth:`Scene.add_game_object` — поддерево
    переносится целиком с сохранением локальных значений.

    Args:
        name: Отображаемое имя (не обязано быть уникальным).
        scene: Сцена-владелец; ``None`` — объект создан «свободным».
        position: Начальная локальная позиция.
        rotation: Начальное локальное вращение.
        scale: Начальный локальный масштаб.
    """

    __slots__ = (
        "_active_self",
        "_by_type",
        "_children",
        "_components",
        "_id",
        "_is_destroyed",
        "_parent_ref",
        "_scene_ref",
        "_transform",
        "_uid",
        "is_static",
        "layer",
        "pooled",
        "name",
        "_tag",
        "__weakref__",
    )

    def __init__(
        self,
        name: str = "GameObject",
        scene: Scene | None = None,
        *,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: Vector3 | None = None,
    ) -> None:
        self.name = str(name)
        self._tag = "Untagged"
        self.layer = DEFAULT_LAYER
        self.is_static = False
        #: Экземпляр принадлежит пулу: поддерево не сохраняется в сцену.
        self.pooled = False

        self._id = 0
        self._uid = new_uid()
        self._scene_ref: weakref.ref[Scene] | None = None
        self._parent_ref: weakref.ref[GameObject] | None = None
        self._children: list[GameObject] = []
        self._components: list[Component] = []
        self._by_type: dict[type, list[Component]] = {}
        self._active_self = True
        self._is_destroyed = False

        store = _ORPHAN_STORE if scene is None else scene.transforms
        index = store.create(ROOT_INDEX, position, rotation, scale)
        self._transform = Transform(store, index, self)
        if scene is not None:
            self._scene_ref = weakref.ref(scene)
            scene._register_game_object(self)  # noqa: SLF001 - сцена владеет реестром

    # ------------------------------------------------------------------ #
    # Идентичность
    # ------------------------------------------------------------------ #
    @property
    def id(self) -> int:
        """Уникальный идентификатор в сцене (``0``, если объект вне сцены)."""
        return self._id

    @property
    def uid(self) -> str:
        """Стабильный идентификатор объекта (16 hex), сохраняется в сцене.

        В отличие от :attr:`id` (номер в рантайме, выдаётся сценой заново при
        каждой загрузке) ``uid`` переживает сохранение, загрузку и undo.
        """
        return self._uid

    @uid.setter
    def uid(self, value: str) -> None:
        """Задать ``uid`` (для загрузчика сцены и префабов).

        Raises:
            ValueError: Если значение не 16 hex-символов или ``uid`` уже занят
                другим объектом сцены.
        """
        if not is_valid_uid(value):
            raise ValueError(
                f"uid должен состоять из {UID_LENGTH} hex-символов "
                f"(0-9, a-f), получено {value!r}"
            )
        if value == self._uid:
            return
        scene = self.scene
        if scene is not None:
            scene._reassign_uid(self, value)  # noqa: SLF001
        self._uid = value

    @property
    def scene(self) -> Scene | None:
        """Сцена-владелец или ``None``."""
        if self._scene_ref is None:
            return None
        return self._scene_ref()

    @property
    def transform(self) -> Transform:
        """Трансформ объекта."""
        return self._transform

    @property
    def is_destroyed(self) -> bool:
        """``True``, если объект помечен на уничтожение или уже уничтожен."""
        return self._is_destroyed

    @property
    def is_valid(self) -> bool:
        """``True``, пока объект жив и его трансформ существует."""
        return not self._is_destroyed and self._transform.is_valid

    # ------------------------------------------------------------------ #
    # Активность
    # ------------------------------------------------------------------ #
    @property
    def tag(self) -> str:
        """Тег объекта (индексируется сценой для ``find_all_with_tag``)."""
        return self._tag

    @tag.setter
    def tag(self, value: str) -> None:
        value = str(value)
        old = self._tag
        if value == old:
            return
        self._tag = value
        scene = self.scene
        if scene is not None:
            scene._retag(self, old)  # noqa: SLF001 - сцена ведёт индекс тегов

    @property
    def active_self(self) -> bool:
        """Локальная активность объекта (без учёта родителей)."""
        return self._active_self

    @active_self.setter
    def active_self(self, value: bool) -> None:
        value = bool(value)
        if value == self._active_self:
            return
        self._active_self = value
        scene = self.scene
        if scene is not None:
            scene._on_active_changed(self)  # noqa: SLF001 - сцена владеет активностью

    @property
    def active_in_hierarchy(self) -> bool:
        """``True``, если активен сам объект и все его родители."""
        if not self._active_self:
            return False
        parent = self.parent
        return True if parent is None else parent.active_in_hierarchy

    def set_active(self, value: bool) -> None:
        """Включить или выключить объект вместе с поддеревом."""
        self.active_self = value

    # ------------------------------------------------------------------ #
    # Иерархия
    # ------------------------------------------------------------------ #
    @property
    def parent(self) -> GameObject | None:
        """Родительский объект или ``None``."""
        if self._parent_ref is None:
            return None
        return self._parent_ref()

    @property
    def children(self) -> tuple[GameObject, ...]:
        """Дочерние объекты в порядке добавления."""
        return tuple(self._children)

    @property
    def root(self) -> GameObject:
        """Корневой объект ветки, к которой принадлежит этот объект."""
        current = self
        while True:
            parent = current.parent
            if parent is None:
                return current
            current = parent

    def set_parent(
        self,
        parent: GameObject | None,
        *,
        keep_world: bool = True,
        index: int | None = None,
    ) -> None:
        """Перевесить объект под другого родителя.

        Args:
            parent: Новый родитель или ``None`` (сделать корневым объектом).
            keep_world: Сохранить мировое положение (по умолчанию) или
                оставить локальные значения как есть.
            index: Позиция среди новых братьев (после изъятия объекта из
                старого места); ``None`` - в конец. Значение за границами
                обрезается. При ``parent`` равном текущему и заданном
                ``index`` объект переставляется среди братьев.

        Raises:
            ValueError: Если родитель принадлежит другой сцене или создаётся цикл.
        """
        if parent is self:
            raise ValueError("Объект не может быть родителем сам себе")
        if parent is not None and parent.scene is not self.scene:
            raise ValueError("Родитель принадлежит другой сцене")
        if parent is not None and self._is_ancestor_of(parent):
            raise ValueError("Нельзя сделать объект потомком собственного потомка")

        current = self.parent
        if current is parent and index is None:
            return

        if current is not parent:
            self._transform.set_parent(
                None if parent is None else parent.transform, keep_world=keep_world
            )
        if current is not None and self in current._children:  # noqa: SLF001
            current._children.remove(self)  # noqa: SLF001
        if parent is not None:
            siblings = parent._children  # noqa: SLF001
            if index is None:
                siblings.append(self)
            else:
                siblings.insert(max(0, min(index, len(siblings))), self)
            self._parent_ref = weakref.ref(parent)
        else:
            self._parent_ref = None

        scene = self.scene
        if scene is not None:
            scene._on_parent_changed(self, current, parent)  # noqa: SLF001
            if parent is None and index is not None:
                scene._place_root(self, index)  # noqa: SLF001

    def _is_ancestor_of(self, other: GameObject) -> bool:
        """Лежит ли ``other`` в поддереве этого объекта (включая сам объект)."""
        node: GameObject | None = other
        while node is not None:
            if node is self:
                return True
            node = node.parent
        return False

    def sibling_index(self) -> int:
        """Позиция объекта среди братьев (для корней - среди корней сцены)."""
        parent = self.parent
        if parent is not None:
            return parent._children.index(self)  # noqa: SLF001
        scene = self.scene
        if scene is None:
            return 0
        return list(scene._roots).index(self.id)  # noqa: SLF001

    def find_child(self, name: str, *, recursive: bool = True) -> GameObject | None:
        """Найти первого потомка с указанным именем.

        Args:
            name: Имя искомого объекта.
            recursive: Искать на всю глубину поддерева.

        Returns:
            Найденный объект или ``None``.
        """
        for child in self._children:
            if child.name == name:
                return child
            if recursive:
                found = child.find_child(name, recursive=True)
                if found is not None:
                    return found
        return None

    # ------------------------------------------------------------------ #
    # Компоненты
    # ------------------------------------------------------------------ #
    def add_component(
        self, component: type[ComponentT] | ComponentT, **kwargs: object
    ) -> ComponentT:
        """Добавить компонент объекту.

        Args:
            component: Класс компонента (будет создан с ``kwargs``) или готовый
                экземпляр.
            **kwargs: Аргументы конструктора, если передан класс.

        Returns:
            Добавленный компонент.

        Raises:
            TypeError: Если передан не компонент и не его класс.
            ValueError: Если этот же экземпляр уже добавлен объекту.
        """
        instance: Component
        if isinstance(component, Component):
            if kwargs:
                raise ValueError("kwargs нельзя использовать с готовым экземпляром")
            instance = component
        elif isinstance(component, type) and issubclass(component, Component):
            instance = component(**kwargs)  # type: ignore[arg-type]
        else:
            raise TypeError(
                f"Ожидался класс Component или экземпляр, получено {component!r}"
            )

        if instance in self._components:
            raise ValueError(
                f"{type(instance).__name__} уже добавлен объекту '{self.name}'"
            )

        instance._attach(self)  # noqa: SLF001 - GameObject управляет жизненным циклом
        self._components.append(instance)
        self._by_type.setdefault(type(instance), []).append(instance)

        scene = self.scene
        if scene is not None:
            scene._register_component(instance)  # noqa: SLF001
        return instance  # type: ignore[return-value]

    def get_component(self, component_type: type[ComponentT]) -> ComponentT | None:
        """Вернуть первый компонент указанного типа или ``None``."""
        for component in self._by_type.get(component_type, ()):
            return component  # type: ignore[return-value]
        return None

    def get_components(self, component_type: type[ComponentT]) -> list[ComponentT]:
        """Вернуть все компоненты указанного типа в порядке добавления."""
        return list(self._by_type.get(component_type, ()))  # type: ignore[arg-type]

    def get_component_in_children(
        self, component_type: type[ComponentT], *, include_inactive: bool = False
    ) -> ComponentT | None:
        """Первый компонент типа у объекта или его потомков (DFS, сам объект первым).

        Args:
            component_type: Тип компонента.
            include_inactive: Заходить и в выключенные поддеревья.
        """
        for component in self.get_components_in_children(
            component_type, include_inactive=include_inactive
        ):
            return component
        return None

    def get_components_in_children(
        self, component_type: type[ComponentT], *, include_inactive: bool = False
    ) -> list[ComponentT]:
        """Все компоненты типа у объекта и потомков в порядке DFS.

        Выключенные (``active_self=False``) объекты вместе с поддеревом
        пропускаются, если не задано ``include_inactive``; сам объект
        проверяется так же.
        """
        result: list[ComponentT] = []
        stack: list[GameObject] = [self]
        while stack:
            current = stack.pop()
            if not include_inactive and not current._active_self:
                continue
            result.extend(current.get_components(component_type))
            stack.extend(reversed(current._children))
        return result

    def get_component_in_parent(
        self, component_type: type[ComponentT]
    ) -> ComponentT | None:
        """Первый компонент типа у объекта или ближайшего предка (вверх по цепочке)."""
        current: GameObject | None = self
        while current is not None:
            found = current.get_component(component_type)
            if found is not None:
                return found
            current = current.parent
        return None

    def has_component(self, component_type: type[Component]) -> bool:
        """``True``, если компонент такого типа есть у объекта."""
        return bool(self._by_type.get(component_type))

    def remove_component(self, component: Component | type[Component]) -> bool:
        """Удалить компонент (экземпляр или первый компонент указанного типа).

        Returns:
            ``True``, если компонент был найден и удалён.
        """
        target: Component | None
        if isinstance(component, type):
            target = self.get_component(component)
        else:
            target = component if component in self._components else None
        if target is None:
            return False

        scene = self.scene
        if scene is not None:
            scene._unregister_component(target)  # noqa: SLF001
        target._detach()  # noqa: SLF001
        self._components.remove(target)
        bucket = self._by_type.get(type(target))
        if bucket is not None:
            bucket.remove(target)
            if not bucket:
                self._by_type.pop(type(target), None)
        return True

    def reorder_component(self, component: Component, index: int) -> None:
        """Переставить компонент на позицию ``index`` (границы обрезаются).

        Нужна откату удаления компонента: порядок компонентов определяет порядок
        обновления и показа в инспекторе.
        """
        if component not in self._components:
            raise ValueError("Компонент не принадлежит объекту")
        self._components.remove(component)
        self._components.insert(max(0, min(index, len(self._components))), component)
        bucket = self._by_type.get(type(component))
        if bucket is not None:
            bucket[:] = [c for c in self._components if type(c) is type(component)]

    @property
    def components(self) -> tuple[Component, ...]:
        """Все компоненты объекта в порядке добавления."""
        return tuple(self._components)

    def num_components(self) -> int:
        """Число компонентов объекта."""
        return len(self._components)

    # ------------------------------------------------------------------ #
    # Уничтожение
    # ------------------------------------------------------------------ #
    def destroy(self, *, immediate: bool = False) -> None:
        """Уничтожить объект вместе с поддеревом.

        По умолчанию уничтожение откладывается до конца кадра — так безопасно
        удалять объекты прямо из обработчиков событий и логики компонентов.

        Args:
            immediate: Уничтожить немедленно.
        """
        if self._is_destroyed:
            return
        scene = self.scene
        if scene is not None:
            scene.destroy_game_object(self, immediate=immediate)
        else:
            self._destroy_orphan()

    def _destroy_orphan(self) -> None:
        """Уничтожить объект вне сцены (рекурсивно, немедленно)."""
        if self._is_destroyed:
            return
        self._is_destroyed = True
        for child in list(self._children):
            child._destroy_orphan()  # noqa: SLF001
        self._children.clear()
        for component in list(self._components):
            if component._is_started:  # noqa: SLF001
                component.on_destroy()
            component._detach()  # noqa: SLF001
        self._components.clear()
        self._by_type.clear()

        store = self._transform.store
        if store is _ORPHAN_STORE:
            store.discard(self._transform.index)
        self._parent_ref = None

    # ------------------------------------------------------------------ #
    # Служебный интерфейс для Scene
    # ------------------------------------------------------------------ #
    def _bind_to_scene(self, scene: Scene, parent_index: int) -> int:
        """Перенести трансформ в хранилище сцены, сохранив локальные значения.

        Реестр объектов ведёт сцена (см. :meth:`Scene.add_game_object`), поэтому
        метод не трогает ``_objects``.

        Args:
            scene: Сцена-получатель.
            parent_index: Индекс трансформа родителя в хранилище сцены.

        Returns:
            Индекс собственного трансформа в хранилище сцены.
        """
        old_transform = self._transform
        index = scene.transforms.create(
            parent_index,
            old_transform.position,
            old_transform.rotation,
            old_transform.scale,
        )
        self._transform = Transform(scene.transforms, index, self)
        self._scene_ref = weakref.ref(scene)

        old_store = old_transform.store
        if old_store is _ORPHAN_STORE:
            old_store.discard(old_transform.index)
        return index

    def _set_id(self, value: int) -> None:
        """Назначить идентификатор в сцене."""
        self._id = int(value)

    def _detach_from_parent(self) -> None:
        """Убрать объект из списка детей родителя."""
        parent = self.parent
        if parent is not None and self in parent._children:  # noqa: SLF001
            parent._children.remove(self)  # noqa: SLF001
        self._parent_ref = None

    def _mark_destroyed(self) -> None:
        """Пометить объект уничтоженным (вызывает сцена)."""
        self._is_destroyed = True

    # ------------------------------------------------------------------ #
    # Контейнерные протоколы
    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[Component]:
        return iter(self._components)

    def __len__(self) -> int:
        return len(self._components)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, type):
            return self.has_component(item)
        return item in self._components

    def __repr__(self) -> str:
        scene = self.scene
        where = scene.name if scene is not None else "<вне сцены>"
        return (
            f"<GameObject '{self.name}' id={self._id} scene='{where}' "
            f"components={len(self._components)} children={len(self._children)}>"
        )
