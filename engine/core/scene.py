"""Сцена: контейнер игрового мира (объекты, компоненты, системы, события).

Сцена владеет тремя реестрами и синхронизирует их:

* :class:`~engine.core.transform.TransformStore` — плоское numpy-хранилище
  трансформов с пакетным пересчётом мировых матриц;
* реестр игровых объектов (:class:`~engine.core.game_object.GameObject`);
* список **активных** компонентов — один плоский список, по которому кадр
  обновляет поведение без обхода дерева объектов.

Уничтожение по умолчанию откладывается до конца кадра, поэтому объекты можно
безопасно удалять прямо из логики компонентов и обработчиков событий.

Example:
    >>> scene = Scene("Level1")
    >>> player = scene.create_game_object("Player")
    >>> scene.add_system(MoveSystem())
    >>> scene.update(1 / 60)
"""

from __future__ import annotations

import bisect
import logging
import random
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from engine.core.component import Component
from engine.core.event_bus import EventBus
from engine.core.game_object import GameObject, new_uid
from engine.core.math3d import Quaternion, Vector3
from engine.core.scheduling.scheduler import Scheduler
from engine.core.system import System
from engine.core.transform import TransformStore

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.physics_adapter import ScenePhysicsAdapter
    from engine.core.save.context import GameState
    from engine.core.save.manager import SaveManager
    from engine.core.scene_manager import SceneManager
    from engine.core.screen_fx import ScreenFx

__all__ = ["Scene", "SceneStats"]

_log = logging.getLogger("engine.core.scene")

SystemT = TypeVar("SystemT", bound=System)
ComponentT = TypeVar("ComponentT", bound=Component)


@dataclass(slots=True)
class SceneStats:
    """Снимок состояния сцены (для отладки, профайлера и тестов)."""

    game_objects: int
    root_objects: int
    components: int
    active_components: int
    systems: int
    transforms_alive: int
    transforms_capacity: int
    pending_destroy: int
    structure_version: int
    total_created: int
    total_destroyed: int


class Scene:
    """Игровой мир: объекты, компоненты, системы и шина событий.

    Args:
        name: Имя сцены (для отладки и сериализации).
        event_bus: Готовая шина событий; по умолчанию создаётся своя.
        transform_capacity: Начальное число слотов трансформов.
    """

    __slots__ = (
        "_active_components",
        "_by_tag",
        "_by_uid",
        "_fixed_batches",
        "_is_alive",
        "_next_object_id",
        "_objects",
        "_pending_destroy",
        "_physics",
        "_query_cache",
        "_random",
        "_screen",
        "_roots",
        "_spawn_serial",
        "_state",
        "save_manager",
        "scene_manager",
        "_structure_version",
        "activity_version",
        "_systems",
        "_total_created",
        "_total_destroyed",
        "__weakref__",
        "events",
        "name",
        "prefab_loader",
        "asset_source",
        "scheduler",
        "transforms",
    )

    def __init__(
        self,
        name: str = "Scene",
        *,
        event_bus: EventBus | None = None,
        transform_capacity: int = 64,
    ) -> None:
        self.name = str(name)
        self.events = event_bus if event_bus is not None else EventBus()
        self.transforms = TransformStore(initial_capacity=transform_capacity)
        #: Таймеры, корутины и твины сцены (docs/CORE.md, «Планировщик»).
        self.scheduler = Scheduler()

        self._objects: dict[int, GameObject] = {}
        #: Индекс ``uid`` -> объект. Производный: не сериализуется, строится
        #: заново при загрузке и ведётся при создании/удалении объектов.
        self._by_uid: dict[str, GameObject] = {}
        #: Индекс тег -> {id: объект}; производный, порядок - порядок добавления.
        self._by_tag: dict[str, dict[int, GameObject]] = {}
        # Корни и активные компоненты хранятся в словарях: порядок вставки
        # сохраняется (dict), а удаление стоит O(1) — иначе уничтожение
        # объектов на большой сцене вырождается в O(n²).
        self._roots: dict[int, GameObject] = {}
        self._systems: list[System] = []
        self._active_components: dict[Component, None] = {}
        self._pending_destroy: dict[int, GameObject] = {}
        self._query_cache: dict[
            tuple[type, ...], tuple[int, tuple[GameObject, ...]]
        ] = {}
        self._next_object_id = 1
        self._structure_version = 0
        #: Растёт при смене активности/родителя объектов (для кэшей рендера).
        self.activity_version = 0
        self._total_created = 0
        self._total_destroyed = 0
        self._is_alive = True
        self._physics: ScenePhysicsAdapter | None = None
        self._state: GameState | None = None
        self._random: random.Random | None = None
        self._screen: ScreenFx | None = None
        #: Менеджер слотов сохранения; ставится снаружи (плеер, редактор, тесты).
        self.save_manager: SaveManager | None = None
        #: Менеджер сцен, в который сцена загружена (ставит ``SceneManager``);
        #: через него слот сохранения охватывает все загруженные сцены.
        self.scene_manager: SceneManager | None = None
        #: Счётчик экземпляров префабов (для детерминированных uid); в файл
        #: не пишется: после загрузки коллизии uid обходятся перебором.
        self._spawn_serial = 0
        #: ``(prefab_guid, prefab_path) -> Prefab`` для вложенных и строковых
        #: ссылок в :meth:`instantiate` (``None`` - загрузчик по умолчанию).
        self.prefab_loader: Callable[[str, str], Any] | None = None
        #: База ассетов/VFS, из которой сцена загружена (ставит ``scene_io``);
        #: физика берёт из неё геометрию mesh-коллайдеров.
        self.asset_source: Any | None = None
        self._fixed_batches: dict[object, tuple[Callable[..., Any], list]] = {}

    # ------------------------------------------------------------------ #
    # Физика
    # ------------------------------------------------------------------ #
    @property
    def physics(self) -> ScenePhysicsAdapter:
        """Физика сцены; создаётся при первом обращении (сцена без неё не платит)."""
        if self._physics is None:
            from engine.core.physics_adapter import ScenePhysicsAdapter

            self._physics = ScenePhysicsAdapter(self)
        return self._physics

    @property
    def screen(self) -> ScreenFx:
        """Fade/flash экрана; создаётся лениво, хост может подставить общий.

        Плеер кладёт во все сцены один ``ScreenFx``, поэтому затемнение
        переживает смену сцены (``fade_to`` → загрузка → ``fade_to`` обратно).
        """
        if self._screen is None:
            from engine.core.screen_fx import ScreenFx

            self._screen = ScreenFx()
        return self._screen

    @screen.setter
    def screen(self, value: ScreenFx) -> None:
        self._screen = value

    @property
    def random(self) -> random.Random:
        """Детерминированный генератор сцены (seed ``0``; см. :meth:`seed_random`).

        Игровая случайность (варианты звука, разброс) берётся отсюда, чтобы
        прогон воспроизводился при одинаковом вводе.
        """
        if self._random is None:
            self._random = random.Random(0)
        return self._random

    def seed_random(self, seed: int) -> None:
        """Пересоздать :attr:`random` с заданным зерном."""
        self._random = random.Random(seed)

    @property
    def state(self) -> GameState:
        """Глобальное состояние игры (``GameState``); создаётся при первом обращении."""
        if self._state is None:
            from engine.core.save.context import GameState

            self._state = GameState()
        return self._state

    def save(self, slot_name: str, **kwargs: Any) -> Any:
        """Записать слот через :attr:`save_manager` (``GameState`` этой сцены).

        Args:
            slot_name: Имя слота.
            **kwargs: ``playtime_sec``, ``screenshot`` для ``SaveManager.save``.

        Returns:
            ``SaveSlot``.

        Raises:
            RuntimeError: ``save_manager`` не задан.
        """
        manager = self._require_save_manager()
        if self.scene_manager is not None and self.scene_manager.active_scene:
            return self.scene_manager.save(manager, slot_name, **kwargs)
        return manager.save(slot_name, self, self.state, **kwargs)

    def load(self, slot_name: str) -> Any:
        """Применить слот к этой сцене (она уже построена из шаблона).

        Returns:
            ``SaveSlot``.

        Raises:
            RuntimeError: ``save_manager`` не задан.
            SaveError: Файл повреждён или версия не поддерживается.
        """
        manager = self._require_save_manager()
        if self.scene_manager is not None and self.scene_manager.active_scene:
            return self.scene_manager.apply_slot(manager, slot_name)
        return manager.load(slot_name, self)

    def quick_save(self) -> Any:
        """Быстрое сохранение в слот ``_quick`` (F5); ``None`` при ошибке."""
        from engine.core.save.service import quick_save

        return quick_save(self)

    def quick_load(self) -> Any:
        """Быстрая загрузка ``_quick`` (F9), иначе ``_autosave``; ``None`` - нет."""
        from engine.core.save.service import quick_load

        return quick_load(self)

    def _require_save_manager(self) -> SaveManager:
        if self.save_manager is None:
            raise RuntimeError(f"У сцены '{self.name}' не задан save_manager")
        return self.save_manager

    @property
    def physics_if_exists(self) -> ScenePhysicsAdapter | None:
        """Физика сцены, если она уже создана (без ленивого создания)."""
        return self._physics

    # ------------------------------------------------------------------ #
    # Объекты
    # ------------------------------------------------------------------ #
    def create_game_object(
        self,
        name: str = "GameObject",
        parent: GameObject | None = None,
        *,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: Vector3 | None = None,
        tag: str | None = None,
    ) -> GameObject:
        """Создать объект в сцене.

        Args:
            name: Имя объекта.
            parent: Родительский объект (``None`` — корневой).
            position: Локальная позиция.
            rotation: Локальное вращение.
            scale: Локальный масштаб.
            tag: Тег объекта; по умолчанию не задан.

        Returns:
            Созданный объект.

        Raises:
            RuntimeError: Если сцена уже уничтожена.
            ValueError: Если родитель принадлежит другой сцене.
        """
        self._ensure_alive()
        game_object = GameObject(
            name, self, position=position, rotation=rotation, scale=scale
        )
        game_object._set_id(self._allocate_object_id())  # noqa: SLF001
        self._objects[game_object.id] = game_object
        self._index_uid(game_object, log_conflict=False)
        self._by_tag.setdefault(game_object.tag, {})[game_object.id] = game_object
        self._total_created += 1

        if tag is not None:
            game_object.tag = str(tag)
        if parent is None:
            self._roots[game_object.id] = game_object
        else:
            game_object.set_parent(parent, keep_world=False)
        self.invalidate_structure()
        return game_object

    def add_game_object(
        self, game_object: GameObject, parent: GameObject | None = None
    ) -> GameObject:
        """Добавить готовый объект (вместе с поддеревом) в эту сцену.

        Локальные позиции, вращения и масштабы сохраняются.

        Raises:
            ValueError: Если объект уже принадлежит сцене.
            RuntimeError: Если сцена уничтожена.
        """
        self._ensure_alive()
        if game_object.scene is not None:
            raise ValueError(
                f"Объект '{game_object.name}' уже принадлежит сцене "
                f"'{game_object.scene.name}'"
            )
        if parent is not None and parent.scene is not self:
            raise ValueError("Родитель принадлежит другой сцене")

        parent_index = -1 if parent is None else parent.transform.index
        self._adopt_subtree(game_object, parent_index)
        if parent is None:
            self._roots[game_object.id] = game_object

        self._on_active_changed(game_object)
        self.invalidate_structure()
        return game_object

    def _adopt_subtree(self, game_object: GameObject, parent_index: int) -> int:
        """Принять объект и всё его поддерево в реестры этой сцены.

        Args:
            game_object: Корневой объект переносимого поддерева.
            parent_index: Индекс трансформа родителя в хранилище сцены.

        Returns:
            Индекс трансформа принятого объекта.
        """
        index = game_object._bind_to_scene(self, parent_index)  # noqa: SLF001
        if self._objects.get(game_object.id) is not game_object:
            game_object._set_id(self._allocate_object_id())  # noqa: SLF001
            self._objects[game_object.id] = game_object
            self._index_uid(game_object, log_conflict=True)
            self._by_tag.setdefault(game_object.tag, {})[game_object.id] = game_object
            self._total_created += 1
        for component in game_object.components:
            self._register_component(component)
        for child in game_object.children:
            self._adopt_subtree(child, index)
        return index

    def instantiate(
        self,
        prefab: Any,
        parent: GameObject | None = None,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
    ) -> GameObject:
        """Создать в сцене экземпляр префаба (с вложенными).

        Args:
            prefab: ``engine.prefab.Prefab`` или строка - GUID / путь файла
                (разрешается через :attr:`prefab_loader`).
            parent: Родитель корня (``None`` - корневой объект сцены).
            position: Локальная позиция корня вместо записанной в префабе.
            rotation: Локальное вращение корня вместо записанного.

        Returns:
            Корень созданного экземпляра (на нём компонент ``PrefabInstance``).

        Raises:
            PrefabCycleError: Цикл вложенных префабов или глубина больше 8.
            PrefabError: Префаб не найден или некорректен.
        """
        self._ensure_alive()
        # Сборка объектов живёт в engine.prefab (там же формат сцены);
        # импорт отложен, чтобы core не зависел от него на этапе загрузки.
        from engine.prefab import instantiate_prefab

        return instantiate_prefab(self, prefab, parent, position, rotation)

    def destroy_game_object(
        self, game_object: GameObject, *, immediate: bool = False
    ) -> None:
        """Уничтожить объект вместе с поддеревом и компонентами.

        Args:
            game_object: Объект этой сцены.
            immediate: Удалить сразу; иначе — в конце текущего кадра.

        Raises:
            ValueError: Если объект принадлежит другой сцене.
        """
        if game_object.scene is not self:
            raise ValueError("Объект принадлежит другой сцене")
        if game_object.is_destroyed:
            return

        game_object._mark_destroyed()  # noqa: SLF001
        self._deactivate_subtree(game_object)

        if immediate:
            self._finalize_destroy(game_object)
        elif game_object.id not in self._pending_destroy:
            self._pending_destroy[game_object.id] = game_object

    def process_pending(self) -> int:
        """Применить отложенные уничтожения.

        Returns:
            Число фактически удалённых корневых объектов.
        """
        if not self._pending_destroy:
            return 0
        pending, self._pending_destroy = self._pending_destroy, {}
        removed = 0
        for game_object in pending.values():
            if game_object.scene is self and game_object.is_destroyed:
                self._finalize_destroy(game_object)
                removed += 1
        return removed

    def _finalize_destroy(self, game_object: GameObject) -> None:
        """Физически удалить объект и его поддерево."""
        if game_object.id not in self._objects:
            return

        for child in list(game_object.children):
            self._finalize_destroy(child)

        for component in list(game_object._components):  # noqa: SLF001
            if component._is_started:  # noqa: SLF001
                component.on_destroy()
            self._deactivate_component(component)
            component._detach()  # noqa: SLF001
        game_object._components.clear()  # noqa: SLF001
        game_object._by_type.clear()  # noqa: SLF001

        self._pending_destroy.pop(game_object.id, None)
        game_object._detach_from_parent()  # noqa: SLF001
        self.transforms.discard(game_object.transform.index)
        self._objects.pop(game_object.id, None)
        self._untag(game_object, game_object.tag)
        if self._by_uid.get(game_object.uid) is game_object:
            del self._by_uid[game_object.uid]
        self._roots.pop(game_object.id, None)
        self._total_destroyed += 1
        self.invalidate_structure()

    def get_game_object(self, object_id: int) -> GameObject | None:
        """Найти объект по идентификатору."""
        return self._objects.get(int(object_id))

    def find_by_uid(self, uid: str) -> GameObject | None:
        """Найти объект по ``uid`` за O(1).

        Args:
            uid: Стабильный идентификатор объекта (16 hex).

        Returns:
            Объект или ``None``, если такого нет в сцене.
        """
        return self._by_uid.get(uid)

    @property
    def objects(self) -> tuple[GameObject, ...]:
        """Все объекты сцены (порядок создания сохраняется)."""
        return tuple(self._objects.values())

    @property
    def root_objects(self) -> tuple[GameObject, ...]:
        """Корневые объекты в порядке создания."""
        return tuple(self._roots.values())

    def find(self, name: str) -> GameObject | None:
        """Найти первый объект с указанным именем (обход в глубину от корней)."""
        for root in self._roots.values():
            if root.name == name:
                return root
            found = root.find_child(name)
            if found is not None:
                return found
        return None

    def find_all(self, name: str) -> list[GameObject]:
        """Найти все объекты с указанным именем."""
        return [
            game_object
            for game_object in self._objects.values()
            if game_object.name == name
        ]

    def find_with_tag(self, tag: str) -> GameObject | None:
        """Найти первый объект с указанным тегом."""
        for game_object in self._objects.values():
            if game_object.tag == tag:
                return game_object
        return None

    def find_all_with_tag(self, tag: str) -> list[GameObject]:
        """Найти все активные объекты с указанным тегом.

        Объекты под выключенным родителем (``active_in_hierarchy`` ложно)
        не попадают в результат. Порядок - порядок добавления в сцену.
        """
        bucket = self._by_tag.get(tag)
        if not bucket:
            return []
        return [o for o in bucket.values() if o.active_in_hierarchy]

    def _untag(self, game_object: GameObject, tag: str) -> None:
        bucket = self._by_tag.get(tag)
        if bucket is not None:
            bucket.pop(game_object.id, None)
            if not bucket:
                del self._by_tag[tag]

    def _retag(self, game_object: GameObject, old_tag: str) -> None:
        """Перенести объект в индексе при смене ``GameObject.tag``."""
        if self._objects.get(game_object.id) is not game_object:
            return
        self._untag(game_object, old_tag)
        self._by_tag.setdefault(game_object.tag, {})[game_object.id] = game_object

    # ------------------------------------------------------------------ #
    # Запросы (ECS-стиль)
    # ------------------------------------------------------------------ #
    def query(self, *component_types: type[Component]) -> tuple[GameObject, ...]:
        """Найти объекты, у которых есть **все** указанные компоненты.

        Результат кэшируется до следующего структурного изменения сцены
        (добавление/удаление объектов и компонентов) и возвращается **тем же
        кортежем** — без копирования, поэтому вызов в цикле кадра бесплатен.

        Args:
            *component_types: Типы компонентов; пустой вызов вернёт все объекты.

        Returns:
            Неизменяемый кортеж объектов. Хранить его дольше одного кадра не
            следует: после изменения состава сцены кэш пересобирается.
        """
        key = tuple(component_types)
        cached = self._query_cache.get(key)
        if cached is not None and cached[0] == self._structure_version:
            return cached[1]

        if not key:
            result = tuple(self._objects.values())
        else:
            result = tuple(
                game_object
                for game_object in self._objects.values()
                if all(
                    game_object.has_component(component_type)
                    for component_type in key
                )
            )
        self._query_cache[key] = (self._structure_version, result)
        return result

    @property
    def structure_version(self) -> int:
        """Счётчик изменений состава сцены (объекты и компоненты)."""
        return self._structure_version

    def invalidate_structure(self) -> None:
        """Сбросить кэш запросов (вызывается при изменении состава сцены)."""
        self._structure_version += 1
        self._query_cache.clear()

    # ------------------------------------------------------------------ #
    # Системы
    # ------------------------------------------------------------------ #
    def add_system(self, system: SystemT) -> SystemT:
        """Добавить систему; порядок определяется её ``priority``.

        Returns:
            Добавленную систему.

        Raises:
            ValueError: Если система уже добавлена в сцену.
        """
        self._ensure_alive()
        if system in self._systems:
            raise ValueError(f"{type(system).__name__} уже добавлена в сцену")
        bisect.insort(self._systems, system, key=lambda item: -item.priority)
        system._scene_ref = weakref.ref(self)  # noqa: SLF001
        system.on_added(self)
        return system

    def remove_system(self, system: System) -> bool:
        """Удалить систему из сцены.

        Returns:
            ``True``, если система была найдена и удалена.
        """
        if system not in self._systems:
            return False
        self._systems.remove(system)
        system.on_removed(self)
        system._scene_ref = None  # noqa: SLF001
        return True

    def get_system(self, system_type: type[SystemT]) -> SystemT | None:
        """Вернуть первую систему указанного типа (с учётом наследования)."""
        for system in self._systems:
            if isinstance(system, system_type):
                return system  # type: ignore[return-value]
        return None

    @property
    def systems(self) -> tuple[System, ...]:
        """Системы в порядке обновления."""
        return tuple(self._systems)

    # ------------------------------------------------------------------ #
    # Кадр
    # ------------------------------------------------------------------ #
    def fixed_update(self, dt: float) -> None:
        """Фаза фиксированного шага: физика и зависящая от неё логика."""
        self.process_pending()
        self.scheduler.tick_fixed()
        for system in list(self._systems):
            if system.enabled:
                system.on_fixed_update(self, dt)
        for component in list(self._active_components):
            component.on_fixed_update(dt)
        if self._fixed_batches:
            batches, self._fixed_batches = self._fixed_batches, {}
            for func, items in batches.values():
                func(items, dt)
        # Физика — после пользовательской логики фиксированного шага.
        if self._physics is not None:
            self._physics.step(dt)
        self.transforms.update_world_matrices()

    def add_fixed_batch(self, key: object, func: Callable[..., Any], item: Any) -> None:
        """Отложить ``item`` в пакет фиксированного шага.

        Во время ``fixed_update`` компоненты с одинаковым ``key`` складываются в
        один список; после обхода компонентов ``func(items, dt)`` вызывается
        один раз на пакет (раньше физики). Так однотипная логика сотен объектов
        идёт одним вызовом, а не сотней.

        Args:
            key: Идентификатор пакета (например, путь скрипта).
            func: Функция ``func(items, dt)``; берётся у первого добавленного.
            item: Элемент пакета.
        """
        entry = self._fixed_batches.get(key)
        if entry is None:
            self._fixed_batches[key] = (func, [item])
        else:
            entry[1].append(item)

    def tick_scheduler_ui(self, dt: float) -> int:
        """Продвинуть UI-часы и ``GAME_RAW`` планировщика вне ``GameLoop``.

        Игровые часы двигает :meth:`update`, поэтому метод их не трогает
        (двойной вызов удвоил бы игровое время). Нужен хосту без ``GameLoop``
        (тесты, утилиты); внутри ``GameLoop`` вызывать не нужно.

        Args:
            dt: Реальное время кадра в секундах.

        Returns:
            Число сработавших задач.
        """
        return self.scheduler.update_ui(dt)

    def update(self, dt: float) -> None:
        """Основная фаза кадра: системы, затем компоненты, затем иерархия."""
        self.process_pending()
        self.events.flush()
        # Планировщик - до хуков on_update: сработавшая корутина/таймер меняет
        # состояние раньше логики этого же кадра. UI-часы ведёт GameLoop.
        self.scheduler.update_game(dt)
        for system in list(self._systems):
            if system.enabled:
                system.on_update(self, dt)
        for component in list(self._active_components):
            component.on_update(dt)
        self.transforms.update_world_matrices()
        self.process_pending()

    def late_update(self, dt: float) -> None:
        """Поздняя фаза кадра: камеры, IK и прочие зависимые вычисления."""
        for system in list(self._systems):
            if system.enabled:
                system.on_late_update(self, dt)
        for component in list(self._active_components):
            component.on_late_update(dt)
        self.transforms.update_world_matrices()

    # ------------------------------------------------------------------ #
    # Управление активностью
    # ------------------------------------------------------------------ #
    def _activate_subtree(self, root: GameObject) -> None:
        """Активировать компоненты объекта и его потомков."""
        stack = [root]
        while stack:
            current = stack.pop()
            if current.is_destroyed:
                continue
            for component in current._components:  # noqa: SLF001
                self._activate_component(component)
            stack.extend(current.children)

    def _deactivate_subtree(self, root: GameObject) -> None:
        """Деактивировать компоненты объекта и его потомков."""
        stack = [root]
        while stack:
            current = stack.pop()
            for component in current._components:  # noqa: SLF001
                self._deactivate_component(component)
            stack.extend(current.children)

    def _activate_component(self, component: Component) -> None:
        """Включить компонент: ``on_enable``, затем однократный ``on_start``."""
        if component in self._active_components:
            return
        if not component.is_active:
            return
        self._active_components[component] = None
        try:
            component.on_enable()
            if not component._is_started:  # noqa: SLF001
                component._mark_started()  # noqa: SLF001
                component.on_start()
        except Exception:  # noqa: BLE001 - ошибка компонента не должна ломать сцену
            _log.exception(
                "Ошибка активации компонента %r объекта '%s'",
                type(component).__name__,
                getattr(component.game_object, "name", "?"),
            )

    def _deactivate_component(self, component: Component) -> None:
        """Выключить компонент, если он был активен."""
        if component not in self._active_components:
            return
        del self._active_components[component]
        try:
            component.on_disable()
        except Exception:  # noqa: BLE001
            _log.exception(
                "Ошибка деактивации компонента %r",
                type(component).__name__,
            )

    def _on_active_changed(self, game_object: GameObject) -> None:
        """Пересчитать активность поддерева после смены флага или родителя."""
        self.activity_version += 1
        if game_object.is_destroyed:
            return
        if game_object.active_in_hierarchy:
            self._activate_subtree(game_object)
        else:
            self._deactivate_subtree(game_object)

    def _place_root(self, game_object: GameObject, index: int) -> None:
        """Поставить корневой объект на позицию ``index`` среди корней.

        Args:
            game_object: Корневой объект этой сцены.
            index: Позиция после изъятия объекта; обрезается до границ.
        """
        others = [r for r in self._roots.values() if r is not game_object]
        others.insert(max(0, min(index, len(others))), game_object)
        self._roots = {root.id: root for root in others}

    def _on_parent_changed(
        self,
        game_object: GameObject,
        old_parent: GameObject | None,
        new_parent: GameObject | None,
    ) -> None:
        """Обновить набор корней и активность после смены родителя."""
        if self._objects.get(game_object.id) is game_object:
            if new_parent is None:
                self._roots[game_object.id] = game_object
            else:
                self._roots.pop(game_object.id, None)
        self._on_active_changed(game_object)

    def _on_component_enabled_changed(self, component: Component) -> None:
        """Реакция на переключение ``Component.enabled``."""
        if component.is_active:
            self._activate_component(component)
        else:
            self._deactivate_component(component)

    def _register_component(self, component: Component) -> None:
        """Зарегистрировать добавленный компонент."""
        if component.is_active:
            self._activate_component(component)
        self.invalidate_structure()

    def _unregister_component(self, component: Component) -> None:
        """Снять компонент с учёта перед удалением."""
        self._deactivate_component(component)
        self.invalidate_structure()

    def _register_game_object(self, game_object: GameObject) -> None:
        """Зарегистрировать объект, созданный с этой сценой.

        Реестр ``_objects`` заполняет сама сцена (:meth:`create_game_object`,
        :meth:`add_game_object`); здесь достаточно сбросить кэш запросов.
        """
        self.invalidate_structure()

    # ------------------------------------------------------------------ #
    # Служебное
    # ------------------------------------------------------------------ #
    def _allocate_object_id(self) -> int:
        """Выдать следующий идентификатор объекта."""
        object_id = self._next_object_id
        self._next_object_id += 1
        return object_id

    def _index_uid(self, game_object: GameObject, *, log_conflict: bool) -> None:
        """Внести ``uid`` объекта в индекс; при занятом ``uid`` выдать новый.

        Args:
            game_object: Объект, только что принятый сценой.
            log_conflict: Писать ли ERROR (``uid`` пришёл из данных, а не из
                генератора, поэтому конфликт - ошибка данных).
        """
        uid = game_object.uid
        other = self._by_uid.get(uid)
        if other is not None and other is not game_object:
            fresh = new_uid()
            while fresh in self._by_uid:
                fresh = new_uid()
            if log_conflict:
                _log.error(
                    "uid %s объекта '%s' уже занят объектом '%s'; выдан новый uid %s",
                    uid,
                    game_object.name,
                    other.name,
                    fresh,
                )
            game_object._uid = fresh  # noqa: SLF001
            uid = fresh
        self._by_uid[uid] = game_object

    def _reassign_uid(self, game_object: GameObject, new: str) -> None:
        """Сменить ``uid`` объекта в индексе (вызывает сеттер ``GameObject.uid``).

        Raises:
            ValueError: Если ``uid`` занят другим объектом сцены.
        """
        other = self._by_uid.get(new)
        if other is not None and other is not game_object:
            raise ValueError(f"uid {new} уже занят объектом '{other.name}'")
        if self._by_uid.get(game_object.uid) is game_object:
            del self._by_uid[game_object.uid]
        self._by_uid[new] = game_object

    def _ensure_alive(self) -> None:
        """Проверить, что сцена ещё не уничтожена."""
        if not self._is_alive:
            raise RuntimeError(f"Сцена '{self.name}' уже уничтожена")

    @property
    def active_components(self) -> tuple[Component, ...]:
        """Компоненты, которые обновляются в текущем кадре."""
        return tuple(self._active_components)

    @property
    def stats(self) -> SceneStats:
        """Снимок статистики сцены."""
        return SceneStats(
            game_objects=len(self._objects),
            root_objects=len(self._roots),
            components=sum(len(go._components) for go in self._objects.values()),  # noqa: SLF001
            active_components=len(self._active_components),
            systems=len(self._systems),
            transforms_alive=self.transforms.alive_count,
            transforms_capacity=self.transforms.capacity,
            pending_destroy=len(self._pending_destroy),
            structure_version=self._structure_version,
            total_created=self._total_created,
            total_destroyed=self._total_destroyed,
        )

    def clear(self) -> None:
        """Уничтожить все объекты сцены, оставив системы и шину событий."""
        for game_object in list(self._roots.values()):
            if not game_object.is_destroyed:
                game_object._mark_destroyed()  # noqa: SLF001
                self._deactivate_subtree(game_object)
                self._finalize_destroy(game_object)
        self._pending_destroy.clear()
        self._roots.clear()

    def shutdown(self) -> None:
        """Полностью остановить сцену: объекты, системы и подписки на события."""
        if not self._is_alive:
            return
        self.clear()
        for system in list(self._systems):
            system.on_removed(self)
            system._scene_ref = None  # noqa: SLF001
        self._systems.clear()
        self.events.clear()
        self.scheduler.clear()
        self._physics = None
        self._is_alive = False

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self) -> Iterator[GameObject]:
        return iter(self._objects.values())

    def __contains__(self, item: object) -> bool:
        if isinstance(item, GameObject):
            return self._objects.get(item.id) is item
        return item in self._objects

    def __repr__(self) -> str:
        stats = self.stats
        return (
            f"<Scene '{self.name}' objects={stats.game_objects} "
            f"components={stats.components} active={stats.active_components} "
            f"systems={stats.systems}>"
        )
