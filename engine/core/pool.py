"""Пул объектов: переиспользование экземпляров префаба без пересборки.

``ObjectPool.get`` берёт готовый экземпляр из очереди возврата (LIFO), включает
его и вызывает ``on_spawn`` у компонентов; ``release`` вызывает ``on_despawn``,
выключает объект и возвращает его в очередь. Создание экземпляра
(``Scene.instantiate``) происходит только когда свободных нет.

Экземпляры пула помечены ``GameObject.pooled`` и в файл сцены не попадают:
состояние пула восстанавливает владелец (скрипт) через свои поля.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from engine.core.game_object import GameObject
from engine.core.math3d import Quaternion, Vector3
from engine.core.physics_components import RigidBodyComponent

if TYPE_CHECKING:
    from engine.core.scene import Scene

_log = logging.getLogger("engine.core.pool")

#: Предупреждение при прогреве больше этого числа экземпляров.
PREWARM_WARN_LIMIT = 50

ExhaustedMode = Literal["grow", "reuse_oldest", "return_none"]
_MODES = ("grow", "reuse_oldest", "return_none")

_Pose = tuple[GameObject, Vector3, Quaternion, Vector3]


class ObjectPool:
    """Пул экземпляров одного префаба в сцене.

    Args:
        scene: Сцена, в которой живут экземпляры.
        prefab: ``engine.prefab.Prefab`` или GUID / путь (через загрузчик сцены).
        prewarm: Сколько выключенных экземпляров создать сразу (синхронно).
        max_size: Предел общего числа экземпляров (занятых + свободных);
            ``None`` - без предела.
        on_exhausted: Поведение при достижении ``max_size``: ``"grow"`` -
            ``get`` возвращает ``None`` с предупреждением; ``"reuse_oldest"`` -
            возвращается самый давний занятый объект; ``"return_none"`` -
            ``None`` без предупреждения.

    Raises:
        ValueError: Неизвестный режим или ``max_size`` < 1.
    """

    def __init__(
        self,
        scene: Scene,
        prefab: Any,
        *,
        prewarm: int = 0,
        max_size: int | None = None,
        on_exhausted: ExhaustedMode = "grow",
    ) -> None:
        if on_exhausted not in _MODES:
            raise ValueError(
                f"on_exhausted должен быть одним из {_MODES}, получено {on_exhausted!r}"
            )
        if max_size is not None and max_size < 1:
            raise ValueError("max_size должен быть не меньше 1")
        self._scene = scene
        self._prefab = prefab
        self._max_size = max_size
        self._mode = on_exhausted
        self._free: list[GameObject] = []
        self._active: list[GameObject] = []
        self._poses: dict[str, list[_Pose]] = {}
        self._parents: dict[str, GameObject | None] = {}
        if prewarm > PREWARM_WARN_LIMIT:
            _log.warning(
                "Прогрев пула на %d экземпляров выполняется синхронно "
                "и может занять заметное время",
                prewarm,
            )
        for _ in range(max(0, prewarm)):
            if max_size is not None and self._total() >= max_size:
                break
            obj = self._create()
            obj.active_self = False
            self._free.append(obj)

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        """Сколько объектов выдано сейчас (занято)."""
        return len(self._active)

    @property
    def free(self) -> int:
        """Сколько объектов лежит в очереди возврата."""
        return len(self._free)

    def _total(self) -> int:
        return len(self._active) + len(self._free)

    # ------------------------------------------------------------------ #
    # Выдача и возврат
    # ------------------------------------------------------------------ #
    def get(
        self,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
    ) -> GameObject | None:
        """Выдать объект из пула.

        Args:
            position: Локальная позиция корня; ``None`` - как в префабе.
            rotation: Локальное вращение корня; ``None`` - как в префабе.

        Returns:
            Включённый корень экземпляра или ``None`` при исчерпании пула.
        """
        self._prune_destroyed()
        obj = self._pop_free()
        if obj is None:
            obj = self._acquire_when_empty()
            if obj is None:
                return None
        if position is not None:
            obj.transform.position = position
        if rotation is not None:
            obj.transform.rotation = rotation
        obj.active_self = True
        self._active.append(obj)
        self._stop_bodies(obj)
        self._notify(obj, "on_spawn")
        return obj

    def release(self, obj: GameObject) -> bool:
        """Вернуть объект в пул.

        Args:
            obj: Корень экземпляра, выданный этим пулом.

        Returns:
            ``True``, если объект возвращён; ``False`` - объект не занят
            этим пулом (в лог пишется предупреждение).
        """
        if obj not in self._active:
            _log.warning("release: объект %r не выдан этим пулом", obj.name)
            return False
        self._active.remove(obj)
        self._return(obj)
        return True

    def _return(self, obj: GameObject) -> None:
        self._notify(obj, "on_despawn")
        obj.active_self = False
        self._reset(obj)
        self._free.append(obj)

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #
    def _acquire_when_empty(self) -> GameObject | None:
        if self._max_size is not None and self._total() >= self._max_size:
            if self._mode == "reuse_oldest" and self._active:
                oldest = self._active.pop(0)
                self._return(oldest)
                return self._pop_free()
            if self._mode == "grow":
                _log.warning(
                    "Пул исчерпан: достигнут max_size=%d, get вернул None",
                    self._max_size,
                )
            return None
        obj = self._create()
        obj.active_self = False
        return obj

    def _pop_free(self) -> GameObject | None:
        while self._free:
            obj = self._free.pop()
            if self._alive(obj):
                return obj
            _log.error(
                "Объект пула %r уничтожен вне пула; создаётся новый", obj.name
            )
            self._forget(obj)
        return None

    def _prune_destroyed(self) -> None:
        for obj in [o for o in self._active if not self._alive(o)]:
            _log.error("Объект пула %r уничтожен вне пула", obj.name)
            self._active.remove(obj)
            self._forget(obj)

    def _alive(self, obj: GameObject) -> bool:
        return not obj.is_destroyed and obj.scene is self._scene

    def _forget(self, obj: GameObject) -> None:
        self._poses.pop(obj.uid, None)
        self._parents.pop(obj.uid, None)

    def _create(self) -> GameObject:
        root = self._scene.instantiate(self._prefab)
        root.pooled = True
        self._parents[root.uid] = root.parent
        self._poses[root.uid] = [
            (o, o.transform.position, o.transform.rotation, o.transform.scale)
            for o in _walk(root)
        ]
        return root

    def _reset(self, root: GameObject) -> None:
        base_parent = self._parents.get(root.uid)
        if root.parent is not base_parent:
            root.set_parent(base_parent, keep_world=False)
        for obj, position, rotation, scale in self._poses.get(root.uid, ()):
            if obj.is_destroyed:
                continue
            transform = obj.transform
            transform.position = position
            transform.rotation = rotation
            transform.scale = scale

    def _stop_bodies(self, root: GameObject) -> None:
        """Обнулить скорости тел: двойник мог пережить короткое выключение.

        Если объект выключили и включили между двумя синхронизациями физики,
        тело в мире не пересоздаётся и хранит скорость прошлой «жизни».
        Синхронизацию структуры не запускаем: это дорого на каждый ``get``.
        """
        physics = self._scene._physics  # noqa: SLF001 - адаптер создаётся лениво
        if physics is None:
            return
        for obj in _walk(root):
            if obj.has_component(RigidBodyComponent):
                physics.stop_body(obj)

    def _notify(self, root: GameObject, hook: str) -> None:
        """Вызвать ``on_spawn`` / ``on_despawn`` у компонентов поддерева (DFS)."""
        for obj in _walk(root):
            for component in list(obj.components):
                method = getattr(component, hook, None)
                if method is None:
                    continue
                try:
                    method()
                except Exception:  # noqa: BLE001 - хук пользователя не ломает пул
                    _log.exception(
                        "Исключение в %s компонента %s объекта %r",
                        hook,
                        type(component).__name__,
                        obj.name,
                    )


def _walk(root: GameObject) -> list[GameObject]:
    """Объекты поддерева в порядке DFS (корень первым)."""
    result: list[GameObject] = []
    stack = [root]
    while stack:
        current = stack.pop()
        result.append(current)
        stack.extend(reversed(current.children))
    return result
