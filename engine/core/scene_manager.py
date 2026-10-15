"""Менеджер сцен: загрузка, выгрузка и переключение активной сцены.

Ядро не знает формат файла сцены (``engine.scene_io`` зависит от рендера и
скриптов), поэтому загрузчик передаётся снаружи: ``SceneManager(loader)``.
Переход по запросу UI (``SceneLoadRequested``) приложение выполняет в конце кадра,
чтобы не уничтожать сцену посреди её же обновления.

Example:
    >>> manager = SceneManager(lambda path: load_scene(path))
    >>> manager.load(Path("main.scene.json"))
    >>> manager.active_scene
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.core.event_bus import EventBus
from engine.core.scene_load import SceneLoadOperation
from engine.core.ui_events import (
    SceneActivated,
    SceneLoaded,
    SceneLoadProgress,
    SceneUnloaded,
)

if TYPE_CHECKING:
    from engine.core.jobs import JobSystem
    from engine.core.save.slot import SaveScene
    from engine.core.scene import Scene

__all__ = ["SceneManager", "SceneTransition"]

_log = logging.getLogger("engine.core.scene_manager")


class SceneTransition:
    """Переход на другую сцену (создаётся :meth:`SceneManager.transition`).

    Attributes:
        operation: Операция загрузки новой сцены (``progress``, ``cancel()``).
        error: Исключение, если загрузка не удалась или отменена; старая сцена
            при этом остаётся активной.
    """

    def __init__(
        self,
        manager: SceneManager,
        target: Path,
        source: Path | None,
        unload_source: bool,
        loading: Path | None,
        operation: SceneLoadOperation,
    ) -> None:
        self._manager = manager
        self.target = target
        self.source = source
        self._unload_source = unload_source
        self._loading = loading
        self.operation = operation
        self.error: BaseException | None = None
        self._done = False

    @property
    def is_done(self) -> bool:
        """Переход закончен (успехом или ошибкой)."""
        return self._done

    @property
    def progress(self) -> float:
        """Прогресс загрузки новой сцены ``0..1``."""
        return self.operation.progress

    def cancel(self) -> None:
        """Отменить переход: старая сцена остаётся активной."""
        self.operation.cancel()

    def _poll(self) -> None:
        if self._done or not self.operation.is_done:
            return
        manager = self._manager
        self._done = True
        try:
            if self.operation.is_cancelled or self.operation.error is not None:
                self.error = self.operation.error or RuntimeError("переход отменён")
                return
            manager.set_active(self.target)
            if (
                self._unload_source
                and self.source is not None
                and self.source != self.target
            ):
                manager.unload(self.source)
        finally:
            if self._loading is not None:
                manager.unload(self._loading)


class SceneManager:
    """Набор загруженных сцен с одной активной.

    Сцены ``additive`` живут рядом с активной: каждая - отдельный :class:`Scene`
    со своим планировщиком, объекты разных сцен друг с другом не связаны.

    Args:
        loader: Функция ``путь -> Scene`` (синхронная загрузка целиком).
        parse: ``путь -> данные`` для порционной загрузки (рабочий поток).
        build: ``(данные, путь) -> генератор`` долей прогресса с готовой
            сценой в ``StopIteration.value``. Без ``parse``/``build``
            порционная загрузка делает всё одним шагом через ``loader``.
        jobs: Пул потоков для ``parse`` в :meth:`load_async`.
        root: Корень проекта: пути сцен в слотах сохранения пишутся
            относительно него (без корня - как есть, с ``/``).
        exists: Проверка наличия файла сцены (по умолчанию ``Path.is_file``;
            плеер подставляет проверку по паку).

    Attributes:
        events: Шина событий менеджера (``SceneLoaded/Unloaded/Activated/
            LoadProgress``).
    """

    def __init__(
        self,
        loader: Callable[[Path], Scene],
        *,
        parse: Callable[[Path], Any] | None = None,
        build: Callable[[Any, Path], Generator[float, None, Scene]] | None = None,
        jobs: JobSystem | None = None,
        root: Path | None = None,
        exists: Callable[[Path], bool] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self._exists = exists or Path.is_file
        self._loader = loader
        self._parse = parse
        self._build = build
        self._jobs = jobs
        self._scenes: dict[Path, Scene] = {}
        self._additive: set[Path] = set()
        self._active: Path | None = None
        self._loading: list[SceneLoadOperation] = []
        self._transitions: list[SceneTransition] = []
        self.events = EventBus()
        #: Хуки ``(прежняя активная | None, новая)`` перед сменой активной сцены.
        self.before_activate: list[Callable[[Scene | None, Scene], None]] = []

    @property
    def active_scene(self) -> Scene | None:
        """Активная сцена или ``None``."""
        return None if self._active is None else self._scenes[self._active]

    @property
    def active_path(self) -> Path | None:
        """Путь активной сцены."""
        return self._active

    @property
    def loaded(self) -> list[Path]:
        """Пути загруженных сцен."""
        return list(self._scenes)

    @property
    def additive_scenes(self) -> dict[Path, Scene]:
        """Загруженные дополнительные (не основные) сцены, в порядке загрузки."""
        return {p: s for p, s in self._scenes.items() if p in self._additive}

    @property
    def all_scenes(self) -> list[Scene]:
        """Все загруженные сцены: активная первой, затем остальные по порядку."""
        active = self.active_scene
        rest = [s for s in self._scenes.values() if s is not active]
        return ([active] if active is not None else []) + rest

    @property
    def loading(self) -> list[SceneLoadOperation]:
        """Операции загрузки, ещё не завершённые."""
        return list(self._loading)

    def get(self, path: Path) -> Scene | None:
        """Загруженная сцена по пути или ``None``."""
        return self._scenes.get(Path(path))

    def load(
        self, path: Path, *, activate: bool | None = None, additive: bool = False
    ) -> Scene:
        """Загрузить сцену целиком (повторная загрузка заменяет прежнюю).

        Реализовано через :meth:`load_async` с бесконечным бюджетом.

        Args:
            path: Файл сцены.
            activate: Сделать сцену активной; по умолчанию ``True``, а для
                ``additive`` - только если активной сцены ещё нет.
            additive: Загрузить как дополнительную сцену рядом с активной.

        Returns:
            Загруженная сцена.
        """
        operation = self.load_async(
            path, activate=activate, additive=additive, use_jobs=False
        )
        operation.step()
        return operation.result()

    def load_async(
        self,
        path: Path,
        *,
        activate: bool | None = None,
        additive: bool = False,
        use_jobs: bool = True,
    ) -> SceneLoadOperation:
        """Начать порционную загрузку; продвигать через ``op.step(budget_ms)``.

        Сцена регистрируется (события ``SceneLoaded``/``SceneActivated``) в
        последнем шаге операции. Прогресс - события ``SceneLoadProgress``.

        Args:
            path: Файл сцены.
            activate: См. :meth:`load`.
            additive: См. :meth:`load`.
            use_jobs: Разбирать файл в рабочем потоке (если пул задан).

        Returns:
            Запущенная операция.
        """
        path = Path(path)
        parse = self._parse or (lambda source: source)
        build = self._build or self._build_with_loader

        def finish(scene: Scene) -> None:
            self._register(path, scene, activate, additive)

        def progress(op: SceneLoadOperation) -> None:
            self.events.emit(SceneLoadProgress(op, op.progress, op.phase, str(path)))

        operation = SceneLoadOperation(
            path,
            parse,
            build,
            jobs=self._jobs if use_jobs else None,
            on_finish=finish,
            on_progress=progress,
        )
        self._loading.append(operation)
        operation.start()
        return operation

    def update_loading(self, budget_ms: float) -> list[SceneLoadOperation]:
        """Продвинуть текущие загрузки на один кадр (по очереди, в общий бюджет).

        Args:
            budget_ms: Бюджет кадра на загрузку.

        Returns:
            Операции, завершившиеся на этом вызове (в том числе отменённые).
        """
        finished: list[SceneLoadOperation] = []
        remaining = budget_ms
        for operation in list(self._loading):
            started = time.perf_counter()
            if operation.step(max(remaining, 0.0)):
                finished.append(operation)
            remaining -= (time.perf_counter() - started) * 1000.0
        self._loading = [op for op in self._loading if not op.is_done]
        for transition in list(self._transitions):
            transition._poll()
        self._transitions = [t for t in self._transitions if not t.is_done]
        return finished

    def transition(
        self,
        to_scene: Path,
        *,
        from_scene: Path | None = None,
        unload_from: bool = True,
        loading_scene: Path | None = None,
    ) -> SceneTransition:
        """Перейти на сцену: загрузка порциями, затем смена активной.

        Старая сцена остаётся живой, пока новая не построена; при желании
        поверх неё показывается лёгкая сцена загрузки (``loading_scene``,
        грузится сразу целиком и выгружается в конце). Переход продвигается
        вызовами :meth:`update_loading`.

        Args:
            to_scene: Файл новой сцены.
            from_scene: Выгружаемая сцена (по умолчанию - активная сейчас).
            unload_from: Выгрузить прежнюю сцену после перехода (иначе она
                остаётся дополнительной).
            loading_scene: Сцена экрана загрузки или ``None``.

        Returns:
            Объект перехода (``is_done``, ``error``, ``operation``).
        """
        target = Path(to_scene)
        source = Path(from_scene) if from_scene is not None else self._active
        loading = Path(loading_scene) if loading_scene is not None else None
        if loading is not None and loading != source and loading != target:
            self.load(loading, additive=True, activate=False)
        else:
            loading = None
        operation = self.load_async(target, activate=False, additive=True)
        transition = SceneTransition(
            self, target, source, unload_from, loading, operation
        )
        self._transitions.append(transition)
        return transition

    def _build_with_loader(
        self, _source: Any, path: Path
    ) -> Generator[float, None, Scene]:
        """Одношаговая сборка через синхронный ``loader``."""
        yield 0.0
        return self._loader(path)

    def _register(
        self, path: Path, scene: Scene, activate: bool | None, additive: bool
    ) -> None:
        """Последний шаг загрузки: заменить прежнюю сцену, события, активация."""
        if path in self._scenes:
            self.unload(path)
        self._scenes[path] = scene
        scene.scene_manager = self
        if additive:
            self._additive.add(path)
        self.events.emit(SceneLoaded(scene.name, str(path), additive))
        if activate is None:
            activate = not additive
        if activate or self._active is None:
            self.set_active(path)

    def set_active(self, path: Path) -> None:
        """Сделать загруженную сцену активной.

        Raises:
            KeyError: Если сцена не загружена.
        """
        path = Path(path)
        if path not in self._scenes:
            raise KeyError(f"Сцена не загружена: {path}")
        new = self._scenes[path]
        old = self.active_scene
        if old is not None and old is not new:
            for hook in list(self.before_activate):
                try:
                    hook(old, new)
                except Exception:  # noqa: BLE001 - хук не ломает переход
                    _log.exception("Хук before_activate упал")
        if self._active is not None and self._active != path:
            self._additive.add(self._active)
        self._active = path
        self._additive.discard(path)
        self.events.emit(SceneActivated(self._scenes[path].name, str(path)))

    def unload(self, path: Path) -> None:
        """Выгрузить сцену (``shutdown``); если была активной — активной не будет."""
        path = Path(path)
        scene = self._scenes.pop(path, None)
        if scene is None:
            return
        additive = path in self._additive
        self._additive.discard(path)
        if self._active == path:
            self._active = None
        name = scene.name
        scene.scene_manager = None
        scene.shutdown()
        self.events.emit(SceneUnloaded(name, str(path), additive))

    # ------------------------------------------------------------------ #
    # Слоты сохранения (docs/CORE.md §7.6)
    # ------------------------------------------------------------------ #
    def _ref_path(self, path: Path) -> str:
        """Путь сцены для слота: относительно корня проекта, с ``/``."""
        if self.root is not None:
            try:
                return path.relative_to(self.root).as_posix()
            except ValueError:
                pass
        return path.as_posix()

    def save_refs(self) -> list[SaveScene]:
        """Загруженные сцены для ``meta.scenes``: активная, затем остальные."""
        from engine.core.save.slot import SaveScene

        refs: list[SaveScene] = []
        if self._active is not None:
            refs.append(SaveScene(self._ref_path(self._active), False))
        for path in self._scenes:
            if path != self._active:
                refs.append(SaveScene(self._ref_path(path), True))
        return refs

    def save(self, save_manager: Any, slot_name: str, **kwargs: Any) -> Any:
        """Записать слот со всеми загруженными сценами.

        Args:
            save_manager: ``SaveManager`` (слоты).
            slot_name: Имя слота.
            **kwargs: ``playtime_sec``, ``screenshot``.

        Returns:
            ``SaveSlot``.

        Raises:
            RuntimeError: Нет активной сцены.
        """
        active = self.active_scene
        if active is None:
            raise RuntimeError("Нет активной сцены")
        extra = [s for s in self.all_scenes if s is not active]
        return save_manager.save(
            slot_name,
            active,
            active.state,
            scenes=self.save_refs(),
            extra_scenes=extra,
            **kwargs,
        )

    def apply_slot(self, save_manager: Any, slot_name: str) -> Any:
        """Применить слот к уже загруженным сценам (набор сцен не меняется)."""
        active = self.active_scene
        if active is None:
            raise RuntimeError("Нет активной сцены")
        extra = [s for s in self.all_scenes if s is not active]
        return save_manager.load(slot_name, active, extra)

    def restore(
        self,
        save_manager: Any,
        slot_name: str,
        resolve: Callable[[str], Path | None] | None = None,
        *,
        load_main: Callable[[Path], Any] | None = None,
        load_extra: Callable[[Path], Any] | None = None,
    ) -> Any:
        """Загрузить слот: привести набор сцен к записанному и применить данные.

        Активной становится ``scenes[0]``, остальные грузятся аддитивно в
        записанном порядке; лишние загруженные дополнительные сцены
        выгружаются. Если набор и порядок уже совпадают, сцены не трогаются.

        Args:
            save_manager: ``SaveManager``.
            slot_name: Имя слота.
            resolve: ``путь из слота -> файл`` (``None`` - не найден); по
                умолчанию путь берётся от корня проекта.
            load_main: Как загрузить активную сцену (по умолчанию ``load``).
            load_extra: Как загрузить дополнительную (по умолчанию
                ``load(additive=True)``).

        Returns:
            ``SaveSlot``.

        Raises:
            FileNotFoundError: Сцена из слота не найдена.
        """
        slot, _state, _objects = save_manager.read(slot_name)
        resolver = resolve or self.resolve_path
        files: list[Path] = []
        for ref in slot.meta.scenes:
            file = resolver(ref.path)
            if file is None:
                raise FileNotFoundError(f"Сцена из слота не найдена: {ref.path}")
            files.append(Path(file))
        main, extras = files[0], files[1:]
        current_extras = [p for p in self._scenes if p != self._active]
        if self._active != main or current_extras != extras:
            if self._active != main:
                (load_main or self.load)(main)
            for path in list(self._scenes):
                if path != main:
                    self.unload(path)
            for path in extras:
                if load_extra is not None:
                    load_extra(path)
                else:
                    self.load(path, additive=True)
        return self.apply_slot(save_manager, slot_name)

    def resolve_path(self, text: str) -> Path | None:
        """Файл сцены по пути из слота (от корня проекта); ``None`` - нет файла."""
        path = Path(text)
        if not path.is_absolute() and self.root is not None:
            path = self.root / path
        return path if self._exists(path) else None

    def unload_all(self) -> None:
        """Выгрузить все сцены."""
        for path in list(self._scenes):
            self.unload(path)
