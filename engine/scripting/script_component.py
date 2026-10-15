"""Компонент-скрипт: привязка Python-файла к игровому объекту.

:class:`ScriptComponent` связывает ECS-объект с пользовательским скриптом.
Скрипт пишется как обычный модуль Python и объявляет функции жизненного цикла:

.. code-block:: python

    def on_start(self):
        self.log("готов")

    def on_update(self, delta_time):
        self.transform.rotate(Vector3.up(), 45.0 * delta_time)

    def on_destroy(self):
        self.log("пока")

``self`` — это :class:`ScriptComponent`, от его лица скрипт получает доступ к
движку: ``self.game_object``, ``self.scene``, ``self.transform``,
``self.input``, ``self.time``, ``self.log()``. Дополнительно конструктор
компонента отдаёт имена прямо в пространство имён скрипта (``Vector3``,
``Quaternion``, ``scene``, ``game_object``, ``input``, ``time``), поэтому
короткая запись ``scene.find("Cube")`` тоже работает.

Каждый экземпляр компонента получает **своё** пространство имён: два скрипта
(и даже два экземпляра одного файла на разных объектах) не делят переменные.

Example:
    >>> component = ScriptComponent(path="scripts/spin.py")
    >>> cube.add_component(component)      # yjе: on_attach → загрузка модуля
    >>> component.reload()                  # горячая перезагрузка
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from engine.core.component import Component
from engine.core.game_object import GameObject
from engine.core.input import InputState
from engine.core.math3d import Quaternion, Vector3
from engine.core.reflection import FieldDescriptor, FieldKind, to_plain
from engine.core.resources.asset_ref import AssetRef
from engine.core.scheduling import (
    Handle,
    TimeCategory,
    TweenHandle,
    Wait,
    WaitForFixed,
    WaitForFrame,
    WaitUntil,
)
from engine.scripting.audio_api import ScriptAudio
from engine.scripting.behaviour import Behaviour
from engine.scripting.checkpoint_api import ScriptCheckpoint
from engine.scripting.fields import (
    ScriptField,
    coerce_field,
    descriptors_from_behaviour,
    scan_fields,
)
from engine.scripting.physics_api import ScriptPhysics
from engine.scripting.runtime import (
    ScriptError,
    ScriptHandle,
    ScriptRuntime,
    format_traceback,
)
from engine.scripting.screen_api import ScriptScreen

__all__ = [
    "API_NAMES",
    "InputState",
    "ScriptComponent",
    "ScriptTime",
    "default_runtime",
    "runtime_for",
    "set_default_runtime",
]

_log = logging.getLogger("engine.scripting.component")

# Служебные атрибуты экземпляра Behaviour: при горячей перезагрузке не переносятся.
_ENGINE_ATTRS = frozenset({"_component", "_subscriptions"})

#: Общий реестр на случай, когда сцена не предоставила свой.
_DEFAULT_RUNTIME: ScriptRuntime | None = None


def set_default_runtime(runtime: ScriptRuntime | None) -> None:
    """Назначить общий реестр скриптов.

    Args:
        runtime: Реестр или ``None`` — вернуть ленивое создание по умолчанию.
    """
    global _DEFAULT_RUNTIME  # noqa: PLW0603 - единственная точка настройки
    _DEFAULT_RUNTIME = runtime


_DEFAULT_INPUT: InputState | None = None


def set_default_input(state: InputState | None) -> None:
    """Назначить общее состояние ввода для новых скриптов (плеер).

    Args:
        state: Общий ввод; ``None`` — у каждого скрипта свой.
    """
    global _DEFAULT_INPUT  # noqa: PLW0603 - единственная точка настройки
    _DEFAULT_INPUT = state


def default_runtime() -> ScriptRuntime:
    """Вернуть общий реестр скриптов, создав его при первом обращении."""
    global _DEFAULT_RUNTIME  # noqa: PLW0603
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = ScriptRuntime()
    return _DEFAULT_RUNTIME


def runtime_for(component: ScriptComponent) -> ScriptRuntime:
    """Реестр скриптов для компонента.

    Порядок поиска: явно переданный реестр → реестр, назначенный через
    :func:`set_default_runtime` → общий ленивый реестр. Сцена сама реестр не
    хранит (у неё ``__slots__``), поэтому владельцем выступает редактор.

    Args:
        component: Компонент-скрипт.

    Returns:
        Реестр, в котором компонент регистрирует свой модуль.
    """
    return component.explicit_runtime or default_runtime()


#: Имена API, которые скрипт получает от лица ``self``. Используются подсветкой
#: и автодополнением редактора (``ScriptComponent.API_NAMES``).
API_NAMES: tuple[str, ...] = (
    "game_object",
    "scene",
    "transform",
    "input",
    "physics",
    "checkpoint",
    "audio",
    "screen",
    "time",
    "log",
    "name",
    "path",
    "instance",
    "delta_time",
    "debug",
    "warning",
    "error",
    "after",
    "every",
    "wait",
    "wait_unscaled",
    "wait_for_frame",
    "wait_for_fixed",
    "wait_until",
    "start_coroutine",
    "tween",
    "cancel",
    "cancel_all",
)


#: Размер списка задач, после которого завершённые Handle вычищаются.
_TASKS_PRUNE_MIN = 32

#: Сколько раз полностью логируется одинаковая ошибка задачи; далее - молчание.
_TASK_ERROR_FULL_LOGS = 2
#: Предел различных сигнатур ошибок в счётчике (затем он сбрасывается).
_TASK_ERROR_SIGNATURES = 64


#: Вид поля ``export`` -> имя вида в ``ScriptField`` (остальные - по ``kind.value``).
_FIELD_KIND_NAMES: dict[FieldKind, str] = {
    FieldKind.BOOL: "bool",
    FieldKind.INT: "int",
    FieldKind.FLOAT: "float",
    FieldKind.STR: "str",
    FieldKind.OBJECT_REF: "object",
    FieldKind.ASSET_REF: "asset",
}


class ScriptTime:
    """Время кадра, доступное скрипту как ``self.time``.

    Attributes:
        delta: Длительность последнего кадра в секундах.
        elapsed: Сколько секунд прошло с первого кадра.
        frame: Номер кадра.
        fixed_delta: Шаг фиксированного обновления (с последнего ``on_fixed_update``).
        fixed_frame: Сколько фиксированных шагов получил этот скрипт.
    """

    __slots__ = ("delta", "elapsed", "fixed_delta", "fixed_frame", "frame")

    def __init__(
        self, delta: float = 0.0, elapsed: float = 0.0, frame: int = 0
    ) -> None:
        self.delta = float(delta)
        self.elapsed = float(elapsed)
        self.frame = int(frame)
        self.fixed_delta = 0.0
        self.fixed_frame = 0

    def advance_fixed(self, fixed_delta: float) -> None:
        """Учесть фиксированный шаг.

        Args:
            fixed_delta: Длительность фиксированного шага в секундах.
        """
        self.fixed_delta = float(fixed_delta)
        self.fixed_frame += 1

    def advance(self, delta: float) -> None:
        """Учесть новый кадр.

        Args:
            delta: Длительность кадра в секундах.
        """
        self.delta = float(delta)
        self.elapsed += self.delta
        self.frame += 1

    def __float__(self) -> float:
        return self.elapsed

    def __repr__(self) -> str:
        return (
            f"<ScriptTime elapsed={self.elapsed:.3f} delta={self.delta:.4f} "
            f"frame={self.frame}>"
        )


class ScriptComponent(Component):
    """Компонент, исполняющий Python-скрипт.

    Args:
        path: Путь к файлу скрипта (``*.py``).
        source: Исходный код строкой (альтернатива ``path``).
        runtime: Реестр скриптов; ``None`` — взять у сцены или общий.
        auto_start: Вызывать ``on_start`` при активации (по умолчанию да).
        strict: Пробрасывать ошибки загрузки наружу; иначе они попадают в лог,
            а компонент остаётся неактивным (удобно для редактора).

    Attributes:
        handle: Учётная запись скрипта в реестре (``None``, если не загружен).
        input: Состояние ввода для скрипта.
        time: Тайминги кадра для скрипта.

    Raises:
        ValueError: Если не передан ни ``path``, ни ``source``.
        ScriptError: Если ``strict`` и скрипт не загрузился.
    """

    __slots__ = (
        "_auto_start",
        "_behaviour",
        "_block_reported",
        "_blocked",
        "_delta_time",
        "_fields",
        "_inputs",
        "_name",
        "_path",
        "_pending_refs",
        "_start_deferred",
        "_physics",
        "_audio",
        "_screen",
        "_checkpoint",
        "_runtime",
        "_running",
        "_script_class",
        "_source",
        "_started",
        "_stop_error",
        "_strict",
        "_task_error_counts",
        "_tasks",
        "_tasks_prune_at",
        "handle",
        "time",
    )

    #: Имена API движка (совпадает с :data:`API_NAMES`); доступны без экземпляра.
    API_NAMES: tuple[str, ...] = API_NAMES

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        source: str | None = None,
        name: str | None = None,
        runtime: ScriptRuntime | None = None,
        auto_start: bool = True,
        strict: bool = False,
        script_class: str | None = None,
    ) -> None:
        super().__init__()
        if path is None and source is None:
            raise ValueError("нужен путь к скрипту (path) или исходник (source)")
        self._path = Path(path) if path is not None else None
        self._source = source
        self._name = name
        self._runtime = runtime
        self._auto_start = bool(auto_start)
        self._strict = bool(strict)
        self._delta_time = 0.0
        self._started = False
        self._running = False
        self._stop_error: str = ""
        self.handle: ScriptHandle | None = None
        self.time = ScriptTime()
        self._inputs = _DEFAULT_INPUT if _DEFAULT_INPUT is not None else InputState()
        self._physics: ScriptPhysics | None = None
        self._audio: ScriptAudio | None = None
        self._screen: ScriptScreen | None = None
        self._checkpoint: ScriptCheckpoint | None = None
        self._fields: dict[str, Any] = {}
        self._pending_refs: dict[str, int | str] = {}
        self._start_deferred = False
        self._script_class = str(script_class) if script_class else None
        self._behaviour: Behaviour | None = None
        self._blocked = False
        self._block_reported = False
        self._tasks: list[Handle] = []
        self._tasks_prune_at = _TASKS_PRUNE_MIN
        self._task_error_counts: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------ #
    # Свойства
    # ------------------------------------------------------------------ #
    @property
    def path(self) -> Path | None:
        """Путь к файлу скрипта (``None`` для скрипта из строки)."""
        return self._path

    @property
    def name(self) -> str:
        """Имя скрипта: явное, имя файла или ``"<inline>"``."""
        if self._name:
            return self._name
        if self._path is not None:
            return self._path.name
        return "<inline>"

    @property
    def source(self) -> str:
        """Текущий исходный текст скрипта (читается с диска)."""
        if self._path is not None:
            try:
                virtual = self.runtime.read_virtual(self._path)
            except ScriptError:
                virtual = None
            if virtual is not None:
                return virtual
            try:
                return self._path.read_text(encoding="utf-8")
            except OSError:
                return ""
        return self._source or ""

    @property
    def instance(self) -> ScriptComponent:
        """Сам компонент: то, что скрипт видит как ``self``."""
        return self

    @property
    def script_class(self) -> str | None:
        """Явно выбранное имя класса ``Behaviour`` (``None`` — выбор автоматический)."""
        return self._script_class

    @script_class.setter
    def script_class(self, value: str | None) -> None:
        self._script_class = str(value) if value else None

    @property
    def script_kind(self) -> str:
        """Вид скрипта: ``"behaviour"`` (класс) или ``"functions"`` (хуки модуля)."""
        return "functions" if self._behaviour is None else "behaviour"

    @property
    def behaviour(self) -> Behaviour | None:
        """Экземпляр ``Behaviour`` (``None`` для скрипта из функций)."""
        return self._behaviour

    @property
    def module(self) -> Any:
        """Загруженный модуль скрипта или ``None``."""
        return self.handle.module if self.handle is not None else None

    @property
    def explicit_runtime(self) -> ScriptRuntime | None:
        """Реестр, переданный явно в конструкторе (``None``, если его не было)."""
        return self._runtime

    @property
    def runtime(self) -> ScriptRuntime:
        """Реестр скриптов: явный или общий (лениво создаётся)."""
        if self._runtime is None:
            self._runtime = default_runtime()
        return self._runtime

    @runtime.setter
    def runtime(self, value: ScriptRuntime) -> None:
        self._runtime = value

    @property
    def is_loaded(self) -> bool:
        """``True``, если модуль скрипта загружен без ошибок."""
        return self.handle is not None

    @property
    def last_error(self) -> str:
        """Текст последней ошибки загрузки (пустая строка, если её не было)."""
        return self._stop_error

    @property
    def auto_start(self) -> bool:
        """Запускать ли скрипт автоматически при старте сцены.

        Флаг хранится в файле сцены, поэтому скрипт можно прикрепить к объекту
        и оставить выключенным: он загрузится, но ``on_start``/``on_update``
        не получат управления, пока флаг не включат.
        """
        return self._auto_start

    @auto_start.setter
    def auto_start(self, value: bool) -> None:
        self._auto_start = bool(value)

    @property
    def is_running(self) -> bool:
        """``True``, если скрипт уже получил ``on_start``.

        Флаг принадлежит компоненту, а не сцене: сцена помечает компонент
        запущенным всегда, но выключенный на старте скрипт кадры не получает.
        """
        return self._running

    @property
    def input(self) -> InputState:
        """Состояние ввода для скрипта."""
        return self._inputs

    def bind_input(self, state: InputState) -> None:
        """Подменить состояние ввода общим (плеер делит одно на все скрипты)."""
        self._inputs = state
        if self.handle is not None:
            self.handle.module.__dict__["input"] = state

    @property
    def physics(self) -> ScriptPhysics:
        """Физический API для скрипта (``self.physics.raycast(...)`` и т. д.)."""
        if self._physics is None:
            self._physics = ScriptPhysics(self)
        return self._physics

    @property
    def audio(self) -> ScriptAudio:
        """Звук: ``self.audio.play(клип)``, ``self.audio.set_bus_gain(...)``."""
        if self._audio is None:
            self._audio = ScriptAudio(self)
        return self._audio

    @property
    def screen(self) -> ScriptScreen:
        """Экран: ``self.screen.fade_to(цвет, alpha, время)``, ``.flash()``."""
        if self._screen is None:
            self._screen = ScriptScreen(self)
        return self._screen

    @property
    def checkpoint(self) -> ScriptCheckpoint:
        """Контрольные точки: ``self.checkpoint.save("имя")``."""
        if self._checkpoint is None:
            self._checkpoint = ScriptCheckpoint(self)
        return self._checkpoint

    @property
    def delta_time(self) -> float:
        """Длительность последнего кадра, полученная компонентом."""
        return self._delta_time

    @property
    def api_names(self) -> tuple[str, ...]:
        """Имена API движка — подсказка для подсветки и автодополнения."""
        return API_NAMES

    # ------------------------------------------------------------------ #
    # Поля (публичные переменные скрипта)
    # ------------------------------------------------------------------ #
    @property
    def fields(self) -> dict[str, Any]:
        """Значения полей, заданные вручную (нет ключа — значение из кода)."""
        return self._fields

    def script_fields(self) -> list[ScriptField]:
        """Поля, найденные в текущем исходнике скрипта."""
        if self._behaviour is not None:
            return [
                ScriptField(
                    d.name,
                    _FIELD_KIND_NAMES.get(d.kind, d.kind.value),
                    d.default,
                    range=d.range,
                    tooltip=d.tooltip,
                    step=d.step,
                    choices=d.choices,
                    asset_kind=d.asset_kind,
                )
                for d in descriptors_from_behaviour(type(self._behaviour))
            ]
        return scan_fields(self.source)

    def _behaviour_descriptor(self, name: str) -> FieldDescriptor | None:
        """Дескриптор поля ``export`` у текущего экземпляра ``Behaviour``."""
        if self._behaviour is None:
            return None
        for desc in descriptors_from_behaviour(type(self._behaviour)):
            if desc.name == name:
                return desc
        return None

    def _set_behaviour_field(self, name: str, value: Any) -> bool:
        """``set_field`` для ``Behaviour``: проверка типа и зажим по дескриптору."""
        inst = self._behaviour
        desc = self._behaviour_descriptor(name)
        if inst is None or desc is None:
            return False
        is_ref = desc.kind in (FieldKind.OBJECT_REF, FieldKind.ASSET_REF)
        if value is None and not is_ref:
            return False
        try:
            prepared = inst.prepare_field(name, value)
        except (ValueError, TypeError, KeyError) as exc:
            _log.warning("[%s] поле '%s' не принято: %s", self.name, name, exc)
            return False
        self._pending_refs.pop(name, None)
        if prepared is None:
            self._fields.pop(name, None)
        else:
            keep = isinstance(prepared, (GameObject, AssetRef))
            self._fields[name] = prepared if keep else to_plain(prepared)
        setattr(inst, name, prepared)
        return True

    def set_field(self, name: str, value: Any) -> bool:
        """Задать значение поля вручную и сразу записать его в модуль.

        Args:
            name: Имя поля.
            value: Новое значение (тип проверяется по исходнику); ``None``
                для ссылки — снять ссылку.

        Returns:
            ``True``, если значение принято.
        """
        if self._behaviour is not None:
            return self._set_behaviour_field(name, value)
        kind = next((f.kind for f in self.script_fields() if f.name == name), None)
        if kind is None:
            return False
        if value is None:
            if kind not in ("object", "asset"):
                return False
            self._fields.pop(name, None)
            self._pending_refs.pop(name, None)
            self._write_field(name, None)
            return True
        coerced = coerce_field(kind, value)
        if coerced is None:
            return False
        self._fields[name] = coerced
        self._pending_refs.pop(name, None)
        self._write_field(name, coerced)
        return True

    def clear_field(self, name: str) -> bool:
        """Сбросить поле к значению из кода.

        Returns:
            ``True``, если переопределение было.
        """
        had = name in self._fields or name in self._pending_refs
        self._fields.pop(name, None)
        self._pending_refs.pop(name, None)
        if self._behaviour is not None:
            desc = self._behaviour_descriptor(name)
            if had and desc is not None:
                setattr(self._behaviour, name, copy.deepcopy(desc.default))
            return had
        if had and self.handle is not None:
            fresh = next((f for f in self.script_fields() if f.name == name), None)
            if fresh is not None:
                self._write_field(name, fresh.default)
        return had

    def resolve_refs(self, objects: dict[int | str, Any]) -> None:
        """Достроить ссылки на объекты после загрузки сцены.

        Args:
            objects: Сохранённая ссылка → созданный объект: ``uid`` (строка,
                формат v2) или целый id (формат v1). Ссылка на отсутствующий
                объект остаётся незаданной.
        """
        pending, self._pending_refs = self._pending_refs, {}
        for name, source_id in pending.items():
            target = objects.get(source_id)
            if target is not None:
                self._fields[name] = target
                self._write_field(name, target)
        if self._start_deferred:
            self._start_deferred = False
            self.on_start()

    def apply_fields(self) -> None:
        """Записать ручные значения в пространство имён скрипта.

        Вызывается после каждой загрузки (в том числе горячей): тело модуля
        при этом заново выставило значения из кода. Значение неподходящего
        типа (в коде тип поля поменяли) отбрасывается с записью в лог.
        """
        if self.handle is None or not self._fields:
            return
        if self._behaviour is not None:
            try:
                unknown = self._behaviour.apply_field_values(dict(self._fields))
            except ValueError as exc:
                _log.error("[%s] %s", self.name, exc)
                return
            for name in unknown:  # поля больше нет в классе; значение ждёт возвращения
                _log.debug("[%s] поле '%s' не найдено в классе", self.name, name)
            return
        kinds = {f.name: f.kind for f in self.script_fields()}
        for name, value in list(self._fields.items()):
            kind = kinds.get(name)
            if kind is None:
                continue  # поля больше нет в коде; значение ждёт возвращения
            coerced = coerce_field(kind, value)
            if coerced is None:
                _log.warning(
                    "[%s] поле '%s': сохранённое значение %r не подходит типу %s, "
                    "используется значение из кода",
                    self.name,
                    name,
                    value,
                    kind,
                )
                del self._fields[name]
                continue
            self._write_field(name, coerced)

    def _write_field(self, name: str, value: Any) -> None:
        if self._behaviour is not None:
            try:
                self._behaviour.apply_field_values({name: value})
            except ValueError as exc:
                _log.error("[%s] %s", self.name, exc)
            return
        if self.handle is not None:
            self.handle.module.__dict__[name] = value

    # ------------------------------------------------------------------ #
    # Загрузка и перезагрузка
    # ------------------------------------------------------------------ #
    def load(self, *, keep_state: bool = False) -> ScriptHandle | None:
        """Загрузить (или перезагрузить) модуль скрипта.

        Args:
            keep_state: Перенести состояние прежнего экземпляра ``Behaviour``
                в новый (горячая перезагрузка).

        Returns:
            Учётную запись скрипта или ``None``, если загрузка не удалась.

        Raises:
            ScriptError: Если ``strict`` и скрипт не загрузился.
        """
        runtime = self.runtime
        token = self._runtime_token()
        previous = self.handle.module.__dict__ if self.handle is not None else None
        try:
            if self._path is not None:
                handle = runtime.load(
                    self._path,
                    token=token,
                    name=self.name,
                    previous=previous,
                    meta={"component": self},
                )
            else:
                module = runtime.load_source(
                    self._source or "",
                    name=f"inline_{token}",
                    filename=f"<{token}>",
                    previous=previous,
                )
                handle = runtime.register(
                    token, module, name=self.name, meta={"component": self}
                )
        except ScriptError as exc:
            self._stop_error = exc.summary()
            self.handle = None
            self._release_behaviour()
            self._report(exc)
            if self._strict:
                raise
            return None
        self._install(handle, keep_state=keep_state)
        return handle

    def unload(self, *, destroy: bool = True) -> None:
        """Остановить скрипт и убрать его из реестра.

        Args:
            destroy: Вызывать ли ``on_destroy`` модуля.
        """
        if self.handle is None:
            return
        self._unsubscribe_events()
        self.cancel_all()
        if destroy:
            self._call("on_destroy")
        self.runtime.unregister(self.handle.token)
        self.handle = None
        self._release_behaviour()
        self._started = False
        self._running = False

    def _destroy_old_version(self) -> None:
        """Вызвать ``on_destroy`` текущей версии, не трогая реестр."""
        self._unsubscribe_events()
        self.cancel_all()
        self._call("on_destroy")

    # -- события и требования Behaviour -------------------------------------
    def _unsubscribe_events(self) -> None:
        """Снять подписки ``@event_handler`` текущего экземпляра."""
        if self._behaviour is not None:
            self._behaviour._unsubscribe_events()  # noqa: SLF001

    def _release_behaviour(self) -> None:
        """Отпустить экземпляр ``Behaviour``: подписки сняты, состояние сброшено."""
        self._unsubscribe_events()
        self._behaviour = None
        self._blocked = False
        self._block_reported = False

    def _ensure_subscribed(self) -> None:
        """Поставить подписки ``@event_handler``, когда у объекта есть сцена."""
        scene = self.scene
        if self._behaviour is not None and scene is not None:
            self._behaviour._subscribe_events(  # noqa: SLF001
                scene.events, self._deliver_event
            )

    def _deliver_event(self, method: str, event: Any) -> None:
        """Вызвать ``@event_handler``-метод (ошибка уходит в лог скрипта)."""
        instance = self._behaviour
        handle = self.handle
        if instance is None or handle is None or self._blocked:
            return
        self.runtime.invoke(handle, getattr(instance, method), method, event)

    def missing_requirements(self) -> list[type]:
        """Компоненты из ``@require_component``, которых нет на объекте."""
        instance = self._behaviour
        if instance is None:
            return []
        return type(instance).missing_components(self.game_object)

    def _requirements_met(self, *, report: bool = True) -> bool:
        """Проверить ``@require_component``.

        Args:
            report: Писать ли ERROR в лог (один раз, пока требования не
                выполнены). При старте объекта не пишется: компоненты
                добавляются по очереди, и нужный может прийти следом.
        """
        missing = self.missing_requirements()
        if not missing:
            self._blocked = False
            self._block_reported = False
            return True
        self._blocked = True
        if report and not self._block_reported:
            self._block_reported = True
            owner = self.game_object
            _log.error(
                "[%s] объект '%s': %s требует компонент(ы) %s, скрипт не запущен",
                self.name,
                owner.name if owner is not None else "?",
                type(self._behaviour).__name__,
                ", ".join(t.__name__ for t in missing),
            )
        return False

    def _try_unblock(self, *, report: bool) -> None:
        """Повторить проверку требований у заблокированного скрипта и запустить его."""
        if not self._requirements_met(report=report):
            return
        if self._auto_start and self.handle is not None and self.is_active:
            self._started = True
            self._running = True
            self._call("on_start")

    def _install(self, handle: ScriptHandle, *, keep_state: bool = False) -> None:
        """Заменить учётную запись и обновить имена в пространстве имён.

        Args:
            handle: Новая запись (обычно после горячей перезагрузки).
            keep_state: Перенести ``__dict__`` прежнего экземпляра ``Behaviour``
                в новый: после ``__init__``, до применения export-полей из
                сцены и до подписок ``@event_handler``.
        """
        previous = self._behaviour if keep_state else None
        self._unsubscribe_events()
        self._blocked = False
        self._block_reported = False
        self.handle = handle
        self._stop_error = ""
        self._publish_namespace()
        self._behaviour = self._create_behaviour(handle)
        if previous is not None and self._behaviour is not None:
            self._carry_state(previous, self._behaviour)
        self.apply_fields()
        if self._behaviour is not None:
            # on_attach - после применения полей, но до on_start
            self.runtime.invoke(
                handle, self._behaviour._attach, "on_attach", self  # noqa: SLF001
            )
            self._ensure_subscribed()

    @staticmethod
    def _module_behaviours(handle: ScriptHandle) -> list[type[Behaviour]]:
        """Классы ``Behaviour``, объявленные в модуле (без предков друг друга)."""
        namespace = handle.module.__dict__
        module_name = namespace.get("__name__")
        found = [
            value
            for value in namespace.values()
            if isinstance(value, type)
            and issubclass(value, Behaviour)
            and value is not Behaviour
            and value.__module__ == module_name
        ]
        return [
            cls
            for cls in found
            if not any(other is not cls and issubclass(other, cls) for other in found)
        ]

    def behaviour_class_names(self) -> list[str]:
        """Имена классов ``Behaviour`` загруженного модуля (для выбора в редакторе)."""
        if self.handle is None:
            return []
        return [cls.__name__ for cls in self._module_behaviours(self.handle)]

    def _find_behaviour_class(self, handle: ScriptHandle) -> type[Behaviour] | None:
        """Выбрать класс ``Behaviour`` модуля.

        Берутся классы, объявленные в самом модуле (импортированные базы не
        считаются) и не являющиеся предком другого найденного класса. Явное
        :attr:`script_class` приоритетнее. Если классов несколько, а имя не
        задано, берётся первый по порядку объявления с предупреждением в лог:
        точный выбор требует ``script_class``.

        Returns:
            Класс или ``None`` - скрипт из функций.
        """
        namespace = handle.module.__dict__
        if self._script_class:
            chosen = namespace.get(self._script_class)
            if (
                isinstance(chosen, type)
                and issubclass(chosen, Behaviour)
                and chosen is not Behaviour
            ):
                return chosen
            _log.warning(
                "[%s] класс '%s' не найден или не наследует Behaviour, "
                "скрипт работает как набор функций",
                self.name,
                self._script_class,
            )
            return None
        leaves = self._module_behaviours(handle)
        if not leaves:
            return None
        if len(leaves) > 1:
            _log.warning(
                "[%s] в модуле несколько Behaviour (%s), взят '%s'; "
                "укажите script_class",
                self.name,
                ", ".join(c.__name__ for c in leaves),
                leaves[0].__name__,
            )
        return leaves[0]

    def _create_behaviour(self, handle: ScriptHandle) -> Behaviour | None:
        """Создать экземпляр ``Behaviour`` (``None`` для скрипта из функций)."""
        cls = self._find_behaviour_class(handle)
        if cls is None:
            return None
        instance = self.runtime.invoke(handle, cls, "__init__")
        if instance is None:
            return None
        instance._component = self  # noqa: SLF001
        return instance

    def _carry_state(self, old: Behaviour, new: Behaviour) -> None:
        """Перенести обычные атрибуты старого экземпляра в новый.

        Не переносятся служебные атрибуты движка и ``@export``-поля (их значение
        берётся из сцены). Атрибут, тип которого в новой версии изменился, всё
        равно переносится; об этом один раз за перезагрузку пишется warning.
        """
        skip = {d.name for d in type(new).__behaviour_fields__}
        skip.update(_ENGINE_ATTRS)
        changed: list[str] = []
        target = new.__dict__
        for name, value in old.__dict__.items():
            if name in skip:
                continue
            fresh = target.get(name)
            if (
                fresh is not None
                and value is not None
                and type(fresh) is not type(value)
            ):
                changed.append(name)
            target[name] = value
        if changed:
            _log.warning(
                "[%s] при горячей перезагрузке изменился тип атрибутов: %s "
                "(перенесены прежние значения)",
                self.name,
                ", ".join(changed),
            )

    def reload(self) -> ScriptHandle | None:
        """Горячая перезагрузка: перечитать файл без перезапуска сцены.

        Порядок такой же, как у редактора Unity: старая версия получает
        ``on_destroy``, новая — ``on_start`` (если скрипт уже работал). В лог
        уходит предупреждение о возможных утечках: объекты функций прежней
        версии живут, пока на них ссылаются сигналы, таймеры и замыкания.

        Returns:
            Новую учётную запись или ``None`` при ошибке.
        """
        was_running = self._running and self.is_active
        already_started = self._started
        if was_running:
            self._destroy_old_version()
        self.cancel_all()  # задачи старой версии не переживают перезагрузку
        self._started = False
        self._running = False
        handle = self.load(keep_state=True)
        if handle is None:
            return None
        if (was_running or (self._auto_start and self.is_active)) and (
            self._requirements_met()
        ):
            self._started = True
            self._running = True
            if self._behaviour is None or not already_started:
                self._call("on_start")
            self._call("on_reload")
        return handle

    def adopt(self, handle: ScriptHandle) -> None:
        """Принять перезагруженную версию скрипта из реестра.

        Вызывается реестром после горячей перезагрузки: старая версия получает
        ``on_destroy``, новая — ``on_start`` (если скрипт работал).

        Args:
            handle: Новая учётная запись с тем же токеном.
        """
        live = getattr(handle.module, "on_start", None)
        was_started = self._running
        if was_started and self.handle is not None:
            self._destroy_old_version()
        self.cancel_all()
        self._install(handle, keep_state=True)
        self._started = False
        self._running = False
        if was_started and self.is_active and self._requirements_met():
            self._started = True
            self._running = True
            # экземпляр Behaviour уже стартовал: состояние перенесено, on_start
            # второй раз не зовётся
            if callable(live) and self._behaviour is None:
                self._call("on_start")
            self._call("on_reload")

    def reset(self) -> None:
        """Перезапустить скрипт с нуля (без перечитывания файла)."""
        self._destroy_old_version()
        self._started = False
        self._running = False
        self.load()
        if self.is_active and self._auto_start and self._requirements_met():
            self._started = True
            self._running = True
            self._call("on_start")

    # ------------------------------------------------------------------ #
    # Хуки жизненного цикла компонента
    # ------------------------------------------------------------------ #
    def on_attach(self) -> None:
        """Загрузить скрипт при добавлении компонента объекту."""
        self.load()

    def on_detach(self) -> None:
        """Выгрузить скрипт при снятии компонента с объекта."""
        self.unload(destroy=True)

    def on_enable(self) -> None:
        """Включение: ``on_enable`` класса ``Behaviour`` (у функций хука нет).

        Вызывается до ``on_start``; при загрузке сцены ссылки на объекты в этот
        момент могут быть ещё не достроены.
        """
        if self._behaviour is None:
            return
        self._ensure_subscribed()  # объект мог попасть в сцену после загрузки
        self._call("on_enable")
        if self._blocked:
            self._try_unblock(report=False)  # компонент мог добавиться за это время

    def on_disable(self) -> None:
        """Выключение: ``on_disable`` класса ``Behaviour``."""
        if self._behaviour is not None:
            self._call("on_disable")

    def on_late_update(self, dt: float) -> None:
        """Поздняя фаза кадра: ``on_late_update`` класса ``Behaviour``."""
        if self._behaviour is not None and self._running:
            self._call("on_late_update", dt)

    def on_start(self) -> None:
        """Первый кадр жизни компонента: запустить скрипт, если он автостартует.

        Сцена помечает компонент запущенным до этого вызова, поэтому состояние
        скрипта ведёт собственный флаг :attr:`is_running`.
        """
        if self.handle is None:
            return
        if self._pending_refs:
            # ссылки на объекты ещё не достроены (идёт загрузка сцены):
            # стартуем из resolve_refs, когда все объекты уже созданы
            self._start_deferred = True
            return
        if not self._auto_start:
            self._running = False
            return
        if not self._requirements_met(report=False):
            self._running = False
            return
        self._started = True
        self._running = True
        self._call("on_start")

    def on_update(self, dt: float) -> None:
        """Кадр: обновить тайминги и вызвать ``on_update`` скрипта.

        Скрипт, выключенный на старте (``auto_start = False``), кадры не
        получает: он остаётся загруженным и ждёт включения.
        """
        self._delta_time = float(dt)
        self.time.advance(dt)
        if self._blocked:
            self._try_unblock(report=True)
        if not self._running:
            return
        self._call("on_update", dt)

    def on_fixed_update(self, dt: float) -> None:
        """Фиксированный шаг: вызвать ``on_fixed_update`` скрипта (физика)."""
        self.time.advance_fixed(dt)
        if not self._running:
            return
        handle = self.handle
        if handle is None:
            return
        instance = self._behaviour
        batched = (
            "on_fixed_update_batch" in instance._overridden_hooks  # noqa: SLF001
            if instance is not None
            else "on_fixed_update_batch" in handle.module.__dict__
        )
        if batched:
            # Пакетный хук: один вызов на все компоненты этого скрипта.
            owner = self.scene
            if owner is not None:
                key = self._path if self._path is not None else self._source
                if instance is not None:
                    key = (key, type(instance).__name__)
                owner.add_fixed_batch(key, self._call_fixed_batch, self)
                return
        self._call("on_fixed_update", dt)

    def _call_fixed_batch(self, items: list, dt: float) -> None:
        """Пакетный фиксированный шаг (``self`` - первый компонент пакета).

        Для функций вызывает ``on_fixed_update_batch(self, items, dt)`` модуля,
        для ``Behaviour`` - classmethod ``on_fixed_update_batch(items, dt)``,
        где ``items`` - экземпляры ``Behaviour`` пакета.
        """
        instance = self._behaviour
        if instance is None:
            self._call("on_fixed_update_batch", items, dt)
        elif self.handle is not None:
            instances = [c._behaviour for c in items if c._behaviour is not None]  # noqa: SLF001
            self.runtime.invoke(
                self.handle,
                instance.on_fixed_update_batch,
                "on_fixed_update_batch",
                instances,
                dt,
            )

    def on_destroy(self) -> None:
        """Уничтожение: вызвать ``on_destroy`` скрипта до потери владельца."""
        self._unsubscribe_events()
        self.cancel_all()
        self._call("on_destroy")
        if self.handle is not None:
            self.runtime.unregister(self.handle.token)
            self.handle = None
        self._release_behaviour()
        self._started = False
        self._running = False

    def on_spawn(self) -> None:
        """Объект взят из пула: вызвать ``on_spawn`` скрипта."""
        self._call("on_spawn")

    def on_despawn(self) -> None:
        """Объект возвращается в пул: отменить задачи, вызвать ``on_despawn``."""
        self.cancel_all()
        self._call("on_despawn")

    def on_save(self, ctx: Any) -> None:
        """Сохранение: вызвать ``on_save(ctx)`` скрипта."""
        self._call("on_save", ctx)

    def on_load(self, ctx: Any) -> None:
        """Загрузка: вызвать ``on_load(ctx)`` скрипта."""
        self._call("on_load", ctx)

    # ------------------------------------------------------------------ #
    # API для скрипта
    # ------------------------------------------------------------------ #
    # ``game_object`` и ``scene`` уже определены в Component — скрипт видит их
    # через ``self`` без дополнительного кода.
    @property
    def transform(self) -> Any | None:
        """Трансформ владельца или ``None``, если компонент ни к чему не привязан.

        Базовая реализация бросает исключение, но скрипт может выполняться и без
        объекта — при разовом запуске из редактора (F5). Возврат ``None``
        позволяет писать привычную проверку ``if self.transform is None`` вместо
        обработки исключения.
        """
        owner = self.game_object
        return None if owner is None else owner.transform

    def log(self, *values: object) -> None:
        """Написать сообщение в консоль редактора от имени скрипта.

        Args:
            *values: Что напечатать (склеивается пробелами).
        """
        _log.info("[%s] %s", self.name, _join(values))

    def debug(self, *values: object) -> None:
        """Подробное сообщение (уровень DEBUG)."""
        _log.debug("[%s] %s", self.name, _join(values))

    def warning(self, *values: object) -> None:
        """Предупреждение (уровень WARNING)."""
        _log.warning("[%s] %s", self.name, _join(values))

    def error(self, *values: object) -> None:
        """Ошибка (уровень ERROR)."""
        _log.error("[%s] %s", self.name, _join(values))

    # ------------------------------------------------------------------ #
    # Планировщик: таймеры, корутины, твины
    # ------------------------------------------------------------------ #
    # Владелец задач - компонент (для Behaviour его отдаёт прокси). Все задачи
    # отменяются до пользовательского on_destroy и при горячей перезагрузке.
    # ВНИМАНИЕ: отключённый (enabled=False) компонент задачи не теряет -
    # корутина/таймер сработают, хотя on_update не вызывается.
    def _scheduler(self):
        scene = self.scene
        if scene is None:
            raise RuntimeError("компонент не привязан к сцене: нет планировщика")
        return scene.scheduler

    def _task_error(self, error: BaseException) -> None:
        """Записать исключение таймера/корутины в лог скрипта с троттлингом.

        Одинаковая ошибка (тип + сообщение) логируется с трассировкой
        ``_TASK_ERROR_FULL_LOGS`` раза, затем один раз сообщается о подавлении.
        Компонент после ошибки не останавливается, ``handle.errors`` растёт.
        """
        if self.handle is not None:
            self.handle.errors += 1
        counts = self._task_error_counts
        if len(counts) >= _TASK_ERROR_SIGNATURES:
            counts.clear()
        key = (type(error).__name__, str(error))
        count = counts.get(key, 0) + 1
        counts[key] = count
        if count <= _TASK_ERROR_FULL_LOGS:
            filename = str(self._path) if self._path is not None else None
            _log.error(
                "[%s] ошибка в задаче планировщика: %s: %s",
                self.name,
                key[0],
                key[1],
            )
            _log.error("[script] %s", format_traceback(error, filename=filename))
        elif count == _TASK_ERROR_FULL_LOGS + 1:
            _log.error(
                "[%s] ошибка %s повторяется, дальнейшие одинаковые сообщения "
                "подавлены",
                self.name,
                key[0],
            )

    def _track(self, handle: Handle) -> Handle:
        tasks = self._tasks
        tasks.append(handle)
        if len(tasks) > self._tasks_prune_at:
            tasks[:] = [h for h in tasks if not h.is_done]
            self._tasks_prune_at = max(_TASKS_PRUNE_MIN, 2 * len(tasks))
        return handle

    def after(
        self,
        seconds: float,
        callback: Any,
        *,
        category: TimeCategory = TimeCategory.GAME,
    ) -> Handle:
        """Вызвать ``callback()`` один раз через ``seconds`` секунд."""
        return self._track(
            self._scheduler().after(
                seconds,
                callback,
                category=category,
                owner=self,
                on_error=self._task_error,
            )
        )

    def every(
        self,
        seconds: float,
        callback: Any,
        *,
        category: TimeCategory = TimeCategory.GAME,
        immediate: bool = False,
    ) -> Handle:
        """Вызывать ``callback()`` каждые ``seconds`` секунд до отмены."""
        return self._track(
            self._scheduler().every(
                seconds,
                callback,
                category=category,
                immediate=immediate,
                owner=self,
                on_error=self._task_error,
            )
        )

    def start_coroutine(
        self, generator: Any, *, category: TimeCategory = TimeCategory.GAME
    ) -> Handle:
        """Запустить генератор-корутину (``yield self.wait(1.0)`` и т. п.)."""
        return self._track(
            self._scheduler().start_coroutine(
                generator, category=category, owner=self, on_error=self._task_error
            )
        )

    def tween(
        self,
        target: object,
        attr: str,
        end: Any,
        duration: float,
        *,
        ease: str | Any = "ease_in_out_quad",
        category: TimeCategory = TimeCategory.GAME,
    ) -> TweenHandle:
        """Плавно привести ``target.attr`` к ``end`` за ``duration`` секунд."""
        handle = self._scheduler().tween(
            target, attr, end, duration, ease=ease, category=category, owner=self
        )
        self._track(handle)
        return handle

    @staticmethod
    def wait(seconds: float) -> Wait:
        """Инструкция корутине: ждать игровые секунды (с учётом паузы/масштаба)."""
        return Wait(seconds)

    @staticmethod
    def wait_unscaled(seconds: float) -> Wait:
        """Ждать реальные секунды (идут и на паузе)."""
        return Wait(seconds, unscaled=True)

    @staticmethod
    def wait_for_frame() -> WaitForFrame:
        """Ждать следующий кадр."""
        return WaitForFrame()

    @staticmethod
    def wait_for_fixed(count: int = 1) -> WaitForFixed:
        """Ждать ``count`` фиксированных шагов физики."""
        return WaitForFixed(count)

    @staticmethod
    def wait_until(predicate: Any) -> WaitUntil:
        """Ждать, пока ``predicate()`` не вернёт истину (проверка раз в кадр)."""
        return WaitUntil(predicate)

    def cancel(self, handle: Handle) -> None:
        """Отменить задачу; список компонента чистится при ближайшей очистке."""
        handle.cancel()

    def cancel_all(self) -> None:
        """Отменить все задачи этого компонента."""
        scene = self.scene
        if scene is not None:
            scene.scheduler.cancel_all(owner=self)
        else:
            for handle in self._tasks:
                handle.cancel()
        self._tasks.clear()
        self._tasks_prune_at = _TASKS_PRUNE_MIN
        self._task_error_counts.clear()

    # ------------------------------------------------------------------ #
    # Сериализация
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, object]:
        """Сохранить компонент в словарь (для файла сцены)."""
        payload: dict[str, object] = {"auto_start": self._auto_start}
        if self._path is not None:
            payload["path"] = str(self._path)
        else:
            payload["source"] = self._source or ""
        if self._script_class:
            payload["script_class"] = self._script_class
        fields = self._encode_fields()
        if fields:
            payload["fields"] = fields
        return payload

    def _encode_fields(self) -> dict[str, Any]:
        """Значения полей в виде, пригодном для JSON."""
        result: dict[str, Any] = {}
        for name, value in self._fields.items():
            if isinstance(value, AssetRef):
                result[name] = value.to_dict()
            elif isinstance(value, GameObject):
                result[name] = {"uid": value.uid}
            else:
                result[name] = value
        for name, source in self._pending_refs.items():
            # ``int`` - ссылка v1 (id), так и не разрешённая; пишем как была
            ref = {"uid": source} if isinstance(source, str) else {"object": source}
            result.setdefault(name, ref)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ScriptComponent:
        """Восстановить компонент из словаря, полученного :meth:`to_dict`.

        Args:
            data: Словарь компонента.

        Returns:
            Новый компонент с указанным скриптом.
        """
        path = data.get("path")
        source = data.get("source")
        script_class = data.get("script_class")
        component = cls(
            Path(str(path)) if path else None,
            source=str(source) if source is not None else None,
            auto_start=bool(data.get("auto_start", True)),
            script_class=str(script_class) if script_class else None,
        )
        raw_fields = data.get("fields")
        if isinstance(raw_fields, dict):
            for name, value in raw_fields.items():
                if isinstance(value, dict) and "uid" in value:
                    component._pending_refs[str(name)] = str(value["uid"])
                elif isinstance(value, dict) and "object" in value:
                    component._pending_refs[str(name)] = int(value["object"])
                elif isinstance(value, dict) and (
                    "asset" in value or "guid" in value or "path" in value
                ):
                    ref = AssetRef.coerce(value)
                    if ref is not None:
                        component._fields[str(name)] = ref
                else:
                    component._fields[str(name)] = value
        return component

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #
    def _runtime_token(self) -> str:
        """Уникальный ключ экземпляра в реестре скриптов.

        Ключ включает ``id`` компонента: один и тот же файл можно повесить на
        несколько объектов, и каждый получит собственное пространство имён.
        """
        base = str(self._path) if self._path is not None else self.name
        return f"{base}#{id(self):x}"

    def _publish_namespace(self) -> None:
        """Положить имена движка прямо в пространство имён скрипта.

        Скрипт может писать ``Vector3.up()`` или ``scene.find("Cube")`` — но
        ровно до тех пор, пока не переопределит имя у себя.
        """
        if self.handle is None:
            return
        namespace = self.handle.module.__dict__
        namespace.setdefault("Vector3", Vector3)
        namespace.setdefault("Quaternion", Quaternion)
        namespace.setdefault("scene", self.scene)
        namespace.setdefault("game_object", self.game_object)
        namespace.setdefault("transform", self.transform)
        namespace.setdefault("input", self._inputs)
        namespace.setdefault("physics", self.physics)
        namespace.setdefault("checkpoint", self.checkpoint)
        namespace.setdefault("audio", self.audio)
        namespace.setdefault("screen", self.screen)
        namespace.setdefault("time", self.time)
        namespace.setdefault("component", self)

    def _call(self, function: str, *arguments: object) -> Any:
        """Вызвать хук скрипта через реестр (с перехватом исключений).

        Args:
            function: Имя функции скрипта (``on_start``, ``on_update``, ...).
            *arguments: Дополнительные аргументы после ``self``.

        Returns:
            Результат функции или ``None``, если её нет или она упала.
        """
        if self.handle is None:
            return None
        instance = self._behaviour
        if instance is not None:
            # Behaviour: только переопределённые хуки, без передачи ``self``
            if function not in instance._overridden_hooks:  # noqa: SLF001
                return None
            return self.runtime.invoke(
                self.handle, getattr(instance, function), function, *arguments
            )
        return self.runtime.call(self.handle, function, self, *arguments)

    def _report(self, error: ScriptError) -> None:
        """Записать ошибку загрузки в лог редактора."""
        _log.error("[%s] %s", error.name or self.name, error)
        text = error.text or format_traceback(error, filename=str(self._path or ""))
        if text:
            _log.error("[script] %s", text)

    def __repr__(self) -> str:
        owner = self.game_object
        target = owner.name if owner is not None else "<свободен>"
        state = "загружен" if self.is_loaded else "не загружен"
        return f"<ScriptComponent '{self.name}' объекта '{target}' {state}>"


def _join(values: Iterable[object]) -> str:
    """Склеить значения сообщения в строку, не падая на плохом ``__str__``."""
    try:
        return " ".join(str(value) for value in values)
    except Exception:  # noqa: BLE001 - сообщение важнее форматирования
        return "<непечатаемое сообщение>"
