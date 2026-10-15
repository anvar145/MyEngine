"""Behaviour - класс-скрипт: хуки жизненного цикла и поля ``export``.

Альтернатива скриптам из функций-хуков. Класс наследует :class:`Behaviour`,
переопределяет нужные хуки и объявляет настраиваемые поля через
:func:`export`::

    class Spinner(Behaviour):
        speed: float = export(90.0, range=(0, 360), tooltip="град/с")

        def on_update(self, delta_time):
            self.transform.rotate(Vector3.up(), self.speed * delta_time)

Два соглашения, которые нельзя ломать:

* ``Behaviour`` **не** регистрируется в глобальном ``registry`` и не использует
  ``@component_fields``: при горячей перезагрузке класс создаётся заново, и
  реестр накапливал бы устаревшие версии с тем же именем. Дескрипторы живут в
  самом классе (``__behaviour_fields__``), читать их - через
  :func:`engine.scripting.fields.descriptors_from_behaviour`;
* доступ к движку (``game_object``, ``scene`` и т.д.) идёт через прокси только по
  явному списку :attr:`Behaviour.API_NAMES`: опечатка в имени поля даст
  ``AttributeError``, а не тихо вернёт чужой атрибут компонента.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any

from engine.core.component import Component
from engine.core.event_bus import EventBus, Subscription
from engine.core.game_object import GameObject
from engine.core.math3d import Quaternion, Vector3
from engine.core.reflection import (
    FieldDescriptor,
    FieldKind,
    FieldRegistry,
    coerce_value,
    describe_class_fields,
    field,
    to_plain,
)
from engine.core.resources.asset_ref import AssetRef

__all__ = [
    "HOOK_NAMES",
    "Behaviour",
    "event_handler",
    "export",
    "require_component",
]

#: Хуки жизненного цикла; вызываются только переопределённые.
HOOK_NAMES: tuple[str, ...] = (
    "on_attach",
    "on_enable",
    "on_disable",
    "on_start",
    "on_update",
    "on_fixed_update",
    "on_late_update",
    "on_destroy",
    "on_spawn",
    "on_despawn",
    "on_reload",
    "on_fixed_update_batch",
    "on_save",
    "on_load",
)

_log = logging.getLogger("engine.scripting.behaviour")

_API_NAMES: tuple[str, ...] = (
    "game_object",
    "scene",
    "transform",
    "input",
    "time",
    "physics",
    "log",
    "get_component",
    "get_components",
    "checkpoint",
    "audio",
    "screen",
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

#: Атрибут метода, помеченного ``@event_handler``: кортеж ``(тип, приоритет)``.
_EVENT_MARK = "__event_marks__"


def event_handler(event_type: type, priority: int = 0) -> Callable[[Any], Any]:
    """Пометить метод ``Behaviour`` обработчиком события сцены.

    Подписка в ``scene.events`` ставится вместе с экземпляром и снимается
    до ``on_destroy``; при горячей перезагрузке переносится на новую версию.
    Подходит любой класс события (физические события не наследуют ``Event``);
    обработчик базового класса получает и события-наследники. Декоратор можно
    ставить несколько раз на один метод.

    Args:
        event_type: Класс события.
        priority: Чем больше, тем раньше вызывается обработчик.

    Returns:
        Декоратор метода.

    Raises:
        TypeError: Если ``event_type`` не класс.
    """
    if not isinstance(event_type, type):
        raise TypeError(
            f"event_handler: ожидался класс события, получено {event_type!r}"
        )

    def decorate(func: Any) -> Any:
        marks = getattr(func, _EVENT_MARK, ())
        setattr(func, _EVENT_MARK, ((event_type, int(priority)), *marks))
        return func

    return decorate


def require_component(component_type: type) -> Callable[[type], type]:
    """Потребовать компонент на том же объекте (декоратор класса ``Behaviour``).

    Несколько декораторов (и требования базовых классов) объединяются в
    ``cls.__required_components__``. Пока компонента нет, скрипт не запускается
    (ERROR в лог), в инспекторе показана подсказка. Проверка идёт при
    запуске и при включении компонента; удаление требуемого компонента во
    время работы скрипт не останавливает. Тип сопоставляется точно, как в
    ``GameObject.get_component``.

    Args:
        component_type: Класс компонента.

    Returns:
        Декоратор класса.

    Raises:
        TypeError: Если ``component_type`` не подкласс ``Component``.
    """
    if not (isinstance(component_type, type) and issubclass(component_type, Component)):
        raise TypeError(
            f"require_component: ожидался класс компонента, получено {component_type!r}"
        )

    def decorate(cls: type) -> type:
        existing = tuple(getattr(cls, "__required_components__", ()))
        if component_type not in existing:
            cls.__required_components__ = (*existing, component_type)
        return cls

    return decorate


def export(default: Any, **options: Any) -> Any:
    """Объявить настраиваемое поле класса-скрипта.

    Тонкая обёртка над :func:`engine.core.reflection.field`: те же параметры
    (``range``, ``tooltip``, ``kind``, ``step``, ``choices``, ...).

    Args:
        default: Значение по умолчанию.
        **options: Поля :class:`FieldDescriptor`.

    Returns:
        Заглушку; :class:`Behaviour` заменит её значением по умолчанию.
    """
    return field(default, **options)


class Behaviour:
    """Базовый класс игрового скрипта.

    Экземпляр создаёт ``ScriptComponent`` и привязывает к себе
    (:meth:`_attach`). Хуки по умолчанию пусты и не вызываются: диспетчер
    смотрит в :attr:`_overridden_hooks`.

    Attributes:
        API_NAMES: Имена, которые прокси отдаёт из компонента.
    """

    API_NAMES: tuple[str, ...] = _API_NAMES

    #: Хуки, переопределённые классом или его предками (строится сам).
    _overridden_hooks: frozenset[str] = frozenset()
    #: Дескрипторы полей ``export`` с учётом баз (строятся сами).
    __behaviour_fields__: tuple[FieldDescriptor, ...] = ()
    #: Обработчики ``@event_handler``: ``(имя метода, тип события, приоритет)``.
    __event_handlers__: tuple[tuple[str, type, int], ...] = ()
    #: Компоненты, требуемые ``@require_component`` (с учётом баз).
    __required_components__: tuple[type[Component], ...] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__event_handlers__ = _collect_event_handlers(cls)
        own = describe_class_fields(cls)
        for name, desc in own.items():
            setattr(cls, name, desc.default)  # атрибут класса = умолчание
        merged: dict[str, FieldDescriptor] = {}
        for klass in reversed(cls.__mro__):
            if klass is not cls and "__behaviour_fields__" in klass.__dict__:
                merged.update((d.name, d) for d in klass.__behaviour_fields__)
        merged.update(own)
        cls.__behaviour_fields__ = tuple(merged.values())
        cls._overridden_hooks = frozenset(
            name
            for name in HOOK_NAMES
            if any(
                name in klass.__dict__
                for klass in cls.__mro__
                if klass is not Behaviour and klass is not object
            )
        )

    def __init__(self) -> None:
        self._component: Any = None
        self._subscriptions: list[Subscription] = []
        for desc in self.__behaviour_fields__:
            if isinstance(desc.default, (list, dict, set)):
                setattr(self, desc.name, copy.deepcopy(desc.default))

    # ------------------------------------------------------------- поля

    @classmethod
    def clamp_field(cls, name: str, value: Any, *, strict: bool = False) -> Any:
        """Привести значение поля к допустимому по дескриптору (``range``).

        Raises:
            ValueError: При ``strict`` или политике ``ERROR``; в тексте имя поля.
        """
        for desc in cls.__behaviour_fields__:
            if desc.name == name:
                return FieldRegistry.clamp_descriptor(
                    cls.__name__, desc, value, strict=strict
                )
        return value

    @classmethod
    def prepare_field(
        cls,
        name: str,
        raw: Any,
        objects: dict[int | str, Any] | None = None,
        *,
        strict: bool = False,
    ) -> Any:
        """Значение поля из JSON-вида/инспектора -> готовое значение (с зажимом).

        В отличие от :meth:`apply_field_values` ничего не записывает и не
        прощает несоответствие типа.

        Raises:
            KeyError: Поля с таким именем нет.
            ValueError: Значение не подходит типу поля, либо политика ``ERROR``
                (в тексте имя поля).
            TypeError: Значение не подходит типу поля.
        """
        for desc in cls.__behaviour_fields__:
            if desc.name == name:
                value = _decode(desc, raw, objects)
                return FieldRegistry.clamp_descriptor(
                    cls.__name__, desc, value, strict=strict
                )
        raise KeyError(name)

    def apply_field_values(
        self,
        values: dict[str, Any],
        *,
        objects: dict[int | str, Any] | None = None,
        strict: bool = False,
    ) -> list[str]:
        """Записать значения из сцены (JSON-вид) в поля экземпляра.

        Не заданное поле остаётся с умолчанием. Значение неподходящего типа
        отбрасывается с записью в лог, число вне диапазона зажимается.

        Args:
            values: Имя поля -> значение в виде, который пишет
                :meth:`serialize_fields`.
            objects: Сохранённая ссылка (``uid`` или id из v1) -> объект.
            strict: Бросать ``ValueError`` вместо зажима.

        Returns:
            Имена из ``values``, которых нет среди полей класса.

        Raises:
            ValueError: Для политики ``ERROR`` или при ``strict``.
        """
        known = {d.name: d for d in self.__behaviour_fields__}
        unknown: list[str] = []
        for name, raw in values.items():
            desc = known.get(name)
            if desc is None:
                unknown.append(name)
                continue
            try:
                value = _decode(desc, raw, objects)
            except (ValueError, TypeError):
                _log.warning(
                    "%s: поле '%s': значение %r не подходит типу %s, "
                    "используется текущее",
                    type(self).__name__, name, raw, desc.kind.value,
                )
                continue
            setattr(
                self,
                name,
                FieldRegistry.clamp_descriptor(
                    type(self).__name__, desc, value, strict=strict
                ),
            )
        return unknown

    def serialize_fields(self) -> dict[str, Any]:
        """Значения полей, отличающиеся от умолчания (для ``fields`` в сцене).

        Поля с ``serialize=False`` пропускаются. Ссылка на объект пишется как
        ``{"uid": uid}``, на файл - как ``{"asset": путь}``.
        """
        result: dict[str, Any] = {}
        for desc in self.__behaviour_fields__:
            if not desc.serialize:
                continue
            encoded = _encode(getattr(self, desc.name))
            if encoded != _encode(desc.default):
                result[desc.name] = encoded
        return result

    def serialize_save_fields(self) -> dict[str, Any]:
        """Значения полей с ``save=True`` (все, не только отличные от умолчания)."""
        return {
            d.name: _encode(getattr(self, d.name))
            for d in self.__behaviour_fields__
            if d.save
        }

    # ------------------------------------------------------------ движок

    def _attach(self, component: Any) -> None:
        """Привязать экземпляр к компоненту и вызвать ``on_attach``."""
        self._component = component
        self._dispatch("on_attach")

    def _subscribe_events(
        self, bus: EventBus, deliver: Callable[[str, Any], None]
    ) -> None:
        """Подписать ``@event_handler``-методы на шину (повторный вызов не дублирует).

        Args:
            bus: Шина событий сцены.
            deliver: ``deliver(имя_метода, событие)`` - вызов метода с
                перехватом ошибок.
        """
        subscriptions = self.__dict__.setdefault("_subscriptions", [])
        if subscriptions:
            return
        for name, event_type, priority in self.__event_handlers__:
            subscriptions.append(
                bus.subscribe(
                    event_type,
                    lambda event, method=name: deliver(method, event),
                    priority=priority,
                )
            )

    def _unsubscribe_events(self) -> None:
        """Снять все подписки ``@event_handler``."""
        subscriptions = self.__dict__.get("_subscriptions")
        if not subscriptions:
            return
        for subscription in subscriptions:
            subscription.cancel()
        subscriptions.clear()

    @classmethod
    def missing_components(cls, game_object: GameObject | None) -> list[type]:
        """Требуемые ``@require_component`` компоненты, которых нет на объекте."""
        required = cls.__required_components__
        if not required:
            return []
        if game_object is None:
            return list(required)
        return [t for t in required if game_object.get_component(t) is None]

    def get_component(self, component_type: type) -> Any:
        """Компонент объекта-владельца или ``None`` (``GameObject.get_component``)."""
        owner = self._owner()
        return None if owner is None else owner.get_component(component_type)

    def get_components(self, component_type: type) -> list:
        """Все компоненты указанного типа объекта-владельца."""
        owner = self._owner()
        return [] if owner is None else owner.get_components(component_type)

    def _owner(self) -> GameObject | None:
        component = self.__dict__.get("_component")
        return None if component is None else component.game_object

    def __getattr__(self, name: str) -> Any:
        # вызывается только для отсутствующих атрибутов
        if name in _API_NAMES:
            component = self.__dict__.get("_component")
            if component is not None:
                return getattr(component, name)
        raise AttributeError(
            f"'{type(self).__name__}' не имеет атрибута '{name}'"
        )

    def _dispatch(self, name: str, *arguments: Any) -> Any:
        """Вызвать хук, если он переопределён; иначе ничего не делать."""
        if name in self._overridden_hooks:
            return getattr(self, name)(*arguments)
        return None

    # ------------------------------------------------------------- хуки

    def on_attach(self) -> None:
        """Экземпляр создан и привязан к компоненту."""

    def on_enable(self) -> None:
        """Компонент включён."""

    def on_disable(self) -> None:
        """Компонент выключен."""

    def on_start(self) -> None:
        """Первый кадр жизни (поля и ссылки на объекты уже применены)."""

    def on_update(self, delta_time: float) -> None:
        """Кадр с переменным шагом."""

    def on_fixed_update(self, dt: float) -> None:
        """Фиксированный шаг (физика)."""

    def on_late_update(self, dt: float) -> None:
        """Конец кадра, после всех ``on_update``."""

    def on_destroy(self) -> None:
        """Уничтожение компонента или выгрузка скрипта."""

    def on_spawn(self) -> None:
        """Объект взят из пула (``ObjectPool.get``), уже включён."""

    def on_despawn(self) -> None:
        """Объект возвращается в пул (``ObjectPool.release``), ещё включён."""

    def on_reload(self) -> None:
        """Скрипт перезагружен (новая версия уже запущена)."""

    def on_save(self, ctx: Any) -> None:
        """Записать собственные данные в ``ctx`` (``SaveContext`` скрипта)."""

    def on_load(self, ctx: Any) -> None:
        """Восстановить состояние из ``ctx`` (поля ``save=True`` уже применены)."""

    @classmethod
    def on_fixed_update_batch(cls, items: list, dt: float) -> None:
        """Пакетный фиксированный шаг: один вызов на все экземпляры класса."""


def _collect_event_handlers(cls: type) -> tuple[tuple[str, type, int], ...]:
    """Собрать ``@event_handler`` класса и баз (переопределение без метки снимает)."""
    found: dict[str, tuple[tuple[type, int], ...]] = {}
    for klass in reversed(cls.__mro__):
        for name, value in vars(klass).items():
            marks = getattr(value, _EVENT_MARK, None)
            if marks:
                found[name] = marks
            elif name in found:
                del found[name]  # потомок переопределил метод без декоратора
    return tuple(
        (name, event_type, priority)
        for name, marks in found.items()
        for event_type, priority in marks
    )


def _encode(value: Any) -> Any:
    """Значение поля -> вид для ``fields`` сцены."""
    if isinstance(value, AssetRef):
        return value.to_dict()
    if isinstance(value, GameObject):
        return {"uid": value.uid}
    return to_plain(value)


def _decode(
    desc: FieldDescriptor, raw: Any, objects: dict[int | str, Any] | None
) -> Any:
    """Значение из сцены -> значение поля (ссылки разрешаются по ``objects``)."""
    kind = desc.kind
    if raw is None:
        is_ref = kind in (FieldKind.OBJECT_REF, FieldKind.ASSET_REF)
        return None if is_ref else desc.default
    if kind is FieldKind.OBJECT_REF:
        if isinstance(raw, GameObject):
            return raw
        if isinstance(raw, dict) and "uid" in raw:
            return (objects or {}).get(str(raw["uid"]))
        if isinstance(raw, dict) and "object" in raw:  # формат v1
            return (objects or {}).get(int(raw["object"]))
        raise ValueError("ожидалась ссылка на объект")
    if kind is FieldKind.ASSET_REF:
        try:
            return AssetRef.coerce(raw)
        except ValueError:
            raise ValueError("ожидалась ссылка на файл") from None
    value = coerce_value(desc, raw)
    if kind is FieldKind.VEC3 and isinstance(desc.default, Vector3):
        if len(value) != 3:
            raise ValueError("ожидалось 3 числа")
        return Vector3(*value)
    if kind is FieldKind.QUAT and isinstance(desc.default, Quaternion):
        if len(value) != 4:
            raise ValueError("ожидалось 4 числа")
        return Quaternion(*value)
    return value
