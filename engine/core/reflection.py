"""Рефлексия полей компонентов: единый источник правды о данных.

``FieldDescriptor`` описывает одно поле (тип, умолчание, диапазон, подсказки для
инспектора, флаги сериализации). Дескрипторы собирает декоратор
:func:`component_fields` из аннотаций класса, а хранит и применяет
:class:`FieldRegistry`: сериализация в словарь сцены, чтение с приведением
типов и зажимом диапазонов, проверка зависимых полей.

Пример::

    @component_fields
    class Probe(Component):
        speed: float = field(4.5, range=(0.0, math.inf), tooltip="Скорость")
        use_gravity: bool = True

        @validator
        def _check(self) -> None:
            ...

Порядок дескрипторов равен порядку объявления аннотаций (базовые классы
идут первыми) — от него зависит порядок ключей в ``.scene.json``.

Модуль не знает про Qt, GL, рендер и редактор.
"""

from __future__ import annotations

import ast
import inspect
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from engine.core.math3d import Quaternion, Vector3
from engine.core.resources.asset_ref import AssetRef

__all__ = [
    "coerce_value",
    "to_plain",
    "FieldDescriptor",
    "FieldKind",
    "FieldRegistry",
    "InvalidPolicy",
    "component_fields",
    "declare",
    "describe_class_fields",
    "evaluate_visible_if",
    "field",
    "registry",
    "validator",
]

_log = logging.getLogger("engine.core.reflection")

_T = TypeVar("_T")
_MIN_POSITIVE = 1e-3  # замена недопустимому значению поля с открытым минимумом


class FieldKind(str, Enum):
    """Вид значения поля (определяет формат в сцене и виджет инспектора)."""

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"
    VEC2 = "vec2"
    VEC3 = "vec3"
    VEC4 = "vec4"
    QUAT = "quat"
    COLOR = "color"
    ENUM = "enum"
    ASSET_REF = "asset_ref"
    OBJECT_REF = "object_ref"
    COMPONENT_REF = "component_ref"
    MASK32 = "mask32"
    TUPLE = "tuple"
    LIST = "list"
    DICT = "dict"
    STRUCT = "struct"


class InvalidPolicy(str, Enum):
    """Что делать с недопустимым значением при чтении."""

    RANGE = "range"  # зажать в диапазон (нечисло -> умолчание), с предупреждением
    ERROR = "error"  # бросить ``ValueError`` с именем поля
    DEFAULT_IF_NAN = "default_if_nan"  # нечисло -> умолчание, остальное как есть


_VECTOR_KINDS = frozenset(
    {FieldKind.VEC2, FieldKind.VEC3, FieldKind.VEC4, FieldKind.QUAT, FieldKind.COLOR}
)
_REF_KINDS = frozenset({FieldKind.OBJECT_REF, FieldKind.COMPONENT_REF})


@dataclass(frozen=True, slots=True)
class FieldDescriptor:
    """Метаданные одного поля компонента.

    Args:
        name: Имя атрибута в классе (и ключ в словаре сцены).
        kind: Вид значения.
        default: Значение по умолчанию.
        type_hint: Подсказка типа для UI.
        range: ``(минимум, максимум)`` числового поля (максимум может быть inf).
        tooltip: Подсказка в инспекторе.
        group: Группа для визуальной группировки.
        step: Шаг спинбокса.
        choices: Допустимые значения ``ENUM``.
        asset_kind: Вид ассета для ``ASSET_REF``.
        ref_kind: Вид цели для ссылок.
        readonly: Поле только для чтения в инспекторе.
        serialize: Писать ли в сцену.
        save: Сохранять ли в runtime-сохранение (заготовка под A7).
        visible_if: Простое условие видимости (см. :func:`evaluate_visible_if`).
        on_invalid: Политика недопустимых значений.
        ctor_error: Шаблон сообщения ``ValueError`` (``{name}``, ``{value}``);
            по умолчанию сообщение строится из диапазона.
        live: Правка на работающем компоненте не требует пересоздания
            внутреннего объекта (например, физического двойника).
        omit_if_default: Не писать в сцену, пока значение равно умолчанию.
        exclusive_min: Минимум диапазона не достижим (``> min``); недопустимое
            значение заменяется на ``1e-3``.
    """

    name: str
    kind: FieldKind
    default: Any
    type_hint: str | None = None
    range: tuple[float, float] | None = None
    tooltip: str | None = None
    group: str | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    asset_kind: str | None = None
    ref_kind: str | None = None
    readonly: bool = False
    serialize: bool = True
    save: bool = False
    visible_if: str | None = None
    on_invalid: InvalidPolicy = InvalidPolicy.RANGE
    ctor_error: str | None = None
    live: bool = True
    omit_if_default: bool = False
    exclusive_min: bool = False


class _FieldSpec:
    """Объявление поля в теле класса: заменяется дескриптором в декораторе."""

    __slots__ = ("default", "kind", "name", "options")

    def __init__(
        self,
        default: Any,
        kind: FieldKind | None,
        options: dict,
        name: str | None = None,
    ) -> None:
        self.name = name
        self.default = default
        self.kind = kind
        self.options = options


def field(default: Any, *, kind: FieldKind | None = None, **options: Any) -> Any:
    """Объявляет поле с метаданными (аргументы - как у :class:`FieldDescriptor`).

    Args:
        default: Значение по умолчанию.
        kind: Вид значения; по умолчанию выводится из аннотации/умолчания.
        **options: Остальные поля :class:`FieldDescriptor`.

    Returns:
        Заглушку, которую :func:`component_fields` заменит дескриптором.
    """
    return _FieldSpec(default, kind, options)


def declare(name: str, default: Any, *, kind: FieldKind | None = None, **options: Any):
    """Объявляет поле списком ``__field_specs__`` (для классов с ``__slots__``).

    Атрибут класса с умолчанием нельзя совместить со слотом того же имени, поэтому
    такие классы перечисляют поля в ``__field_specs__ = (declare(...), ...)``.

    Args:
        name: Имя поля (оно же слот).
        default: Значение по умолчанию.
        kind: Вид значения; по умолчанию выводится из умолчания.
        **options: Остальные поля :class:`FieldDescriptor`.

    Returns:
        Заглушку для :func:`component_fields`.
    """
    return _FieldSpec(default, kind, options, name)


def validator(func: Callable[[Any], None]) -> Callable[[Any], None]:
    """Помечает метод проверки зависимых полей (бросает ``ValueError``)."""
    func.__reflection_validator__ = True  # type: ignore[attr-defined]
    return func


def _infer_kind(annotation: str, default: Any) -> FieldKind:
    ann = annotation.replace(" ", "").lower()
    simple = {
        "bool": FieldKind.BOOL, "int": FieldKind.INT, "float": FieldKind.FLOAT,
        "str": FieldKind.STR,
    }
    if ann in simple:
        return simple[ann]
    if ann in ("asset", "prefab"):
        return FieldKind.ASSET_REF
    if ann == "gameobject":
        return FieldKind.OBJECT_REF
    if ann == "vector3":
        return FieldKind.VEC3
    if ann == "quaternion":
        return FieldKind.QUAT
    if isinstance(default, Vector3):
        return FieldKind.VEC3
    if isinstance(default, Quaternion):
        return FieldKind.QUAT
    if isinstance(default, bool):
        return FieldKind.BOOL
    if isinstance(default, int):
        return FieldKind.INT
    if isinstance(default, float):
        return FieldKind.FLOAT
    if isinstance(default, str):
        return FieldKind.STR
    if isinstance(default, tuple) and len(default) in (2, 3, 4):
        return {2: FieldKind.VEC2, 3: FieldKind.VEC3, 4: FieldKind.VEC4}[len(default)]
    if isinstance(default, (list, tuple)):
        return FieldKind.LIST
    if isinstance(default, dict):
        return FieldKind.DICT
    raise TypeError(f"не удалось вывести kind по аннотации {annotation!r}")


def describe_class_fields(
    cls: type, *, consume: bool = False
) -> dict[str, FieldDescriptor]:
    """Дескрипторы полей, объявленных аннотациями в теле самого класса.

    Args:
        cls: Класс с аннотациями ``имя: тип = умолчание`` / ``field(...)``.
        consume: Удалить атрибуты класса с умолчаниями (значение живёт в
            дескрипторе); иначе класс не меняется.

    Returns:
        Дескрипторы в порядке объявления (базовые классы не обходятся).

    Raises:
        TypeError: Если у поля нет значения по умолчанию.
    """
    found: dict[str, FieldDescriptor] = {}
    for name, annotation in inspect.get_annotations(cls).items():
        if name.startswith("_") or _is_classvar(annotation):
            continue
        raw = cls.__dict__.get(name, _MISSING)
        if raw is _MISSING:
            raise TypeError(f"{cls.__name__}.{name}: нет значения по умолчанию")
        if isinstance(raw, _FieldSpec):
            default, kind, options = raw.default, raw.kind, raw.options
        else:
            default, kind, options = raw, None, {}
        if kind is None:
            # без ``from __future__ import annotations`` аннотация - сам класс
            hint = annotation.__name__ if isinstance(annotation, type) else annotation
            kind = _infer_kind(str(hint), default)
            if kind in _REF_KINDS and "ref_kind" not in options:
                options = {**options, "ref_kind": "GameObject"}
            if (
                kind is FieldKind.ASSET_REF
                and "asset_kind" not in options
                and str(hint).replace(" ", "").lower() == "prefab"
            ):
                options = {**options, "asset_kind": "prefab"}
        found[name] = FieldDescriptor(name, kind, default, **options)
        if consume:
            delattr(cls, name)
    return found


def _is_classvar(annotation: Any) -> bool:
    return isinstance(annotation, str) and annotation.lstrip().startswith("ClassVar")


def _to_plain(value: Any) -> Any:
    """Значение -> JSON-совместимое (кортежи в списки, ссылки в id)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    to_tuple = getattr(value, "to_tuple", None)
    if to_tuple is not None:  # Vector3 / Quaternion
        return list(to_tuple())
    if isinstance(value, AssetRef):
        return value.to_dict()
    tolist = getattr(value, "tolist", None)
    if tolist is not None:  # numpy
        return tolist()
    game_object = getattr(value, "game_object", None)
    if game_object is not None:  # компонент -> id владельца
        return game_object.id
    obj_id = getattr(value, "id", None)
    if obj_id is not None:  # GameObject -> id (заглушка до object_uid)
        return obj_id
    return value


def _coerce(desc: FieldDescriptor, value: Any) -> Any:
    """Приводит прочитанное значение к типу поля."""
    kind = desc.kind
    if kind is FieldKind.BOOL:
        return bool(value)
    if kind in (FieldKind.INT, FieldKind.MASK32):
        return int(value)
    if kind is FieldKind.FLOAT:
        return float(value)
    if kind in (FieldKind.STR, FieldKind.ENUM):
        return str(value)
    if kind in _VECTOR_KINDS:
        return tuple(float(v) for v in value)
    if kind is FieldKind.TUPLE:
        return tuple(_plain_to_tuple(v) for v in value)
    if kind is FieldKind.LIST:
        return [_plain_to_tuple(v) for v in value]
    if kind is FieldKind.ASSET_REF:
        return _coerce_asset_ref(value)
    return value


def _coerce_asset_ref(value: Any) -> Any:
    """``{"guid","path"}`` -> ``AssetRef``; пусто -> ``None``."""
    return AssetRef.coerce(value)


def _plain_to_tuple(value: Any) -> Any:
    """Вложенные списки из JSON -> кортежи (списки верхнего уровня - списки)."""
    if isinstance(value, list):
        return tuple(_plain_to_tuple(v) for v in value)
    return value


def _bound_text(desc: FieldDescriptor) -> str:
    assert desc.range is not None
    low, high = desc.range
    text = f"{'>' if desc.exclusive_min else '>='} {low:g}"
    if math.isfinite(high):
        text += f" и <= {high:g}"
    return text


class FieldRegistry:
    """Реестр дескрипторов полей классов компонентов."""

    def __init__(self) -> None:
        self._fields: dict[type, tuple[FieldDescriptor, ...]] = {}
        self._validators: dict[type, tuple[Callable[[Any], None], ...]] = {}

    # ------------------------------------------------------------ регистрация

    def register(self, cls: type) -> type:
        """Собирает дескрипторы класса (с базовыми) и запоминает их.

        Поля объявляются аннотациями; значение класса - умолчание или
        :func:`field`. Атрибуты класса с умолчаниями удаляются (значение
        живёт в дескрипторе), поэтому класс не должен перечислять эти поля
        в ``__slots__`` - при необходимости слоты обходятся ``__dict__``.

        Args:
            cls: Класс компонента.

        Returns:
            Тот же класс (``__fields__`` проставлен).
        """
        merged: dict[str, FieldDescriptor] = {}
        for base in reversed(cls.__mro__[1:]):
            for desc in self._fields.get(base, ()):
                merged[desc.name] = desc
        merged.update(describe_class_fields(cls, consume=True))
        for spec in cls.__dict__.get("__field_specs__", ()):
            kind = spec.kind or _infer_kind("", spec.default)
            merged[spec.name] = FieldDescriptor(
                spec.name, kind, spec.default, **spec.options
            )
        descriptors = tuple(merged.values())
        self._fields[cls] = descriptors
        cls.__fields__ = descriptors  # type: ignore[attr-defined]
        self._validators[cls] = self._collect_validators(cls)
        if "__init__" not in cls.__dict__:
            cls.__init__ = _make_init(cls, descriptors)  # type: ignore[misc]
        return cls

    def _collect_validators(self, cls: type) -> tuple[Callable[[Any], None], ...]:
        found: dict[str, Callable[[Any], None]] = {}
        for base in reversed(cls.__mro__):
            for name, value in base.__dict__.items():
                if getattr(value, "__reflection_validator__", False):
                    found[name] = value
        return tuple(found.values())

    # ---------------------------------------------------------------- запросы

    def fields(self, cls: type) -> tuple[FieldDescriptor, ...]:
        """Дескрипторы класса в порядке объявления (пусто, если не зарегистрирован)."""
        for klass in cls.__mro__:
            if klass in self._fields:
                return self._fields[klass]
        return ()

    def for_name(self, cls: type, name: str) -> FieldDescriptor | None:
        """Дескриптор поля по имени или ``None``."""
        for desc in self.fields(cls):
            if desc.name == name:
                return desc
        return None

    # ------------------------------------------------------------ валидация

    def clamp(
        self,
        cls: type,
        name: str,
        value: Any,
        *,
        strict: bool = False,
        warn: bool = True,
    ) -> Any:
        """Приводит значение поля к допустимому диапазону.

        Допустимое значение возвращается без изменений. Значение вне диапазона
        заменяется границей (для ``exclusive_min`` - ``1e-3``), нечисло - умолчанием;
        замена пишется в лог предупреждением. Поле без диапазона и без политики
        ``DEFAULT_IF_NAN`` не трогается.

        Args:
            cls: Класс компонента.
            name: Имя поля.
            value: Исходное значение.
            strict: Бросать ``ValueError`` вместо замены (конструктор).
            warn: Писать ли замену в лог (``False`` - тихая замена).

        Returns:
            Допустимое значение.

        Raises:
            ValueError: При ``strict`` или политике ``ERROR`` для недопустимого
                значения; в сообщении имя поля.
        """
        desc = self.for_name(cls, name)
        if desc is None:
            return value
        # enum зарегистрированных компонентов не проверяется: неизвестное значение
        # (например, форма коллайдера) читается как есть, объект пропускает симуляция
        return self.clamp_descriptor(
            cls.__name__, desc, value, strict=strict, warn=warn, check_enum=False
        )

    @staticmethod
    def clamp_descriptor(
        owner: str,
        desc: FieldDescriptor,
        value: Any,
        *,
        strict: bool = False,
        warn: bool = True,
        check_enum: bool = True,
    ) -> Any:
        """То же, что :meth:`clamp`, но по готовому дескриптору.

        Нужен классам, которые нельзя регистрировать (скрипты ``Behaviour``).

        Args:
            owner: Имя класса для сообщений.
            desc: Дескриптор поля.
            value: Исходное значение.
            strict: Бросать ``ValueError`` вместо замены.
            warn: Писать ли замену в лог.
            check_enum: Проверять ли значение ``ENUM`` по ``choices``.
        """
        strict = strict or desc.on_invalid is InvalidPolicy.ERROR
        if check_enum and desc.kind is FieldKind.ENUM and desc.choices:
            if value in desc.choices:
                return value
            if strict:
                raise ValueError(
                    f"{desc.name}={value!r}: допустимо одно из "
                    f"{', '.join(desc.choices)}"
                )
            if warn:
                _log.warning(
                    "%s: %s=%r не из %s, используется %r",
                    owner, desc.name, value, desc.choices, desc.default,
                )
            return desc.default
        if desc.kind not in (FieldKind.INT, FieldKind.FLOAT):
            return value
        name = desc.name
        number = float(value)
        fixed: float
        if not math.isfinite(number):
            keeps = desc.on_invalid is not InvalidPolicy.DEFAULT_IF_NAN
            if desc.range is None and keeps:
                return value
            fixed = float(desc.default)
        elif desc.range is None or desc.on_invalid is InvalidPolicy.DEFAULT_IF_NAN:
            return value
        else:
            low, high = desc.range
            inside = number > low if desc.exclusive_min else number >= low
            if inside:
                fixed = min(number, high)
            else:
                fixed = _MIN_POSITIVE if desc.exclusive_min else low
        if fixed == number:
            return value
        if strict:
            raise ValueError(FieldRegistry._error_text(desc, number))
        if warn:
            _log.warning(
                "%s: %s=%s недопустим, используется %s",
                owner, name, number, fixed,
            )
        return int(fixed) if desc.kind is FieldKind.INT else fixed

    @staticmethod
    def _error_text(desc: FieldDescriptor, value: float) -> str:
        if desc.ctor_error is not None:
            return desc.ctor_error.format(name=desc.name, value=value)
        if desc.range is None:
            return f"{desc.name}={value!r}: недопустимое значение"
        return f"{desc.name}={value!r}: допустимо значение {_bound_text(desc)}"

    def validate_dependencies(self, obj: object) -> None:
        """Проверяет правила, связывающие несколько полей (``@validator``).

        Raises:
            ValueError: Если правило нарушено (текст задаёт сам валидатор).
        """
        for check in self._validators.get(type(obj), self._inherited(type(obj))):
            check(obj)

    def _inherited(self, cls: type) -> tuple[Callable[[Any], None], ...]:
        for klass in cls.__mro__:
            if klass in self._validators:
                return self._validators[klass]
        return ()

    # --------------------------------------------------------- сериализация

    def serialize(self, obj: object) -> dict[str, Any]:
        """Словарь сцены: поля с ``serialize=True`` в порядке объявления.

        Поля с ``omit_if_default`` пропускаются, пока равны умолчанию.
        """
        data: dict[str, Any] = {}
        for desc in self.fields(type(obj)):
            if not desc.serialize:
                continue
            plain = _to_plain(getattr(obj, desc.name))
            if desc.omit_if_default and plain == _to_plain(desc.default):
                continue
            data[desc.name] = plain
        return data

    def deserialize(self, cls: type, data: dict[str, Any]) -> Any:
        """Создаёт компонент из словаря сцены.

        Отсутствующее и ``null`` поле даёт умолчание; значения приводятся к
        типу поля и зажимаются политикой дескриптора. Затем проверяются
        зависимые поля.

        Args:
            cls: Класс компонента (конструктор принимает поля как kwargs).
            data: Словарь из сцены.

        Returns:
            Новый компонент.
        """
        kwargs: dict[str, Any] = {}
        for desc in self.fields(cls):
            if not desc.serialize:
                continue
            raw = data.get(desc.name)
            if raw is None:
                kwargs[desc.name] = desc.default
                continue
            kwargs[desc.name] = self.clamp(cls, desc.name, _coerce(desc, raw))
        obj = cls(**kwargs)
        self.validate_dependencies(obj)
        return obj


def to_plain(value: Any) -> Any:
    """Значение поля в JSON-совместимый вид (публичная обёртка)."""
    return _to_plain(value)


def coerce_value(desc: FieldDescriptor, value: Any) -> Any:
    """Привести прочитанное значение к типу поля (публичная обёртка).

    Raises:
        ValueError: Если значение не приводится.
        TypeError: Если значение неподходящего типа.
    """
    return _coerce(desc, value)


_MISSING = object()


def _make_init(cls: type, descriptors: tuple[FieldDescriptor, ...]) -> Callable:
    """Конструктор по умолчанию: поля kwargs -> атрибуты, затем проверки."""
    defaults = {d.name: d.default for d in descriptors}
    base_init = next(
        (k.__init__ for k in cls.__mro__[1:] if "__init__" in k.__dict__), None
    )

    def __init__(self, **kwargs: Any) -> None:
        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise TypeError(f"{cls.__name__}: неизвестные поля {sorted(unknown)}")
        if base_init is not None and base_init is not object.__init__:
            base_init(self)
        for name, default in defaults.items():
            setattr(self, name, kwargs.get(name, default))
        registry.validate_dependencies(self)

    return __init__


registry = FieldRegistry()


def component_fields(cls: type[_T]) -> type[_T]:
    """Декоратор класса: регистрирует поля в общем :data:`registry`."""
    return registry.register(cls)  # type: ignore[return-value]


def evaluate_visible_if(expression: str | None, get: Callable[[str], Any]) -> bool:
    """Безопасно вычисляет условие видимости поля.

    Поддерживается ``имя ==/!= литерал`` и ``and`` между такими сравнениями;
    ``eval`` не используется.

    Args:
        expression: Условие (``None`` или пустое - всегда видно).
        get: Читает значение поля по имени.

    Returns:
        ``True``, если поле нужно показывать.

    Raises:
        ValueError: Если выражение не из поддерживаемого набора.
    """
    if not expression or not expression.strip():
        return True
    for part in expression.split(" and "):
        tokens = part.split()
        if len(tokens) < 3 or tokens[1] not in ("==", "!="):
            raise ValueError(f"visible_if: неподдерживаемое выражение {part!r}")
        literal_text = " ".join(tokens[2:])
        try:
            literal = ast.literal_eval(literal_text)
        except (ValueError, SyntaxError):
            literal = literal_text
        equal = get(tokens[0]) == literal
        if equal != (tokens[1] == "=="):
            return False
    return True
