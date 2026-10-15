"""Поля скрипта: публичные глобальные переменные, настраиваемые в инспекторе.

Поля находятся разбором исходника (``ast``), скрипт при этом не исполняется:

* ``speed = 3.0``, ``count = 5``, ``label = "hi"``, ``on = True`` — поля
  типов ``float`` / ``int`` / ``str`` / ``bool`` со значением из кода по умолчанию;
* ``target: GameObject = None`` — ссылка на объект сцены;
* ``icon: Asset = None`` — ссылка на файл проекта.

Всё остальное (функции, списки, вычисляемые значения) полем не является.
Имена, начинающиеся с ``_``, приватные и в инспекторе не показываются.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from engine.core.reflection import FieldDescriptor, describe_class_fields
from engine.core.resources.asset_ref import AssetRef

__all__ = [
    "Asset",
    "Prefab",
    "FieldKind",
    "ScriptField",
    "coerce_field",
    "descriptors_from_behaviour",
    "scan_fields",
]

#: Тип поля: ``bool`` / ``int`` / ``float`` / ``str`` / ``object`` / ``asset``.
FieldKind = str

_PRIMITIVES = {"bool": bool, "int": int, "float": float, "str": str}


def Asset(path: str) -> AssetRef:  # noqa: N802 - имя типа для аннотаций
    """Ссылка на файл проекта в поле скрипта - :class:`AssetRef` по пути.

    Имя ``Asset`` доступно скрипту без импорта и служит аннотацией:
    ``icon: Asset = None``. Значение поля, которое видит скрипт, - ``AssetRef``.
    """
    return AssetRef(path=path)


def Prefab(path: str) -> AssetRef:  # noqa: N802 - имя типа для аннотаций
    """Ссылка на файл префаба (``*.prefab.json``): тот же :class:`AssetRef`.

    ``Scene.instantiate`` и ``ObjectPool`` принимают его. Аннотация
    ``bullet: Prefab = None`` в инспекторе допускает только файлы префабов.
    """
    return AssetRef(path=path)


#: Вид ассета для ``ScriptField.asset_kind`` / ``FieldDescriptor.asset_kind``.
PREFAB_KIND = "prefab"
PREFAB_SUFFIX = ".prefab.json"


@dataclass(frozen=True)
class ScriptField:
    """Описание поля скрипта.

    Attributes:
        name: Имя глобальной переменной.
        kind: Тип поля (см. :data:`FieldKind`).
        default: Значение из кода (для ссылок — ``None``).
        range: Границы числа (только поля ``export`` у ``Behaviour``).
        tooltip: Подсказка поля (там же).
        step: Шаг правки числа (там же).
        choices: Допустимые значения перечисления (там же).
        asset_kind: Вид ассета (``"prefab"``) у ссылок на файл; иначе ``None``.
    """

    name: str
    kind: FieldKind
    default: Any = None
    range: tuple[float, float] | None = None
    tooltip: str | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    asset_kind: str | None = None


def scan_fields(source: str) -> list[ScriptField]:
    """Найти поля скрипта в исходнике.

    Args:
        source: Текст скрипта.

    Returns:
        Поля в порядке объявления; при повторном присваивании одного имени
        побеждает последнее. Пустой список при синтаксической ошибке.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    found: dict[str, ScriptField] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name, annotation, value = node.targets[0].id, None, node.value
        elif isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name) or node.value is None:
                continue
            name, annotation, value = node.target.id, node.annotation, node.value
        else:
            continue
        if name.startswith("_"):
            continue
        field = _make_field(name, annotation, value)
        if field is None:
            found.pop(name, None)
        else:
            found[name] = field
    return list(found.values())


def _annotation_name(annotation: ast.expr | None) -> str:
    """Имя аннотации (``int``, ``Asset``, ``"GameObject"``) или пустая строка."""
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    return ""


def _make_field(
    name: str, annotation: ast.expr | None, value: ast.expr
) -> ScriptField | None:
    """Собрать поле из присваивания или ``None``, если это не поле."""
    hint = _annotation_name(annotation)
    if hint in ("GameObject", "Asset", "Prefab"):
        if isinstance(value, ast.Constant) and value.value is None:
            if hint == "GameObject":
                return ScriptField(name, "object")
            return ScriptField(
                name, "asset", asset_kind=PREFAB_KIND if hint == "Prefab" else None
            )
        return None
    try:
        literal = ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    kind = _kind_of(literal)
    if kind is None:
        return None
    if hint in _PRIMITIVES and hint != kind:
        coerced = coerce_field(hint, literal)
        if coerced is None:
            return None
        kind, literal = hint, coerced
    return ScriptField(name, kind, literal)


def _kind_of(value: object) -> FieldKind | None:
    """Тип поля по литералу (``bool`` раньше ``int``: это его подкласс)."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return None


def coerce_field(kind: FieldKind, value: object) -> Any:
    """Привести значение к типу поля.

    Args:
        kind: Тип поля.
        value: Значение (из сцены или из инспектора).

    Returns:
        Значение нужного типа или ``None``, если оно не подходит. Допустимо
        только расширение ``int`` → ``float``; ``True`` — не число, строка не
        превращается в число. Для ссылок ``None`` — «не задано» и тоже
        возвращается как ``None``.
    """
    if kind == "bool":
        return value if isinstance(value, bool) else None
    if kind == "int":
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)
    if kind == "str":
        return str(value) if type(value) is str else None
    if kind == "asset":
        try:
            return AssetRef.coerce(value)
        except ValueError:
            return None
    if kind == "object":
        from engine.core.game_object import GameObject

        return value if isinstance(value, GameObject) else None
    return None


def descriptors_from_behaviour(cls: type) -> tuple[FieldDescriptor, ...]:
    """Дескрипторы полей класса-скрипта (заготовка под ``Behaviour``).

    Поля объявляются аннотациями ``имя: тип = умолчание`` или ``field(...)``;
    ``GameObject`` и ``Asset`` дают ссылки. Класс не регистрируется и не
    меняется. Поля баз идут первыми, переопределение в наследнике сохраняет
    позицию поля базы.

    Args:
        cls: Класс скрипта.

    Returns:
        Дескрипторы в порядке объявления.

    Raises:
        TypeError: Если у поля нет умолчания.
    """
    if hasattr(cls, "__behaviour_fields__"):
        return tuple(cls.__behaviour_fields__)  # класс Behaviour: уже собрано
    merged: dict[str, FieldDescriptor] = {}
    for klass in reversed(cls.__mro__):
        if klass is not object:
            merged.update(describe_class_fields(klass))
    return tuple(merged.values())
