"""Схема настроек импорта: описание полей и проверка значений.

Схема нужна автогенерируемому интерфейсу (инспектор настроек импорта) и
валидации перед записью в ``.meta``. Импортёр может определить свой метод
``settings_schema() -> list[SettingSpec]``; иначе схема выводится из типов
значений ``default_settings()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from engine.core.resources.errors import AssetImportError
from engine.core.resources.importer import AssetImporter

__all__ = ["SettingSpec", "settings_schema", "validate_settings"]

_KINDS = ("bool", "int", "float", "str", "choice", "list", "any")


@dataclass(frozen=True, slots=True)
class SettingSpec:
    """Описание одной настройки импорта.

    Attributes:
        name: Ключ в ``.meta``.
        kind: ``bool``/``int``/``float``/``str``/``choice``/``list``/``any``.
        default: Значение по умолчанию.
        label: Подпись для интерфейса (по умолчанию — имя).
        minimum: Нижняя граница числа (``None`` — без границы).
        maximum: Верхняя граница числа (``None`` — без границы).
        choices: Допустимые значения для ``choice``.
    """

    name: str
    kind: str
    default: Any = None
    label: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"Неизвестный вид настройки '{self.kind}'")


def settings_schema(importer: AssetImporter) -> list[SettingSpec]:
    """Схема настроек импортёра.

    Args:
        importer: Импортёр формата.

    Returns:
        Свой ``settings_schema()`` импортёра или схема, выведенная из умолчаний.
    """
    custom = getattr(importer, "settings_schema", None)
    if callable(custom):
        return list(custom())
    return [
        SettingSpec(name, _kind_of(value), value, name)
        for name, value in importer.default_settings().items()
    ]


def validate_settings(
    importer: AssetImporter, settings: Mapping[str, Any]
) -> dict[str, Any]:
    """Проверить настройки по схеме и оставить только отличные от умолчаний.

    Args:
        importer: Импортёр, чьи настройки проверяются.
        settings: Предложенные значения (можно частично).

    Returns:
        Нормализованные значения, отличающиеся от умолчаний (для ``.meta``).

    Raises:
        AssetImportError: Неизвестный ключ, неверный тип или выход за границы.
    """
    schema = {spec.name: spec for spec in settings_schema(importer)}
    result: dict[str, Any] = {}
    for name, value in settings.items():
        spec = schema.get(name)
        if spec is None:
            known = ", ".join(sorted(schema)) or "нет"
            raise AssetImportError(
                f"Неизвестная настройка '{name}' импортёра '{importer.name}'. "
                f"Доступно: {known}"
            )
        checked = _check(spec, value)
        if checked != spec.default:
            result[name] = checked
    return result


def _kind_of(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)):
        return "list"
    return "any"


def _check(spec: SettingSpec, value: Any) -> Any:
    kind = spec.kind
    name = spec.name
    if kind == "bool":
        if not isinstance(value, bool):
            raise AssetImportError(f"Настройка '{name}' должна быть bool")
        return value
    if kind in ("int", "float"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AssetImportError(f"Настройка '{name}' должна быть числом")
        if kind == "int":
            if int(value) != value:
                raise AssetImportError(f"Настройка '{name}' должна быть целой")
            value = int(value)
        else:
            value = float(value)
        if spec.minimum is not None and value < spec.minimum:
            raise AssetImportError(f"Настройка '{name}' меньше {spec.minimum}")
        if spec.maximum is not None and value > spec.maximum:
            raise AssetImportError(f"Настройка '{name}' больше {spec.maximum}")
        return value
    if kind == "str":
        if not isinstance(value, str):
            raise AssetImportError(f"Настройка '{name}' должна быть строкой")
        return value
    if kind == "choice":
        if value not in spec.choices:
            raise AssetImportError(
                f"Настройка '{name}': {value!r} не из {list(spec.choices)}"
            )
        return value
    if kind == "list":
        if not isinstance(value, (list, tuple)):
            raise AssetImportError(f"Настройка '{name}' должна быть списком")
        return list(value)
    return value
