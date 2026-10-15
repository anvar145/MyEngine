"""Контекст сохранения и глобальное состояние игры.

``SaveContext`` - словарь с типизированными значениями и вложенными ключами
``"a.b.c"``. Допустимы ``None``, ``bool``, ``int``, ``float``, ``str``,
``list``, ``dict`` (ключи - строки без ``.`` и не начинающиеся с ``$``),
``tuple`` и :class:`~engine.core.math3d.Vector3`. Остальное отвергается при
``set`` - ошибка видна там, где значение положили, а не при записи файла.

``GameState`` - обёртка над ``SaveContext`` с подпиской на изменения; живёт в
``Scene`` и целиком попадает в сохранение (без схемы).
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterator
from typing import Any

from engine.core.math3d import Vector3

__all__ = ["GameState", "SaveContext", "decode_value", "encode_value"]

_log = logging.getLogger("engine.core.save")
_MISSING = object()
_TAG = "$t"


def _check_key(key: str) -> list[str]:
    """Разобрать ключ ``"a.b.c"`` на части.

    Raises:
        TypeError: Ключ не строка.
        ValueError: Пустой ключ или пустая часть, либо часть начинается с ``$``.
    """
    if not isinstance(key, str):
        raise TypeError(f"ключ должен быть строкой, получено {type(key).__name__}")
    parts = key.split(".")
    if not key or any(not p for p in parts):
        raise ValueError(f"некорректный ключ сохранения: {key!r}")
    if any(p.startswith("$") for p in parts):
        raise ValueError(f"ключ не может начинаться с '$': {key!r}")
    return parts


def _validate(value: Any, path: str) -> Any:
    """Проверить значение и вернуть его независимую копию.

    Raises:
        TypeError: Тип значения не поддерживается.
        ValueError: Некорректный ключ вложенного словаря.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Vector3):
        return Vector3(value.x, value.y, value.z)
    if isinstance(value, (list, tuple)):
        items = [_validate(v, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str) or not k or "." in k or k.startswith("$"):
                raise ValueError(f"недопустимый ключ словаря в {path!r}: {k!r}")
            result[k] = _validate(v, f"{path}.{k}")
        return result
    raise TypeError(f"значение {path!r} типа {type(value).__name__} нельзя сохранить")


def encode_value(value: Any) -> Any:
    """Значение -> JSON-совместимый вид (``tuple`` и ``Vector3`` с меткой типа)."""
    if isinstance(value, Vector3):
        return {_TAG: "vec3", "v": [value.x, value.y, value.z]}
    if isinstance(value, tuple):
        return {_TAG: "tuple", "v": [encode_value(v) for v in value]}
    if isinstance(value, list):
        return [encode_value(v) for v in value]
    if isinstance(value, dict):
        return {k: encode_value(v) for k, v in value.items()}
    return value


def decode_value(raw: Any) -> Any:
    """JSON-вид -> значение (обратное к :func:`encode_value`)."""
    if isinstance(raw, list):
        return [decode_value(v) for v in raw]
    if isinstance(raw, dict):
        tag = raw.get(_TAG)
        if tag == "vec3":
            x, y, z = raw["v"]
            return Vector3(float(x), float(y), float(z))
        if tag == "tuple":
            return tuple(decode_value(v) for v in raw["v"])
        return {k: decode_value(v) for k, v in raw.items()}
    return raw


class SaveContext:
    """Типизированное хранилище ключ-значение с вложенными ключами.

    Example:
        >>> ctx = SaveContext()
        >>> ctx.set("player.hp", 80)
        >>> ctx.get("player.hp")
        80
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        if data:
            for key, value in data.items():
                self.set(key, value)

    # ------------------------------------------------------------- доступ
    def set(self, key: str, value: Any) -> None:
        """Записать значение; вложенные словари создаются по пути ключа.

        Raises:
            TypeError: Тип значения не поддерживается или ключ не строка.
            ValueError: Некорректный ключ, либо промежуточный узел - не словарь.
        """
        parts = _check_key(key)
        checked = _validate(value, key)
        node = self._data
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            elif not isinstance(child, dict):
                raise ValueError(f"ключ {key!r}: {part!r} уже хранит не словарь")
            node = child
        node[parts[-1]] = checked

    def get(self, key: str, default: Any = None) -> Any:
        """Значение по ключу или ``default``, если пути нет (возвращается копия)."""
        node: Any = self._data
        for part in _check_key(key):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return copy.deepcopy(node)

    def get_typed(self, key: str, expected: type, default: Any = _MISSING) -> Any:
        """Значение заданного типа.

        ``int`` принимается как ``float`` (приводится); ``bool`` не считается
        ``int``.

        Raises:
            ValueError: Ключа нет (и нет ``default``) либо тип не совпал.
        """
        value = self.get(key, _MISSING)
        if value is _MISSING:
            if default is _MISSING:
                raise ValueError(f"ключ {key!r} не найден")
            return default
        is_int = isinstance(value, int) and not isinstance(value, bool)
        if expected is float and is_int:
            return float(value)
        if expected is int and isinstance(value, bool):
            ok = False
        else:
            ok = isinstance(value, expected)
        if not ok:
            raise ValueError(
                f"ключ {key!r}: ожидался {expected.__name__}, "
                f"в сохранении {type(value).__name__}"
            )
        return value

    def has(self, key: str) -> bool:
        """Есть ли значение по ключу."""
        return self.get(key, _MISSING) is not _MISSING

    def remove(self, key: str) -> bool:
        """Удалить ключ; ``True``, если он был."""
        parts = _check_key(key)
        node: Any = self._data
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        if isinstance(node, dict) and parts[-1] in node:
            del node[parts[-1]]
            return True
        return False

    def clear(self) -> None:
        """Удалить всё."""
        self._data.clear()

    def keys(self) -> Iterator[str]:
        """Ключи верхнего уровня."""
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SaveContext) and self._data == other._data

    __hash__ = None  # type: ignore[assignment]

    # ------------------------------------------------------- сериализация
    def to_dict(self) -> dict[str, Any]:
        """JSON-совместимый словарь (детерминированный порядок ключей)."""
        return {k: encode_value(self._data[k]) for k in sorted(self._data)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SaveContext:
        """Восстановить из :meth:`to_dict`."""
        ctx = cls()
        for key, value in raw.items():
            ctx._data[key] = _validate(decode_value(value), key)
        return ctx


class GameState:
    """Глобальное состояние игры сцены: ``SaveContext`` + подписки.

    Подписка на ключ срабатывает при записи и удалении самого ключа, его
    родителя и потомков (``on_changed("quest")`` слышит ``quest.stage``).
    """

    __slots__ = ("_ctx", "_subs")

    def __init__(self) -> None:
        self._ctx = SaveContext()
        self._subs: list[tuple[str, Callable[[str, Any], None]]] = []

    @property
    def context(self) -> SaveContext:
        """Нижележащий контекст (для сериализации)."""
        return self._ctx

    def set(self, key: str, value: Any) -> None:
        """Записать значение и уведомить подписчиков, если оно изменилось."""
        old = self._ctx.get(key, _MISSING)
        self._ctx.set(key, value)
        new = self._ctx.get(key)
        if old is _MISSING or old != new:
            self._notify(key, new)

    def get(self, key: str, default: Any = None) -> Any:
        """Значение по ключу или ``default``."""
        return self._ctx.get(key, default)

    def get_typed(self, key: str, expected: type, default: Any = _MISSING) -> Any:
        """См. :meth:`SaveContext.get_typed`."""
        return self._ctx.get_typed(key, expected, default)

    def has(self, key: str) -> bool:
        """Есть ли значение по ключу."""
        return self._ctx.has(key)

    def remove(self, key: str) -> bool:
        """Удалить ключ (подписчики получают ``None``)."""
        removed = self._ctx.remove(key)
        if removed:
            self._notify(key, None)
        return removed

    def on_changed(
        self, key: str, callback: Callable[[str, Any], None]
    ) -> Callable[[], None]:
        """Подписаться на изменение ключа.

        Args:
            key: Ключ (``"a.b"``); слышны также родитель и потомки.
            callback: ``callback(изменённый_ключ, новое_значение)``.

        Returns:
            Функция отписки.
        """
        _check_key(key)
        entry = (key, callback)
        self._subs.append(entry)

        def unsubscribe() -> None:
            if entry in self._subs:
                self._subs.remove(entry)

        return unsubscribe

    def to_dict(self) -> dict[str, Any]:
        """JSON-совместимый вид всего состояния."""
        return self._ctx.to_dict()

    def load_dict(self, raw: dict[str, Any]) -> None:
        """Заменить состояние содержимым ``raw`` и уведомить о каждом ключе."""
        old_keys = set(self._ctx.keys())
        self._ctx = SaveContext.from_dict(raw)
        for key in sorted(old_keys | set(self._ctx.keys())):
            self._notify(key, self._ctx.get(key))

    def clear(self) -> None:
        """Очистить состояние (подписки остаются)."""
        self.load_dict({})

    def _notify(self, key: str, value: Any) -> None:
        for sub, callback in list(self._subs):
            if key == sub or key.startswith(sub + ".") or sub.startswith(key + "."):
                try:
                    callback(key, value)
                except Exception:  # noqa: BLE001 - подписчик не ломает игру
                    _log.exception("Исключение в подписчике GameState %r", key)
