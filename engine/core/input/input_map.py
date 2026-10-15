"""Набор действий: загрузка/сохранение JSON, дефолты, ремаппинг.

Формат файла::

    {"version": 1, "actions": {"Jump": {"kind": "button",
        "bindings": [{"device": "keyboard", "code": "space"}]}}}

Дефолты лежат в ``assets/default_input_map.json``. Пользовательский файл
(``Settings/input.json``) накладывается поверх дефолтов: действие из файла
заменяет одноимённое целиком. ``save(path, only_changes=True)`` пишет только
отличия от дефолтов; ``load`` восстанавливает карту, идентичную исходной.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from engine.core.input.action import InputAction
from engine.core.input.binding import Binding

__all__ = ["DEFAULT_MAP_PATH", "InputMap", "default_input_map"]

_log = logging.getLogger("engine.input")

#: Файл дефолтного маппинга.
DEFAULT_MAP_PATH = (
    Path(__file__).resolve().parents[3] / "assets" / "default_input_map.json"
)

_VERSION = 1


class InputMap:
    """Именованные действия с привязками и дефолтами для сброса.

    Args:
        actions: Начальные действия.
        defaults: Действия для :meth:`reset_to_default`; по умолчанию - копия
            ``actions``.
    """

    def __init__(
        self,
        actions: Iterable[InputAction] = (),
        defaults: Iterable[InputAction] | None = None,
    ) -> None:
        self._actions: dict[str, InputAction] = {a.name: a.copy() for a in actions}
        source = self._actions.values() if defaults is None else defaults
        self._defaults: dict[str, InputAction] = {a.name: a.copy() for a in source}

    # ------------------------------------------------------------------ #
    # Доступ
    # ------------------------------------------------------------------ #
    @property
    def actions(self) -> dict[str, InputAction]:
        """Действия по имени (изменяемый словарь)."""
        return self._actions

    def names(self) -> list[str]:
        """Имена действий в порядке добавления."""
        return list(self._actions)

    def get(self, name: str) -> InputAction | None:
        """Действие или ``None``."""
        return self._actions.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._actions

    def __len__(self) -> int:
        return len(self._actions)

    def add(self, action: InputAction) -> None:
        """Добавить или заменить действие."""
        self._actions[action.name] = action

    def copy(self) -> InputMap:
        """Независимая копия вместе с дефолтами."""
        return InputMap(self._actions.values(), self._defaults.values())

    # ------------------------------------------------------------------ #
    # Ремаппинг
    # ------------------------------------------------------------------ #
    def rebind(self, action: str, binding_index: int, new_binding: Binding) -> None:
        """Заменить привязку; индекс, равный числу привязок, добавляет новую.

        Raises:
            KeyError: Нет такого действия.
            IndexError: Индекс вне диапазона.
        """
        target = self._require(action)
        count = len(target.bindings)
        if binding_index == count:
            target.bindings.append(new_binding)
        elif 0 <= binding_index < count:
            target.bindings[binding_index] = new_binding
        else:
            raise IndexError(f"{action}: нет привязки #{binding_index}")

    def remove_binding(self, action: str, binding_index: int) -> Binding:
        """Удалить привязку и вернуть её."""
        return self._require(action).bindings.pop(binding_index)

    def unbind(self, action: str, binding_index: int) -> Binding:
        """Синоним :meth:`remove_binding`."""
        return self.remove_binding(action, binding_index)

    def add_binding(self, action: str, binding: Binding) -> None:
        """Добавить привязку в конец списка действия.

        Raises:
            KeyError: Нет такого действия.
        """
        self._require(action).bindings.append(binding)

    def reset_to_default(self, action: str) -> None:
        """Вернуть привязки действия к дефолту.

        Raises:
            KeyError: У действия нет дефолта.
        """
        if action not in self._defaults:
            raise KeyError(f"У действия {action!r} нет значения по умолчанию")
        self._actions[action] = self._defaults[action].copy()

    def reset_all(self) -> None:
        """Вернуть все действия к дефолту (пользовательские без дефолта остаются)."""
        for name in self._defaults:
            self.reset_to_default(name)

    def is_default(self, action: str) -> bool:
        """Совпадают ли привязки с дефолтом."""
        current, base = self._actions.get(action), self._defaults.get(action)
        return current is not None and base is not None and current == base

    def _require(self, name: str) -> InputAction:
        try:
            return self._actions[name]
        except KeyError:
            raise KeyError(f"Нет действия {name!r}") from None

    # ------------------------------------------------------------------ #
    # JSON
    # ------------------------------------------------------------------ #
    def to_dict(self, only_changes: bool = False) -> dict[str, Any]:
        """Словарь формата файла.

        Args:
            only_changes: Писать только действия, отличающиеся от дефолта
                (и действия без дефолта).
        """
        return {
            "version": _VERSION,
            "actions": {
                n: a.to_dict()
                for n, a in self._actions.items()
                if not (only_changes and self.is_default(n))
            },
        }

    @staticmethod
    def actions_from_dict(data: dict[str, Any]) -> list[InputAction]:
        """Разобрать секцию ``actions``.

        Raises:
            ValueError: Формат неверен.
        """
        raw = data.get("actions") if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            raise ValueError("В файле маппинга нет секции 'actions'")
        return [InputAction.from_dict(str(n), d) for n, d in raw.items()]

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], defaults: InputMap | None = None
    ) -> InputMap:
        """Создать карту: дефолты + действия из ``data`` поверх них.

        Args:
            data: Содержимое файла.
            defaults: Источник дефолтов (по умолчанию - ``default_input_map()``).
        """
        base = defaults if defaults is not None else default_input_map()
        result = base.copy()
        for action in cls.actions_from_dict(data):
            result.add(action)
        return result

    def save(self, path: Path | str, only_changes: bool = False) -> None:
        """Записать JSON (каталог создаётся); ``only_changes`` - только отличия."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(only_changes), ensure_ascii=False, indent=2)
        target.write_text(text + "\n", encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: Path | str,
        defaults: InputMap | None = None,
        reader: Callable[[Path], str | None] | None = None,
    ) -> InputMap:
        """Прочитать JSON поверх дефолтов.

        Args:
            path: Файл маппинга.
            defaults: Базовый маппинг.
            reader: Чтение текста не с диска (из пака): ``путь -> текст | None``.

        Raises:
            OSError: Файл не читается.
            ValueError: Не JSON или неверный формат.
        """
        try:
            text = reader(Path(path)) if reader is not None else None
            data = json.loads(
                text if text is not None else Path(path).read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: некорректный JSON ({exc})") from exc
        return cls.from_dict(data, defaults)


_DEFAULT_CACHE: InputMap | None = None


def default_input_map() -> InputMap:
    """Копия дефолтного маппинга (файл читается один раз за процесс).

    Если файла нет или он повреждён - пустая карта и предупреждение в логе.
    """
    global _DEFAULT_CACHE  # noqa: PLW0603 - кэш разбора
    if _DEFAULT_CACHE is None:
        try:
            data = json.loads(DEFAULT_MAP_PATH.read_text(encoding="utf-8"))
            _DEFAULT_CACHE = InputMap(InputMap.actions_from_dict(data))
        except (OSError, ValueError) as exc:
            _log.warning("Дефолтный маппинг ввода не загружен: %s", exc)
            _DEFAULT_CACHE = InputMap()
    return _DEFAULT_CACHE.copy()
