"""Миграции схемы сохранений.

Миграция - функция ``fn(payload) -> payload`` над словарём
``{"meta": {...}, "state": {...}, "objects": {...}}`` в JSON-виде; она
переводит данные из версии ``from`` в ``to``. Реестр - ``MIGRATIONS``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from itertools import pairwise
from typing import Any

from engine.core.save.slot import SaveVersionError

__all__ = ["MIGRATIONS", "Migration", "migrate_chain", "register_migration"]

Migration = Callable[[dict[str, Any]], dict[str, Any]]

#: ``(from_version, to_version) -> миграция``.
MIGRATIONS: dict[tuple[int, int], Migration] = {}


def register_migration(
    from_version: int,
    to_version: int,
    registry: dict[tuple[int, int], Migration] | None = None,
) -> Callable[[Migration], Migration]:
    """Декоратор: зарегистрировать миграцию ``from_version -> to_version``.

    Args:
        from_version: Исходная версия.
        to_version: Целевая версия (строго больше исходной).
        registry: Свой реестр вместо глобального ``MIGRATIONS``.

    Raises:
        ValueError: ``to_version <= from_version`` или пара уже занята.
    """
    target = MIGRATIONS if registry is None else registry
    if to_version <= from_version:
        raise ValueError("миграция должна повышать версию")

    def decorator(func: Migration) -> Migration:
        key = (int(from_version), int(to_version))
        if key in target:
            raise ValueError(f"миграция {key[0]} -> {key[1]} уже зарегистрирована")
        target[key] = func
        return func

    return decorator


def migrate_chain(
    from_v: int,
    to_v: int,
    registry: dict[tuple[int, int], Migration] | None = None,
) -> list[Migration]:
    """Кратчайшая цепочка миграций ``from_v -> to_v``.

    Args:
        from_v: Версия сохранения.
        to_v: Текущая версия игры.
        registry: Свой реестр вместо глобального ``MIGRATIONS``.

    Returns:
        Функции в порядке применения; пусто, если версии равны.

    Raises:
        SaveVersionError: Сохранение новее игры или цепочки нет.
    """
    source = MIGRATIONS if registry is None else registry
    if from_v == to_v:
        return []
    if from_v > to_v:
        raise SaveVersionError(
            f"Сохранение создано более новой версией игры (формат {from_v}), "
            f"эта версия понимает до {to_v}. Обновите игру."
        )
    edges: dict[int, list[int]] = {}
    for src, dst in source:
        edges.setdefault(src, []).append(dst)
    previous: dict[int, int | None] = {from_v: None}
    queue = deque([from_v])
    while queue:
        node = queue.popleft()
        if node == to_v:
            break
        for nxt in sorted(edges.get(node, ()), reverse=True):
            if nxt <= to_v and nxt not in previous:
                previous[nxt] = node
                queue.append(nxt)
    if to_v not in previous:
        raise SaveVersionError(f"Нет миграции сохранения с формата {from_v} на {to_v}.")
    path = [to_v]
    while (parent := previous[path[-1]]) is not None:
        path.append(parent)
    path.reverse()
    return [source[(a, b)] for a, b in pairwise(path)]
