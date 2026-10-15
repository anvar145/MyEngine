"""Острова: связные компоненты графа «динамические тела — контакты».

Статические и кинематические тела острова не объединяют (иначе весь мир с полом
стал бы одним островом): контакт с ними лишь добавляется к острову динамического
тела.
"""

from __future__ import annotations

from collections.abc import Sequence


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # меньший индекс становится корнем: порядок детерминирован
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def build_islands(
    body_count: int,
    pairs: Sequence[tuple[int, int]],
    dynamic: Sequence[bool],
) -> list[tuple[list[int], list[int]]]:
    """Строит острова.

    Args:
        body_count: Число тел (индексы 0..body_count-1).
        pairs: Пары тел, у каждого контакта (``pairs[i]`` — индексы тел контакта).
        dynamic: Признак «динамическое тело» для каждого индекса.

    Returns:
        Список ``(индексы_тел, индексы_контактов)``, отсортированный по
        минимальному индексу тела. Острова состоят из динамических тел; контакты
        динамика со статикой входят в остров динамика.
    """
    uf = _UnionFind(body_count)
    for a, b in pairs:
        if dynamic[a] and dynamic[b]:
            uf.union(a, b)
    bodies: dict[int, list[int]] = {}
    for i in range(body_count):
        if dynamic[i]:
            bodies.setdefault(uf.find(i), []).append(i)
    contacts: dict[int, list[int]] = {root: [] for root in bodies}
    for k, (a, b) in enumerate(pairs):
        if dynamic[a]:
            contacts[uf.find(a)].append(k)
        elif dynamic[b]:
            contacts[uf.find(b)].append(k)
    return [(bodies[root], contacts[root]) for root in sorted(bodies)]
