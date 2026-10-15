"""Запросы: find_all_with_tag (индекс тегов), поиск компонентов в детях/родителе."""

from __future__ import annotations

from engine.core.component import Component
from engine.core.scene import Scene


class _A(Component):
    pass


class _B(Component):
    pass


def _tree() -> tuple[Scene, dict]:
    scene = Scene("s")
    names = ["root", "a", "b", "a1", "b1"]
    n = {name: scene.create_game_object(name) for name in names}
    n["a"].set_parent(n["root"])
    n["b"].set_parent(n["root"])
    n["a1"].set_parent(n["a"])
    n["b1"].set_parent(n["b"])
    return scene, n


class TestTags:
    def test_find_all_with_tag_active_only_in_order(self) -> None:
        scene, n = _tree()
        for name in ("a", "b1", "root"):
            n[name].tag = "X"
        assert [o.name for o in scene.find_all_with_tag("X")] == ["a", "b1", "root"]
        n["b"].active_self = False
        n["root"].active_self = False
        assert scene.find_all_with_tag("X") == []
        n["root"].active_self = True
        assert [o.name for o in scene.find_all_with_tag("X")] == ["a", "root"]

    def test_retag_and_destroy_update_index(self) -> None:
        scene, n = _tree()
        n["a"].tag = "X"
        n["a"].tag = "Y"
        assert scene.find_all_with_tag("X") == []
        assert scene.find_all_with_tag("Y") == [n["a"]]
        scene.destroy_game_object(n["a"], immediate=True)
        assert scene.find_all_with_tag("Y") == []
        assert scene.find_all_with_tag("nope") == []

    def test_tag_given_on_create(self) -> None:
        scene = Scene("s")
        obj = scene.create_game_object("e", tag="Enemy")
        assert scene.find_all_with_tag("Enemy") == [obj]


class TestComponentSearch:
    def test_in_children_dfs_order(self) -> None:
        _scene, n = _tree()
        comps = {k: n[k].add_component(_A()) for k in ("root", "a1", "b", "b1")}
        found = n["root"].get_components_in_children(_A)
        assert found == [comps["root"], comps["a1"], comps["b"], comps["b1"]]
        assert n["root"].get_component_in_children(_A) is comps["root"]
        assert n["a"].get_component_in_children(_A) is comps["a1"]
        assert n["a"].get_component_in_children(_B) is None

    def test_inactive_subtree_skipped_unless_included(self) -> None:
        _scene, n = _tree()
        hidden = n["a1"].add_component(_A())
        shown = n["b1"].add_component(_A())
        n["a"].active_self = False
        assert n["root"].get_components_in_children(_A) == [shown]
        assert n["root"].get_components_in_children(_A, include_inactive=True) == [
            hidden,
            shown,
        ]

    def test_in_parent(self) -> None:
        _scene, n = _tree()
        top = n["root"].add_component(_A())
        mid = n["a"].add_component(_A())
        assert n["a1"].get_component_in_parent(_A) is mid
        assert n["b1"].get_component_in_parent(_A) is top
        assert n["b1"].get_component_in_parent(_B) is None
