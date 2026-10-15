"""Базовый класс Behaviour: хуки, прокси к движку, поля export."""

from __future__ import annotations

import pytest

from engine.core.reflection import FieldKind, registry
from engine.scripting.behaviour import HOOK_NAMES, Behaviour, export
from engine.scripting.fields import descriptors_from_behaviour


class _Owner:
    """Заглушка компонента: то, что Behaviour берёт у ScriptComponent."""

    game_object = "go"
    scene = "scene"
    transform = "tr"
    input = "in"
    time = "time"
    physics = "phys"
    secret = "не из API"

    def __init__(self) -> None:
        self.messages: list[tuple] = []

    def log(self, *values: object) -> None:
        self.messages.append(values)


def _make(cls: type[Behaviour]) -> Behaviour:
    instance = cls()
    instance._attach(_Owner())
    return instance


# ---------------------------------------------------------------- хуки


def test_hook_names_cover_lifecycle() -> None:
    assert set(HOOK_NAMES) == {
        "on_attach", "on_enable", "on_disable", "on_start", "on_update",
        "on_fixed_update", "on_late_update", "on_destroy", "on_reload",
        "on_fixed_update_batch", "on_spawn", "on_despawn", "on_save", "on_load",
    }


def test_overridden_hooks_detected() -> None:
    class Spinner(Behaviour):
        def on_start(self) -> None: ...
        def on_update(self, delta_time: float) -> None: ...

    assert isinstance(Spinner._overridden_hooks, frozenset)
    assert Spinner._overridden_hooks == {"on_start", "on_update"}
    assert Behaviour._overridden_hooks == frozenset()


def test_overridden_hooks_include_ancestors() -> None:
    class Base(Behaviour):
        def on_update(self, delta_time: float) -> None: ...

    class Child(Base):
        def on_destroy(self) -> None: ...

    assert Child._overridden_hooks == {"on_update", "on_destroy"}
    assert Base._overridden_hooks == {"on_update"}


def test_batch_hook_detected_as_classmethod() -> None:
    class Patrol(Behaviour):
        @classmethod
        def on_fixed_update_batch(cls, items, dt) -> None: ...

    assert Patrol._overridden_hooks == {"on_fixed_update_batch"}


def test_empty_base_hooks_are_not_called() -> None:
    calls: list[str] = []

    class Counted(Behaviour):
        def on_update(self, delta_time: float) -> None:
            calls.append("update")

    instance = _make(Counted)
    instance._dispatch("on_start")
    instance._dispatch("on_update", 0.016)
    instance._dispatch("on_late_update", 0.016)
    instance._dispatch("on_destroy")
    assert calls == ["update"]  # остальные хуки не переопределены


def test_dispatch_skips_hook_even_if_called_directly_on_base() -> None:
    class Plain(Behaviour):
        pass

    assert _make(Plain)._dispatch("on_update", 0.1) is None


def test_lifecycle_order() -> None:
    order: list[str] = []

    class Tracked(Behaviour):
        def on_attach(self) -> None:
            order.append("attach")

        def on_enable(self) -> None:
            order.append("enable")

        def on_start(self) -> None:
            order.append("start")

        def on_update(self, delta_time: float) -> None:
            order.append("update")

        def on_destroy(self) -> None:
            order.append("destroy")

    instance = _make(Tracked)  # _attach вызывает on_attach
    for hook in ("on_enable", "on_start"):
        instance._dispatch(hook)
    instance._dispatch("on_update", 0.016)
    instance._dispatch("on_update", 0.016)
    instance._dispatch("on_destroy")
    assert order == ["attach", "enable", "start", "update", "update", "destroy"]


def test_batch_hook_called_once_with_all_items() -> None:
    seen: list[tuple] = []

    class Patrol(Behaviour):
        @classmethod
        def on_fixed_update_batch(cls, items, dt) -> None:
            seen.append((cls, list(items), dt))

    items = [_make(Patrol), _make(Patrol), _make(Patrol)]
    Patrol.on_fixed_update_batch(items, 0.02)
    assert len(seen) == 1
    assert seen[0] == (Patrol, items, 0.02)


# ------------------------------------------------------- прокси к движку


def test_api_proxy_reads_component() -> None:
    instance = _make(Behaviour)
    for name in ("game_object", "scene", "transform", "input", "time", "physics"):
        assert getattr(instance, name) == getattr(_Owner, name)
    instance.log("привет", 1)
    assert instance._component.messages == [("привет", 1)]


def test_api_names_is_public_tuple() -> None:
    assert Behaviour.API_NAMES[:9] == (
        "game_object", "scene", "transform", "input", "time", "physics", "log",
        "get_component", "get_components",
    )
    assert {"after", "every", "start_coroutine", "tween", "cancel_all"} <= set(
        Behaviour.API_NAMES
    )


def test_unknown_attribute_is_not_proxied() -> None:
    instance = _make(Behaviour)
    with pytest.raises(AttributeError):
        instance.secret  # noqa: B018 - есть у заглушки, но не в API
    with pytest.raises(AttributeError):
        instance.spped  # noqa: B018 - опечатка в имени поля


def test_proxy_before_attach_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        Behaviour().game_object  # noqa: B018


# --------------------------------------------------------------- export


def test_export_fields_are_instance_attributes() -> None:
    class Spinner(Behaviour):
        speed: float = export(90.0, range=(0, 360), tooltip="град/с")
        label: str = export("abc")

    assert Spinner().speed == 90.0
    instance = _make(Spinner)
    assert (instance.speed, instance.label) == (90.0, "abc")
    instance.speed = 10.0
    assert instance.speed == 10.0
    assert Spinner().speed == 90.0  # умолчание класса не затронуто


def test_export_descriptors_in_declaration_order() -> None:
    class Spinner(Behaviour):
        speed: float = export(90.0, range=(0, 360), tooltip="град/с")
        enabled_flag: bool = export(True)
        target: object = export(None, kind=FieldKind.OBJECT_REF)

    descs = descriptors_from_behaviour(Spinner)
    assert [d.name for d in descs] == ["speed", "enabled_flag", "target"]
    assert descs[0].kind is FieldKind.FLOAT
    assert descs[0].range == (0, 360) and descs[0].tooltip == "град/с"
    assert descs[1].kind is FieldKind.BOOL


def test_behaviour_is_not_in_global_registry() -> None:
    class Spinner(Behaviour):
        speed: float = export(1.0)

    assert registry.fields(Spinner) == ()


def test_mutable_default_is_copied_per_instance() -> None:
    class Bag(Behaviour):
        items: list = export([])

    first, second = Bag(), Bag()
    first.items.append(1)
    assert second.items == []
