"""descriptors_from_behaviour: дескрипторы из класса скрипта без регистрации."""

from __future__ import annotations

import math

import pytest

from engine.core.reflection import FieldKind, field, registry
from engine.scripting.fields import (
    Asset,
    descriptors_from_behaviour,
    scan_fields,
)


class Base:
    health: int = 100
    _private: int = 5


class Hero(Base):
    speed: float = field(3.0, range=(0.0, math.inf), tooltip="м/с")
    name: str = "hero"
    alive: bool = True
    target: GameObject = None  # noqa: F821
    icon: Asset = None
    mode: str = field("a", kind=FieldKind.ENUM, choices=("a", "b"))
    health: int = 150  # переопределение умолчания базы
    start: tuple = (0.0, 1.0, 2.0)

    def on_update(self, delta_time):
        pass


def test_order_kinds_and_defaults() -> None:
    descs = descriptors_from_behaviour(Hero)
    assert [d.name for d in descs] == [
        "health", "speed", "name", "alive", "target", "icon", "mode", "start",
    ]
    by = {d.name: d for d in descs}
    assert by["speed"].kind is FieldKind.FLOAT and by["speed"].tooltip == "м/с"
    assert by["health"].kind is FieldKind.INT and by["health"].default == 150
    assert by["target"].kind is FieldKind.OBJECT_REF
    assert by["icon"].kind is FieldKind.ASSET_REF
    assert by["mode"].choices == ("a", "b")
    assert by["start"].kind is FieldKind.VEC3


def test_class_is_not_registered_or_mutated() -> None:
    descriptors_from_behaviour(Hero)
    assert registry.fields(Hero) == ()
    assert "__fields__" not in vars(Hero) and "speed" in vars(Hero)
    assert Hero.name == "hero"


def test_missing_default_raises() -> None:
    class Bad:
        x: int

    with pytest.raises(TypeError, match="нет значения"):
        descriptors_from_behaviour(Bad)


def test_scan_fields_unchanged() -> None:
    src = "speed = 3.0\ntarget: GameObject = None\n_p = 1\n"
    assert [(f.name, f.kind) for f in scan_fields(src)] == [
        ("speed", "float"), ("target", "object"),
    ]
