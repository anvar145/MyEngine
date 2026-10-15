"""Тесты рефлексии полей (engine.core.reflection)."""

from __future__ import annotations

import json
import logging
import math

import pytest

from engine.core.component import Component
from engine.core.game_object import GameObject
from engine.core.reflection import (
    FieldKind,
    InvalidPolicy,
    component_fields,
    evaluate_visible_if,
    field,
    registry,
    validator,
)


@component_fields
class Probe(Component):
    """Компонент с полями разных видов."""

    name_tag: str = "a"
    count: int = 3
    flag: bool = True
    speed: float = field(4.5, range=(0.0, math.inf), step=0.5, tooltip="скорость")
    power: float = field(5.0, range=(0.0, math.inf), exclusive_min=True)
    air: float = field(0.3, range=(0.0, 1.0))
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tint: tuple[float, ...] = field((1.0, 1.0, 1.0, 1.0), kind=FieldKind.COLOR)
    rot: tuple[float, float, float, float] = field(
        (0.0, 0.0, 0.0, 1.0), kind=FieldKind.QUAT, omit_if_default=True
    )
    mode: str = field("one", kind=FieldKind.ENUM, choices=("one", "two"))
    mask: int = field(0xFFFFFFFF, kind=FieldKind.MASK32)
    points: list = field((), kind=FieldKind.LIST)
    target: object = field(None, kind=FieldKind.OBJECT_REF, ref_kind="GameObject")
    hidden: int = field(7, serialize=False)
    strict: float = field(
        1.0, range=(0.0, 10.0), on_invalid=InvalidPolicy.ERROR
    )
    nanonly: float = field(2.0, on_invalid=InvalidPolicy.DEFAULT_IF_NAN)


def test_registration_and_order() -> None:
    names = [d.name for d in registry.fields(Probe)]
    assert names == [
        "name_tag", "count", "flag", "speed", "power", "air", "pos", "tint",
        "rot", "mode", "mask", "points", "target", "hidden", "strict", "nanonly",
    ]
    assert registry.for_name(Probe, "speed").step == 0.5
    assert registry.for_name(Probe, "pos").kind is FieldKind.VEC3
    assert registry.for_name(Probe, "nope") is None
    assert Probe.__fields__ == registry.fields(Probe)


def test_serialize_only_serializable_fields() -> None:
    data = registry.serialize(Probe())
    assert "hidden" not in data
    assert "rot" not in data  # omit_if_default
    assert data["pos"] == [0.0, 0.0, 0.0]
    assert list(data)[:3] == ["name_tag", "count", "flag"]
    json.dumps(data)  # JSON-совместимо


def test_omit_if_default_written_when_changed() -> None:
    probe = Probe(rot=(0.0, 0.7, 0.0, 0.7))
    assert registry.serialize(probe)["rot"] == [0.0, 0.7, 0.0, 0.7]


def test_deserialize_missing_and_null_use_default() -> None:
    probe = registry.deserialize(Probe, {"count": None, "speed": 2})
    assert probe.count == 3 and probe.speed == 2.0
    assert probe.mode == "one" and probe.hidden == 7


def test_round_trip() -> None:
    src = Probe(count=9, pos=(1.0, 2.0, 3.0), mode="two", points=[(1.0, 2.0)])
    data = registry.serialize(src)
    back = registry.deserialize(Probe, json.loads(json.dumps(data)))
    assert registry.serialize(back) == data
    assert back.pos == (1.0, 2.0, 3.0)


def test_nan_becomes_default_and_range_clamps_with_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="engine.core.reflection"):
        probe = registry.deserialize(
            Probe, {"speed": float("nan"), "air": 5.0, "power": -1.0}
        )
    assert probe.speed == 4.5
    assert probe.air == 1.0
    assert probe.power == 1e-3  # открытый минимум
    assert "Probe: air=5.0" in caplog.text


def test_default_if_nan_leaves_other_values() -> None:
    assert registry.clamp(Probe, "nanonly", float("nan")) == 2.0
    assert registry.clamp(Probe, "nanonly", -50.0) == -50.0


def test_error_policy_and_strict() -> None:
    with pytest.raises(ValueError, match=r"strict=11.0"):
        registry.clamp(Probe, "strict", 11.0)
    with pytest.raises(ValueError, match=r"speed=-1.0: допустимо значение >= 0"):
        registry.clamp(Probe, "speed", -1.0, strict=True)
    with pytest.raises(ValueError, match=r"air=2.0.*<= 1"):
        registry.clamp(Probe, "air", 2.0, strict=True)
    with pytest.raises(ValueError, match=r"power=0.0: допустимо значение > 0"):
        registry.clamp(Probe, "power", 0.0, strict=True)


def test_valid_value_passes_unchanged_and_unknown_field() -> None:
    assert registry.clamp(Probe, "speed", 3) == 3
    assert registry.clamp(Probe, "unknown", "x") == "x"


def test_object_reference_serializes_as_id() -> None:
    owner = GameObject("t")
    owner._id = 42
    probe = Probe(target=owner)
    assert registry.serialize(probe)["target"] == 42


def test_nested_lists_are_recursive() -> None:
    probe = Probe(points=[(1.0, 2.0), [3.0, (4.0, 5.0)]])
    assert registry.serialize(probe)["points"] == [[1.0, 2.0], [3.0, [4.0, 5.0]]]


def test_custom_ctor_and_validator() -> None:
    @component_fields
    class Dep(Component):
        radius: float = 0.4
        height: float = 1.8

        def __init__(self, radius: float = 0.4, height: float = 1.8) -> None:
            super().__init__()
            self.radius = radius
            self.height = height
            registry.validate_dependencies(self)

        @validator
        def _check(self) -> None:
            if self.height < 2 * self.radius:
                raise ValueError("height меньше 2*radius")

    with pytest.raises(ValueError, match="2\\*radius"):
        Dep(radius=1.0, height=1.0)
    with pytest.raises(ValueError, match="2\\*radius"):
        registry.deserialize(Dep, {"radius": 1.0, "height": 1.0})
    assert [d.name for d in registry.fields(Dep)] == ["radius", "height"]


def test_inheritance_keeps_base_fields_first() -> None:
    @component_fields
    class Child(Probe):
        extra: int = 1
        count: int = 10

    names = [d.name for d in registry.fields(Child)]
    assert names[0] == "name_tag" and names[-1] == "extra"
    assert registry.for_name(Child, "count").default == 10


def test_unannotated_default_raises() -> None:
    with pytest.raises(TypeError, match="нет значения"):

        @component_fields
        class Bad(Component):
            x: int


def test_visible_if() -> None:
    values = {"enabled": True, "mode": "two"}
    assert evaluate_visible_if(None, values.get)
    assert evaluate_visible_if("enabled == True", values.get)
    assert not evaluate_visible_if("enabled != True", values.get)
    assert evaluate_visible_if("enabled == True and mode == 'two'", values.get)
    assert not evaluate_visible_if("mode == 'one' and enabled == True", values.get)
    with pytest.raises(ValueError):
        evaluate_visible_if("__import__('os')", values.get)
