"""Поля export у Behaviour: дескрипторы, умолчания, зажим, (де)сериализация."""

from __future__ import annotations

import json
import logging

import pytest

from engine.core.game_object import GameObject
from engine.core.math3d import Quaternion, Vector3
from engine.core.reflection import FieldKind, InvalidPolicy
from engine.core.resources import AssetRef
from engine.scripting.behaviour import Behaviour, export
from engine.scripting.fields import Asset, descriptors_from_behaviour


class Mover(Behaviour):
    speed: float = export(6.0, range=(0, 20), tooltip="м/с")
    count: int = export(3, range=(0, 10))
    strict_speed: float = export(
        1.0, range=(0, 5), on_invalid=InvalidPolicy.ERROR
    )
    flag: bool = export(True)
    label: str = export("abc")
    offset: Vector3 = export(Vector3(0.0, 1.0, 0.0))
    spin: Quaternion = export(Quaternion())
    tint: tuple = export((1.0, 0.5, 0.0, 1.0), kind=FieldKind.COLOR)
    target: GameObject = export(None)
    icon: Asset = export(None)
    secret: float = export(7.0, serialize=False)


def _by_name(cls):
    return {d.name: d for d in descriptors_from_behaviour(cls)}


def test_fields_visible_through_descriptors() -> None:
    descs = _by_name(Mover)
    assert list(descs) == [
        "speed", "count", "strict_speed", "flag", "label", "offset", "spin",
        "tint", "target", "icon", "secret",
    ]
    assert descs["speed"].kind is FieldKind.FLOAT
    assert descs["speed"].range == (0, 20)
    assert descs["count"].kind is FieldKind.INT
    assert descs["offset"].kind is FieldKind.VEC3
    assert descs["spin"].kind is FieldKind.QUAT
    assert descs["tint"].kind is FieldKind.COLOR
    assert descs["target"].kind is FieldKind.OBJECT_REF
    assert descs["icon"].kind is FieldKind.ASSET_REF


def test_defaults_apply_when_scene_has_no_value() -> None:
    mover = Mover()
    mover.apply_field_values({})
    assert mover.speed == 6.0 and mover.label == "abc"
    assert mover.offset == Vector3(0.0, 1.0, 0.0)
    assert mover.target is None and mover.icon is None


def test_clamp_to_range(caplog: pytest.LogCaptureFixture) -> None:
    mover = Mover()
    with caplog.at_level(logging.WARNING):
        mover.apply_field_values({"speed": 30, "count": -4})
    assert mover.speed == 20.0
    assert mover.count == 0 and isinstance(mover.count, int)
    assert "speed" in caplog.text


def test_error_policy_raises_with_field_name() -> None:
    with pytest.raises(ValueError, match="strict_speed"):
        Mover().apply_field_values({"strict_speed": 9.0})
    assert Mover.clamp_field("strict_speed", 3.0) == 3.0


def test_unknown_field_is_reported_not_applied() -> None:
    mover = Mover()
    assert mover.apply_field_values({"nope": 1, "speed": 2}) == ["nope"]
    assert mover.speed == 2.0
    assert not hasattr(mover, "nope")


def test_inheritance_keeps_base_fields_first() -> None:
    class Child(Mover):
        extra: float = export(1.0)
        speed: float = export(2.0, range=(0, 5))  # переопределение

    names = list(_by_name(Child))
    assert names[:2] == ["speed", "count"]  # позиция поля базы сохранена
    assert names[-1] == "extra" and len(names) == len(_by_name(Mover)) + 1
    assert Child().speed == 2.0
    assert _by_name(Child)["speed"].range == (0, 5)
    assert _by_name(Mover)["speed"].range == (0, 20)  # база не изменилась


def test_serialize_writes_only_changed_serializable_values() -> None:
    mover = Mover()
    assert mover.serialize_fields() == {}
    mover.speed = 12.5
    mover.offset = Vector3(1.0, 2.0, 3.0)
    mover.spin = Quaternion(0.0, 1.0, 0.0, 0.0)
    mover.tint = (0.0, 0.0, 1.0, 1.0)
    mover.secret = 1.0  # serialize=False
    data = mover.serialize_fields()
    assert data == {
        "speed": 12.5,
        "offset": [1.0, 2.0, 3.0],
        "spin": [0.0, 1.0, 0.0, 0.0],
        "tint": [0.0, 0.0, 1.0, 1.0],
    }
    json.dumps(data)  # JSON-совместимо


def test_round_trip_values() -> None:
    source = Mover()
    source.speed = 12.5
    source.label = "xyz"
    source.offset = Vector3(1.0, 2.0, 3.0)
    source.spin = Quaternion(0.0, 1.0, 0.0, 0.0)
    source.tint = (0.0, 0.0, 1.0, 1.0)
    source.icon = Asset("C:/p/icon.png")
    target = GameObject("T")
    source.target = target
    data = json.loads(json.dumps(source.serialize_fields()))
    assert data["icon"] == {"guid": None, "path": "C:/p/icon.png"}
    assert data["target"] == {"uid": target.uid}

    copy = Mover()
    copy.apply_field_values(data, objects={target.uid: target})
    assert copy.speed == 12.5 and copy.label == "xyz"
    assert copy.offset == Vector3(1.0, 2.0, 3.0)
    assert isinstance(copy.offset, Vector3)
    assert copy.spin == Quaternion(0.0, 1.0, 0.0, 0.0)
    assert copy.tint == (0.0, 0.0, 1.0, 1.0)
    assert copy.icon == AssetRef(path="C:/p/icon.png")
    assert copy.target is target


def test_object_ref_missing_in_scene_stays_unset() -> None:
    mover = Mover()
    mover.apply_field_values({"target": {"object": 999}}, objects={})
    assert mover.target is None


def test_type_mismatch_keeps_default(caplog: pytest.LogCaptureFixture) -> None:
    mover = Mover()
    with caplog.at_level(logging.WARNING):
        mover.apply_field_values({"offset": "bad", "speed": "fast"})
    assert mover.offset == Vector3(0.0, 1.0, 0.0)
    assert mover.speed == 6.0
    assert "speed" in caplog.text and "offset" in caplog.text


def test_prefab_field_round_trip_and_scan() -> None:
    from engine.scripting import Prefab
    from engine.scripting.fields import scan_fields

    class Spawner(Behaviour):
        bullet: Prefab = export(None)

    desc = _by_name(Spawner)["bullet"]
    assert desc.kind is FieldKind.ASSET_REF and desc.asset_kind == "prefab"

    source = Spawner()
    source.bullet = Prefab("C:/p/bullet.prefab.json")
    data = json.loads(json.dumps(source.serialize_fields()))
    assert data["bullet"] == {"guid": None, "path": "C:/p/bullet.prefab.json"}
    copy = Spawner()
    copy.apply_field_values(data, objects={})
    assert isinstance(copy.bullet, AssetRef) and copy.bullet == source.bullet

    [field] = scan_fields("bullet: Prefab = None\n")
    assert (field.kind, field.asset_kind) == ("asset", "prefab")
