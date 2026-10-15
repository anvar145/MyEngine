"""Флаги теней ``MeshRenderer``: значения по умолчанию и сериализация."""

from __future__ import annotations

from engine.core.components import MeshRenderer


def test_shadow_flags_default_to_true() -> None:
    component = MeshRenderer()
    assert component.cast_shadow is True
    assert component.receive_shadow is True


def test_shadow_flags_round_trip() -> None:
    source = MeshRenderer(cast_shadow=False, receive_shadow=False)
    restored = MeshRenderer.from_dict(source.to_dict())
    assert restored.cast_shadow is False
    assert restored.receive_shadow is False


def test_sorting_layer_round_trip() -> None:
    assert MeshRenderer().sorting_layer == 0
    restored = MeshRenderer.from_dict(MeshRenderer(sorting_layer=3).to_dict())
    assert restored.sorting_layer == 3
    assert "sorting_layer" not in MeshRenderer().to_dict()


def test_old_scene_without_flags_loads_with_true() -> None:
    restored = MeshRenderer.from_dict({"mesh": "cube"})
    assert restored.cast_shadow is True
    assert restored.receive_shadow is True
