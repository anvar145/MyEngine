"""Тесты SpriteBatch: сборка вершин на CPU (без OpenGL) и компонент SpriteRenderer."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from engine.core.components import SpriteRenderer
from engine.core.resources import SpriteAtlas
from engine.core.resources.cookers import SpriteFrame
from engine.rendering import SpriteBatch


@pytest.fixture()
def atlas() -> SpriteAtlas:
    frames = {
        "a": SpriteFrame("a", 0, 0, 16, 8),
        "b": SpriteFrame(
            "b", 16, 0, 8, 8, pivot=(0.0, 1.0), source_size=(16, 16), offset=(4, 2)
        ),
    }
    return SpriteAtlas(32, 16, frames, Path("texture.bin"))


def test_empty_batch(atlas: SpriteAtlas) -> None:
    vertices, indices = SpriteBatch(atlas).build()
    assert vertices.shape == (0, 9)
    assert indices.size == 0


def test_quad_geometry_and_uv(atlas: SpriteAtlas) -> None:
    batch = SpriteBatch(atlas, pixels_per_unit=16.0)
    batch.add("a", (1.0, 2.0, 3.0))
    vertices, indices = batch.build()
    assert vertices.shape == (4, 9)
    assert indices.tolist() == [0, 1, 2, 0, 2, 3]
    expected = [(0.5, 2.25), (1.5, 2.25), (1.5, 1.75), (0.5, 1.75)]
    assert np.allclose(vertices[:, :2], expected)
    assert np.allclose(vertices[:, 2], 3.0)
    assert np.allclose(vertices[:, 3:5], [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)])
    assert np.allclose(vertices[:, 5:], 1.0)


def test_pivot_and_trim_offset(atlas: SpriteAtlas) -> None:
    batch = SpriteBatch(atlas, pixels_per_unit=16.0)
    batch.add("b")
    vertices, _ = batch.build()
    # pivot = (0, 16) px: левый край (4-0)/16, верх (16-2)/16
    assert np.allclose(vertices[0, :2], (0.25, 0.875))
    assert np.allclose(vertices[2, :2], (0.75, 0.375))


def test_scale_rotation_flip_color(atlas: SpriteAtlas) -> None:
    batch = SpriteBatch(atlas, pixels_per_unit=16.0)
    batch.add("a", rotation=math.pi / 2, scale=(2.0, 1.0), color=(1, 0, 0, 0.5))
    batch.add("a", flip_x=True)
    vertices, indices = batch.build()
    assert vertices.shape == (8, 9)
    assert indices[6:].tolist() == [4, 5, 6, 4, 6, 7]
    # TL (-0.5, 0.25) * scale (2, 1) = (-1, 0.25) -> поворот на 90°: (-0.25, -1)
    assert np.allclose(vertices[0, :2], (-0.25, -1.0), atol=1e-6)
    assert np.allclose(vertices[0, 5:], (1, 0, 0, 0.5))
    # flip_x: левый верх теперь справа
    assert np.allclose(vertices[4, :2], (0.5, 0.25))
    assert np.allclose(vertices[4, 3:5], (0, 0))


def test_clear_and_unknown_sprite(atlas: SpriteAtlas) -> None:
    batch = SpriteBatch(atlas)
    batch.add("a")
    assert len(batch) == 1
    batch.clear()
    assert len(batch) == 0
    with pytest.raises(KeyError):
        batch.add("missing")


def test_invalid_pixels_per_unit(atlas: SpriteAtlas) -> None:
    from engine.rendering import RenderError

    with pytest.raises(RenderError):
        SpriteBatch(atlas, pixels_per_unit=0)


def test_sprite_renderer_roundtrip() -> None:
    sprite = SpriteRenderer("hero", "ab" * 16, (1, 0.5, 0.25, 1), flip_x=True)
    restored = SpriteRenderer.from_dict(sprite.to_dict())
    assert restored.sprite == "hero"
    assert restored.atlas_guid == "ab" * 16
    assert restored.color == (1.0, 0.5, 0.25, 1.0)
    assert restored.flip_x and not restored.flip_y and restored.visible
