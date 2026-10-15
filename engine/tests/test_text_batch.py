"""TextBatch: сборка вершин и индексов, группировка по атласам (без OpenGL)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.math3d import Vector3
from engine.rendering import Camera
from engine.rendering.text import DEFAULT_FONT, TextBatch, clear_cache, get_atlas

pytestmark = pytest.mark.skipif(not DEFAULT_FONT.is_file(), reason="нет шрифта")


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_cache()
    yield
    clear_cache()


def _camera(z: float = 5.0) -> Camera:
    camera = Camera()
    camera.position = Vector3(0.0, 0.0, z)
    camera.look_at(Vector3.zero())
    return camera


def test_two_letters_make_two_quads() -> None:
    atlas = get_atlas("", 32)
    batch = TextBatch()
    batch.add_text(atlas, "Hi", (0.0, 0.0, 0.0), 0.01)
    vertices, indices = batch.build()
    assert vertices.shape == (8, 9)
    assert vertices.dtype == np.float32
    assert indices.shape == (12,)
    assert indices.dtype == np.uint32
    assert indices[:6].tolist() == [0, 1, 2, 0, 2, 3]
    assert indices[6:].tolist() == [4, 5, 6, 4, 6, 7]


def test_empty_text_gives_empty_buffers() -> None:
    batch = TextBatch()
    batch.add_text(get_atlas("", 32), "", (0.0, 0.0, 0.0), 0.01)
    vertices, indices = batch.build()
    assert vertices.shape == (0, 9) and indices.size == 0
    batch.add_text(get_atlas("", 32), "   ", (0.0, 0.0, 0.0), 0.01)
    assert batch.build()[1].size == 0


def test_color_and_alpha_go_to_vertices() -> None:
    batch = TextBatch()
    batch.add_text(get_atlas("", 32), "A", color=(1.0, 0.5, 0.25), alpha=0.5)
    vertices, _ = batch.build()
    assert np.allclose(vertices[:, 5:9], [1.0, 0.5, 0.25, 0.5])


def test_add_glyph_makes_one_quad() -> None:
    batch = TextBatch()
    batch.add_glyph(get_atlas("", 32), ord("A"), (1.0, 2.0, 3.0), 0.1)
    vertices, indices = batch.build()
    assert vertices.shape == (4, 9) and indices.size == 6
    assert np.allclose(vertices[:, 2], 3.0)


def test_hundred_texts_one_atlas_one_group() -> None:
    atlas = get_atlas("", 32)
    batch = TextBatch()
    for i in range(100):
        batch.add_text(atlas, f"Text {i}", (float(i), 0.0, 0.0), 0.01)
    assert len(batch) == 100
    assert len(batch._groups(None)) == 1  # noqa: SLF001
    vertices, indices = batch.build()
    assert vertices.shape[0] == indices.size // 6 * 4


def test_two_atlases_two_groups_with_global_indices() -> None:
    batch = TextBatch()
    batch.add_text(get_atlas("", 32), "A")
    batch.add_text(get_atlas("", 16), "B")
    assert len(batch._groups(None)) == 2  # noqa: SLF001
    _, indices = batch.build()
    assert indices[6:].min() == 4


def test_billboard_faces_camera() -> None:
    camera = _camera()
    camera.position = Vector3(5.0, 0.0, 0.0)
    camera.look_at(Vector3.zero())
    flat = TextBatch()
    flat.add_text(get_atlas("", 32), "I", scale=1.0, billboard=False)
    turned = TextBatch()
    turned.add_text(get_atlas("", 32), "I", scale=1.0, billboard=True)
    flat_vertices, _ = flat.build(camera)
    turned_vertices, _ = turned.build(camera)
    # Обычный текст лежит в плоскости XY (x вдоль глифа), billboard — в плоскости
    # экрана камеры, смотрящей вдоль X: ширина идёт вдоль Z.
    assert np.ptp(flat_vertices[:, 2]) < 1e-5
    assert np.ptp(turned_vertices[:, 0]) < 1e-4
    assert np.ptp(turned_vertices[:, 2]) > 1.0


def test_matrix_rotation_applies_when_not_billboard() -> None:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 0] = (0.0, 0.0, 1.0)  # локальный X -> мировой Z
    batch = TextBatch()
    batch.add_text(get_atlas("", 32), "I", scale=1.0, matrix=matrix)
    vertices, _ = batch.build()
    assert np.ptp(vertices[:, 0]) < 1e-5 and np.ptp(vertices[:, 2]) > 1.0


def test_sorting_layer_then_distance() -> None:
    atlas = get_atlas("", 32)
    batch = TextBatch()
    batch.add_text(atlas, "I", (0.0, 0.0, 4.0), 0.01, sorting_layer=0)  # близко
    batch.add_text(atlas, "I", (0.0, 0.0, -4.0), 0.01, sorting_layer=0)  # далеко
    batch.add_text(atlas, "I", (0.0, 0.0, -8.0), 0.01, sorting_layer=1)  # выше слой
    vertices, _ = batch.build(_camera(10.0))
    z = vertices[::4, 2]
    assert z[0] < 0 and z[1] > 0 and z[2] < -7.0


def test_outline_adds_eight_copies_before_fill() -> None:
    batch = TextBatch()
    batch.add_text(get_atlas("", 32), "A", outline_width=2.0, outline_color=(1, 0, 0))
    vertices, indices = batch.build()
    assert indices.size // 6 == 9
    assert np.allclose(vertices[0, 5:8], [1, 0, 0])
    assert np.allclose(vertices[-1, 5:8], [1, 1, 1])


def test_wrap_width_is_in_world_units() -> None:
    atlas = get_atlas("", 32)
    wide = TextBatch()
    wide.add_text(atlas, "aa bb cc dd", scale=0.01)
    narrow = TextBatch()
    narrow.add_text(atlas, "aa bb cc dd", scale=0.01, wrap_width=0.5)
    assert np.ptp(narrow.build()[0][:, 1]) > np.ptp(wide.build()[0][:, 1])


def test_pass_keeps_empty_batch_instance() -> None:
    from engine.rendering.text import TextPass

    batch = TextBatch()  # пустой пакет ложен по __len__, но это тот же объект
    assert TextPass(batch).batch is batch


def _scene_with_text():
    from engine.core.components import TextRenderer
    from engine.core.scene import Scene

    scene = Scene("s")
    game_object = scene.create_game_object("T")
    text = game_object.add_component(TextRenderer(text="Hi"))
    return scene, game_object, text


def test_fill_skips_when_scene_unchanged() -> None:
    from engine.rendering.text import fill_text_batch

    scene, _, _ = _scene_with_text()
    batch = TextBatch()
    assert fill_text_batch(batch, scene.objects, (), scene) == 1
    first = batch._items[0]  # noqa: SLF001
    assert fill_text_batch(batch, scene.objects, (), scene) == 1
    assert batch._items[0] is first  # noqa: SLF001 - пакет не пересобирался


def test_fill_rebuilds_on_changes() -> None:
    from engine.core.math3d import Vector3
    from engine.rendering.text import fill_text_batch

    scene, game_object, text = _scene_with_text()
    batch = TextBatch()
    fill_text_batch(batch, scene.objects, (), scene)
    first = batch._items[0]  # noqa: SLF001

    text.text = "Other"  # правка компонента
    fill_text_batch(batch, scene.objects, (), scene)
    second = batch._items[0]  # noqa: SLF001
    assert second is not first

    game_object.transform.position = Vector3(1, 0, 0)  # правка трансформа
    fill_text_batch(batch, scene.objects, (), scene)
    assert batch._items[0] is not second  # noqa: SLF001

    game_object.set_active(False)  # активность
    assert fill_text_batch(batch, scene.objects, (), scene) == 0
    game_object.set_active(True)
    assert fill_text_batch(batch, scene.objects, (), scene) == 1
