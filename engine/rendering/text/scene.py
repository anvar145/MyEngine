"""Сбор ``TextRenderer`` сцены в :class:`TextBatch` (общий для плеера и редактора)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from engine.core.components import TextRenderer
from engine.rendering.text.atlas import get_atlas

if TYPE_CHECKING:
    from engine.core.game_object import GameObject
    from engine.core.scene import Scene
    from engine.rendering.text.batch import TextBatch

__all__ = ["fill_text_batch"]

_log = logging.getLogger("engine.rendering.text.scene")


def fill_text_batch(
    batch: TextBatch,
    objects: Iterable[GameObject],
    roots: tuple[Path, ...] = (),
    scene: Scene | None = None,
) -> int:
    """Очистить пакет и добавить в него тексты активных объектов.

    Args:
        batch: Пакет текста.
        objects: Объекты сцены.
        roots: Каталоги для поиска относительных путей шрифтов.
        scene: Сцена-источник объектов. Если задана и с прошлого вызова не менялись
            трансформы, состав, активность и ``TextRenderer``, пакет остаётся как
            есть (пересборка и повторная загрузка в GPU пропускаются).

    Returns:
        Число добавленных текстов.
    """
    token = None
    if scene is not None:
        token = (
            id(scene),
            scene.transforms.version,
            scene.structure_version,
            scene.activity_version,
            TextRenderer.revision,
            roots,
        )
        if token == batch.fill_token:
            return batch.fill_count
    batch.clear()
    count = 0
    for game_object in objects:
        if not game_object.active_in_hierarchy:
            continue
        text = game_object.get_component(TextRenderer)
        if text is None or not text.visible or not text.text:
            continue
        try:
            atlas = get_atlas(
                text.font_path,
                text.size_px,
                bold=text.bold,
                italic=text.italic,
                roots=roots,
            )
        except Exception as error:  # noqa: BLE001
            _log.warning("Текст '%s': шрифт недоступен: %s", game_object.name, error)
            continue
        matrix = game_object.transform.world_matrix
        position = matrix[:3, 3].tolist()
        batch.add_text(
            atlas,
            text.text,
            position,
            text.units_per_pixel,
            text.color,
            text.alpha,
            text.align,
            text.wrap_width,
            vertical_align=text.vertical_align,
            line_spacing=text.line_spacing,
            billboard=text.billboard,
            matrix=matrix,
            sorting_layer=text.sorting_layer,
            outline_color=text.outline_color,
            outline_width=text.outline_width,
        )
        count += 1
    batch.fill_token = token
    batch.fill_count = count
    return count
