"""Сохранение и загрузка сцены в JSON.

Реализация живёт в :mod:`engine.scene_io` (она нужна и плееру без Qt);
этот модуль сохраняет прежний путь импорта для редактора.
"""

from __future__ import annotations

from engine.scene_io import (
    SCENE_FORMAT,
    SCENE_VERSION,
    load_scene,
    load_scene_full,
    load_scene_with_sky,
    read_post,
    read_sky,
    save_scene,
    scene_from_dict,
    scene_to_dict,
)

__all__ = [
    "SCENE_FORMAT",
    "SCENE_VERSION",
    "load_scene",
    "load_scene_full",
    "load_scene_with_sky",
    "read_post",
    "read_sky",
    "save_scene",
    "scene_from_dict",
    "scene_to_dict",
]
