"""Ядро MyEngine: игровые объекты, ECS-подобные запросы, ресурсы и цикл игры.

Модуль собирает публичный API ядра. Важно: ядро не зависит ни от PySide6, ни
от OpenGL — его можно использовать в headless-режиме (тесты, серверные
сборки, инструменты импорта).

Пример минимального приложения:

    >>> from engine.core import GameLoop, Scene, Vector3
    >>> scene = Scene("Demo")
    >>> player = scene.create_game_object("Player", position=Vector3(0, 1, 0))
    >>> loop = GameLoop(scene, target_fps=60)
    >>> _ = loop.run(max_frames=120)
"""

from __future__ import annotations

from engine.core.component import Component
from engine.core.event_bus import Event, EventBus, EventBusStats, Subscription
from engine.core.game_loop import GameLoop, LoopStats, Time
from engine.core.game_object import GameObject
from engine.core.math3d import Quaternion, Vector3
from engine.core.resources import (
    Asset,
    AssetLoaded,
    AssetLoader,
    AssetType,
    AssetUnloaded,
    MeshAsset,
    ResourceError,
    ResourceManager,
    ResourceStats,
    ShaderAsset,
    ShaderStage,
    TextureAsset,
)
from engine.core.scene import Scene, SceneStats
from engine.core.system import System
from engine.core.transform import Transform, TransformStore

__all__ = [
    "Asset",
    "AssetLoaded",
    "AssetLoader",
    "AssetType",
    "AssetUnloaded",
    "Component",
    "Event",
    "EventBus",
    "EventBusStats",
    "GameLoop",
    "GameObject",
    "LoopStats",
    "MeshAsset",
    "Quaternion",
    "ResourceError",
    "ResourceManager",
    "ResourceStats",
    "Scene",
    "SceneStats",
    "ShaderAsset",
    "ShaderStage",
    "Subscription",
    "System",
    "TextureAsset",
    "Time",
    "Transform",
    "TransformStore",
    "Vector3",
]
