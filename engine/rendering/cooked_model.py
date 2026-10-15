"""Части импортированной модели из ``Library/``: меши ``mesh.bin`` и материалы.

Общий код плеера (:class:`~engine.player.scene_renderer.SceneRenderer`) и
вьюпорта редактора: модель из glTF/FBX импортируется в несколько мешей, каждый
со своим материалом.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.core.resources.cookers.mesh_format import read_cooked_mesh
from engine.core.resources.vfs import as_assets
from engine.rendering.mesh import GpuMesh

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources import AssetDatabase

__all__ = ["load_cooked_parts"]


def load_cooked_parts(
    assets: AssetDatabase,
    guid: str,
    pbr_material: Callable[[str, str], Any | None],
) -> list[tuple[GpuMesh, Any]]:
    """Загрузить все меши импортированного ассета на GPU.

    Args:
        assets: База ассетов проекта или VFS.
        guid: GUID ассета-модели.
        pbr_material: ``(guid, имя артефакта материала) -> материал | None``.

    Returns:
        Список ``(меш, материал или None)`` по артефактам ``mesh``.

    Raises:
        FileNotFoundError: У ассета нет мешей.
    """
    handle = as_assets(assets).resolve(guid)
    result: list[tuple[GpuMesh, Any]] = []
    for artifact in handle.artifacts:
        if artifact.kind != "mesh":
            continue
        cooked = read_cooked_mesh(handle.artifact_path(artifact))
        lod = cooked.lods[0]
        mesh = GpuMesh.from_arrays(
            cooked.positions(0),
            normals=cooked.normals(0),
            uvs=None if lod.uvs is None else np.asarray(lod.uvs, np.float32),
            indices=np.asarray(lod.indices, np.uint32),
            name=f"{guid}:{artifact.name}",
        )
        material = None
        material_name = artifact.info.get("material_artifact")
        if material_name:
            material = pbr_material(guid, str(material_name))
        result.append((mesh, material))
    if not result:
        raise FileNotFoundError(f"у ассета {guid} нет мешей")
    return result
