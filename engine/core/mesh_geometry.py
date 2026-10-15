"""Геометрия mesh-ассета для физики: LOD0 из ``mesh.bin`` в виде массивов.

Модель из glTF/FBX импортируется в несколько мешей-артефактов; для коллайдера
они склеиваются в один набор вершин и треугольников.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.resources.asset_ref import AssetRef
from engine.core.resources.cookers.mesh_format import read_cooked_mesh
from engine.core.resources.vfs import as_assets

__all__ = ["load_mesh_geometry"]


def load_mesh_geometry(
    assets: Any, ref: AssetRef
) -> tuple[np.ndarray, np.ndarray]:
    """Прочитать вершины и треугольники LOD0 ассета-меша.

    Args:
        assets: База ассетов проекта или VFS.
        ref: Ссылка на ассет (нужен ``guid`` либо путь, известный базе).

    Returns:
        ``(vertices float64 (n, 3), triangles int64 (m, 3))``.

    Raises:
        FileNotFoundError: Ассет не найден или у него нет мешей.
    """
    guid = ref.guid
    if guid is None and ref.path:
        finder = getattr(assets, "guid_of", None)
        root = getattr(assets, "root", None)
        if finder is not None and root is not None:
            guid = finder(root / ref.path)
    if guid is None:
        raise FileNotFoundError(f"ассет меша не найден: {ref.path}")
    handle = as_assets(assets).resolve(guid)
    parts_v: list[np.ndarray] = []
    parts_t: list[np.ndarray] = []
    offset = 0
    for artifact in handle.artifacts:
        if artifact.kind != "mesh":
            continue
        cooked = read_cooked_mesh(handle.artifact_path(artifact))
        positions = np.asarray(cooked.positions(0), np.float64).reshape(-1, 3)
        indices = np.asarray(cooked.lods[0].indices, np.int64).reshape(-1, 3)
        parts_v.append(positions)
        parts_t.append(indices + offset)
        offset += len(positions)
    if not parts_v:
        raise FileNotFoundError(f"у ассета {guid} нет мешей")
    return np.concatenate(parts_v), np.concatenate(parts_t)
