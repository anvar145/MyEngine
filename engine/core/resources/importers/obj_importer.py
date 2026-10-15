"""Импортёр Wavefront OBJ: оборачивает существующий парсер в пайплайн ассетов."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from engine.core.resources.errors import AssetImportError, ResourceFormatError
from engine.core.resources.importer import Artifact, AssetPreview, ImportContext
from engine.core.resources.loaders.mesh_loader import parse_obj

__all__ = ["ObjImporter"]

MESH_FILE = "mesh.bin"


class ObjImporter:
    """``.obj`` → артефакт ``mesh`` (``mesh.bin``: LOD, квантизация).

    Настройки:
        scale (float): множитель координат вершин;
        flip_uv (bool): инвертировать координату ``v``;
        generate_lods (bool), lod_ratios (list[float]): параметры LOD.

    Версия 2: вместо промежуточного ``mesh.npz`` результат кукуется
    :func:`~engine.core.resources.cookers.cook_mesh`.
    """

    name = "obj"
    extensions = (".obj",)
    version = 2

    def default_settings(self) -> dict[str, Any]:
        """Настройки по умолчанию."""
        return {
            "scale": 1.0,
            "flip_uv": False,
            "generate_lods": True,
            "lod_ratios": [0.5, 0.25],
        }

    def inspect(self, path: Path) -> AssetPreview:
        """Посчитать вершины и грани, не разбирая файл целиком."""
        vertices = faces = 0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if line.startswith("v "):
                        vertices += 1
                    elif line.startswith("f "):
                        faces += 1
        except OSError as exc:
            raise AssetImportError(f"Не удалось прочитать '{path}': {exc}") from exc
        return AssetPreview("mesh", {"vertices": vertices, "faces": faces})

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        """Разобрать OBJ и записать ``mesh.bin``."""
        from engine.core.resources.cookers import (
            MeshCookSettings,
            cook_mesh,
            read_cooked_mesh,
        )

        try:
            mesh = parse_obj(
                ctx.source,
                scale=float(ctx.setting("scale", 1.0)),
                flip_uv=bool(ctx.setting("flip_uv", False)),
            )
        except ResourceFormatError as exc:
            raise AssetImportError(str(exc)) from exc

        blob = cook_mesh(
            mesh.positions,
            mesh.indices,
            mesh.normals,
            mesh.uvs,
            MeshCookSettings.from_mapping(dict(ctx.settings)),
        )
        (ctx.output_dir / MESH_FILE).write_bytes(blob)

        low, high = mesh.bounds
        return [
            ctx.artifact(
                "mesh",
                MESH_FILE,
                vertices=mesh.vertex_count,
                triangles=mesh.triangle_count,
                lods=len(read_cooked_mesh(blob).lods),
                has_normals=mesh.normals is not None,
                has_uvs=mesh.uvs is not None,
                bounds_min=[low.x, low.y, low.z],
                bounds_max=[high.x, high.y, high.z],
            )
        ]
