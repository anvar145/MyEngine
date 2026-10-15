"""Импортёр glTF 2.0 (``.gltf`` / ``.glb``) на базе ``pygltflib`` (extra ``[gltf]``).

Что делает:

* обходит узлы сцены, запекает мировые матрицы в вершины (позиции и нормали);
* каждый (узел, примитив) становится отдельным артефактом ``mesh_NNN.bin``
  (кукованный меш с LOD и квантизацией), с описанием материала в ``info``;
* поддерживает режимы TRIANGLES / TRIANGLE_STRIP / TRIANGLE_FAN, строгие и
  нормализованные акцессоры, ``byteStride``, data-URI и внешние ``.bin``;
* материалы конвертируются в ``material_NNN.json`` (+ упакованную текстуру
  ``material_NNN_orm.bin``), см. ``material_importer``; файлы текстур
  записываются как зависимости ассета.

Не поддержано (с предупреждением в логе или ошибкой): sparse-акцессоры, скиннинг,
морф-таргеты, анимации, не-треугольные примитивы, встроенные (data-URI /
bufferView) текстуры материалов.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import numpy as np

from engine.core.resources.errors import AssetImportError
from engine.core.resources.importer import Artifact, AssetPreview, ImportContext
from engine.core.resources.importers.material_importer import build_material_artifacts
from engine.core.resources.primitives import compute_normals

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources.cookers.material_cooker import MaterialSpec

__all__ = ["GltfImporter"]

_GLB_MAGIC = b"glTF"
_GLB_JSON = 0x4E4F534A
_COMPONENT = {
    5120: np.dtype("<i1"),
    5121: np.dtype("<u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_TYPE_SIZE = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
_NORMALIZE = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}
_MODE_TRIANGLES, _MODE_STRIP, _MODE_FAN = 4, 5, 6
_MAX_NODE_DEPTH = 256


class GltfImporter:
    """``.gltf``/``.glb`` → набор артефактов ``mesh`` (``mesh_NNN.bin``).

    Настройки:
        scale (float): множитель координат (после применения матриц узлов);
        generate_normals (bool): считать нормали, если их нет в файле;
        generate_lods (bool), lod_ratios (list[float]): параметры LOD;
        extract_materials (bool): конвертировать материалы в ``material_NNN.json``.

    Версия 2: добавлено извлечение материалов.
    """

    name = "gltf"
    extensions = (".gltf", ".glb")
    version = 2

    def default_settings(self) -> dict[str, Any]:
        """Настройки по умолчанию."""
        return {
            "scale": 1.0,
            "generate_normals": True,
            "generate_lods": True,
            "lod_ratios": [0.5, 0.25],
            "extract_materials": True,
        }

    def inspect(self, path: Path) -> AssetPreview:
        """Посчитать меши, узлы и материалы по JSON-части (без ``pygltflib``)."""
        document = _read_json(path)
        meshes = document.get("meshes", [])
        return AssetPreview(
            "model",
            {
                "meshes": len(meshes),
                "primitives": sum(len(m.get("primitives", [])) for m in meshes),
                "nodes": len(document.get("nodes", [])),
                "materials": len(document.get("materials", [])),
                "generator": document.get("asset", {}).get("generator"),
            },
        )

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        """Разобрать модель и записать по артефакту на каждый примитив узла."""
        from engine.core.resources.cookers import MeshCookSettings, cook_mesh

        try:
            from pygltflib import GLTF2
        except ImportError as exc:
            raise AssetImportError(
                "Для импорта glTF нужен пакет pygltflib: "
                "установите extra [gltf] (pip install myengine[gltf])"
            ) from exc
        try:
            gltf = GLTF2().load(str(ctx.source))
        except Exception as exc:  # noqa: BLE001 - pygltflib бросает разные типы
            raise AssetImportError(f"'{ctx.source.name}': {exc}") from exc
        if gltf is None:
            raise AssetImportError(f"'{ctx.source.name}': не удалось разобрать glTF")

        buffers = _load_buffers(gltf, ctx.source)
        scale = float(ctx.setting("scale", 1.0))
        cook = MeshCookSettings.from_mapping(dict(ctx.settings))
        make_normals = bool(ctx.setting("generate_normals", True))

        artifacts: list[Artifact] = []
        material_artifacts: list[Artifact] = []
        material_names: dict[int, str | None] = {}

        def material_json(index: int | None) -> str | None:
            """Собрать материал при первом обращении; ``None`` — не удалось."""
            if index is None or not bool(ctx.setting("extract_materials", True)):
                return None
            if index not in material_names:
                material_names[index] = None
                if index < len(gltf.materials):
                    try:
                        spec, name = _material_spec(gltf, index, ctx)
                        built = build_material_artifacts(
                            ctx, spec, ctx.source.parent, f"material_{index:03d}", name
                        )
                    except AssetImportError as exc:
                        ctx.log.warning("Материал %d пропущен: %s", index, exc)
                    else:
                        material_artifacts.extend(built)
                        material_names[index] = built[0].name
            return material_names[index]

        for node_index, matrix in _walk_scene(gltf):
            node = gltf.nodes[node_index]
            if node.mesh is None:
                continue
            mesh = gltf.meshes[node.mesh]
            for prim_index, primitive in enumerate(mesh.primitives):
                geometry = _read_primitive(
                    gltf, buffers, primitive, matrix, scale, make_normals, ctx
                )
                if geometry is None:
                    continue
                positions, indices, normals, uvs = geometry
                name = f"mesh_{len(artifacts):03d}.bin"
                blob = cook_mesh(positions, indices, normals, uvs, cook)
                (ctx.output_dir / name).write_bytes(blob)
                low, high = positions.min(axis=0), positions.max(axis=0)
                artifacts.append(
                    ctx.artifact(
                        "mesh",
                        name,
                        node=node.name or f"node{node_index}",
                        mesh=mesh.name or f"mesh{node.mesh}",
                        primitive=prim_index,
                        vertices=int(positions.shape[0]),
                        triangles=int(indices.size // 3),
                        has_normals=normals is not None,
                        has_uvs=uvs is not None,
                        material=_material_info(gltf, primitive.material, ctx),
                        material_artifact=material_json(primitive.material),
                        bounds_min=low.tolist(),
                        bounds_max=high.tolist(),
                    )
                )
        if not artifacts:
            raise AssetImportError(
                f"'{ctx.source.name}': в сцене нет треугольных мешей"
            )
        return artifacts + material_artifacts


# ---------------------------------------------------------------------- #
# Чтение файла и акцессоров
# ---------------------------------------------------------------------- #
def _read_json(path: Path) -> dict[str, Any]:
    """Прочитать JSON-часть ``.gltf`` или ``.glb``."""
    try:
        if path.suffix.lower() == ".glb":
            with path.open("rb") as stream:
                header = stream.read(20)
                if len(header) < 20 or header[:4] != _GLB_MAGIC:
                    raise AssetImportError(f"'{path.name}': это не GLB")
                length, kind = struct.unpack_from("<II", header, 12)
                if kind != _GLB_JSON:
                    raise AssetImportError(f"'{path.name}': первый блок GLB не JSON")
                return json.loads(stream.read(length).decode("utf-8"))
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AssetImportError(f"'{path.name}': {exc}") from exc


def _load_buffers(gltf: Any, source: Path) -> list[bytes]:
    """Прочитать все буферы (встроенный GLB-блок, data-URI или внешние файлы)."""
    result: list[bytes] = []
    for index, buffer in enumerate(gltf.buffers):
        try:
            if buffer.uri is None:
                blob = gltf.binary_blob()
            else:
                blob = gltf.get_data_from_buffer_uri(buffer.uri)
        except Exception as exc:  # noqa: BLE001
            raise AssetImportError(
                f"'{source.name}': буфер {index} не читается: {exc}"
            ) from exc
        if blob is None or len(blob) < buffer.byteLength:
            raise AssetImportError(f"'{source.name}': буфер {index} короче byteLength")
        result.append(bytes(blob))
    return result


def _accessor(gltf: Any, buffers: list[bytes], index: int) -> np.ndarray:
    """Прочитать акцессор в массив ``(count, components)``."""
    accessor = gltf.accessors[index]
    if accessor.sparse is not None:
        raise AssetImportError("Sparse-акцессоры не поддержаны")
    dtype = _COMPONENT.get(accessor.componentType)
    components = _TYPE_SIZE.get(accessor.type)
    if dtype is None or components is None:
        raise AssetImportError(f"Акцессор {index}: неизвестный тип")
    count = accessor.count
    if accessor.bufferView is None:
        return np.zeros((count, components), dtype=np.float32)

    view = gltf.bufferViews[accessor.bufferView]
    data = buffers[view.buffer]
    element = dtype.itemsize * components
    stride = view.byteStride or element
    start = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    if count and start + stride * (count - 1) + element > len(data):
        raise AssetImportError(f"Акцессор {index} выходит за пределы буфера")
    array = np.ndarray(
        (count, components),
        dtype=dtype,
        buffer=data,
        offset=start,
        strides=(stride, dtype.itemsize),
    )
    if accessor.normalized and accessor.componentType in _NORMALIZE:
        divisor = np.float32(_NORMALIZE[accessor.componentType])
        values = array.astype(np.float32) / divisor
        return np.maximum(values, -1.0)
    return array


def _read_primitive(
    gltf: Any,
    buffers: list[bytes],
    primitive: Any,
    matrix: np.ndarray,
    scale: float,
    make_normals: bool,
    ctx: ImportContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None:
    """Прочитать примитив и применить матрицу узла; ``None`` — пропустить."""
    mode = 4 if primitive.mode is None else primitive.mode
    if mode not in (_MODE_TRIANGLES, _MODE_STRIP, _MODE_FAN):
        ctx.log.warning("Примитив с режимом %d пропущен (нужны треугольники)", mode)
        return None
    attributes = primitive.attributes
    if attributes.POSITION is None:
        raise AssetImportError("Примитив без атрибута POSITION")

    positions = _accessor(gltf, buffers, attributes.POSITION).astype(np.float32)
    count = positions.shape[0]
    if primitive.indices is None:
        indices = np.arange(count, dtype=np.uint32)
    else:
        raw = _accessor(gltf, buffers, primitive.indices)
        indices = raw.reshape(-1).astype(np.uint32)
    indices = _to_triangles(indices, mode)
    if indices.size == 0:
        return None
    if int(indices.max()) >= count:
        raise AssetImportError("Индекс примитива выходит за пределы POSITION")

    normals = None
    if attributes.NORMAL is not None:
        normals = _accessor(gltf, buffers, attributes.NORMAL).astype(np.float32)
    uvs = None
    if attributes.TEXCOORD_0 is not None:
        uvs = _accessor(gltf, buffers, attributes.TEXCOORD_0).astype(np.float32)

    linear = matrix[:3, :3]
    positions = positions @ linear.T + matrix[:3, 3]
    if normals is not None:
        invertible = abs(np.linalg.det(linear)) > 1e-12
        normal_matrix = np.linalg.inv(linear).T if invertible else linear
        normals = normals @ normal_matrix.T
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = (normals / np.maximum(length, 1e-12)).astype(np.float32)
    if np.linalg.det(linear) < 0:  # зеркальное отражение переворачивает обход
        indices = indices.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1)
    positions = (positions * scale).astype(np.float32)
    if normals is None and make_normals:
        normals = compute_normals(positions, indices)
    return positions, indices, normals, uvs


def _to_triangles(indices: np.ndarray, mode: int) -> np.ndarray:
    """Привести индексы любого треугольного режима к списку треугольников."""
    if mode == _MODE_TRIANGLES:
        return indices[: indices.size // 3 * 3]
    if indices.size < 3:
        return indices[:0]
    first = np.arange(indices.size - 2)
    if mode == _MODE_FAN:
        triangles = np.stack(
            [np.full_like(first, 0), first + 1, first + 2], axis=1
        )
    else:
        flip = first % 2
        triangles = np.stack([first, first + 1 + flip, first + 2 - flip], axis=1)
    return indices[triangles].reshape(-1)


# ---------------------------------------------------------------------- #
# Сцена и материалы
# ---------------------------------------------------------------------- #
def _local_matrix(node: Any) -> np.ndarray:
    """Локальная матрица узла 4x4 (столбцовая конвенция)."""
    if node.matrix:
        return np.asarray(node.matrix, dtype=np.float64).reshape(4, 4).T
    x, y, z, w = node.rotation or [0.0, 0.0, 0.0, 1.0]
    norm = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    result = np.eye(4)
    result[:3, :3] = rotation * np.asarray(node.scale or [1.0, 1.0, 1.0])
    result[:3, 3] = node.translation or [0.0, 0.0, 0.0]
    return result


def _walk_scene(gltf: Any) -> list[tuple[int, np.ndarray]]:
    """Обойти сцену и вернуть ``(индекс узла, мировая матрица)``."""
    if gltf.scenes:
        scene_index = gltf.scene if gltf.scene is not None else 0
        roots = list(gltf.scenes[scene_index].nodes)
    else:
        children = {c for node in gltf.nodes for c in (node.children or [])}
        roots = [i for i in range(len(gltf.nodes)) if i not in children]
    result: list[tuple[int, np.ndarray]] = []
    stack = [(root, np.eye(4), 0) for root in reversed(roots)]
    while stack:
        index, parent, depth = stack.pop()
        if depth > _MAX_NODE_DEPTH:
            raise AssetImportError("Иерархия узлов слишком глубокая или зациклена")
        world = parent @ _local_matrix(gltf.nodes[index])
        result.append((index, world))
        for child in reversed(gltf.nodes[index].children or []):
            stack.append((child, world, depth + 1))
    return result


def _texture_uri(gltf: Any, reference: Any, ctx: ImportContext) -> str | None:
    """Относительный путь внешнего файла текстуры или ``None``."""
    if reference is None or reference.index is None:
        return None
    source = gltf.textures[reference.index].source
    image = gltf.images[source] if source is not None else None
    if image is None or not image.uri:
        return None
    if image.uri.startswith("data:"):
        ctx.log.warning("Встроенная текстура материала пропущена (data-URI)")
        return None
    return unquote(image.uri)


def _material_spec(
    gltf: Any, index: int, ctx: ImportContext
) -> tuple[MaterialSpec, str]:
    """Перевести материал glTF в :class:`MaterialSpec`."""
    from engine.core.resources.cookers.material_cooker import MaterialSpec

    material = gltf.materials[index]
    pbr = material.pbrMetallicRoughness
    name = material.name or f"material{index}"

    def number(value: Any, default: float) -> float:
        return default if value is None else float(value)

    textures: dict[str, str] = {}
    pairs = [
        ("base_color", pbr.baseColorTexture if pbr else None),
        ("metallic_roughness", pbr.metallicRoughnessTexture if pbr else None),
        ("normal", material.normalTexture),
        ("ao", material.occlusionTexture),
        ("emissive", material.emissiveTexture),
    ]
    for slot, reference in pairs:
        uri = _texture_uri(gltf, reference, ctx)
        if uri:
            textures[slot] = uri
    normal = material.normalTexture
    occlusion = material.occlusionTexture
    base_color = pbr.baseColorFactor if pbr and pbr.baseColorFactor else [1, 1, 1, 1]
    data: dict[str, Any] = {
        "name": name,
        "base_color": list(base_color),
        "metallic": number(pbr.metallicFactor if pbr else None, 1.0),
        "roughness": number(pbr.roughnessFactor if pbr else None, 1.0),
        "emissive": list(material.emissiveFactor or [0, 0, 0]),
        "normal_scale": number(normal.scale if normal else None, 1.0),
        "occlusion_strength": number(occlusion.strength if occlusion else None, 1.0),
        "alpha_mode": (material.alphaMode or "OPAQUE").lower(),
        "alpha_cutoff": number(material.alphaCutoff, 0.5),
        "double_sided": bool(material.doubleSided),
        "textures": textures,
    }
    return MaterialSpec.from_mapping(data), name


def _material_info(gltf: Any, index: int | None, ctx: ImportContext) -> dict[str, Any]:
    """Краткое описание материала; файл текстуры становится зависимостью."""
    if index is None or index >= len(gltf.materials):
        return {}
    material = gltf.materials[index]
    pbr = material.pbrMetallicRoughness
    info: dict[str, Any] = {
        "name": material.name or f"material{index}",
        "base_color": list(pbr.baseColorFactor) if pbr else [1.0, 1.0, 1.0, 1.0],
        "base_color_texture": None,
    }
    texture = pbr.baseColorTexture if pbr else None
    if texture is not None and texture.index is not None:
        source = gltf.textures[texture.index].source
        uri = gltf.images[source].uri if source is not None else None
        if uri and not uri.startswith("data:"):
            info["base_color_texture"] = uri
            candidate = ctx.source.parent / uri
            if candidate.is_file():
                ctx.add_dependency(candidate)
    return info
