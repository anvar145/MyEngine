"""Импортёр FBX (``.fbx``, бинарный и ASCII) на базе ``ufbx`` (extra ``[fbx]``).

Работает как glTF-импортёр: обходит узлы сцены, запекает мировые матрицы в
вершины, а каждую пару (узел, материал меша) кукует в отдельный артефакт
``mesh_NNN.bin`` (LOD, квантизация) с описанием материала в ``info``.
Материалы конвертируются в ``material_NNN.json`` (+ упакованную ORM-текстуру).

Систему координат и единицы ``ufbx`` приводит сам: оси правая Y-вверх, единица
измерения — метр (как в glTF и остальном движке).

Не поддержано (с предупреждением в логе): скиннинг и анимация (меш берётся в
позе привязки), морф-таргеты, NURBS, вогнутые многоугольники (триангулируются
веером от первой вершины), встроенные в файл текстуры.

``ufbx`` — нативная библиотека: временные объекты её списков (например,
``mesh.materials[0].pbr``) нельзя использовать цепочкой, поэтому каждый
промежуточный объект здесь хранится в переменной.
"""

from __future__ import annotations

import gc
import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.core.resources.errors import AssetImportError
from engine.core.resources.importer import Artifact, AssetPreview, ImportContext
from engine.core.resources.importers.fbx_worker import run_fbx_worker
from engine.core.resources.importers.material_importer import build_material_artifacts
from engine.core.resources.primitives import compute_normals

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources.cookers.material_cooker import MaterialSpec

__all__ = ["FbxImporter", "triangulate_faces", "weld_corners"]

_BINARY_MAGIC = b"Kaydara FBX Binary  \x00"
_ASCII_MAGIC = b"; FBX"

_NO_UFBX = (
    "Для импорта FBX нужен пакет ufbx: "
    "установите extra [fbx] (pip install myengine[fbx])"
)


class FbxImporter:
    """``.fbx`` → набор артефактов ``mesh`` (``mesh_NNN.bin``).

    Настройки:
        scale (float): множитель координат (после перевода в метры);
        generate_normals (bool): считать нормали, если их нет в файле;
        generate_lods (bool), lod_ratios (list[float]): параметры LOD;
        extract_materials (bool): конвертировать материалы в ``material_NNN.json``.
    """

    name = "fbx"
    extensions = (".fbx",)
    version = 1

    #: Разбирать ли файл в дочернем процессе (защита от краша нативной ``ufbx``).
    isolated = True

    #: Предельное время разбора в дочернем процессе, секунды.
    worker_timeout = 300.0

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
        """Определить вид (бинарный / ASCII) и версию по заголовку, без ``ufbx``."""
        try:
            with path.open("rb") as stream:
                head = stream.read(64)
        except OSError as exc:
            raise AssetImportError(f"'{path.name}': {exc}") from exc
        if head.startswith(_BINARY_MAGIC) and len(head) >= 27:
            version = int.from_bytes(head[23:27], "little")
            return AssetPreview("model", {"format": "binary", "version": version})
        if head.startswith(_ASCII_MAGIC):
            return AssetPreview("model", {"format": "ascii", "version": None})
        raise AssetImportError(f"'{path.name}': это не FBX")

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        """Разобрать модель и записать по артефакту на каждую часть меша.

        По умолчанию разбор идёт в дочернем процессе (:attr:`isolated`): нативный
        краш ``ufbx`` не должен ронять редактор.
        """
        if self.isolated:
            if importlib.util.find_spec("ufbx") is None:
                raise AssetImportError(_NO_UFBX)
            return run_fbx_worker(ctx, self.worker_timeout)
        return self.import_in_process(ctx)

    def import_in_process(self, ctx: ImportContext) -> list[Artifact]:
        """То же, что :meth:`import_asset`, но в текущем процессе (без изоляции)."""
        try:
            import ufbx
        except ImportError as exc:
            raise AssetImportError(_NO_UFBX) from exc
        self.inspect(ctx.source)
        make_normals = bool(ctx.setting("generate_normals", True))
        try:
            scene = ufbx.load_file(
                str(ctx.source),
                target_axes=ufbx.axes_right_handed_y_up,
                target_unit_meters=1.0,
                generate_missing_normals=make_normals,
                ignore_animation=True,
            )
        except Exception as exc:  # noqa: BLE001 - ufbx бросает свои типы ошибок
            raise AssetImportError(f"'{ctx.source.name}': {exc}") from exc
        try:
            return self._build(scene, ctx, make_normals)
        finally:
            # Объекты ufbx не держат сцену живой: все производные должны быть
            # собраны до неё, иначе сборщик мусора обращается к освобождённой памяти.
            gc.collect()
            del scene

    def _build(
        self, scene: Any, ctx: ImportContext, make_normals: bool
    ) -> list[Artifact]:
        """Записать артефакты по загруженной сцене ``ufbx``."""
        from engine.core.resources.cookers import MeshCookSettings, cook_mesh

        scale = float(ctx.setting("scale", 1.0))
        cook = MeshCookSettings.from_mapping(dict(ctx.settings))
        hold: list[Any] = []  # обёртки ufbx живут до конца разбора
        materials = list(scene.materials)
        hold.append(materials)
        material_artifacts: list[Artifact] = []
        material_files: dict[int, str | None] = {}

        def material_json(index: int | None) -> str | None:
            """Собрать материал при первом обращении; ``None`` — не удалось."""
            if index is None or not bool(ctx.setting("extract_materials", True)):
                return None
            if index not in material_files:
                material_files[index] = None
                try:
                    spec, name = _material_spec(materials[index], ctx)
                    built = build_material_artifacts(
                        ctx, spec, ctx.source.parent, f"material_{index:03d}", name
                    )
                except AssetImportError as exc:
                    ctx.log.warning("Материал %d пропущен: %s", index, exc)
                else:
                    material_artifacts.extend(built)
                    material_files[index] = built[0].name
            return material_files[index]

        artifacts: list[Artifact] = []
        nodes = list(scene.nodes)
        hold.append(nodes)
        for node in nodes:
            mesh = node.mesh
            hold.append(mesh)
            if mesh is None:
                continue
            if len(mesh.skin_deformers) or len(mesh.blend_deformers):
                ctx.log.warning(
                    "Меш '%s': скиннинг и морфы не поддержаны, поза привязки",
                    node.name,
                )
            matrix = _world_matrix(node.geometry_to_world)
            for part_index, part in enumerate(_mesh_parts(mesh, hold)):
                positions, indices, normals, uvs = part.geometry
                positions, normals, indices = _apply_matrix(
                    positions, normals, indices, matrix, scale
                )
                if normals is None and make_normals:
                    normals = compute_normals(positions, indices)
                name = f"mesh_{len(artifacts):03d}.bin"
                (ctx.output_dir / name).write_bytes(
                    cook_mesh(positions, indices, normals, uvs, cook)
                )
                low, high = positions.min(axis=0), positions.max(axis=0)
                material = None if part.material is None else materials[part.material]
                artifacts.append(
                    ctx.artifact(
                        "mesh",
                        name,
                        node=node.name or f"node{node.typed_id}",
                        mesh=mesh.name or f"mesh{mesh.typed_id}",
                        primitive=part_index,
                        vertices=int(positions.shape[0]),
                        triangles=int(indices.size // 3),
                        has_normals=normals is not None,
                        has_uvs=uvs is not None,
                        material=_material_info(material, ctx),
                        material_artifact=material_json(part.material),
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
# Геометрия
# ---------------------------------------------------------------------- #
class _Part:
    """Часть меша с одним материалом (после сварки вершин)."""

    __slots__ = ("geometry", "material")

    def __init__(self, geometry: Any, material: int | None) -> None:
        self.geometry = geometry
        self.material = material


def triangulate_faces(
    begins: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Триангулировать многоугольники веером от первой вершины.

    Args:
        begins: Индекс первого угла каждой грани.
        counts: Число углов каждой грани (грани с числом углов < 3 пропускаются).

    Returns:
        Кортеж ``(corners, faces)``: ``corners`` формы ``(T, 3)`` — индексы углов
        каждого треугольника, ``faces`` формы ``(T,)`` — номер исходной грани.
    """
    begins = np.asarray(begins, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    per_face = np.maximum(counts - 2, 0)
    total = int(per_face.sum())
    faces = np.repeat(np.arange(begins.size), per_face)
    first = np.cumsum(per_face) - per_face
    step = np.arange(total) - np.repeat(first, per_face)
    base = begins[faces]
    corners = np.stack([base, base + step + 1, base + step + 2], axis=1)
    return corners, faces


def weld_corners(
    corners: np.ndarray, *attribute_indices: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Склеить углы с одинаковым набором индексов атрибутов в вершины.

    Args:
        corners: Индексы углов треугольников формы ``(T, 3)``.
        *attribute_indices: Для каждого атрибута массив «угол → индекс значения»
            (``None`` — атрибута нет).

    Returns:
        ``(first_corner, triangles)``: для каждой уникальной вершины индекс
        первого угла, который её породил, и треугольники в новых индексах.
    """
    flat = corners.reshape(-1)
    columns = [a[flat] for a in attribute_indices if a is not None]
    keys = np.stack(columns, axis=1) if columns else flat[:, None]
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return flat[first], inverse.reshape(-1, 3).astype(np.uint32)


def _mesh_parts(mesh: Any, hold: list[Any]) -> list[_Part]:
    """Разбить меш ufbx по материалам и свести к плоским массивам numpy."""
    faces = mesh.faces
    if not len(faces):
        return []
    spans = np.array([(f.index_begin, f.num_indices) for f in faces], dtype=np.int64)
    corners, face_of = triangulate_faces(spans[:, 0], spans[:, 1])
    if corners.size == 0:
        return []

    position = mesh.vertex_position
    pos_values = np.array(position.values, dtype=np.float32).reshape(-1, 3)
    pos_index = np.array(position.indices, dtype=np.int64)
    normal = mesh.vertex_normal
    uv = mesh.vertex_uv
    nrm_values = nrm_index = uv_values = uv_index = None
    if normal.exists:
        nrm_values = np.array(normal.values, dtype=np.float32).reshape(-1, 3)
        nrm_index = np.array(normal.indices, dtype=np.int64)
    if uv.exists:
        uv_values = np.array(uv.values, dtype=np.float32).reshape(-1, 2)
        uv_index = np.array(uv.indices, dtype=np.int64)

    face_material = np.array(mesh.face_material, dtype=np.int64)
    if face_material.size != len(faces):
        face_material = np.zeros(len(faces), dtype=np.int64)
    triangle_material = face_material[face_of]
    mesh_materials = list(mesh.materials)
    hold.append(mesh_materials)
    scene_materials = [m.typed_id for m in mesh_materials]

    parts: list[_Part] = []
    for slot in np.unique(triangle_material):
        chosen = corners[triangle_material == slot]
        first, triangles = weld_corners(chosen, pos_index, nrm_index, uv_index)
        positions = pos_values[pos_index[first]]
        normals = nrm_values[nrm_index[first]] if nrm_values is not None else None
        uvs = uv_values[uv_index[first]] if uv_values is not None else None
        material = scene_materials[slot] if slot < len(scene_materials) else None
        parts.append(
            _Part((positions, triangles.reshape(-1), normals, uvs), material)
        )
    return parts


def _world_matrix(matrix: Any) -> np.ndarray:
    """Матрица ufbx (три столбца базиса и сдвиг) в 4x4."""
    result = np.eye(4)
    for column, vec in enumerate((matrix.c0, matrix.c1, matrix.c2, matrix.c3)):
        result[:3, column] = (vec.x, vec.y, vec.z)
    return result


def _apply_matrix(
    positions: np.ndarray,
    normals: np.ndarray | None,
    indices: np.ndarray,
    matrix: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Запечь мировую матрицу и масштаб в позиции и нормали."""
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
    return (positions * scale).astype(np.float32), normals, indices


# ---------------------------------------------------------------------- #
# Материалы
# ---------------------------------------------------------------------- #
_TEXTURE_SLOTS = (
    ("base_color", "base_color"),
    ("normal", "normal_map"),
    ("ao", "ambient_occlusion"),
    ("emissive", "emission_color"),
    ("metallic", "metalness"),
    ("roughness", "roughness"),
)


def _scalar(map_: Any, default: float) -> float:
    return float(map_.value_vec4.x) if map_.has_value else default


def _texture_file(texture: Any, base_dir: Path) -> str | None:
    """Найти файл текстуры рядом с моделью (FBX хранит пути автора).

    Ищется только внутри каталога модели: по относительному пути, затем по
    имени файла в нём самом, в ``textures/`` и в ``<каталог>.fbm/``.
    """
    if texture is None:
        return None
    for raw in (texture.relative_filename, texture.filename):
        if not raw:
            continue
        path = Path(raw.replace("\\", "/"))
        options = [base_dir / "textures" / path.name, base_dir / path.name]
        options.append(base_dir / (base_dir.name + ".fbm") / path.name)
        if not path.is_absolute():
            options.insert(0, base_dir / path)
        for option in options:
            if option.is_file():
                return Path(os.path.relpath(option, base_dir)).as_posix()
    return None


def _material_spec(material: Any, ctx: ImportContext) -> tuple[MaterialSpec, str]:
    """Перевести материал ufbx в :class:`MaterialSpec`."""
    from engine.core.resources.cookers.material_cooker import MaterialSpec

    pbr = material.pbr
    name = material.name or f"material{material.typed_id}"
    base = pbr.base_color
    color = base.value_vec4 if base.has_value else None
    rgb = [1.0, 1.0, 1.0] if color is None else [color.x, color.y, color.z]
    emission = pbr.emission_color
    factor = _scalar(pbr.emission_factor, 1.0)
    glow = emission.value_vec4 if emission.has_value else None
    emissive = [0.0, 0.0, 0.0] if glow is None else [
        glow.x * factor, glow.y * factor, glow.z * factor
    ]
    opacity = _scalar(pbr.opacity, 1.0)

    textures: dict[str, str] = {}
    for slot, attribute in _TEXTURE_SLOTS:
        map_ = getattr(pbr, attribute)
        texture = map_.texture if map_.texture_enabled else None
        found = _texture_file(texture, ctx.source.parent)
        if found:
            textures[slot] = found
        elif texture is not None:
            ctx.log.warning("Текстура '%s' материала '%s' не найдена", slot, name)
    data: dict[str, Any] = {
        "name": name,
        "base_color": [*rgb, opacity],
        "metallic": min(max(_scalar(pbr.metalness, 0.0), 0.0), 1.0),
        "roughness": min(max(_scalar(pbr.roughness, 0.8), 0.0), 1.0),
        "emissive": emissive,
        "alpha_mode": "blend" if opacity < 1.0 else "opaque",
        "double_sided": bool(material.features.double_sided.enabled),
        "textures": textures,
    }
    return MaterialSpec.from_mapping(data), name


def _material_info(material: Any, ctx: ImportContext) -> dict[str, Any]:
    """Краткое описание материала; файл текстуры становится зависимостью."""
    if material is None:
        return {}
    pbr = material.pbr
    base = pbr.base_color
    color = base.value_vec4 if base.has_value else None
    info: dict[str, Any] = {
        "name": material.name or f"material{material.typed_id}",
        "base_color": [1.0] * 4 if color is None else [color.x, color.y, color.z, 1.0],
        "base_color_texture": None,
    }
    texture = base.texture if base.texture_enabled else None
    found = _texture_file(texture, ctx.source.parent)
    if found:
        info["base_color_texture"] = found
        ctx.add_dependency(ctx.source.parent / found)
    return info
