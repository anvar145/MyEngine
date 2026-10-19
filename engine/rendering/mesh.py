"""Меши на GPU: вершинный формат, VAO/VBO/EBO и отрисовка.

Формат вершин описывается :class:`VertexLayout`, данные упаковываются в один
interleaved-буфер (лучшая локальность кэша вершин) и один раз загружаются в
``GL_STATIC_DRAW``. Меш умеет рисоваться обычным и инстансным вызовом — это
задел для батчинга повторяющихся объектов (трава, пропсы, деревья).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from engine.core.math3d import Vector3
from engine.rendering.errors import MeshUploadError
from engine.rendering.gl_debug import get_gl as _gl
from engine.rendering.gl_debug import void_pointer

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources import MeshAsset

__all__ = [
    "GpuInstanceBuffer",
    "GpuMesh",
    "VertexAttribute",
    "VertexLayout",
]

_log = logging.getLogger("engine.rendering.mesh")

#: Сколько float занимает каждый стандартный атрибут вершины.
ATTRIBUTE_SIZES: dict[str, int] = {
    "position": 3,
    "normal": 3,
    "tangent": 4,
    "uv": 2,
    "uv2": 2,
    "color": 4,
    "bone_weights": 4,
    "bone_indices": 4,
}


@dataclass(frozen=True, slots=True)
class VertexAttribute:
    """Один атрибут вершины в interleaved-буфере."""

    name: str
    location: int
    components: int
    offset: int

    @property
    def size_bytes(self) -> int:
        """Размер атрибута в байтах (все атрибуты — float32)."""
        return self.components * 4

    def __repr__(self) -> str:
        return (
            f"<attr {self.name} loc={self.location}"
            f" x{self.components} @{self.offset}B>"
        )


class VertexLayout:
    """Описание формата вершин.

    Args:
        attributes: Атрибуты в порядке упаковки (они же занимают locations).
        dtype: Тип компонент; движок использует ``float32``.

    Raises:
        ValueError: Если атрибут неизвестен или список пуст.
    """

    __slots__ = ("_attributes", "_by_name", "dtype", "stride")

    def __init__(
        self,
        attributes: Sequence[VertexAttribute],
        dtype: type = np.float32,
    ) -> None:
        if not attributes:
            raise ValueError("Формат вершин не может быть пустым")
        self._attributes = tuple(attributes)
        self._by_name = {attribute.name: attribute for attribute in attributes}
        self.dtype = np.dtype(dtype)
        self.stride = sum(attribute.size_bytes for attribute in attributes)

    # ------------------------------------------------------------------ #
    # Создание
    # ------------------------------------------------------------------ #
    @classmethod
    def from_names(
        cls, names: Sequence[str], *, dtype: type = np.float32, base_location: int = 0
    ) -> VertexLayout:
        """Собрать формат по именам атрибутов.

        Args:
            names: Имена из :data:`ATTRIBUTE_SIZES`
                (например, ``("position", "normal", "uv")``).
            dtype: Тип компонент.
            base_location: С какого location начинать нумерацию.

        Returns:
            Формат вершин.

        Raises:
            ValueError: Если имя атрибута неизвестно.
        """
        attributes: list[VertexAttribute] = []
        offset = 0
        for index, name in enumerate(names):
            components = ATTRIBUTE_SIZES.get(name)
            if components is None:
                raise ValueError(
                    f"Неизвестный атрибут '{name}'; доступны: {sorted(ATTRIBUTE_SIZES)}"
                )
            attributes.append(
                VertexAttribute(
                    name=name,
                    location=base_location + index,
                    components=components,
                    offset=offset,
                )
            )
            offset += components * 4
        return cls(attributes, dtype)

    # ------------------------------------------------------------------ #
    # Свойства
    # ------------------------------------------------------------------ #
    @property
    def attributes(self) -> tuple[VertexAttribute, ...]:
        """Атрибуты в порядке упаковки."""
        return self._attributes

    @property
    def names(self) -> tuple[str, ...]:
        """Имена атрибутов."""
        return tuple(attribute.name for attribute in self._attributes)

    @property
    def components(self) -> int:
        """Суммарное число компонент на вершину."""
        return sum(attribute.components for attribute in self._attributes)

    def attribute(self, name: str) -> VertexAttribute:
        """Вернуть атрибут по имени.

        Raises:
            KeyError: Если атрибута нет в формате.
        """
        return self._by_name[name]

    def has(self, name: str) -> bool:
        """``True``, если атрибут есть в формате."""
        return name in self._by_name

    def describe(self) -> str:
        """Краткое описание формата для логов."""
        parts = ", ".join(
            f"{attribute.name}:{attribute.components}" for attribute in self._attributes
        )
        return f"{parts} (stride {self.stride}B)"

    # ------------------------------------------------------------------ #
    # Упаковка данных
    # ------------------------------------------------------------------ #
    def interleave(
        self, arrays: Mapping[str, np.ndarray], vertex_count: int | None = None
    ) -> np.ndarray:
        """Упаковать отдельные массивы атрибутов в один interleaved-буфер.

        Args:
            arrays: Отображение «имя атрибута → массив ``(N, components)``».
            vertex_count: Число вершин; по умолчанию определяется по данным.

        Returns:
            Массив ``(N, components)`` типа ``float32``.

        Raises:
            ValueError: Если массивов нет или их длины не совпадают.
        """
        if not arrays:
            raise ValueError("Нет данных для упаковки вершин")
        if vertex_count is None:
            vertex_count = int(next(iter(arrays.values())).shape[0])

        buffer = np.zeros((vertex_count, self.components), dtype=np.float32)
        for attribute in self._attributes:
            data = arrays.get(attribute.name)
            if data is None:
                continue  # атрибут заполнится нулями
            array = np.asarray(data, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            if array.shape[0] != vertex_count:
                raise ValueError(
                    f"Атрибут '{attribute.name}': {array.shape[0]} вершин "
                    f"вместо {vertex_count}"
                )
            if array.shape[1] != attribute.components:
                raise ValueError(
                    f"Атрибут '{attribute.name}': {array.shape[1]} компонент вместо "
                    f"{attribute.components}"
                )
            start = attribute.offset // 4
            buffer[:, start : start + attribute.components] = array
        return buffer

    def __repr__(self) -> str:
        return f"<VertexLayout {self.describe()}>"


class GpuMesh:
    """Меш, загруженный в GPU (VAO + VBO, при наличии — EBO).

    Args:
        layout: Формат вершин.
        vertex_data: Упакованные вершины ``(N, layout.components)``.
        index_data: Индексы ``(M,)``; ``None`` — рисование без индексов.
        name: Имя меша (для логов и панели статистики).
        bounds: Ограничивающий параллелепипед ``(min, max)``.
    """

    __slots__ = (
        "_aabb_max",
        "_aabb_min",
        "_cython_arrays",
        "_index_buffer",
        "_index_count",
        "_is_indexed",
        "_layout",
        "_vao",
        "_vertex_buffer",
        "_vertex_count",
        "_vertex_data",
        "_index_data",
        "name",
    )

    def __init__(
        self,
        layout: VertexLayout,
        vertex_data: np.ndarray,
        index_data: np.ndarray | None = None,
        *,
        name: str = "mesh",
        bounds: tuple[Vector3, Vector3] | None = None,
    ) -> None:
        if vertex_data.ndim != 2 or vertex_data.shape[1] != layout.components:
            raise MeshUploadError(
                f"Ожидался массив ({layout.components} компонент), "
                f"получено {getattr(vertex_data, 'shape', None)}"
            )
        self.name = name
        self._layout = layout
        self._vertex_data = np.ascontiguousarray(vertex_data, dtype=np.float32)
        self._index_data = (
            None
            if index_data is None
            else np.ascontiguousarray(index_data, dtype=np.uint32)
        )
        self._vertex_count = int(self._vertex_data.shape[0])
        self._index_count = (
            0 if self._index_data is None else int(self._index_data.size)
        )
        self._is_indexed = self._index_data is not None
        self._vao = 0
        self._vertex_buffer = 0
        self._index_buffer = 0
        self._cython_arrays: tuple[np.ndarray, ...] = ()
        if bounds is not None:
            self._aabb_min, self._aabb_max = bounds
        else:
            self._aabb_min, self._aabb_max = self._compute_bounds()

    # ------------------------------------------------------------------ #
    # Создание
    # ------------------------------------------------------------------ #
    @classmethod
    def from_mesh_asset(
        cls, asset: MeshAsset, layout: VertexLayout | None = None
    ) -> GpuMesh:
        """Подготовить меш к загрузке из ассета (данные ещё не на GPU).

        Args:
            asset: Загруженный меш (вершины, нормали, UV, индексы).
            layout: Формат вершин; по умолчанию position+normal+uv.

        Returns:
            Меш, готовый к :meth:`upload`.
        """
        attributes: list[str] = ["position"]
        arrays: dict[str, np.ndarray] = {"position": asset.positions}
        if asset.normals is not None:
            attributes.append("normal")
            arrays["normal"] = asset.normals
        if asset.uvs is not None:
            attributes.append("uv")
            arrays["uv"] = asset.uvs
        vertex_layout = layout or VertexLayout.from_names(attributes)
        packed = vertex_layout.interleave(arrays, asset.vertex_count)
        return cls(
            vertex_layout,
            packed,
            asset.indices,
            name=asset.name,
            bounds=asset.bounds,
        )

    @classmethod
    def from_arrays(
        cls,
        positions: np.ndarray,
        *,
        normals: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        name: str = "mesh",
    ) -> GpuMesh:
        """Собрать меш из отдельных массивов.

        Args:
            positions: Позиции ``(N, 3)``.
            normals: Нормали ``(N, 3)``.
            uvs: Текстурные координаты ``(N, 2)``.
            indices: Индексы ``(M,)``.
            name: Имя меша.

        Returns:
            Меш, готовый к :meth:`upload`.
        """
        attributes = ["position"]
        arrays: dict[str, np.ndarray] = {"position": positions}
        if normals is not None:
            attributes.append("normal")
            arrays["normal"] = normals
        if uvs is not None:
            attributes.append("uv")
            arrays["uv"] = uvs
        layout = VertexLayout.from_names(attributes)
        packed = layout.interleave(arrays, int(np.asarray(positions).shape[0]))
        return cls(layout, packed, indices, name=name)

    # ------------------------------------------------------------------ #
    # Свойства
    # ------------------------------------------------------------------ #
    @property
    def layout(self) -> VertexLayout:
        """Формат вершин."""
        return self._layout

    @property
    def vertex_count(self) -> int:
        """Число вершин."""
        return self._vertex_count

    @property
    def index_count(self) -> int:
        """Число индексов (0, если меш неиндексированный)."""
        return self._index_count

    @property
    def triangle_count(self) -> int:
        """Число треугольников."""
        elements = self._index_count if self._is_indexed else self._vertex_count
        return elements // 3

    @property
    def is_indexed(self) -> bool:
        """``True``, если меш рисуется по индексам."""
        return self._is_indexed

    @property
    def is_uploaded(self) -> bool:
        """``True``, если буферы уже созданы в драйвере."""
        return self._vao != 0

    @property
    def bounds(self) -> tuple[Vector3, Vector3]:
        """Ограничивающий параллелепипед ``(min, max)``."""
        return self._aabb_min, self._aabb_max

    def cpu_triangles(self) -> tuple[np.ndarray, np.ndarray]:
        """Позиции ``(n, 3)`` и треугольники ``(m, 3)`` из CPU-копии вершин.

        Нужны точному пикингу редактора; данные остаются в памяти и после
        :meth:`upload`.
        """
        offset = self._layout.attribute("position").offset // 4
        positions = self._vertex_data[:, offset : offset + 3]
        if self._index_data is not None:
            indices = self._index_data
        else:
            indices = np.arange(self._vertex_count, dtype=np.uint32)
        count = (indices.size // 3) * 3
        return positions, indices[:count].reshape(-1, 3)

    @property
    def gpu_bytes(self) -> int:
        """Объём занимаемой видеопамяти (вершины + индексы)."""
        total = int(self._vertex_data.nbytes)
        if self._index_data is not None:
            total += int(self._index_data.nbytes)
        return total

    def _compute_bounds(self) -> tuple[Vector3, Vector3]:
        """Посчитать AABB по позициям вершин."""
        offset = self._layout.attribute("position").offset // 4
        positions = self._vertex_data[:, offset : offset + 3]
        if positions.size == 0:
            return Vector3.zero(), Vector3.zero()
        minimum = positions.min(axis=0)
        maximum = positions.max(axis=0)
        return (
            Vector3(float(minimum[0]), float(minimum[1]), float(minimum[2])),
            Vector3(float(maximum[0]), float(maximum[1]), float(maximum[2])),
        )

    # ------------------------------------------------------------------ #
    # Загрузка и отрисовка
    # ------------------------------------------------------------------ #
    def upload(self) -> int:
        """Создать буферы и VAO в драйвере.

        Returns:
            Идентификатор VAO.

        Raises:
            MeshUploadError: Если буферы не удалось создать.
        """
        GL = _gl()

        if self._vao:
            return self._vao
        if self._vertex_count == 0:
            raise MeshUploadError(f"Меш '{self.name}' не содержит вершин")

        vao = int(GL.glGenVertexArrays(1))
        vbo = int(GL.glGenBuffers(1))
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            self._vertex_data.nbytes,
            self._vertex_data,
            GL.GL_STATIC_DRAW,
        )
        for attribute in self._layout.attributes:
            GL.glEnableVertexAttribArray(attribute.location)
            GL.glVertexAttribPointer(
                attribute.location,
                attribute.components,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                self._layout.stride,
                void_pointer(attribute.offset),
            )

        index_buffer = 0
        if self._index_data is not None:
            index_buffer = int(GL.glGenBuffers(1))
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer)
            GL.glBufferData(
                GL.GL_ELEMENT_ARRAY_BUFFER,
                self._index_data.nbytes,
                self._index_data,
                GL.GL_STATIC_DRAW,
            )

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        self._vao = vao
        self._vertex_buffer = vbo
        self._index_buffer = index_buffer
        _log.debug(
            "Меш '%s' загружен: %d вершин, %d индексов, %d Б",
            self.name,
            self._vertex_count,
            self._index_count,
            self.gpu_bytes,
        )
        return vao

    def bind(self) -> int:
        """Загрузить (если нужно) и привязать VAO.

        Returns:
            Идентификатор VAO.
        """
        GL = _gl()

        vao = self.upload()
        GL.glBindVertexArray(vao)
        return vao

    def draw(self, mode: int | None = None, count: int | None = None) -> int:
        """Нарисовать меш.

        Args:
            mode: Примитив OpenGL (по умолчанию треугольники).
            count: Число элементов; ``None`` — все.

        Returns:
            Число отрисованных элементов.
        """
        GL = _gl()

        primitive = GL.GL_TRIANGLES if mode is None else mode
        self.bind()
        if self._is_indexed:
            elements = self._index_count if count is None else min(
                count, self._index_count
            )
            GL.glDrawElements(primitive, elements, GL.GL_UNSIGNED_INT, None)
        else:
            elements = self._vertex_count if count is None else min(
                count, self._vertex_count
            )
            GL.glDrawArrays(primitive, 0, elements)
        return elements

    def draw_instanced(self, instance_count: int, mode: int | None = None) -> int:
        """Нарисовать меш несколько раз за один вызов (GPU instancing).

        Требует привязки :class:`GpuInstanceBuffer` к тому же VAO.

        Args:
            instance_count: Число экземпляров.
            mode: Примитив OpenGL.

        Returns:
            Число отрисованных элементов геометрии.
        """
        GL = _gl()

        if instance_count <= 0 or not hasattr(GL, "glDrawElementsInstanced"):
            return 0
        primitive = GL.GL_TRIANGLES if mode is None else mode
        self.bind()
        if self._is_indexed:
            # Указатель на индексы берётся из привязанного EBO, поэтому нужен
            # явный нулевой указатель (не ``None``).
            GL.glDrawElementsInstanced(
                primitive,
                self._index_count,
                GL.GL_UNSIGNED_INT,
                void_pointer(0),
                instance_count,
            )
            return self._index_count
        GL.glDrawArraysInstanced(primitive, 0, self._vertex_count, instance_count)
        return self._vertex_count

    def dispose(self) -> None:
        """Освободить буферы и VAO."""
        GL = _gl()

        if self._index_buffer:
            GL.glDeleteBuffers(1, [self._index_buffer])
            self._index_buffer = 0
        if self._vertex_buffer:
            GL.glDeleteBuffers(1, [self._vertex_buffer])
            self._vertex_buffer = 0
        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
            self._vao = 0

    def __repr__(self) -> str:
        state = "gpu" if self.is_uploaded else "cpu"
        return (
            f"<GpuMesh '{self.name}' {self.vertex_count}в/{self.triangle_count}тр "
            f"{state} [{self._layout.describe()}]>"
        )


class GpuInstanceBuffer:
    """Буфер матриц экземпляров для инстансной отрисовки.

    Каждой матрице соответствуют четыре атрибута вершин (locations 8..11) с
    делителем 1 — так данные читаются по одному разу на экземпляр, а не на
    вершину. Это готовый фундамент батчинга: одна геометрия, тысячи объектов.
    """

    __slots__ = (
        "_attached_vao",
        "_buffer",
        "_capacity",
        "_count",
        "_first_location",
        "name",
    )

    def __init__(
        self,
        capacity: int = 256,
        *,
        first_location: int = 8,
        name: str = "instances",
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity должна быть положительной")
        self.name = name
        self._first_location = int(first_location)
        self._capacity = int(capacity)
        self._count = 0
        self._buffer = 0
        self._attached_vao = 0

    @property
    def count(self) -> int:
        """Число записанных экземпляров."""
        return self._count

    @property
    def capacity(self) -> int:
        """Вместимость буфера."""
        return self._capacity

    @property
    def first_location(self) -> int:
        """Первый location, занятый матрицами экземпляров."""
        return self._first_location

    def create(self) -> int:
        """Создать буфер в драйвере."""
        GL = _gl()

        if not self._buffer:
            self._buffer = int(GL.glGenBuffers(1))
        return self._buffer

    def upload(self, matrices: np.ndarray) -> int:
        """Загрузить матрицы экземпляров.

        Матрицы принимаются в соглашениях движка (row-major, сдвиг в последнем
        столбце) и транспонируются при загрузке: GLSL собирает ``mat4`` из
        столбцов, поэтому в буфере должны лежать столбцы исходной матрицы.

        Args:
            matrices: Массив ``(N, 4, 4)``.

        Returns:
            Число загруженных экземпляров.

        Raises:
            ValueError: Если форма массива не подходит или он не влезает.
        """
        GL = _gl()

        array = np.asarray(matrices, dtype=np.float32)
        if array.ndim != 3 or array.shape[1:] != (4, 4):
            raise ValueError(f"Ожидался массив (N, 4, 4), получено {array.shape}")
        count = int(array.shape[0])
        if count > self._capacity:
            raise ValueError(
                f"Экземпляров {count} больше вместимости буфера {self._capacity}"
            )
        self.create()
        payload = np.ascontiguousarray(array.transpose(0, 2, 1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffer)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            int(payload.nbytes),
            payload,
            GL.GL_DYNAMIC_DRAW,
        )
        self._count = count
        return count

    def attach(self, vao: int, *, keep_bound: bool = False) -> None:
        """Настроить атрибуты экземпляров у указанного VAO.

        Args:
            vao: Идентификатор VAO, которому принадлежит буфер.
            keep_bound: Оставить VAO привязанным. Атрибуты экземпляров хранятся
                в состоянии VAO, поэтому отрисовку нужно выполнять с тем же
                привязанным VAO — иначе драйвер возьмёт VAO без атрибутов и
                экземпляры нарисуются в начале координат.
        """
        GL = _gl()

        self.create()
        GL.glBindVertexArray(vao)
        if vao == self._attached_vao:
            # Указатели атрибутов живут в VAO и ссылаются на объект буфера, а
            # не на его содержимое: повторная настройка — 14 лишних вызовов GL.
            if not keep_bound:
                GL.glBindVertexArray(0)
            return
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffer)
        row_size = 4 * 4
        for row in range(4):
            location = self._first_location + row
            GL.glEnableVertexAttribArray(location)
            GL.glVertexAttribPointer(
                location,
                4,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                row_size * 4,
                void_pointer(row * row_size),
            )
            GL.glVertexAttribDivisor(location, 1)
        self._attached_vao = vao
        if not keep_bound:
            GL.glBindVertexArray(0)

    def dispose(self) -> None:
        """Удалить буфер."""
        GL = _gl()

        if self._buffer:
            GL.glDeleteBuffers(1, [self._buffer])
            self._buffer = 0
        self._attached_vao = 0
        self._count = 0

    def __repr__(self) -> str:
        return f"<GpuInstanceBuffer '{self.name}' {self._count}/{self._capacity}>"
