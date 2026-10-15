"""Отладочная визуализация физики: каркасы коллайдеров линиями.

Построение геометрии (:func:`build_debug_lines`) — чистый numpy без GL, его
проверяют тесты без контекста. :class:`PhysicsDebugPass` — проход графа кадра,
который грузит линии в динамический буфер и рисует их одним вызовом.

Цвета: статика — серый, динамика — зелёный, кинематика — синий, триггер —
жёлтый, персонаж — оранжевый. Спящее динамическое тело темнее.

Слои: модуль зависит только от ``engine.physics`` (форм и мира) и GL; Qt и
редактор он не знает.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.physics.shape import BoxShape, CapsuleShape, SphereShape

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.rendering.render_graph import FrameContext

__all__ = ["COLORS", "PhysicsDebugPass", "build_debug_lines", "mesh_outline"]

_log = logging.getLogger("engine.rendering.physics_debug")

#: Цвета по виду коллайдера (RGB, 0..1).
COLORS: dict[str, tuple[float, float, float]] = {
    "static": (0.55, 0.55, 0.58),
    "dynamic": (0.20, 0.90, 0.30),
    "kinematic": (0.25, 0.50, 1.00),
    "trigger": (1.00, 0.90, 0.10),
    "character": (1.00, 0.55, 0.10),
}
_SLEEP_DIM = 0.45
_RING = 24


def _circle(plane: str, radius: float, center_y: float = 0.0) -> np.ndarray:
    """Сегменты окружности ``(_RING, 2, 3)`` в плоскости ``xy``/``yz``/``xz``."""
    angles = np.linspace(0.0, 2.0 * np.pi, _RING + 1)
    a, b = np.cos(angles) * radius, np.sin(angles) * radius
    pts = np.zeros((_RING + 1, 3))
    if plane == "xz":
        pts[:, 0], pts[:, 2] = a, b
    elif plane == "xy":
        pts[:, 0], pts[:, 1] = a, b
    else:  # yz
        pts[:, 1], pts[:, 2] = a, b
    pts[:, 1] += center_y if plane == "xz" else 0.0
    return np.stack([pts[:-1], pts[1:]], axis=1)


@lru_cache(maxsize=256)
def _box_template(half: tuple[float, float, float]) -> np.ndarray:
    signs = np.array(
        [[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], dtype=float
    )
    corners = signs * np.asarray(half)
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),  # вдоль X
        (0, 2), (1, 3), (4, 6), (5, 7),  # вдоль Y
        (0, 4), (1, 5), (2, 6), (3, 7),  # вдоль Z
    ]
    return np.array([[corners[a], corners[b]] for a, b in edges])


@lru_cache(maxsize=256)
def _sphere_template(radius: float) -> np.ndarray:
    return np.concatenate(
        [_circle("xz", radius), _circle("xy", radius), _circle("yz", radius)]
    )


def _arc(plane: str, radius: float, center_y: float, upper: bool) -> np.ndarray:
    """Полуокружность над/под центром в плоскости ``xy`` или ``yz``."""
    count = _RING // 2
    angles = np.linspace(0.0, np.pi, count + 1)
    if not upper:
        angles = -angles
    horiz, vert = np.cos(angles) * radius, np.sin(angles) * radius
    pts = np.zeros((count + 1, 3))
    pts[:, 0 if plane == "xy" else 2] = horiz
    pts[:, 1] = vert + center_y
    return np.stack([pts[:-1], pts[1:]], axis=1)


@lru_cache(maxsize=256)
def _capsule_template(radius: float, half_height: float) -> np.ndarray:
    parts = [
        _circle("xz", radius, half_height),
        _circle("xz", radius, -half_height),
    ]
    for plane in ("xy", "yz"):
        parts.append(_arc(plane, radius, half_height, True))
        parts.append(_arc(plane, radius, -half_height, False))
    sides = []
    for x, z in ((radius, 0), (-radius, 0), (0, radius), (0, -radius)):
        sides.append([[x, half_height, z], [x, -half_height, z]])
    parts.append(np.array(sides, dtype=float))
    return np.concatenate(parts)


_FEATURE_COS = 0.866  # рёбра с углом между гранями > 30° считаются контуром
_OUTLINE_CACHE: weakref.WeakKeyDictionary[Any, np.ndarray] = (
    weakref.WeakKeyDictionary()
)


def mesh_outline(bvh: Any) -> np.ndarray:
    """Контур меша: граничные и острые рёбра, ``(N, 2, 3)`` в локальных осях.

    Полная сетка слишком густа для отладочных линий; плоские участки пола
    остаются пустыми, видны стены, углы и края. Результат кэшируется на BVH.
    """
    cached = _OUTLINE_CACHE.get(bvh)
    if cached is not None:
        return cached
    verts = np.asarray(bvh.vertices, dtype=np.float64)
    tris = np.asarray(bvh.triangles, dtype=np.int64)
    if len(tris) == 0:
        result = np.zeros((0, 2, 3))
        _OUTLINE_CACHE[bvh] = result
        return result
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths > 0, lengths, 1.0)
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    owner = np.tile(np.arange(len(tris)), 3)
    key = np.sort(edges, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, owner, edges = key[order], owner[order], edges[order]
    same = np.all(key[1:] == key[:-1], axis=1)  # пара соседних строк = общее ребро
    keep = np.ones(len(key), dtype=bool)  # граничные (одна грань) - всегда
    keep[1:][same] = False
    keep[:-1][same] = False
    first = np.flatnonzero(same)
    dots = np.einsum("ij,ij->i", normals[owner[first]], normals[owner[first + 1]])
    keep[first[dots < _FEATURE_COS]] = True
    # ребро из трёх и более граней даёт серию пар; лишние дубли безвредны
    chosen = edges[keep]
    result = np.stack([verts[chosen[:, 0]], verts[chosen[:, 1]]], axis=1)
    _OUTLINE_CACHE[bvh] = result
    return result


def _template(shape: Any) -> np.ndarray | None:
    if isinstance(shape, BoxShape):
        return _box_template(tuple(float(v) for v in shape.half_extents))
    if isinstance(shape, SphereShape):
        return _sphere_template(float(shape.radius))
    if isinstance(shape, CapsuleShape):
        return _capsule_template(float(shape.radius), float(shape.half_height))
    return None


def _kind(body: Any, collider: Any, handle: int, characters: Iterable[int]) -> str:
    if handle in characters:
        return "character"
    if collider.is_trigger:
        return "trigger"
    return str(body.body_type)


def build_debug_lines(
    world: Any,
    characters: Iterable[int] = (),
    only_handles: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Собрать линии каркасов всех коллайдеров мира.

    Args:
        world: ``PhysicsWorld`` (нужен ``iter_colliders``).
        characters: Handle'ы тел-персонажей (рисуются оранжевым).
        only_handles: Рисовать только эти тела (``None`` — все).

    Returns:
        ``(positions, colors)``: массивы ``float32`` формы ``(2N, 3)``; пары
        соседних вершин — отрезки ``GL_LINES``. Пусто — ``(0, 3)``.
    """
    characters = frozenset(characters)
    allowed = None if only_handles is None else frozenset(only_handles)
    chunks: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for handle, body, collider in world.iter_colliders():
        if allowed is not None and handle not in allowed:
            continue
        template = _template(collider.shape)
        if template is None:
            continue
        rot = np.asarray(collider.world_rotation, dtype=float)
        pos = np.asarray(collider.world_position, dtype=float)
        points = template.reshape(-1, 3) @ rot.T + pos
        kind = _kind(body, collider, handle, characters)
        color = np.asarray(COLORS[kind], dtype=np.float32)
        if kind == "dynamic" and getattr(body, "sleeping", False):
            color = color * _SLEEP_DIM
        chunks.append(points)
        colors.append(np.broadcast_to(color, points.shape))
    for handle, mesh in getattr(world, "iter_meshes", lambda: ())():
        if allowed is not None and handle not in allowed:
            continue
        edges = mesh_outline(mesh.bvh)
        if len(edges) == 0:
            continue
        points = edges.reshape(-1, 3) @ mesh.matrix.T + mesh.position
        color = np.asarray(
            COLORS["trigger" if mesh.is_trigger else "static"], dtype=np.float32
        )
        chunks.append(points)
        colors.append(np.broadcast_to(color, points.shape))
    if not chunks:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty.copy()
    return (
        np.concatenate(chunks).astype(np.float32),
        np.concatenate(colors).astype(np.float32),
    )


class PhysicsDebugPass:
    """Проход графа кадра: рисует каркасы коллайдеров поверх сцены.

    Args:
        source: Объект с ``world`` и ``character_handles()`` (адаптер сцены) либо
            сам мир; вызываемый объект возвращает источник по требованию (сцена
            может смениться между кадрами).
        depth_test: Скрывать ли линии за геометрией (по умолчанию нет —
            каркасы видны «насквозь»).
    """

    def __init__(self, source: Any = None, *, depth_test: bool = False) -> None:
        self.source = source
        self.enabled = True
        self.depth_test = depth_test
        #: Вызываемый объект: ``source -> handle'ы | None`` (``None`` — все).
        self.handle_filter: Any = None
        self.line_count = 0
        self._vao = 0
        self._vbo = 0
        self._capacity = 0

    @property
    def name(self) -> str:
        """Имя прохода."""
        return "physics_debug"

    def _resolve(self) -> tuple[Any, Iterable[int]] | None:
        source = self.source() if callable(self.source) else self.source
        if source is None:
            return None
        world = getattr(source, "world", source)
        handles = getattr(source, "character_handles", None)
        return world, (handles() if handles is not None else ())

    def _program(self, renderer: Any) -> Any:
        try:
            return renderer.program("debug_line")
        except Exception:  # noqa: BLE001 - регистрируем при первом использовании
            program = renderer._load_program_from_files(  # noqa: SLF001
                "debug_line", "shaders/debug_line.vert", "shaders/debug_line.frag"
            )
            if program is None:
                return None
            return renderer.register_program("debug_line", program)

    def _ensure_buffers(self, vertices: int) -> None:
        from OpenGL import GL

        from engine.rendering.gl_debug import void_pointer

        if not self._vao:
            self._vao = int(GL.glGenVertexArrays(1))
            self._vbo = int(GL.glGenBuffers(1))
            GL.glBindVertexArray(self._vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
            for location, offset in ((0, 0), (1, 12)):
                GL.glEnableVertexAttribArray(location)
                GL.glVertexAttribPointer(
                    location, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, void_pointer(offset)
                )
        if vertices > self._capacity:
            self._capacity = max(vertices, self._capacity * 2, 1024)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, self._capacity * 24, None, GL.GL_DYNAMIC_DRAW
            )

    def execute(self, context: FrameContext) -> None:
        """Нарисовать каркасы коллайдеров."""
        renderer = context.renderer
        if not self.enabled or renderer is None:
            return
        resolved = self._resolve()
        if resolved is None:
            return
        only = None
        if self.handle_filter is not None:
            only = self.handle_filter()
        positions, colors = build_debug_lines(*resolved, only_handles=only)
        self.line_count = positions.shape[0] // 2
        if self.line_count == 0:
            return
        program = self._program(renderer)
        if program is None:
            return
        from OpenGL import GL

        data = np.concatenate([positions, colors], axis=1)
        data = np.ascontiguousarray(data, dtype=np.float32)
        state = context.state
        program.use(state)
        renderer._apply_camera_uniforms(program)  # noqa: SLF001 - тот же пакет
        self._ensure_buffers(positions.shape[0])
        state.bind_vertex_array(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, data.nbytes, data)
        state.set_cull(False)
        state.set_blend(False)
        state.set_depth_write(False)
        state.set_depth_test(self.depth_test)
        GL.glDrawArrays(GL.GL_LINES, 0, positions.shape[0])
        state.set_depth_test(True)
        state.set_depth_write(True)
        state.set_cull(True)
        renderer.invalidate_active_program()

    def dispose(self) -> None:
        """Освободить буфер."""
        from OpenGL import GL

        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
            GL.glDeleteBuffers(1, [self._vbo])
            self._vao = self._vbo = self._capacity = 0
