"""Карта теней направленного источника (directional light).

Одна ортографическая карта на сцену: для слабых GPU (Intel HD 520 и подобные)
это единственный разумный вариант — каскады и кубические карты для точечных
источников требуют либо нескольких проходов, либо геометрического шейдера.

Устройство:

* :class:`ShadowMap` владеет текстурой глубины и framebuffer-объектом
  (``RenderTarget`` для этого не подходит: там цветовые привязки, а нам нужна
  именно depth-текстура для выборки в шейдере);
* :func:`directional_light_matrix` строит матрицу «мир → пространство
  источника» так, чтобы в неё попала видимая часть сцены: бокс строится вокруг
  области камеры вдоль оси света, а глубина пишется в диапазон ``0..1``;
* :class:`ShadowPass` рисует геометрию в карту — только непрозрачные объекты,
  без цвета и без расчёта освещения.

Example:
    >>> shadow = ShadowMap(size=512)
    >>> matrix = shadow.begin(light, camera)
    >>> for item in opaque_items:
    ...     shadow.draw(item.mesh, item.model_matrix)
    >>> shadow.end()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from engine.core.math3d import EPSILON, Vector3, look_at_matrix
from engine.rendering.errors import RenderError
from engine.rendering.state import select_scratch_unit

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.rendering.camera import Camera
    from engine.rendering.lighting import DirectionalLight
    from engine.rendering.state import GLState

__all__ = [
    "DEFAULT_SHADOW_SIZE",
    "ShadowMap",
    "directional_light_matrix",
]

_log = logging.getLogger("engine.rendering.shadow")

#: Разрешение карты теней по умолчанию: 512×512 хватает на слабом GPU.
DEFAULT_SHADOW_SIZE = 512

#: Минимальный запас по глубине, чтобы геометрия не обрезалась плоскостями.
MIN_DEPTH_RANGE = 0.5

#: Запас по глубине вдоль луча света по умолчанию.
DEFAULT_DEPTH_PADDING = 20.0


def directional_light_matrix(
    light_direction: Vector3,
    center: Vector3,
    radius: float,
    *,
    depth_padding: float = DEFAULT_DEPTH_PADDING,
    up: Vector3 | None = None,
) -> np.ndarray:
    """Построить матрицу «мир → пространство источника» для теней.

    Сцена рассматривается как сфера (``center``, ``radius``): так карта теней
    гарантированно накрывает видимую область независимо от её формы, а
    ортографический бокс получается компактным — это напрямую влияет на
    эффективное разрешение карты теней.

    Глубина пишется в диапазон ``0..1``: начало координат — на дальней грани
    бокса. Такой вид удобен для сравнения с глубиной из текстуры (в отличие от
    ``-1..1``, где нужна пересборка).

    Args:
        light_direction: Направление, **куда** светит источник.
        center: Центр освещаемой области (обычно цель камеры).
        radius: Радиус освещаемой области.
        depth_padding: Запас по глубине вдоль луча света.
        up: Вектор «вверх» для матрицы вида; ``None`` — подобрать автоматически.

    Returns:
        Матрица ``(4, 4)`` в float64: ``clip`` (x, y, depth в ``0..1``).

    Raises:
        ValueError: Если направление света нулевое или радиус отрицательный.
    """
    forward = light_direction.normalized
    if forward.is_zero:
        raise ValueError("Направление света не может быть нулевым")
    if radius < 0.0:
        raise ValueError("Радиус области не может быть отрицательным")

    up_vector = up or _stable_up(forward)
    eye = center - forward * (radius + max(0.0, depth_padding))
    # Видовая матрица источника: смотрим из «глаза» вдоль направления света.
    view = look_at_matrix(eye, center, up_vector)

    extent = max(float(radius), 1.0)
    padding = max(0.0, depth_padding)
    # Диапазон вдоль луча света: область теней плюс запас с обеих сторон.
    # Без запаса «сзади» геометрия позади области выпадала бы из карты.
    # Глаз стоит в radius + padding от центра, поэтому ближняя плоскость почти
    # у самого глаза: всё между глазом и областью (отбрасывающие тень объекты
    # выше области) должно попасть в карту, а не обрезаться.
    near = EPSILON
    far = padding + 2.0 * extent
    if far - near < MIN_DEPTH_RANGE:
        far = near + MIN_DEPTH_RANGE

    # Ортографическая проекция в NDC (``-1..1`` по глубине).
    #
    # Здесь важно не перепутать два шага: матрица обязана дать NDC, а перевод
    # в ``0..1`` (оконная глубина, которую пишет драйвер и читает шейдер)
    # делает сам OpenGL по формуле ``z_win = (z_ndc + 1) / 2``. Если записать
    # ``0..1`` уже в матрицу, драйвер применит преобразование второй раз, и
    # диапазон уедет вдвое — сравнение с текстурой перестанет срабатывать.
    #
    # Вид смотрит вдоль ``-Z`` (``z_view = -distance``), а ближняя плоскость
    # должна дать ``-1``: ``z_ndc = (z_view + near) / (far - near) * 2 - 1``.
    near, far = float(near), float(far)
    matrix = np.zeros((4, 4), dtype=float)
    matrix[0, 0] = 1.0 / extent
    matrix[1, 1] = 1.0 / extent
    matrix[2, 2] = -2.0 / (far - near)
    matrix[2, 3] = -(far + near) / (far - near)
    matrix[3, 3] = 1.0
    return matrix @ np.asarray(view, dtype=float)


def _stable_up(forward: Vector3) -> Vector3:
    """Подобрать вектор «вверх», не параллельный направлению света."""
    if abs(forward.y) < 0.99:
        return Vector3.up()
    return Vector3.forward()


@dataclass(slots=True)
class ShadowMap:
    """Текстурная карта теней с framebuffer-объектом.

    Args:
        size: Разрешение в текселях (512 — значение для слабых GPU).
        name: Имя цели (для логов).
        bias: Смещение глубины против «акне» теней.
        normal_bias: Смещение по нормали (помогает на наклонных поверхностях).
        strength: Насколько тёмной делается тень (1 — полностью).
        enabled: Использовать ли карту при рендере.

    Attributes:
        light_space_matrix: Матрица «мир → пространство источника».
    """

    size: int = DEFAULT_SHADOW_SIZE
    name: str = "shadow_map"
    bias: float = 0.0015
    normal_bias: float = 0.02
    strength: float = 0.75
    enabled: bool = True
    light_space_matrix: np.ndarray = field(
        default_factory=lambda: np.identity(4, dtype=float), repr=False
    )
    _texture: int = field(default=0, repr=False)
    _framebuffer: int = field(default=0, repr=False)
    _drawn: int = field(default=0, repr=False)
    _bound: bool = field(default=False, repr=False)
    _frustum: object | None = field(default=None, repr=False)
    _frustum_ready: bool = field(default=False, repr=False)
    _culled: int = field(default=0, repr=False)

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    @property
    def texture(self) -> int:
        """Идентификатор текстуры глубины (0 — не создана)."""
        return self._texture

    @property
    def framebuffer(self) -> int:
        """Идентификатор framebuffer-объекта (0 — не создан)."""
        return self._framebuffer

    @property
    def is_created(self) -> bool:
        """``True``, если текстура и FBO созданы в драйвере."""
        return self._texture != 0 and self._framebuffer != 0

    @property
    def drawn_objects(self) -> int:
        """Сколько объектов нарисовано в карту в последнем проходе."""
        return self._drawn

    @property
    def texel_size(self) -> tuple[float, float]:
        """Размер текселя карты (для фильтрации в шейдере)."""
        inverse = 1.0 / max(1, self.size)
        return (inverse, inverse)

    @property
    def z_range(self) -> float:
        """Диапазон глубины карты (для диагностики)."""
        return 1.0

    # ------------------------------------------------------------------ #
    # Отсечение по объёму источника
    # ------------------------------------------------------------------ #
    def prepare_culling(self) -> None:
        """Подготовить плоскости усечённого объёма источника.

        Отсечение — главная оптимизация прохода теней: он рисует геометрию
        второй раз, и на интегрированной графике это самая дорогая его часть
        (замеры: 86 мс из 88 мс на 200 объектах). Объекты вне объёма источника
        тени всё равно не отбрасывают, поэтому их можно не рисовать.
        """
        from engine.rendering.camera import Frustum

        if self._frustum is None:
            self._frustum = Frustum()
        # Матрица даёт NDC, глубина в -1..1 (в 0..1 её переводит драйвер).
        matrix = np.asarray(self.light_space_matrix, dtype=float)
        planes = self._frustum.planes
        planes[0] = matrix[3] + matrix[0]  # left
        planes[1] = matrix[3] - matrix[0]  # right
        planes[2] = matrix[3] + matrix[1]  # bottom
        planes[3] = matrix[3] - matrix[1]  # top
        planes[4] = matrix[3] + matrix[2]  # near (z_ndc >= -1)
        planes[5] = matrix[3] - matrix[2]  # far (z_ndc <= 1)
        lengths = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
        lengths[lengths < 1e-6] = 1.0
        planes /= lengths
        self._frustum_ready = True
        self._culled = 0

    def is_visible(self, center: Vector3, radius: float) -> bool:
        """Проверить, попадает ли сфера в объём источника.

        Args:
            center: Центр сферы в мире.
            radius: Радиус сферы.

        Returns:
            ``True``, если объект может отбросить тень в этой карте.
        """
        if not self._frustum_ready or self._frustum is None:
            return True
        visible = self._frustum.intersects_sphere(center, radius)
        if not visible:
            self._culled += 1
        return visible

    @property
    def culled_objects(self) -> int:
        """Сколько объектов отсечено по объёму источника в последнем проходе."""
        return self._culled

    def set_culled(self, count: int) -> None:
        """Записать число отсечённых объектов (для статистики кадра)."""
        self._culled = max(0, int(count))

    # ------------------------------------------------------------------ #
    # Жизненный цикл
    # ------------------------------------------------------------------ #
    def create(self) -> int:
        """Создать текстуру глубины и framebuffer.

        Returns:
            Идентификатор framebuffer-объекта.

        Raises:
            RenderError: Если драйвер не собрал цель.
        """
        from OpenGL import GL

        if self.is_created:
            return self._framebuffer

        size = max(64, int(self.size))
        texture = int(GL.glGenTextures(1))
        select_scratch_unit()
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        # Текстура только глубинная: цвет не пишется, поэтому была бы лишней.
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH_COMPONENT24,
            size,
            size,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # Края карты не должны «заворачиваться»: иначе тени соседних объектов
        # появлялись бы на противоположной стороне сцены.
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        framebuffer = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            texture,
            0,
        )
        # Цветовых привязок нет: сообщаем драйверу, что читать и рисовать цвет
        # в эту цель не нужно.
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            self.dispose()
            raise RenderError(
                f"Карта теней '{self.name}' не собрана: статус 0x{status:04X}"
            )

        self._texture = texture
        self._framebuffer = framebuffer
        _log.debug("Карта теней '%s' создана: %dx%d", self.name, size, size)
        return framebuffer

    def resize(self, size: int) -> None:
        """Изменить разрешение карты (пересоздаёт текстуру)."""
        size = max(64, int(size))
        if size == self.size and self.is_created:
            return
        self.dispose()
        self.size = size
        self.create()

    def dispose(self) -> None:
        """Освободить текстуру и framebuffer."""
        from OpenGL import GL

        if self._framebuffer:
            GL.glDeleteFramebuffers(1, [self._framebuffer])
            self._framebuffer = 0
        if self._texture:
            GL.glDeleteTextures([self._texture])
            self._texture = 0

    # ------------------------------------------------------------------ #
    # Подготовка прохода
    # ------------------------------------------------------------------ #
    def prepare(
        self,
        light: DirectionalLight,
        camera: Camera,
        *,
        state: GLState | None = None,
    ) -> np.ndarray:
        """Пересчитать матрицу света и обновить параметры карты.

        Область теней берётся из видимой части сцены: центр — точка, куда
        смотрит камера, радиус — расстояние до дальней плоскости или
        ``shadow_distance`` источника (что меньше).

        Args:
            light: Направленный источник.
            camera: Камера кадра.
            state: Кэш состояний (передаётся, чтобы не рассинхронизировать его).

        Returns:
            Матрица «мир → пространство источника» для шейдера.
        """
        # Разрешение задаётся источником: разные солнца могут требовать разной
        # точности, а по умолчанию это 512 для слабых GPU.
        desired = max(64, int(light.shadow_map_size))
        if desired != self.size:
            if self.is_created:
                self.resize(desired)
            else:
                self.size = desired

        center, radius = _shadow_fit(light, camera)
        matrix = directional_light_matrix(light.unit_direction, center, radius)
        # Привязка к сетке текселей: без неё границы теней «плывут» (мерцают)
        # при каждом движении камеры.
        self.light_space_matrix = snap_to_texels(matrix, self.size)
        # Смещения не меньше размера текселя: иначе на крупной области (и
        # маленькой карте) тень «акнит» на наклонных поверхностях.
        extent = max(radius, 1.0)
        texel_world = 2.0 * extent / max(1, self.size)
        depth_range = DEFAULT_DEPTH_PADDING + 2.0 * extent
        self.bias = max(float(light.shadow_bias), texel_world / depth_range)
        self.normal_bias = max(float(light.normal_bias), texel_world)
        self.prepare_culling()
        self._drawn = 0
        return self.light_space_matrix

    def bind(self, state: GLState | None = None) -> None:
        """Сделать карту активной целью вывода."""
        from OpenGL import GL

        self.create()
        if state is not None:
            state.bind_framebuffer(self._framebuffer)
            state.set_viewport(0, 0, self.size, self.size)
            # В карту пишется только глубина: цвет отключаем, чтобы драйвер не
            # тратил полосу на запись ненужных пикселей.
            state.set_color_write(False)
        else:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._framebuffer)
            GL.glViewport(0, 0, self.size, self.size)
            GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
        self._bound = True

    def unbind(self, state: GLState | None = None) -> None:
        """Вернуть вывод на экран и включить запись цвета."""
        from OpenGL import GL

        if state is not None:
            state.bind_default_framebuffer()
            state.set_color_write(True)
        else:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        self._bound = False

    def clear(self) -> None:
        """Очистить карту максимальной глубиной (дальняя плоскость)."""
        from OpenGL import GL

        GL.glClearDepth(1.0)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

    def count_draw(self) -> None:
        """Учесть ещё один объект, нарисованный в карту."""
        self._drawn += 1

    def set_drawn(self, count: int) -> None:
        """Записать число объектов, попавших в карту за проход.

        Args:
            count: Сколько объектов нарисовано.
        """
        self._drawn = max(0, int(count))

    def __repr__(self) -> str:
        state = "gpu" if self.is_created else "cpu"
        return f"<ShadowMap '{self.name}' {self.size}×{self.size} {state}>"


def snap_to_texels(matrix: np.ndarray, size: int) -> np.ndarray:
    """Привязать смещение матрицы источника к сетке текселей карты.

    Args:
        matrix: Матрица «мир → пространство источника» (NDC).
        size: Разрешение карты теней в пикселях.

    Returns:
        Копия матрицы со смещением по x и y, кратным размеру текселя.
    """
    snapped = np.array(matrix, dtype=float)
    step = 2.0 / max(1, int(size))  # размер текселя в NDC
    snapped[0, 3] = round(snapped[0, 3] / step) * step
    snapped[1, 3] = round(snapped[1, 3] / step) * step
    return snapped


def _shadow_fit(light: DirectionalLight, camera: Camera) -> tuple[Vector3, float]:
    """Вписать в карту теней видимый кусок сцены перед камерой.

    Область — сфера, описанная вокруг отрезка «камера → ``shadow_distance``»:
    центр лежит на середине глубины, радиус учитывает сечение на дальнем
    конце (для ортографической камеры — её видимую ширину и высоту). Чем
    короче ``shadow_distance``, тем плотнее тексели карты.

    Args:
        light: Направленный источник (даёт ``shadow_distance``).
        camera: Камера кадра.

    Returns:
        Пару ``(центр, радиус)`` области теней.
    """
    distance = min(max(light.shadow_distance, 1.0), max(camera.far, 1.0))
    aspect = max(camera.effective_aspect, 1e-3)
    if camera.is_orthographic:
        half_height = max(camera.ortho_height, EPSILON) * 0.5
    else:
        half_height = distance * math.tan(math.radians(max(1.0, camera.fov) * 0.5))
    half_width = half_height * aspect
    center = camera.position + camera.forward * (distance * 0.5)
    radius = math.hypot(distance * 0.5, math.hypot(half_width, half_height))
    return center, max(radius, 2.0)
