"""Процедурное небо: настройки и проход графа кадра.

Небо — полноэкранный треугольник на дальней плоскости. Проход стоит в графе
между ``geometry`` и ``transparent``: рисуется только там, где нет непрозрачной
геометрии (проверка глубины ``LEQUAL``, запись глубины выключена), а
прозрачные объекты затем смешиваются уже с небом.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.rendering.render_graph import FrameContext

__all__ = ["SKY_MODES", "SkyPass", "SkySettings"]

_log = logging.getLogger("engine.rendering.sky")

#: Допустимые режимы неба.
SKY_MODES: tuple[str, ...] = ("procedural", "solid", "texture")

_COLOR_FIELDS = ("zenith_color", "horizon_color", "ground_color", "sun_color")
_FLOAT_FIELDS = ("sun_size", "sun_glow", "exposure", "intensity")


@dataclass
class SkySettings:
    """Параметры неба (хранятся в сцене, секция ``sky``).

    Attributes:
        enabled: Рисовать ли небо.
        mode: ``procedural`` | ``solid`` | ``texture`` (текстурный режим пока
            рисуется как процедурный).
        zenith_color: Цвет в зените (в режиме ``solid`` — единственный цвет).
        horizon_color: Цвет у горизонта.
        ground_color: Цвет «земли» под горизонтом.
        sun_color: Цвет солнца и его свечения.
        sun_size: Угловой размер диска солнца в градусах.
        sun_glow: Степень свечения вокруг солнца.
        exposure: Экспозиция тон-маппинга.
        intensity: Общая яркость.
        texture_guid: GUID текстуры для режима ``texture``.
    """

    enabled: bool = True
    mode: str = "procedural"
    zenith_color: tuple[float, float, float] = (0.25, 0.45, 0.85)
    horizon_color: tuple[float, float, float] = (0.70, 0.80, 0.95)
    ground_color: tuple[float, float, float] = (0.20, 0.18, 0.15)
    sun_color: tuple[float, float, float] = (1.0, 0.95, 0.85)
    sun_size: float = 1.0
    sun_glow: float = 20.0
    exposure: float = 1.2
    intensity: float = 1.0
    texture_guid: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать в словарь для JSON сцены."""
        data = asdict(self)
        for name in _COLOR_FIELDS:
            data[name] = [float(c) for c in data[name]]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SkySettings:
        """Восстановить настройки; неизвестные ключи игнорируются."""
        settings = cls()
        known = {f.name for f in fields(settings)}
        for key, value in (data or {}).items():
            if key in known:
                settings.set_value(key, value)
        return settings

    def set_value(self, name: str, value: Any) -> None:
        """Задать поле по имени с приведением типа.

        Args:
            name: Имя поля.
            value: Новое значение.

        Raises:
            KeyError: Если такого поля нет.
            ValueError: Если режим неба неизвестен.
        """
        if name not in {f.name for f in fields(self)}:
            raise KeyError(f"У неба нет параметра '{name}'")
        if name in _COLOR_FIELDS:
            value = tuple(float(c) for c in value)[:3]
        elif name == "enabled":
            value = bool(value)
        elif name in _FLOAT_FIELDS:
            value = float(value)
        elif name == "mode":
            value = str(value)
            if value not in SKY_MODES:
                raise ValueError(f"Неизвестный режим неба: {value!r}")
        elif name == "texture_guid":
            value = str(value) if value else None
        setattr(self, name, value)


class SkyPass:
    """Проход неба в графе кадра.

    Args:
        settings: Параметры неба (изменяются «на лету»).
        light: Направленный свет, задающий направление солнца. Если не задан,
            берётся свет из контекста кадра.
    """

    def __init__(self, settings: SkySettings | None = None, light: Any = None) -> None:
        self.settings = settings or SkySettings()
        self.light = light
        self._vao = 0
        self._source: object | None = None
        self._inverse = np.identity(4, dtype=np.float32)

    @property
    def name(self) -> str:
        """Имя прохода."""
        return "sky"

    def _inverse_view_projection(self, renderer: Any) -> np.ndarray:
        """Обратная ``projection @ view``; пересчёт только при смене кадра."""
        from engine.rendering.camera import invert_view_projection

        view, projection, view_projection = renderer._camera_uniforms()  # noqa: SLF001
        if view_projection is not self._source:
            self._inverse[:] = invert_view_projection(projection, view)
            self._source = view_projection
        return self._inverse

    def execute(self, context: FrameContext) -> None:
        """Нарисовать небо за непрозрачной геометрией."""
        settings = self.settings
        renderer = context.renderer
        if not settings.enabled or renderer is None:
            return
        from OpenGL import GL

        try:
            program = renderer.program("sky")
        except Exception:  # noqa: BLE001 - без шейдера небо просто не рисуем
            return
        light = self.light if self.light is not None else context.light
        state = context.state
        program.use(state)
        program.set_uniform(
            "u_inv_view_projection", self._inverse_view_projection(renderer)
        )
        program.set_uniform("u_sun_direction", light.unit_direction)
        program.set_uniform("u_sun_color", settings.sun_color)
        program.set_uniform("u_zenith_color", settings.zenith_color)
        program.set_uniform("u_horizon_color", settings.horizon_color)
        program.set_uniform("u_ground_color", settings.ground_color)
        program.set_uniform("u_sun_size", settings.sun_size)
        program.set_uniform("u_sun_glow", settings.sun_glow)
        program.set_uniform("u_exposure", settings.exposure)
        program.set_uniform("u_intensity", settings.intensity)
        program.set_uniform("u_mode", 1 if settings.mode == "solid" else 0)

        if not self._vao:
            self._vao = int(GL.glGenVertexArrays(1))
        state.set_cull(False)
        state.set_depth_test(True)
        state.set_depth_func(GL.GL_LEQUAL)
        state.set_depth_write(False)
        state.bind_vertex_array(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        state.set_depth_write(True)
        state.set_depth_func(None)
        state.set_cull(True)
        renderer.invalidate_active_program()
