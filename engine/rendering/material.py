"""Материалы и источники света.

Материал — это программа плюс значения её uniform-переменных. Он умеет
применять себя к программе и отдавать ключ сортировки: рендерер сортирует
очередь по материалам, чтобы минимизировать переключения состояния (это
первый, самый дешёвый уровень батчинга).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.core.math3d import Vector3
from engine.rendering.shader import ShaderProgram
from engine.rendering.texture import GpuTexture

__all__ = ["DirectionalLight", "Material"]


@dataclass(slots=True)
class DirectionalLight:
    """Направленный источник света (солнце).

    Attributes:
        direction: Направление, **куда** светит источник (нормализуется).
        color: Цвет света.
        intensity: Яркость.
        ambient: Доля фонового освещения.
    """

    direction: Vector3 = field(
        default_factory=lambda: Vector3(-0.45, -1.0, -0.35).normalized
    )
    color: Vector3 = field(default_factory=lambda: Vector3(1.0, 0.98, 0.94))
    intensity: float = 1.0
    ambient: float = 0.22

    @property
    def unit_direction(self) -> Vector3:
        """Нормализованное направление света."""
        return self.direction.normalized

    @property
    def ambient_color(self) -> Vector3:
        """Фоновая подсветка как цвет (используется менеджером света)."""
        level = max(0.0, float(self.ambient))
        return Vector3(level, level, level)

    def apply(self, program: ShaderProgram) -> int:
        """Передать параметры света в программу.

        Фоновая подсветка сюда не входит: в шейдере ``u_ambient`` — это
        ``vec3``, которым управляет
        :class:`~engine.rendering.lighting.LightManager` (он складывает все
        :class:`~engine.rendering.lighting.AmbientLight` сцены). Если менеджер
        не используется, рендерер берёт запасное значение из :attr:`ambient`.

        Args:
            program: Активная шейдерная программа.

        Returns:
            Число установленных uniform-переменных.
        """
        applied = 0
        applied += program.set_uniform("u_light_direction", self.unit_direction)
        applied += program.set_uniform(
            "u_light_color", self.color * max(0.0, self.intensity)
        )
        return applied

    def describe(self) -> str:
        """Описание для панели статистики."""
        return (
            f"direction={self.unit_direction} color={self.color} "
            f"intensity={self.intensity:.2f} ambient={self.ambient:.2f}"
        )


@dataclass(slots=True)
class Material:
    """Материал: программа и её параметры.

    Attributes:
        name: Имя материала (для логов).
        program: Шейдерная программа.
        base_color: Базовый цвет (albedo).
        roughness: Шероховатость для блика.
        metallic: «Металличность» (в базовом шейдере управляет яркостью блика).
        transparent: Рисовать в проходе прозрачности.
        double_sided: Отключать отсечение задних граней.
        texture: Диффузная текстура (необязательно).
        lightmap: Лайтмапа (запечённое освещение); требует варианта шейдера
            ``USE_LIGHTMAP`` — его выбирает рендерер по этому полю.
        lightmap_strength: Доля лайтмапы в освещении (``0`` — не использовать).
        dynamic_lighting: Учитывать ли динамические источники. ``False``
            переводит объект на запечённый свет (вершинные цвета/лайтмапа) —
            это самый дешёвый режим для слабых устройств.
        receive_shadows: Получать ли тени от направленного источника.
        depth_test: Проверять ли глубину при отрисовке. ``False`` рисует поверх
            остальной геометрии (гизмо редактора); такой материал стоит делать
            ``transparent``, чтобы он рисовался после непрозрачных объектов.
        sorting_layer: Слой сортировки прозрачных объектов: меньшие слои
            рисуются раньше (под большими), внутри слоя — от дальнего к
            ближнему. У непрозрачных не влияет.
        alpha: Прозрачность (используется программами без освещения и
            полупрозрачными материалами).
    """

    name: str
    program: ShaderProgram
    base_color: Vector3 = field(default_factory=lambda: Vector3(0.82, 0.82, 0.86))
    roughness: float = 0.5
    metallic: float = 0.0
    transparent: bool = False
    double_sided: bool = False
    texture: GpuTexture | None = None
    lightmap: GpuTexture | None = None
    lightmap_strength: float = 1.0
    dynamic_lighting: bool = True
    receive_shadows: bool = True
    depth_test: bool = True
    alpha: float = 1.0
    sorting_layer: int = 0
    _extra: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def sort_key(self) -> tuple[int, int, int]:
        """Ключ сортировки: программа → текстуры → прозрачность.

        Одинаковые ключи означают, что объекты можно рисовать подряд без
        переключения состояния. Лайтмапа входит в ключ наравне с диффузной
        текстурой: её тоже нужно привязать к своему юниту.
        """
        return (
            id(self.program),
            id(self.texture) if self.texture is not None else 0,
            int(self.transparent) + 2 * (id(self.lightmap) if self.lightmap else 0),
        )

    @property
    def uses_lightmap(self) -> bool:
        """``True``, если материал должен использовать вариант с лайтмапой."""
        return self.lightmap is not None and self.lightmap_strength > 0.0

    def extra_textures(self) -> tuple[tuple[int, GpuTexture], ...]:
        """Дополнительные текстуры материала: пары ``(юнит, текстура)``.

        Базовый материал их не имеет; PBR-материал отдаёт нормали, ORM и
        свечение. Юниты 0 (диффузная), 1 (тени) и 2 (лайтмапа) заняты.
        """
        return ()

    def set_float(self, name: str, value: float) -> Material:
        """Задать числовой параметр материала (для расширений).

        Returns:
            Сам материал (для цепочки вызовов).
        """
        self._extra[name] = float(value)
        return self

    @property
    def extra_uniforms(self) -> dict[str, float]:
        """Дополнительные uniform-переменные материала."""
        return self._extra

    def apply(
        self,
        light: DirectionalLight,
        camera_position: Vector3,
        *,
        program: ShaderProgram | None = None,
    ) -> int:
        """Применить материал, свет и позицию камеры к программе.

        Args:
            light: Источник света сцены; в uniform не пишется (свет грузит
                ``LightManager``), параметр оставлен для совместимости.
            camera_position: Позиция камеры (для блика).
            program: Программа, в которую писать uniform-переменные; ``None`` —
                собственная программа материала. Параметр нужен для вариантов
                шейдера (например, инстансной отрисовки одной геометрией).

        Returns:
            Число установленных uniform-переменных.
        """
        target = program or self.program
        applied = 0
        applied += target.set_uniform("u_base_color", self.base_color)
        applied += target.set_uniform("u_roughness", float(self.roughness))
        applied += target.set_uniform("u_metallic", float(self.metallic))
        applied += target.set_uniform("u_camera_position", camera_position)
        applied += target.set_uniform("u_use_texture", self.texture is not None)
        applied += target.set_uniform("u_alpha", float(self.alpha))
        if self.texture is not None:
            applied += target.set_uniform("u_diffuse_texture", 0)
        if self.lightmap is not None:
            applied += target.set_uniform("u_lightmap", 1)
            applied += target.set_uniform(
                "u_lightmap_strength", float(self.lightmap_strength)
            )
        applied += target.set_uniform(
            "u_dynamic_lighting", bool(self.dynamic_lighting)
        )
        for name, value in self._extra.items():
            applied += target.set_uniform(name, value)
        # Источники света сюда намеренно не пишутся: u_light_* в шейдере — массивы,
        # их грузит LightManager. Скалярная запись затёрла бы элемент [0] (солнце).
        return applied

    def __repr__(self) -> str:
        suffix = " +texture" if self.texture is not None else ""
        return f"<Material '{self.name}' program='{self.program.name}'{suffix}>"
