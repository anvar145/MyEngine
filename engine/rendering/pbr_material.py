"""PBR-материал: металличность/шероховатость, карты нормалей, ORM и свечения.

Данные приходят из пайплайна ассетов (``.mat.json`` → ``material.json`` +
упакованная ORM-текстура), но материал можно собрать и вручную в коде.
Рисуется программой ``pbr`` (``assets/shaders/pbr.frag``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.core.math3d import Vector3
from engine.rendering.material import DirectionalLight, Material
from engine.rendering.shader import ShaderProgram
from engine.rendering.texture import GpuTexture

__all__ = [
    "ALPHA_BLEND",
    "ALPHA_MASK",
    "ALPHA_OPAQUE",
    "NORMAL_UNIT",
    "ORM_UNIT",
    "EMISSIVE_UNIT",
    "PbrMaterial",
]

ALPHA_OPAQUE, ALPHA_MASK, ALPHA_BLEND = 0, 1, 2
_ALPHA_MODES = {"opaque": ALPHA_OPAQUE, "mask": ALPHA_MASK, "blend": ALPHA_BLEND}

#: Юниты текстур PBR (0 — base color, 1 — тени, 2 — лайтмапа заняты).
NORMAL_UNIT, ORM_UNIT, EMISSIVE_UNIT = 3, 4, 5


@dataclass(slots=True)
class PbrMaterial(Material):
    """Материал металличность/шероховатость.

    Унаследованные поля: ``base_color`` (RGB-множитель), ``alpha``
    (альфа-множитель), ``roughness``/``metallic`` (множители карты ORM),
    ``texture`` — карта base color (sRGB), ``double_sided``.

    Attributes:
        normal_texture: Карта нормалей (линейная).
        orm_texture: Упакованная карта: R = occlusion, G = roughness,
            B = metallic (линейная).
        emissive_texture: Карта свечения (sRGB).
        emissive: Цвет свечения (множитель).
        normal_scale: Сила карты нормалей.
        occlusion_strength: Сила ambient occlusion (0..1).
        alpha_mode: ``opaque`` | ``mask`` | ``blend``.
        alpha_cutoff: Порог отсечения для ``mask``.
        uv_scale: Тайлинг UV.
        uv_offset: Смещение UV.
    """

    normal_texture: GpuTexture | None = None
    orm_texture: GpuTexture | None = None
    emissive_texture: GpuTexture | None = None
    emissive: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    normal_scale: float = 1.0
    occlusion_strength: float = 1.0
    alpha_mode: str = "opaque"
    alpha_cutoff: float = 0.5
    uv_scale: tuple[float, float] = (1.0, 1.0)
    uv_offset: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.alpha_mode not in _ALPHA_MODES:
            raise ValueError(f"alpha_mode: допустимы {', '.join(_ALPHA_MODES)}")
        # Смешивание рисуется в проходе прозрачности, остальное — в основном.
        self.transparent = self.alpha_mode == "blend"

    @property
    def alpha_mode_id(self) -> int:
        """Числовой режим альфы для шейдера."""
        return _ALPHA_MODES[self.alpha_mode]

    @property
    def sort_key(self) -> tuple[int, int, int]:
        """Ключ сортировки: программа → набор текстур → режим альфы."""
        textures = (
            self.texture,
            self.normal_texture,
            self.orm_texture,
            self.emissive_texture,
        )
        return (
            id(self.program),
            hash(tuple(id(t) for t in textures)),
            self.alpha_mode_id + 4 * int(self.double_sided),
        )

    def extra_textures(self) -> tuple[tuple[int, GpuTexture], ...]:
        """Нормали, ORM и свечение с их юнитами."""
        pairs = (
            (NORMAL_UNIT, self.normal_texture),
            (ORM_UNIT, self.orm_texture),
            (EMISSIVE_UNIT, self.emissive_texture),
        )
        return tuple((unit, tex) for unit, tex in pairs if tex is not None)

    def apply(
        self,
        light: DirectionalLight,
        camera_position: Vector3,
        *,
        program: ShaderProgram | None = None,
    ) -> int:
        """Применить параметры материала (см. :meth:`Material.apply`)."""
        target = program or self.program
        # Явный вызов: zero-arg super() не работает в dataclass(slots=True).
        applied = Material.apply(self, light, camera_position, program=program)
        applied += target.set_uniform(
            "u_has_normal_map", self.normal_texture is not None
        )
        applied += target.set_uniform("u_has_orm_map", self.orm_texture is not None)
        applied += target.set_uniform(
            "u_has_emissive_map", self.emissive_texture is not None
        )
        applied += target.set_uniform("u_normal_texture", NORMAL_UNIT)
        applied += target.set_uniform("u_orm_texture", ORM_UNIT)
        applied += target.set_uniform("u_emissive_texture", EMISSIVE_UNIT)
        applied += target.set_uniform("u_emissive", self.emissive)
        applied += target.set_uniform("u_normal_scale", float(self.normal_scale))
        applied += target.set_uniform(
            "u_occlusion_strength", float(self.occlusion_strength)
        )
        applied += target.set_uniform("u_alpha_mode", self.alpha_mode_id)
        applied += target.set_uniform("u_alpha_cutoff", float(self.alpha_cutoff))
        applied += target.set_uniform("u_uv_scale", self.uv_scale)
        applied += target.set_uniform("u_uv_offset", self.uv_offset)
        return applied

    def __repr__(self) -> str:
        return f"<PbrMaterial '{self.name}' alpha={self.alpha_mode}>"
