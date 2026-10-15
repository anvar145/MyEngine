"""Пост-обработка кадра: настройки и fused-проходы (A10a).

Порядок после сцены: HDR-цель → tonemap (+ экспозиция, место под LUT) →
FXAA (в LDR) → fade/flash. Fade и flash рисуются поверх уже сглаженного
кадра, поэтому FXAA их не трогает. Проходы объединены: без FXAA кадр делает
один полноэкранный проход, с FXAA — два (tonemap в LDR-цель и FXAA в итоговую).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from engine.core.resources.asset_ref import AssetRef
from engine.rendering.bloom import MAX_BLOOM_LEVELS, BloomChain
from engine.rendering.render_graph import RenderTarget
from engine.rendering.settings import QualityTier

if TYPE_CHECKING:
    from engine.core.screen_fx import ScreenFx
    from engine.rendering.renderer import Renderer
    from engine.rendering.texture import GpuTexture

__all__ = [
    "TONEMAP_MODES",
    "PostSettings",
    "PostStack",
    "RenderScaleHints",
    "tier_render_scale",
]

_log = logging.getLogger("engine.rendering.post")

#: Режимы тональной компрессии; индекс совпадает с ``u_mode`` в ``tonemap.frag``.
TONEMAP_MODES: tuple[str, ...] = ("linear", "reinhard", "aces")

EXPOSURE_RANGE = (0.05, 8.0)

#: Включать ли bloom в профиле medium (решено замером бюджета ≤3.5 мс, A10b).
#: Замер HD 520, 720p: medium+bloom 4.09 мс (> 3.5), без bloom 2.23 мс.
BLOOM_IN_MEDIUM = False

#: Ширина плавного колена порога bloom (``u_soft_knee``).
BLOOM_SOFT_KNEE = 0.5

#: Размер LUT-полосы: 16 срезов 16x16 по синему каналу.
LUT_WIDTH = 256
LUT_HEIGHT = 16


@dataclass(slots=True)
class PostSettings:
    """Настройки пост-обработки (секция сцены ``post``).

    Attributes:
        tonemap: ``linear`` (без сжатия, значения > 1 обрезаются), ``reinhard``
            или ``aces``.
        exposure: Множитель яркости до tonemap.
        fxaa: Включить FXAA после tonemap.
    """

    tonemap: str = "linear"
    exposure: float = 1.0
    fxaa: bool = False
    bloom: bool = False
    bloom_threshold: float = 1.0
    bloom_intensity: float = 1.0
    bloom_levels: int = 3
    vignette: bool = False
    vignette_inner: float = 0.5
    vignette_outer: float = 0.9
    vignette_intensity: float = 0.3
    lut: AssetRef | None = None
    lut_intensity: float = 1.0

    def __post_init__(self) -> None:
        self.tonemap = self.tonemap if self.tonemap in TONEMAP_MODES else "linear"
        low, high = EXPOSURE_RANGE
        self.exposure = min(high, max(low, float(self.exposure)))
        self.fxaa = bool(self.fxaa)
        self.bloom = bool(self.bloom)
        self.bloom_threshold = min(16.0, max(0.0, float(self.bloom_threshold)))
        self.bloom_intensity = min(8.0, max(0.0, float(self.bloom_intensity)))
        self.bloom_levels = min(MAX_BLOOM_LEVELS, max(1, int(self.bloom_levels)))
        self.vignette = bool(self.vignette)
        self.vignette_inner = min(1.0, max(0.0, float(self.vignette_inner)))
        self.vignette_outer = min(1.5, max(0.0, float(self.vignette_outer)))
        self.vignette_intensity = min(1.0, max(0.0, float(self.vignette_intensity)))
        self.lut = _coerce_lut(self.lut)
        self.lut_intensity = min(1.0, max(0.0, float(self.lut_intensity)))

    @classmethod
    def for_quality(cls, tier: QualityTier) -> PostSettings:
        """Умолчания профиля (таблица тиров A10b).

        low — только вывод; medium — ACES + FXAA + bloom (3 уровня) + виньетка;
        high — то же с 4 уровнями bloom. LUT включается заданной ссылкой.
        """
        if tier is QualityTier.LOW:
            return cls()
        levels = 4 if tier is QualityTier.HIGH else 3
        return cls(
            tonemap="aces",
            exposure=1.0,
            fxaa=True,
            bloom=BLOOM_IN_MEDIUM or tier is QualityTier.HIGH,
            bloom_levels=levels,
            vignette=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, base: PostSettings | None = None):
        """Прочитать секцию сцены поверх ``base`` (неизвестные ключи игнорируются)."""
        result = cls()
        if base:
            for item in fields(cls):
                setattr(result, item.name, getattr(base, item.name))
        for item in fields(cls):
            if data and item.name in data:
                setattr(result, item.name, data[item.name])
        result.__post_init__()
        return result

    def to_dict(
        self, base: PostSettings | None = None, *, full: bool = False
    ) -> dict[str, Any]:
        """Словарь настроек.

        Args:
            base: С чем сравнивать (по умолчанию — ``PostSettings()``).
            full: Записать все поля, а не только отличия от ``base``.
        """
        base = base or PostSettings()
        result: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if not (full or value != getattr(base, f.name)):
                continue
            result[f.name] = value.to_dict() if isinstance(value, AssetRef) else value
        return result

    def set_value(self, name: str, value: Any) -> None:
        """Изменить поле «на лету» (``--set post.tonemap=aces``)."""
        names = {f.name for f in fields(self)}
        if name not in names:
            raise KeyError(name)
        kind = type(getattr(self, name))
        if kind is bool and isinstance(value, str):
            value = value.strip().lower() in ("1", "true", "yes", "on")
        elif kind is float:
            value = float(value)
        elif kind is int:
            value = int(value)
        setattr(self, name, value)
        self.__post_init__()


def _coerce_lut(value: Any) -> AssetRef | None:
    """Ссылка на LUT из ``AssetRef``/словаря/пути; мусор — ``None``."""
    try:
        return AssetRef.coerce(value) if value not in ("", {}) else None
    except ValueError:
        return None


@dataclass(slots=True)
class RenderScaleHints:
    """Параметры масштаба рендера: профиль качества + подсказки проекта.

    Attributes:
        scale: Начальный ``render_scale``.
        floor: Нижняя граница динамического разрешения.
        ceiling: Верхняя граница динамического разрешения.
        dynamic: Включено ли динамическое разрешение.
    """

    scale: float = 1.0
    floor: float = 0.5
    ceiling: float = 1.0
    dynamic: bool = True


#: Таблица профилей: ``tier -> (scale, floor, ceiling, dynamic)``.
_TIER_SCALE = {
    QualityTier.LOW: RenderScaleHints(0.5, 0.5, 0.75, True),
    QualityTier.MEDIUM: RenderScaleHints(0.75, 0.5, 1.0, True),
    QualityTier.HIGH: RenderScaleHints(1.0, 1.0, 1.0, False),
}


def tier_render_scale(
    tier: QualityTier,
    *,
    hint: float | None = None,
    dynamic: bool | None = None,
) -> RenderScaleHints:
    """Масштаб рендера профиля с учётом подсказок проекта.

    Args:
        tier: Профиль качества.
        hint: ``render_scale_hint`` проекта (заменяет начальный масштаб).
        dynamic: ``dynamic_resolution`` проекта (``None`` — как в профиле).

    Returns:
        Новые параметры; масштаб зажат в границы floor..ceiling профиля, а
        ceiling не ниже подсказки.
    """
    base = _TIER_SCALE[tier]
    scale = base.scale if hint is None else min(1.0, max(0.25, float(hint)))
    ceiling = max(base.ceiling, scale)
    floor = min(base.floor, scale)
    return RenderScaleHints(
        scale=scale,
        floor=floor,
        ceiling=ceiling,
        dynamic=base.dynamic if dynamic is None else bool(dynamic),
    )


class PostStack:
    """Выполняет fused tonemap, FXAA и fade/flash для текущего кадра."""

    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer
        self._ldr: RenderTarget | None = None
        self._vao = 0
        self.settings = PostSettings()
        self.screen_fx: ScreenFx | None = None
        self._bloom = BloomChain(renderer)
        #: Источник ассетов (``VFS``/``AssetDatabase``) для загрузки LUT.
        self.assets: Any = None
        self._lut_key: tuple[str | None, str | None] | None = None
        self._lut_texture: GpuTexture | None = None
        self._lut_failed: tuple[str | None, str | None] | None = None
        self._lut_pixels: Any = None

    @property
    def use_lut(self) -> bool:
        """LUT задан в настройках и успешно загружен."""
        return self._resolve_lut() is not None

    def dispose(self) -> None:
        """Освободить промежуточные цели, LUT и VAO."""
        self._bloom.dispose()
        if self._lut_texture is not None:
            self._lut_texture.dispose()
            self._lut_texture = None
        self._lut_key = None
        if self._ldr is not None:
            self._ldr.dispose()
            self._ldr = None
        if self._vao:
            from OpenGL import GL

            GL.glDeleteVertexArrays(1, [self._vao])
            self._vao = 0

    # ------------------------------------------------------------------ #
    def apply(
        self,
        hdr: RenderTarget,
        final: RenderTarget | None,
        full_size: tuple[int, int],
    ) -> None:
        """Вывести ``hdr`` в ``final`` (``None`` — экран) полного размера."""
        from OpenGL import GL

        renderer = self._renderer
        state = renderer.state
        width, height = full_size
        settings = self.settings
        fxaa = settings.fxaa and self._has("fxaa")

        state.set_depth_test(False)
        state.set_depth_write(False)
        state.set_cull(False)
        state.set_blend(False)
        if not self._vao:
            self._vao = int(GL.glGenVertexArrays(1))
        state.bind_vertex_array(self._vao)

        bloom = None
        if settings.bloom and settings.bloom_intensity > 0.0:
            bloom = self._bloom.run(
                hdr,
                levels=settings.bloom_levels,
                threshold=settings.bloom_threshold,
                soft_knee=BLOOM_SOFT_KNEE,
            )
        elif self._bloom.level_count:
            self._bloom.dispose()

        if fxaa:
            ldr = self._acquire_ldr(hdr.width, hdr.height)
            ldr.bind(state)
            self._tonemap(hdr, bloom)
            self._bind_output(final, width, height)
            program = renderer.program("fxaa")
            program.use(state)
            state.bind_texture(0, GL.GL_TEXTURE_2D, ldr.color_textures[0])
            program.set_uniform("u_color", 0)
            program.set_uniform("u_texel", (1.0 / ldr.width, 1.0 / ldr.height))
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        else:
            self._bind_output(final, width, height)
            self._tonemap(hdr, bloom)

        self._draw_fx()
        state.set_blend(False)
        state.set_depth_test(True)
        state.set_depth_write(True)
        state.set_cull(True)
        renderer.invalidate_active_program()

    # ------------------------------------------------------------------ #
    def _has(self, name: str) -> bool:
        from engine.rendering.errors import RenderError

        try:
            self._renderer.program(name)
        except RenderError:
            return False
        return True

    def _bind_output(self, final: RenderTarget | None, width: int, height: int) -> None:
        state = self._renderer.state
        if final is not None:
            final.bind(state)
        else:
            state.bind_default_framebuffer()
        state.set_viewport(0, 0, width, height)

    def _acquire_ldr(self, width: int, height: int) -> RenderTarget:
        target = self._ldr
        if target is None:
            target = RenderTarget("post_ldr", width, height, depth=False)
            self._ldr = target
        target.resize(width, height)
        target.create()
        return target

    def _resolve_lut(self) -> GpuTexture | None:
        """Текстура LUT из настроек (кэш по ссылке); ``None`` — нет/не загрузилась."""
        if self._lut_pixels is not None:
            return self._lut_override()
        ref = self.settings.lut
        if ref is None or self.assets is None:
            return None
        key = (ref.guid, ref.path)
        if key == self._lut_key:
            return self._lut_texture
        if key == self._lut_failed:
            return None
        texture = self._load_lut(ref)
        if texture is None:
            self._lut_failed = key
            return None
        if self._lut_texture is not None:
            self._lut_texture.dispose()
        self._lut_texture = texture
        self._lut_key = key
        self._lut_failed = None
        return texture

    def set_lut_pixels(self, pixels: Any) -> None:
        """Задать LUT массивом ``(16, 256, 3)`` в обход ассетов (``None`` — снять).

        Нужен тестам и скриптам; ``settings.lut`` при этом не используется,
        но ``use_lut`` истинно.
        """
        self._lut_pixels = pixels
        if self._lut_texture is not None:
            self._lut_texture.dispose()
            self._lut_texture = None
        self._lut_key = None

    def _lut_override(self) -> GpuTexture | None:
        if self._lut_texture is None:
            from engine.rendering.texture import GpuTexture, TextureFilter

            pixels = self._lut_pixels
            texture = GpuTexture.from_bytes(
                "post_lut",
                pixels.shape[1],
                pixels.shape[0],
                pixels,
                filter_mode=TextureFilter.BILINEAR,
                wrap_repeat=False,
                generate_mips=False,
                srgb=False,
            )
            texture.upload()
            self._lut_texture = texture
        return self._lut_texture

    def _load_lut(self, ref: AssetRef) -> GpuTexture | None:
        from engine.core.resources.cookers import read_cooked_texture
        from engine.core.resources.errors import ResourceError
        from engine.core.resources.vfs import as_assets
        from engine.rendering.texture import GpuTexture, TextureFilter

        try:
            assets = as_assets(self.assets)
            guid = ref.resolve(assets)
            if guid is None:
                raise ValueError("ссылка не разрешается")
            handle = assets.resolve(guid)
            artifact = handle.artifact("texture")
            if artifact is None:
                raise ValueError("у ассета нет артефакта 'texture'")
            cooked = read_cooked_texture(handle.artifact_path(artifact).read_bytes())
            if cooked.compressed:
                raise ValueError("LUT не должен быть сжатым")
            if (cooked.width, cooked.height) != (LUT_WIDTH, LUT_HEIGHT):
                raise ValueError(
                    f"размер {cooked.width}x{cooked.height}, нужен "
                    f"{LUT_WIDTH}x{LUT_HEIGHT}"
                )
            pixels = cooked.pixels(0)
            texture = GpuTexture.from_bytes(
                "post_lut",
                cooked.width,
                cooked.height,
                pixels,
                filter_mode=TextureFilter.BILINEAR,
                wrap_repeat=False,
                generate_mips=False,
                srgb=False,
            )
            texture.upload()
        except (ResourceError, OSError, KeyError, ValueError) as exc:
            _log.warning("LUT %s не загружен: %s", ref.path or ref.guid, exc)
            return None
        return texture

    def _tonemap(self, hdr: RenderTarget, bloom: RenderTarget | None = None) -> None:
        from OpenGL import GL

        renderer = self._renderer
        state = renderer.state
        program = renderer.program("tonemap")
        settings = self.settings
        lut = self._resolve_lut()
        program.use(state)
        state.bind_texture(0, GL.GL_TEXTURE_2D, hdr.color_textures[0])
        program.set_uniform("u_color", 0)
        program.set_uniform("u_mode", TONEMAP_MODES.index(settings.tonemap))
        program.set_uniform("u_exposure", float(settings.exposure))
        program.set_uniform("u_bloom", 1)
        program.set_uniform("u_lut", 2)
        if bloom is not None:
            state.bind_texture(1, GL.GL_TEXTURE_2D, bloom.color_textures[0])
        program.set_uniform("u_bloom_on", 1 if bloom is not None else 0)
        program.set_uniform("u_bloom_intensity", float(settings.bloom_intensity))
        program.set_uniform("u_use_lut", 1 if lut is not None else 0)
        if lut is not None:
            state.bind_texture(2, GL.GL_TEXTURE_2D, lut.handle)
            program.set_uniform("u_lut_intensity", float(settings.lut_intensity))
        program.set_uniform("u_vignette_on", 1 if settings.vignette else 0)
        if settings.vignette:
            program.set_uniform("u_inner", float(settings.vignette_inner))
            program.set_uniform("u_outer", float(settings.vignette_outer))
            program.set_uniform("u_vignette", float(settings.vignette_intensity))
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

    def _draw_fx(self) -> None:
        """Fade и flash: отдельный микро-проход альфа-смешиванием."""
        fx = self.screen_fx
        if fx is None or not fx.active or not self._has("fade"):
            return
        from OpenGL import GL

        renderer = self._renderer
        state = renderer.state
        program = renderer.program("fade")
        program.use(state)
        state.set_blend(True, (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA))
        for color, alpha in (
            (fx.fade_color, fx.fade_alpha),
            (fx.flash_color, fx.flash_alpha),
        ):
            if alpha <= 0.0:
                continue
            program.set_uniform("u_color", (color[0], color[1], color[2], alpha))
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
