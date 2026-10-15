"""Возможности драйвера: лимиты и поддерживаемые технологии.

Рендерер выбирает путь выполнения по этим данным: включать ли инстансинг,
скользить ли к deferred-шейдингу (нужны множественные render targets), какой
размер текстур допустим на «слабом» железе.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Capabilities", "query_capabilities"]


@dataclass(slots=True)
class Capabilities:
    """Лимиты и флаги текущего контекста."""

    max_texture_size: int = 0
    max_cube_map_size: int = 0
    max_texture_units: int = 0
    max_vertex_attribs: int = 0
    max_renderbuffer_size: int = 0
    max_samples: int = 0
    max_uniform_components: int = 0
    supports_instancing: bool = False
    supports_multiple_render_targets: bool = False
    supports_texture_arrays: bool = False
    supports_framebuffer_srgb: bool = False
    supports_anisotropy: bool = False
    supports_hdr_target: bool = False

    @property
    def can_defer_shading(self) -> bool:
        """Хватает ли возможностей для отложенного шейдинга (задел на будущее)."""
        needed_targets = 3  # albedo + normal + material
        return (
            self.supports_multiple_render_targets
            and self.max_texture_size >= 2048
            and needed_targets <= 4
        )

    def describe(self) -> str:
        """Многострочное описание для лога панели статистики."""
        flags = [
            name
            for name, enabled in (
                ("instancing", self.supports_instancing),
                ("MRT", self.supports_multiple_render_targets),
                ("texture arrays", self.supports_texture_arrays),
                ("FBO sRGB", self.supports_framebuffer_srgb),
                ("anisotropy", self.supports_anisotropy),
                ("HDR target", self.supports_hdr_target),
            )
            if enabled
        ]
        return (
            f"tex {self.max_texture_size}px, units {self.max_texture_units}, "
            f"attribs {self.max_vertex_attribs}, msaa x{self.max_samples}; "
            f"features: {', '.join(flags) or 'minimum'}"
        )


def _probe_hdr_target(GL) -> bool:
    """Собрать пробный FBO 4×4 ``RGBA16F`` и проверить его полноту.

    Args:
        GL: Модуль ``OpenGL.GL``.

    Returns:
        ``True``, если драйвер умеет рисовать в полуплавающую текстуру.
        Привязки FBO/текстуры возвращаются как были.
    """
    previous_fbo = int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING) or 0)
    previous_texture = int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D) or 0)
    texture = framebuffer = 0
    try:
        texture = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, 4, 4, 0,
            GL.GL_RGBA, GL.GL_HALF_FLOAT, None,
        )  # fmt: skip
        framebuffer = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, texture, 0
        )
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        return status == GL.GL_FRAMEBUFFER_COMPLETE
    except Exception:  # noqa: BLE001 - любой отказ драйвера = нет HDR
        return False
    finally:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, previous_fbo)
        GL.glBindTexture(GL.GL_TEXTURE_2D, previous_texture)
        if framebuffer:
            GL.glDeleteFramebuffers(1, [framebuffer])
        if texture:
            GL.glDeleteTextures([texture])


def query_capabilities() -> Capabilities:
    """Прочитать лимиты текущего контекста.

    Returns:
        Возможности драйвера.
    """
    from OpenGL import GL

    def integer(name: int) -> int:
        value = GL.glGetIntegerv(name)
        return int(value) if value is not None else 0

    major = 0
    version = GL.glGetString(GL.GL_VERSION)
    if isinstance(version, bytes):
        version = version.decode("utf-8", errors="replace")
    if version:
        head = version.split(" ", 1)[0].split(".")
        if head and head[0].isdigit():
            major = int(head[0])

    extensions = ""
    count = integer(GL.GL_NUM_EXTENSIONS)
    if count and hasattr(GL, "glGetStringi"):
        names: list[str] = []
        for index in range(count):
            raw = GL.glGetStringi(GL.GL_EXTENSIONS, index)
            if isinstance(raw, bytes):
                names.append(raw.decode("utf-8", errors="replace"))
        extensions = " ".join(names)

    return Capabilities(
        supports_hdr_target=_probe_hdr_target(GL),
        max_texture_size=integer(GL.GL_MAX_TEXTURE_SIZE),
        max_cube_map_size=integer(GL.GL_MAX_CUBE_MAP_TEXTURE_SIZE),
        max_texture_units=integer(GL.GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS),
        max_vertex_attribs=integer(GL.GL_MAX_VERTEX_ATTRIBS),
        max_renderbuffer_size=integer(GL.GL_MAX_RENDERBUFFER_SIZE),
        max_samples=integer(GL.GL_MAX_SAMPLES) if major >= 3 else 0,
        max_uniform_components=integer(GL.GL_MAX_VERTEX_UNIFORM_COMPONENTS),
        supports_instancing=major >= 3 or "GL_ARB_instanced_arrays" in extensions,
        supports_multiple_render_targets=major >= 3
        or "GL_ARB_draw_buffers" in extensions,
        supports_texture_arrays=major >= 3 or "GL_EXT_texture_array" in extensions,
        supports_framebuffer_srgb="GL_ARB_framebuffer_sRGB" in extensions
        or major >= 3,
        supports_anisotropy="GL_EXT_texture_filter_anisotropic" in extensions,
    )
