"""Текстуры на GPU.

Модуль загружает пиксели в OpenGL: мип-уровни, фильтрация, анизотропия (если
поддерживается) и — для блочно-сжатых DDS — прямая передача сжатых данных без
распаковки. Это ключ к экономии памяти на слабом железе.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from engine.rendering.errors import RenderError
from engine.rendering.state import select_scratch_unit

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources import TextureAsset
    from engine.core.resources.cookers import CookedTexture

__all__ = ["GpuTexture", "TextureFilter"]

_log = logging.getLogger("engine.rendering.texture")


# Сжатые форматы: (формат cooker'а, sRGB) → internalformat OpenGL.
# S3TC (BC1/BC3) — расширение EXT_texture_compression_s3tc, RGTC (BC5) — ядро 3.0.
_COMPRESSED_FORMATS = {
    ("bc1", False): 0x83F1,
    ("bc1", True): 0x8C4D,
    ("bc3", False): 0x83F3,
    ("bc3", True): 0x8C4F,
    ("bc5", False): 0x8DBD,
    ("bc5", True): 0x8DBD,  # sRGB-варианта у RGTC нет: нормали и так линейные
}


class TextureFilter:
    """Режимы фильтрации текстуры."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    TRILINEAR = "trilinear"


_GL_FILTER = {
    TextureFilter.NEAREST: "GL_NEAREST",
    TextureFilter.BILINEAR: "GL_LINEAR",
    TextureFilter.TRILINEAR: "GL_LINEAR_MIPMAP_LINEAR",
}


class GpuTexture:
    """Текстура в видеопамяти.

    Args:
        name: Имя текстуры (для логов).
        width: Ширина в пикселях.
        height: Высота в пикселях.
        channels: Число каналов (1, 2, 3, 4).
        compressed: ``True`` для блочно-сжатых данных (DDS/BC).
        filter_mode: Режим фильтрации.
        wrap_repeat: Повторять текстуру за пределами UV.
        generate_mips: Генерировать мип-уровни при загрузке.
        srgb: Данные в sRGB: железо переведёт их в линейное пространство при
            выборке (для цветовых текстур; нормали и маски — ``False``).
    """

    __slots__ = (
        "_cooked",
        "_handle",
        "_has_mips",
        "_pending",
        "channels",
        "compressed",
        "filter_mode",
        "height",
        "name",
        "srgb",
        "width",
        "wrap_repeat",
    )

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        *,
        channels: int = 4,
        compressed: bool = False,
        filter_mode: str = TextureFilter.TRILINEAR,
        wrap_repeat: bool = True,
        generate_mips: bool = True,
        srgb: bool = False,
    ) -> None:
        if width <= 0 or height <= 0:
            raise RenderError(
                f"Некорректный размер текстуры '{name}': {width}x{height}"
            )
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.compressed = bool(compressed)
        self.filter_mode = filter_mode
        self.wrap_repeat = bool(wrap_repeat)
        self._handle = 0
        self._has_mips = False
        self._pending: np.ndarray | None = None
        self._cooked: CookedTexture | None = None
        self.srgb = bool(srgb)

    # ------------------------------------------------------------------ #
    # Загрузка
    # ------------------------------------------------------------------ #
    @classmethod
    def from_asset(cls, asset: TextureAsset, **options: object) -> GpuTexture:
        """Создать текстуру из загруженного ассета.

        Args:
            asset: Ассет текстуры (метаданные и, возможно, пиксели).
            **options: Дополнительные параметры :class:`GpuTexture`.

        Returns:
            Текстура, готовая к :meth:`upload`.

        Raises:
            RenderError: Если у ассета нет пиксельных данных.
        """
        if asset.pixels is None:
            raise RenderError(
                f"Текстура '{asset.name}' не декодирована: загрузите её с параметром "
                "decode=True или передайте пиксели явно"
            )
        texture = cls(
            asset.name,
            asset.width,
            asset.height,
            channels=asset.channels,
            compressed=asset.compressed,
            **options,  # type: ignore[arg-type]
        )
        texture._pending = asset.pixels
        return texture

    @classmethod
    def from_cooked(
        cls, name: str, cooked: CookedTexture, **options: object
    ) -> GpuTexture:
        """Создать текстуру из ``texture.bin`` (мипы и BC-блоки загружаются как есть).

        Args:
            name: Имя текстуры.
            cooked: Результат ``read_cooked_texture``.
            **options: Дополнительные параметры :class:`GpuTexture`
                (``srgb`` по умолчанию берётся из файла).

        Returns:
            Текстура, готовая к :meth:`upload`.
        """
        options.setdefault("srgb", cooked.srgb)
        channels = {"r8": 1, "rg8": 2}.get(cooked.pixel_format, 4)
        texture = cls(
            name,
            cooked.width,
            cooked.height,
            channels=channels,
            compressed=cooked.compressed,
            **options,  # type: ignore[arg-type]
        )
        texture._cooked = cooked
        return texture

    @classmethod
    def from_bytes(
        cls,
        name: str,
        width: int,
        height: int,
        data: np.ndarray,
        **options: object,
    ) -> GpuTexture:
        """Создать текстуру из массива пикселей.

        Args:
            name: Имя текстуры.
            width: Ширина.
            height: Высота.
            data: Массив ``(H, W, C)`` или ``(H, W)``.
            **options: Дополнительные параметры.

        Returns:
            Текстура, готовая к :meth:`upload`.
        """
        array = np.asarray(data)
        if array.ndim == 2:
            array = array[:, :, None]
        texture = cls(
            name, width, height, channels=int(array.shape[2]), **options  # type: ignore[arg-type]
        )
        texture._pending = array
        return texture

    # ------------------------------------------------------------------ #
    # Свойства
    # ------------------------------------------------------------------ #
    @property
    def handle(self) -> int:
        """Идентификатор текстуры OpenGL (0 — не загружена)."""
        return self._handle

    @property
    def is_uploaded(self) -> bool:
        """``True``, если текстура уже в видеопамяти."""
        return self._handle != 0

    @property
    def has_mips(self) -> bool:
        """``True``, если у текстуры есть мип-уровни."""
        return self._has_mips

    @property
    def gpu_bytes(self) -> int:
        """Приблизительный объём видеопамяти с учётом мипов."""
        base = self.width * self.height * self.channels
        return int(base * 1.34) if self._has_mips else int(base)

    # ------------------------------------------------------------------ #
    # Работа с драйвером
    # ------------------------------------------------------------------ #
    def upload(self, pixels: np.ndarray | None = None) -> int:
        """Загрузить пиксели в видеопамять.

        Args:
            pixels: Массив ``(H, W, C)``; если не задан, берётся из ассета
                (для :meth:`from_asset` пиксели передаются здесь).

        Returns:
            Идентификатор текстуры.

        Raises:
            RenderError: Если нет данных для загрузки.
        """
        from OpenGL import GL

        data = pixels
        if data is None and self._cooked is not None:
            return self._upload_cooked(self._cooked)
        if data is None:
            data = self._pending
        if data is None:
            raise RenderError(
                f"Для текстуры '{self.name}' не переданы пиксельные данные"
            )
        array = np.asarray(data)
        if array.ndim == 2:
            array = array[:, :, None]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        handle = int(GL.glGenTextures(1))
        select_scratch_unit()
        GL.glBindTexture(GL.GL_TEXTURE_2D, handle)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

        internal_format, source_format = self._pixel_formats(array.shape[2])
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            internal_format,
            self.width,
            self.height,
            0,
            source_format,
            GL.GL_UNSIGNED_BYTE,
            np.ascontiguousarray(array),
        )

        if self._generate_mips(array):
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            self._has_mips = True
        self._apply_sampling()
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self._handle = handle
        _log.debug(
            "Текстура '%s' загружена: %dx%d, %d канал(ов), мипы=%s",
            self.name,
            self.width,
            self.height,
            self.channels,
            self._has_mips,
        )
        return handle

    def _upload_cooked(self, cooked: CookedTexture) -> int:
        """Загрузить готовые мип-уровни (в том числе BC-блоки) по одному."""
        from OpenGL import GL

        handle = int(GL.glGenTextures(1))
        select_scratch_unit()
        GL.glBindTexture(GL.GL_TEXTURE_2D, handle)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        levels = len(cooked.mips)
        for level, mip in enumerate(cooked.mips):
            data = np.frombuffer(mip.data, dtype=np.uint8)
            if cooked.compressed:
                GL.glCompressedTexImage2D(
                    GL.GL_TEXTURE_2D,
                    level,
                    _COMPRESSED_FORMATS[(cooked.pixel_format, self.srgb)],
                    mip.width,
                    mip.height,
                    0,
                    data,
                )
            else:
                internal, source = self._pixel_formats(self.channels)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    level,
                    internal,
                    mip.width,
                    mip.height,
                    0,
                    source,
                    GL.GL_UNSIGNED_BYTE,
                    data,
                )
        wants_mips = (
            self.filter_mode != TextureFilter.NEAREST
            and self.width > 1
            and self.height > 1
        )
        if levels == 1 and not self.compressed and wants_mips:
            # В файле только мип 0 (generate_mips=False у cooker'а не значит
            # «без мипов» для рендера): достроим цепочку драйвером.
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            self._has_mips = True
        else:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, levels - 1)
            self._has_mips = levels > 1
        self._apply_sampling()
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._handle = handle
        return handle

    def _generate_mips(self, array: np.ndarray) -> bool:
        """Решить, нужно ли строить мип-уровни."""
        if self.filter_mode == TextureFilter.NEAREST or self.compressed:
            return False
        return self.width > 1 and self.height > 1 and array.ndim == 3

    def _pixel_formats(self, channels: int) -> tuple[int, int]:
        """Подобрать внутренний и исходный форматы по числу каналов."""
        from OpenGL import GL

        return {
            1: (GL.GL_R8, GL.GL_RED),
            2: (GL.GL_RG8, GL.GL_RG),
            3: (GL.GL_SRGB8 if self.srgb else GL.GL_RGB8, GL.GL_RGB),
            4: (GL.GL_SRGB8_ALPHA8 if self.srgb else GL.GL_RGBA8, GL.GL_RGBA),
        }.get(channels, (GL.GL_RGBA8, GL.GL_RGBA))

    def _apply_sampling(self) -> None:
        """Применить фильтрацию и режим обёртки."""
        from OpenGL import GL

        min_filter_name = _GL_FILTER.get(self.filter_mode, "GL_LINEAR_MIPMAP_LINEAR")
        if min_filter_name == "GL_LINEAR_MIPMAP_LINEAR" and not self._has_mips:
            min_filter_name = "GL_LINEAR"
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_MIN_FILTER,
            getattr(GL, min_filter_name),
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        wrap = GL.GL_REPEAT if self.wrap_repeat else GL.GL_CLAMP_TO_EDGE
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap)

    def set_anisotropy(self, level: float) -> bool:
        """Включить анизотропную фильтрацию, если драйвер её поддерживает.

        Args:
            level: Уровень анизотропии (обычно 2, 4, 8, 16).

        Returns:
            ``True``, если параметр применён.
        """
        from OpenGL import GL

        if not self._handle or not hasattr(GL, "GL_TEXTURE_MAX_ANISOTROPY_EXT"):
            return False
        max_level = 1.0
        if hasattr(GL, "GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT"):
            max_level = float(GL.glGetFloatv(GL.GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT))
        select_scratch_unit()
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._handle)
        GL.glTexParameterf(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_MAX_ANISOTROPY_EXT,
            min(float(level), max_level),
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return True

    def dispose(self) -> None:
        """Удалить текстуру из видеопамяти."""
        from OpenGL import GL

        if self._handle:
            GL.glDeleteTextures([self._handle])
            self._handle = 0

    def __repr__(self) -> str:
        state = "gpu" if self.is_uploaded else "cpu"
        return f"<GpuTexture '{self.name}' {self.width}x{self.height} {state}>"
