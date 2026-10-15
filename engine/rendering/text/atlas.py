"""Атлас глифов: растеризация шрифта Pillow в одну RGBA-текстуру.

Сам атлас (пиксели и метрики) не требует OpenGL и тестируется без контекста;
GPU-текстура создаётся лениво методом :meth:`TextAtlas.texture`. Одна страница:
512×512, при нехватке места 1024×1024, при дальнейшем переполнении — предупреждение
и обрезка набора символов.

Пример:
    >>> atlas = get_atlas("", 32)
    >>> atlas.get_glyph(ord("A")).advance_x
    22.0
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.rendering.texture import GpuTexture

__all__ = [
    "DEFAULT_CHARSET",
    "DEFAULT_FONT",
    "GlyphMetrics",
    "TextAtlas",
    "get_atlas",
    "resolve_font_path",
]

_log = logging.getLogger("engine.rendering.text.atlas")

#: ASCII 32..126, кириллица А–я (0x410..0x44F), Ё и ё.
DEFAULT_CHARSET: str = (
    "".join(chr(code) for code in range(32, 127))
    + "".join(chr(code) for code in range(0x410, 0x450))
    + "Ёё"
)

#: Шрифт по умолчанию (DejaVu Sans, свободная лицензия, см. рядом LICENSE).
DEFAULT_FONT: Path = Path(__file__).resolve().parents[3] / "assets" / "fonts" / (
    "DejaVuSans.ttf"
)

#: Запасной системный шрифт, если файла по умолчанию нет.
_FALLBACK_FONTS: tuple[str, ...] = (
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

#: Зазор между глифами в атласе (защита от протекания при билинейной выборке).
_PADDING = 2
_ATLAS_SIZES = (512, 1024)
_CACHE_LIMIT = 16
_ITALIC_SHEAR = 0.2


@dataclass(frozen=True)
class GlyphMetrics:
    """Метрики и положение глифа в атласе (пиксели, ось Y вверх от базовой линии).

    Attributes:
        codepoint: Код символа Unicode.
        uv_rect: ``(u0, v0, u1, v1)``; ``v0`` — верхняя кромка глифа.
        advance_x: Сдвиг пера к следующему глифу.
        bearing_x: Смещение левой кромки от пера.
        bearing_y: Высота верхней кромки над базовой линией.
        width_px: Ширина растра глифа.
        height_px: Высота растра глифа.
        line_height: Высота строки шрифта.
    """

    codepoint: int
    uv_rect: tuple[float, float, float, float]
    advance_x: float
    bearing_x: float
    bearing_y: float
    width_px: int
    height_px: int
    line_height: float


def resolve_font_path(font_path: str, roots: tuple[Path, ...] = ()) -> Path:
    """Найти файл шрифта.

    Args:
        font_path: Путь из компонента; пустой — шрифт по умолчанию.
        roots: Каталоги, относительно которых ищется относительный путь.

    Returns:
        Путь к существующему файлу. Если ничего не найдено, предупреждение и
        запасной шрифт (``DEFAULT_FONT`` или системный).

    Raises:
        FileNotFoundError: Если нет ни запрошенного, ни запасных шрифтов.
    """
    if font_path:
        path = Path(font_path)
        candidates = [path]
        if not path.is_absolute():
            candidates += [root / path for root in roots]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        _log.warning("Шрифт '%s' не найден, используется шрифт по умолчанию", font_path)
    for fallback in (DEFAULT_FONT, *(Path(p) for p in _FALLBACK_FONTS)):
        if fallback.is_file():
            return fallback.resolve()
    raise FileNotFoundError("Не найден ни один шрифт (assets/fonts/DejaVuSans.ttf)")


class TextAtlas:
    """Растеризованный набор глифов одного шрифта и размера.

    Args:
        font_path: Путь к TTF/OTF-файлу.
        size_px: Размер шрифта в пикселях (> 0).
        bold: Синтетически утолщить глифы.
        italic: Синтетически наклонить глифы.
        charset: Набор символов (``None`` — :data:`DEFAULT_CHARSET`).

    Attributes:
        font_path: Путь к файлу шрифта.
        size_px: Размер шрифта.
        width: Ширина страницы атласа.
        height: Высота страницы атласа.
        pixels: Массив ``uint8`` формы ``(height, width, 4)``: RGB белые,
            альфа — покрытие.
        ascent: Высота над базовой линией.
        line_height: Высота строки.
    """

    def __init__(
        self,
        font_path: str | Path,
        size_px: int,
        *,
        bold: bool = False,
        italic: bool = False,
        charset: str | None = None,
    ) -> None:
        if int(size_px) <= 0:
            raise ValueError("size_px должен быть положительным")
        self.font_path = str(font_path)
        self.size_px = int(size_px)
        self.bold = bool(bold)
        self.italic = bool(italic)
        self._glyphs: dict[int, GlyphMetrics] = {}
        self._texture: GpuTexture | None = None
        font = ImageFont.truetype(self.font_path, self.size_px)
        ascent, descent = font.getmetrics()
        self.ascent = float(ascent)
        self.line_height = float(ascent + descent)
        text = DEFAULT_CHARSET if charset is None else charset
        chars = list(dict.fromkeys(text))  # без повторов, порядок сохранён
        self.width = self.height = _ATLAS_SIZES[0]
        self.pixels = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self._build(font, chars)

    # ------------------------------------------------------------------ #
    # Растеризация
    # ------------------------------------------------------------------ #
    def _rasterize(self, font: ImageFont.FreeTypeFont, char: str):
        """Растр глифа: ``(L-картинка | None, bearing_x, bearing_y, advance)``."""
        advance = float(font.getlength(char))
        left, top, right, bottom = font.getbbox(char, anchor="ls")
        width, height = int(right - left), int(bottom - top)
        if width <= 0 or height <= 0:
            return None, 0.0, 0.0, advance
        embolden = max(1, self.size_px // 16) if self.bold else 0
        shear_pad = math.ceil(height * _ITALIC_SHEAR) if self.italic else 0
        canvas_w = width + embolden + 2 * shear_pad
        image = Image.new("L", (canvas_w, height), 0)
        draw = ImageDraw.Draw(image)
        origin = (shear_pad - left, -top)
        draw.text(origin, char, font=font, fill=255, anchor="ls")
        if embolden:
            shifted = (origin[0] + embolden, origin[1])
            draw.text(shifted, char, font=font, fill=255, anchor="ls")
        bearing_y = float(-top)
        if self.italic:
            image = image.transform(
                image.size,
                Image.Transform.AFFINE,
                (1.0, _ITALIC_SHEAR, -_ITALIC_SHEAR * bearing_y, 0.0, 1.0, 0.0),
                resample=Image.Resampling.BILINEAR,
            )
        return image, float(left - shear_pad), bearing_y, advance + embolden

    def _build(self, font: ImageFont.FreeTypeFont, chars: list[str]) -> None:
        rasters = [(char, *self._rasterize(font, char)) for char in chars]
        for size in _ATLAS_SIZES:
            placed, overflow = self._pack(rasters, size)
            if not overflow:
                break
        if overflow:
            _log.warning(
                "Атлас шрифта %s (%d px): %d символов не помещаются в %d×%d, "
                "набор обрезан",
                Path(self.font_path).name,
                self.size_px,
                len(overflow),
                size,
                size,
            )
        self.width = self.height = size
        self.pixels = np.zeros((size, size, 4), dtype=np.uint8)
        self.pixels[:, :, :3] = 255
        for char, image, bx, by, advance, x, y in placed:
            if image is not None:
                w, h = image.size
                coverage = np.asarray(image, dtype=np.uint8)
                self.pixels[y : y + h, x : x + w, 3] = coverage
                uv = (x / size, y / size, (x + w) / size, (y + h) / size)
                size_wh = (w, h)
            else:
                uv = (0.0, 0.0, 0.0, 0.0)
                size_wh = (0, 0)
            self._glyphs[ord(char)] = GlyphMetrics(
                ord(char), uv, advance, bx, by, size_wh[0], size_wh[1], self.line_height
            )

    @staticmethod
    def _pack(rasters: list, size: int) -> tuple[list, list]:
        """Простая полочная упаковка; возвращает размещённые и не поместившиеся."""
        placed: list = []
        overflow: list = []
        x = y = row_height = 0
        for char, image, bx, by, advance in rasters:
            if image is None:
                placed.append((char, None, bx, by, advance, 0, 0))
                continue
            w, h = image.size
            if x + w + _PADDING > size:
                x, y, row_height = 0, y + row_height + _PADDING, 0
            if y + h + _PADDING > size or w + _PADDING > size:
                overflow.append(char)
                continue
            pad = _PADDING // 2
            placed.append((char, image, bx, by, advance, x + pad, y + pad))
            x += w + _PADDING
            row_height = max(row_height, h)
        return placed, overflow

    # ------------------------------------------------------------------ #
    # Доступ
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._glyphs)

    def __contains__(self, codepoint: int) -> bool:
        return codepoint in self._glyphs

    def get_glyph(self, codepoint: int) -> GlyphMetrics:
        """Метрики глифа; для отсутствующего символа — ``?``, иначе пустой глиф."""
        glyph = self._glyphs.get(codepoint)
        if glyph is not None:
            return glyph
        glyph = self._glyphs.get(ord("?"))
        if glyph is not None:
            return glyph
        return GlyphMetrics(
            codepoint, (0.0, 0.0, 0.0, 0.0), self.size_px * 0.5, 0.0, 0.0, 0, 0,
            self.line_height,
        )

    def texture(self) -> GpuTexture:
        """GPU-текстура атласа (RGBA8, линейная фильтрация, без мипов)."""
        from engine.rendering.texture import GpuTexture, TextureFilter

        if self._texture is None:
            self._texture = GpuTexture.from_bytes(
                f"font_atlas_{Path(self.font_path).stem}_{self.size_px}",
                self.width,
                self.height,
                self.pixels,
                filter_mode=TextureFilter.BILINEAR,
                generate_mips=False,
                wrap_repeat=False,
            )
        if not self._texture.is_uploaded:
            self._texture.upload()
        return self._texture

    def dispose(self) -> None:
        """Освободить GPU-текстуру (если создавалась)."""
        if self._texture is not None:
            self._texture.dispose()
            self._texture = None


_cache: OrderedDict[tuple, TextAtlas] = OrderedDict()
#: Разрешённые пути шрифтов: обращение к файловой системе дорого для кадра.
_paths: dict[tuple[str, tuple[Path, ...]], Path] = {}


def get_atlas(
    font_path: str | Path,
    size_px: int,
    *,
    bold: bool = False,
    italic: bool = False,
    charset: str | None = None,
    roots: tuple[Path, ...] = (),
) -> TextAtlas:
    """Атлас из кэша (LRU на 16 записей) или новый.

    Ключ: абсолютный путь шрифта, размер, bold, italic (и набор символов, если он
    не по умолчанию). Пустой ``font_path`` означает шрифт по умолчанию.
    """
    path_key = (str(font_path), tuple(roots))
    path = _paths.get(path_key)
    if path is None:
        path = _paths[path_key] = resolve_font_path(str(font_path), roots)
    key = (str(path), int(size_px), bool(bold), bool(italic), charset)
    atlas = _cache.get(key)
    if atlas is not None:
        _cache.move_to_end(key)
        return atlas
    atlas = TextAtlas(path, size_px, bold=bold, italic=italic, charset=charset)
    _cache[key] = atlas
    while len(_cache) > _CACHE_LIMIT:
        _, evicted = _cache.popitem(last=False)
        evicted.dispose()
    return atlas


def upload_cached_atlases() -> int:
    """Залить в видеопамять GPU-текстуры всех атласов кэша (прогрев до кадра).

    Returns:
        Число атласов в кэше.
    """
    for atlas in _cache.values():
        atlas.texture()
    return len(_cache)


def clear_cache() -> None:
    """Сбросить кэш атласов (для тестов и смены GL-контекста)."""
    for atlas in _cache.values():
        atlas.dispose()
    _cache.clear()
    _paths.clear()
