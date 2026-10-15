"""Cooker спрайт-атласов: нарезка листа и упаковка спрайтов в одну текстуру.

Три режима описания (файл ``*.atlas.json``):

* ``grid`` — лист ``image`` нарезается сеткой ячеек ``cell`` (с отступом
  ``margin`` по краям и промежутком ``spacing`` между ячейками);
* ``alpha`` — лист нарезается по связным областям непрозрачных пикселей;
* ``pack`` — набор отдельных картинок ``sources`` (или все PNG каталога
  ``folder``) укладывается в общий атлас.

Нарезанные в ``grid``/``alpha`` спрайты остаются на тех же местах, что и в
исходном листе. В ``pack`` спрайты упаковываются «полками» с зазором
``padding``; зазор заполняется краевыми пикселями спрайта (экструзия), чтобы
линейная фильтрация не подмешивала соседей.

Прямоугольники кадров заданы в пикселях от **верхнего левого** угла
изображения (так же лежат строки в файле). Модуль не знает про GL.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from engine.core.resources.errors import AssetImportError

__all__ = [
    "AtlasSpec",
    "BuiltAtlas",
    "Sprite",
    "SpriteFrame",
    "build_atlas",
    "pack_sprites",
    "parse_atlas_spec",
    "slice_alpha",
    "slice_grid",
    "to_rgba",
]

ATLAS_FORMAT = "mysprites"
ATLAS_VERSION = 1
MODES = ("grid", "alpha", "pack")


@dataclass(frozen=True, slots=True)
class SpriteFrame:
    """Кадр атласа.

    Attributes:
        name: Имя спрайта (уникально в атласе).
        x: Левый край кадра в пикселях атласа.
        y: Верхний край кадра в пикселях атласа.
        w: Ширина кадра (после обрезки прозрачных полей).
        h: Высота кадра.
        pivot: Точка привязки в долях исходного (необрезанного) спрайта.
        source_size: Размер спрайта до обрезки.
        offset: Сдвиг обрезанного кадра внутри исходного спрайта (от верха-слева).
    """

    name: str
    x: int
    y: int
    w: int
    h: int
    pivot: tuple[float, float] = (0.5, 0.5)
    source_size: tuple[int, int] = (0, 0)
    offset: tuple[int, int] = (0, 0)

    def uv(
        self, atlas_width: int, atlas_height: int
    ) -> tuple[float, float, float, float]:
        """Нормализованные ``(u0, v0, u1, v1)``; ``v`` растёт сверху вниз."""
        return (
            self.x / atlas_width,
            self.y / atlas_height,
            (self.x + self.w) / atlas_width,
            (self.y + self.h) / atlas_height,
        )

    def to_dict(self, atlas_width: int, atlas_height: int) -> dict[str, Any]:
        """Запись для ``atlas.json`` (с готовыми UV)."""
        u0, v0, u1, v1 = self.uv(atlas_width, atlas_height)
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "uv": [u0, v0, u1, v1],
            "pivot": list(self.pivot),
            "source_size": list(self.source_size),
            "offset": list(self.offset),
        }

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> SpriteFrame:
        """Восстановить кадр из :meth:`to_dict`.

        Raises:
            KeyError: Если нет обязательных полей.
        """
        w, h = int(data["w"]), int(data["h"])
        pivot = data.get("pivot", (0.5, 0.5))
        source = data.get("source_size", (w, h))
        offset = data.get("offset", (0, 0))
        return cls(
            name,
            int(data["x"]),
            int(data["y"]),
            w,
            h,
            pivot=(float(pivot[0]), float(pivot[1])),
            source_size=(int(source[0]), int(source[1])),
            offset=(int(offset[0]), int(offset[1])),
        )


@dataclass(slots=True)
class Sprite:
    """Спрайт до упаковки.

    Attributes:
        name: Имя.
        pixels: ``uint8`` RGBA ``(H, W, 4)``.
        origin: Место на исходном листе ``(x, y)`` или ``None`` (отдельный файл).
    """

    name: str
    pixels: np.ndarray
    origin: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class AtlasSpec:
    """Проверенное описание атласа.

    Attributes:
        mode: ``grid`` | ``alpha`` | ``pack``.
        image: Лист для ``grid``/``alpha`` (путь относительно спецификации).
        sources: Файлы для ``pack`` (пути относительно спецификации).
        folder: Каталог с PNG для ``pack``.
        cell: Размер ячейки сетки ``(w, h)``.
        margin: Отступ сетки от краёв листа.
        spacing: Промежуток между ячейками.
        skip_empty: Пропускать полностью прозрачные ячейки сетки.
        threshold: Порог альфы (0..255) «непрозрачного» пикселя в ``alpha``.
        min_size: Минимальная сторона области в ``alpha`` (мельче — шум).
        prefix: Префикс автоматических имён.
        names: Явные имена по порядку нарезки.
        pivot: Точка привязки всех кадров.
        padding: Зазор между спрайтами в ``pack``.
        max_size: Предел стороны атласа в ``pack``.
        power_of_two: Размер атласа — степень двойки.
        trim: Обрезать прозрачные поля спрайтов в ``pack``.
    """

    mode: str
    image: str | None = None
    sources: tuple[str, ...] = ()
    folder: str | None = None
    cell: tuple[int, int] = (0, 0)
    margin: int = 0
    spacing: int = 0
    skip_empty: bool = True
    threshold: int = 1
    min_size: int = 2
    prefix: str = "sprite"
    names: tuple[str, ...] = ()
    pivot: tuple[float, float] = (0.5, 0.5)
    padding: int = 2
    max_size: int = 2048
    power_of_two: bool = True
    trim: bool = False


@dataclass(slots=True)
class BuiltAtlas:
    """Готовый атлас.

    Attributes:
        pixels: ``uint8`` RGBA ``(H, W, 4)``.
        frames: Кадры в порядке нарезки/упаковки.
        inputs: Исходные файлы, содержимое которых вошло в атлас.
    """

    pixels: np.ndarray
    frames: list[SpriteFrame]
    inputs: list[Path] = field(default_factory=list)

    @property
    def width(self) -> int:
        """Ширина атласа."""
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        """Высота атласа."""
        return int(self.pixels.shape[0])

    def document(self) -> dict[str, Any]:
        """Содержимое ``atlas.json``."""
        return {
            "format": ATLAS_FORMAT,
            "version": ATLAS_VERSION,
            "size": [self.width, self.height],
            "frames": {
                frame.name: frame.to_dict(self.width, self.height)
                for frame in self.frames
            },
        }


# ---------------------------------------------------------------------- #
# Описание атласа
# ---------------------------------------------------------------------- #
def parse_atlas_spec(text: str) -> AtlasSpec:
    """Разобрать и проверить ``*.atlas.json``.

    Raises:
        AssetImportError: При некорректном JSON, режиме или параметрах.
    """
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise AssetImportError(f"Некорректный JSON атласа: {exc}") from exc
    if not isinstance(data, dict):
        raise AssetImportError("Описание атласа должно быть JSON-объектом")
    mode = data.get("mode")
    if mode not in MODES:
        raise AssetImportError(
            f"mode: допустимы {', '.join(MODES)}, получено {mode!r}"
        )

    def integer(key: str, default: int, low: int) -> int:
        try:
            value = int(data.get(key, default))
        except (TypeError, ValueError) as exc:
            raise AssetImportError(f"{key}: ожидается целое число") from exc
        if value < low:
            raise AssetImportError(f"{key}: не меньше {low}")
        return value

    pair = _pair(data.get("cell"), "cell", integer=True) if "cell" in data else (0, 0)
    pivot = _pair(data.get("pivot", (0.5, 0.5)), "pivot", integer=False)
    spec = AtlasSpec(
        mode=mode,
        image=_optional_text(data, "image"),
        sources=tuple(_text_list(data.get("sources", ()), "sources")),
        folder=_optional_text(data, "folder"),
        cell=(int(pair[0]), int(pair[1])),
        margin=integer("margin", 0, 0),
        spacing=integer("spacing", 0, 0),
        skip_empty=bool(data.get("skip_empty", True)),
        threshold=integer("threshold", 1, 0),
        min_size=integer("min_size", 2, 1),
        prefix=str(data.get("prefix", "sprite")),
        names=tuple(_text_list(data.get("names", ()), "names")),
        pivot=(float(pivot[0]), float(pivot[1])),
        padding=integer("padding", 2, 0),
        max_size=integer("max_size", 2048, 1),
        power_of_two=bool(data.get("power_of_two", True)),
        trim=bool(data.get("trim", False)),
    )
    if mode in ("grid", "alpha") and not spec.image:
        raise AssetImportError(f"Для mode={mode!r} нужно поле image")
    if mode == "grid" and min(spec.cell) < 1:
        raise AssetImportError("Для mode='grid' нужно cell: [ширина, высота] > 0")
    if mode == "pack" and not (spec.sources or spec.folder):
        raise AssetImportError("Для mode='pack' нужны sources или folder")
    if len(set(spec.names)) != len(spec.names):
        raise AssetImportError("names: имена спрайтов должны быть уникальны")
    return spec


def _optional_text(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AssetImportError(f"{key}: ожидается непустая строка")
    return value


def _text_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise AssetImportError(f"{key}: ожидается список непустых строк")
    return list(value)


def _pair(value: Any, key: str, *, integer: bool) -> tuple[float, float]:
    try:
        first, second = value
        return (int(first), int(second)) if integer else (float(first), float(second))
    except (TypeError, ValueError) as exc:
        raise AssetImportError(f"{key}: ожидается пара чисел") from exc


# ---------------------------------------------------------------------- #
# Сборка
# ---------------------------------------------------------------------- #
def build_atlas(spec: AtlasSpec, base_dir: Path) -> BuiltAtlas:
    """Собрать атлас по описанию.

    Args:
        spec: Проверенное описание.
        base_dir: Каталог, от которого отсчитаны пути в описании.

    Raises:
        AssetImportError: Если файл не найден или не декодируется, нарезка
            ничего не дала или спрайты не помещаются в ``max_size``.
    """
    from engine.core.resources.cookers.texture_cooker import decode_image

    if spec.mode == "pack":
        files = _pack_files(spec, base_dir)
        sprites = [Sprite(path.stem, to_rgba(decode_image(path))) for path in files]
        built = pack_sprites(
            sprites,
            padding=spec.padding,
            max_size=spec.max_size,
            power_of_two=spec.power_of_two,
            trim=spec.trim,
            pivot=spec.pivot,
        )
        built.inputs = files
        return built

    path = _inside(base_dir, str(spec.image))
    sheet = to_rgba(decode_image(path))
    if spec.mode == "grid":
        cells = slice_grid(
            sheet, spec.cell, spec.margin, spec.spacing, spec.skip_empty, spec.prefix
        )
    else:
        cells = slice_alpha(sheet, spec.threshold, spec.min_size, spec.prefix)
    if not cells:
        raise AssetImportError(f"'{path.name}': нарезка не нашла ни одного спрайта")
    names = _names(cells, spec.names)
    frames = [
        SpriteFrame(
            name,
            *(cell.origin or (0, 0)),
            cell.pixels.shape[1],
            cell.pixels.shape[0],
            pivot=spec.pivot,
            source_size=(cell.pixels.shape[1], cell.pixels.shape[0]),
        )
        for name, cell in zip(names, cells, strict=True)
    ]
    return BuiltAtlas(sheet, frames, [path])


def _names(cells: Sequence[Sprite], explicit: Sequence[str]) -> list[str]:
    names = [
        explicit[i] if i < len(explicit) else cell.name for i, cell in enumerate(cells)
    ]
    if len(set(names)) != len(names):
        raise AssetImportError("Имена спрайтов атласа повторяются")
    return names


def _pack_files(spec: AtlasSpec, base_dir: Path) -> list[Path]:
    files = [_inside(base_dir, item) for item in spec.sources]
    if spec.folder:
        folder = _inside(base_dir, spec.folder, directory=True)
        files += sorted(folder.glob("*.png"), key=lambda item: item.name.lower())
    unique = list(dict.fromkeys(files))
    stems = [path.stem for path in unique]
    if len(set(stems)) != len(stems):
        raise AssetImportError("В атласе есть файлы с одинаковыми именами")
    if not unique:
        raise AssetImportError("В атлас нечего упаковывать")
    return unique


def _inside(base_dir: Path, relative: str, *, directory: bool = False) -> Path:
    """Путь внутри каталога описания: без абсолютных путей и выхода наверх."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise AssetImportError(f"Путь должен быть внутри каталога атласа: '{relative}'")
    full = base_dir / path
    exists = full.is_dir() if directory else full.is_file()
    if not exists:
        raise AssetImportError(f"Не найдено: '{relative}'")
    return full


def to_rgba(pixels: np.ndarray) -> np.ndarray:
    """Привести ``(H, W, 1..4)`` к RGBA ``uint8`` (серый и RGB получают альфу 255)."""
    channels = pixels.shape[2]
    if channels == 4:
        return np.ascontiguousarray(pixels)
    result = np.empty((*pixels.shape[:2], 4), dtype=np.uint8)
    result[..., 3] = 255
    if channels == 3:
        result[..., :3] = pixels
    else:
        result[..., :3] = pixels[..., :1]
        if channels == 2:
            result[..., 3] = pixels[..., 1]
    return result


def _alpha_mask(pixels: np.ndarray, threshold: int) -> np.ndarray:
    return pixels[..., 3] >= max(threshold, 1)


# ---------------------------------------------------------------------- #
# Нарезка
# ---------------------------------------------------------------------- #
def slice_grid(
    sheet: np.ndarray,
    cell: tuple[int, int],
    margin: int = 0,
    spacing: int = 0,
    skip_empty: bool = True,
    prefix: str = "sprite",
) -> list[Sprite]:
    """Нарезать лист сеткой ячеек.

    Args:
        sheet: Лист RGBA ``(H, W, 4)``.
        cell: Размер ячейки ``(ширина, высота)``.
        margin: Отступ от краёв листа.
        spacing: Промежуток между ячейками.
        skip_empty: Пропускать полностью прозрачные ячейки.
        prefix: Префикс имён; индекс считается по всем ячейкам (построчно), в
            том числе пропущенным, поэтому имя не зависит от соседей.

    Returns:
        Спрайты в порядке «строка за строкой».
    """
    width, height = cell
    full_h, full_w = sheet.shape[:2]
    columns = (full_w - 2 * margin + spacing) // (width + spacing)
    rows = (full_h - 2 * margin + spacing) // (height + spacing)
    digits = max(2, len(str(max(columns * rows - 1, 0))))
    sprites: list[Sprite] = []
    for row in range(rows):
        for column in range(columns):
            x = margin + column * (width + spacing)
            y = margin + row * (height + spacing)
            pixels = sheet[y : y + height, x : x + width]
            if skip_empty and not pixels[..., 3].any():
                continue
            index = row * columns + column
            sprites.append(
                Sprite(f"{prefix}_{index:0{digits}d}", pixels.copy(), (x, y))
            )
    return sprites


def slice_alpha(
    sheet: np.ndarray, threshold: int = 1, min_size: int = 2, prefix: str = "sprite"
) -> list[Sprite]:
    """Нарезать лист по связным областям непрозрачных пикселей (8-связность).

    Args:
        sheet: Лист RGBA ``(H, W, 4)``.
        threshold: Минимальная альфа «непрозрачного» пикселя.
        min_size: Области, у которых обе стороны меньше, отбрасываются (шум).
        prefix: Префикс имён.

    Returns:
        Спрайты, упорядоченные по верхнему краю, затем по левому.
    """
    boxes = [
        box
        for box in _regions(_alpha_mask(sheet, threshold))
        if max(box[2] - box[0], box[3] - box[1]) >= min_size
    ]
    boxes.sort(key=lambda box: (box[1], box[0]))
    digits = max(2, len(str(max(len(boxes) - 1, 0))))
    return [
        Sprite(
            f"{prefix}_{index:0{digits}d}",
            sheet[y0:y1, x0:x1].copy(),
            (x0, y0),
        )
        for index, (x0, y0, x1, y1) in enumerate(boxes)
    ]


def _regions(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Ограничивающие рамки ``(x0, y0, x1, y1)`` связных областей маски."""
    height, width = mask.shape
    padded = np.zeros((height, width + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    edges = np.diff(padded, axis=1)
    ys, starts = np.nonzero(edges == 1)
    _, ends = np.nonzero(edges == -1)
    if ys.size == 0:
        return []

    parent = list(range(ys.size))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    row_start = np.searchsorted(ys, np.arange(height + 1))
    for row in range(1, height):
        a0, a1 = row_start[row - 1], row_start[row]
        b0, b1 = row_start[row], row_start[row + 1]
        if a0 == a1 or b0 == b1:
            continue
        previous_starts, previous_ends = starts[a0:a1], ends[a0:a1]
        low = np.searchsorted(previous_ends, starts[b0:b1], side="left")
        high = np.searchsorted(previous_starts, ends[b0:b1], side="right")
        for offset, (first, last) in enumerate(zip(low, high, strict=True)):
            current = find(b0 + offset)
            for other in range(a0 + int(first), a0 + int(last)):
                root = find(other)
                if root != current:
                    parent[root] = current

    boxes: dict[int, list[int]] = {}
    for run in range(ys.size):
        root = find(run)
        y, x0, x1 = int(ys[run]), int(starts[run]), int(ends[run])
        box = boxes.get(root)
        if box is None:
            boxes[root] = [x0, y, x1, y + 1]
        else:
            box[0] = min(box[0], x0)
            box[2] = max(box[2], x1)
            box[3] = y + 1
    return [(b[0], b[1], b[2], b[3]) for b in boxes.values()]


# ---------------------------------------------------------------------- #
# Упаковка
# ---------------------------------------------------------------------- #
def pack_sprites(
    sprites: Sequence[Sprite],
    *,
    padding: int = 2,
    max_size: int = 2048,
    power_of_two: bool = True,
    trim: bool = False,
    pivot: tuple[float, float] = (0.5, 0.5),
) -> BuiltAtlas:
    """Уложить спрайты в один атлас «полками».

    Перебираются ширины атласа (степени двойки); выбирается та, при которой
    площадь минимальна. Зазор ``padding`` заполняется краевыми пикселями.

    Args:
        sprites: Спрайты RGBA; имена уникальны.
        padding: Зазор вокруг каждого спрайта.
        max_size: Предел стороны атласа.
        power_of_two: Округлять высоту атласа до степени двойки.
        trim: Обрезать полностью прозрачные поля (смещение в ``offset``).
        pivot: Точка привязки кадров.

    Raises:
        AssetImportError: Если нет спрайтов, имена повторяются или спрайты не
            помещаются в ``max_size``.
    """
    if not sprites:
        raise AssetImportError("В атлас нечего упаковывать")
    names = [sprite.name for sprite in sprites]
    if len(set(names)) != len(names):
        raise AssetImportError("Имена спрайтов атласа повторяются")

    cropped: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]] = []
    for sprite in sprites:
        pixels = sprite.pixels
        offset = (0, 0)
        if trim and pixels.shape[0] and pixels.shape[1]:
            pixels, offset = _trim(pixels)
        source = (sprite.pixels.shape[1], sprite.pixels.shape[0])
        cropped.append((pixels, offset, source))

    sizes = [
        (p.shape[1] + 2 * padding, p.shape[0] + 2 * padding) for p, _, _ in cropped
    ]
    if max(max(size) for size in sizes) > max_size:
        raise AssetImportError(
            f"Спрайт с зазором не помещается в max_size={max_size}"
        )
    width, positions, height = _best_layout(sizes, max_size, power_of_two)

    atlas = np.zeros((height, width, 4), dtype=np.uint8)
    frames: list[SpriteFrame] = []
    for sprite, (pixels, offset, source), (px, py) in zip(
        sprites, cropped, positions, strict=True
    ):
        h, w = pixels.shape[:2]
        x, y = px + padding, py + padding
        if padding:
            atlas[py : py + h + 2 * padding, px : px + w + 2 * padding] = np.pad(
                pixels, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
            )
        atlas[y : y + h, x : x + w] = pixels
        frames.append(
            SpriteFrame(
                sprite.name, x, y, w, h, pivot=pivot, source_size=source, offset=offset
            )
        )
    return BuiltAtlas(atlas, frames)


def _trim(pixels: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    rows = np.nonzero(pixels[..., 3].any(axis=1))[0]
    columns = np.nonzero(pixels[..., 3].any(axis=0))[0]
    if rows.size == 0:
        return pixels[:1, :1], (0, 0)
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, columns[0], columns[-1] + 1
    return pixels[y0:y1, x0:x1], (int(x0), int(y0))


def _best_layout(
    sizes: Sequence[tuple[int, int]], max_size: int, power_of_two: bool
) -> tuple[int, list[tuple[int, int]], int]:
    """Выбрать ширину атласа с наименьшей площадью и вернуть раскладку."""
    widest = max(width for width, _ in sizes)
    best: tuple[int, int, list[tuple[int, int]]] | None = None
    width = 1
    while width < widest:
        width *= 2
    while width <= max_size:
        positions, used = _shelves(sizes, width)
        height = _ceil_pot(used) if power_of_two else used
        if height <= max_size and (best is None or width * height < best[0] * best[1]):
            best = (width, height, positions)
        width *= 2
    if best is None:
        raise AssetImportError(f"Спрайты не помещаются в атлас max_size={max_size}")
    return best[0], best[2], best[1]


def _shelves(
    sizes: Sequence[tuple[int, int]], width: int
) -> tuple[list[tuple[int, int]], int]:
    """Раскладка полками: от высоких спрайтов к низким, слева направо."""
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i][1], -sizes[i][0], i))
    positions = [(0, 0)] * len(sizes)
    x = y = shelf = 0
    for index in order:
        w, h = sizes[index]
        if x + w > width:
            x, y, shelf = 0, y + shelf, 0
        positions[index] = (x, y)
        x += w
        shelf = max(shelf, h)
    return positions, y + shelf


def _ceil_pot(value: int) -> int:
    result = 1
    while result < value:
        result *= 2
    return result
