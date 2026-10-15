"""Импортёр шейдеров: ``.vert``/``.frag``/``.geom``/``.comp``/``.glsl``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.core.resources.errors import AssetImportError
from engine.core.resources.importer import Artifact, AssetPreview, ImportContext

if TYPE_CHECKING:
    from engine.core.resources.cookers.shader_cooker import ShaderCookSettings

__all__ = ["ShaderImporter"]


class ShaderImporter:
    """Шейдер → артефакт ``shader`` (``shader.glsl``: исходник без ``#include``).

    Файлы, на которые шейдер ссылается через ``#include``, регистрируются как
    «запечённые» зависимости: правка библиотеки переимпортирует все шейдеры,
    которые её включают. Откомпилированные бинарники программ строит и
    кэширует графический слой (``ProgramBinaryCache``), а не импорт.

    Настройки:
        include_dirs (list[str]): Каталоги поиска ``#include`` относительно
            папки шейдера.
        strip_comments (bool): Убрать комментарии и пустые строки.
        variants (list[dict]): Варианты для предварительной сборки:
            ``[{"name": "instanced", "defines": {"INSTANCED": 1}}]``.
    """

    name = "shader"
    extensions = (".vert", ".frag", ".geom", ".comp", ".glsl")
    version = 1

    def default_settings(self) -> dict[str, Any]:
        """Настройки по умолчанию."""
        return {"include_dirs": [], "strip_comments": False, "variants": []}

    def inspect(self, path: Path) -> AssetPreview:
        """Развернуть шейдер и вернуть краткую сводку."""
        from engine.core.resources.cookers.shader_cooker import cook_shader

        cooked = cook_shader(path)
        return AssetPreview(
            "shader",
            {
                "stage": cooked.stage.value,
                "glsl_version": cooked.glsl_version,
                "features": list(cooked.features),
                "library": not cooked.has_main,
            },
        )

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        """Развернуть ``#include`` и записать ``shader.glsl``."""
        from engine.core.resources.cookers.shader_cooker import cook_shader

        settings = _cook_settings(ctx)
        cooked = cook_shader(ctx.source, settings)
        ctx.add_baked_input(ctx.source)
        for include in dict.fromkeys(cooked.includes):
            ctx.add_baked_input(include)
        (ctx.output_dir / "shader.glsl").write_text(
            cooked.source, encoding="utf-8", newline="\n"
        )
        return [
            ctx.artifact(
                "shader",
                "shader.glsl",
                stage=cooked.stage.value,
                glsl_version=cooked.glsl_version,
                features=list(cooked.features),
                defaults=dict(cooked.defaults),
                includes=[Path(item).name for item in cooked.includes],
                library=not cooked.has_main,
                digest=cooked.digest,
                variants=[
                    {
                        "name": variant.name,
                        "defines": dict(variant.defines),
                        "key": variant.key(),
                    }
                    for variant in settings.variants
                ],
            )
        ]


def _cook_settings(ctx: ImportContext) -> ShaderCookSettings:
    from engine.core.resources.cookers.shader_cooker import (
        ShaderCookSettings,
        ShaderVariant,
        variant_key,
    )

    base = ctx.source.parent
    include_dirs = tuple(
        (base / str(item)).resolve() for item in ctx.setting("include_dirs", []) or []
    )
    variants: list[ShaderVariant] = []
    for entry in ctx.setting("variants", []) or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("defines"), dict):
            raise AssetImportError(
                "variants: ожидается список {'name': ..., 'defines': {...}}"
            )
        defines = dict(entry["defines"])
        name = str(entry.get("name") or variant_key(defines))
        variants.append(ShaderVariant(name, defines))
    return ShaderCookSettings(
        include_dirs=include_dirs,
        strip_comments=bool(ctx.setting("strip_comments", False)),
        variants=tuple(variants),
    )
