"""Подсистема ресурсов: ассеты, загрузчики и кэш.

Публичный интерфейс подсистемы — :class:`ResourceManager` и типы ресурсов;
загрузчики подключаются регистрацией, поэтому новые форматы не требуют правок
в ядре. Поверх них работает пайплайн импорта: :class:`AssetDatabase` (GUID,
``.meta``, кэш ``Library/``) и реестр импортёров :class:`ImporterRegistry`.
"""

from __future__ import annotations

from engine.core.resources.asset import (
    Asset,
    AssetType,
    MeshAsset,
    ShaderAsset,
    ShaderStage,
    TextureAsset,
    describe,
)
from engine.core.resources.asset_db import (
    AssetDatabase,
    AssetHandle,
    AssetImported,
    AssetImportFailed,
)
from engine.core.resources.asset_ref import AssetRef
from engine.core.resources.errors import (
    AssetImportError,
    ImporterNotFoundError,
    InvalidGuidError,
    MetaFormatError,
    MissingLoaderError,
    ResourceError,
    ResourceFormatError,
    ResourceLoadError,
    ResourceNotFoundError,
)
from engine.core.resources.guid import is_valid_guid, new_guid, parse_guid
from engine.core.resources.import_manager import (
    AssetMoved,
    AssetRemoved,
    ImportBatchFinished,
    ImportBatchProgress,
    ImportBatchStarted,
    ImportManager,
    ImportState,
)
from engine.core.resources.import_settings import (
    SettingSpec,
    settings_schema,
    validate_settings,
)
from engine.core.resources.importer import (
    Artifact,
    AssetImporter,
    AssetPreview,
    ImportContext,
)
from engine.core.resources.importer_registry import ImporterRegistry
from engine.core.resources.importers import default_importers
from engine.core.resources.loader import AssetLoader, LoadContext, LoaderRegistry
from engine.core.resources.loaders import (
    GlslShaderLoader,
    ImageTextureLoader,
    ObjMeshLoader,
    default_loaders,
)
from engine.core.resources.material_asset import (
    ResolvedMaterial,
    load_material,
    load_material_artifact,
)
from engine.core.resources.meta import (
    AssetMeta,
    meta_path_for,
    read_meta,
    write_meta,
)
from engine.core.resources.pak import PakBuilder, PakReader
from engine.core.resources.primitives import (
    bounds_of,
    compute_normals,
    cube_mesh,
    cylinder_mesh,
    line_grid_mesh,
    plane_mesh,
    uv_sphere_mesh,
    wire_box_mesh,
)
from engine.core.resources.resource_manager import (
    AssetLoaded,
    AssetUnloaded,
    ResourceManager,
    ResourceStats,
)
from engine.core.resources.shader_asset import load_shader
from engine.core.resources.sprite_atlas import SpriteAtlas, load_sprite_atlas
from engine.core.resources.vfs import (
    VFS,
    LibraryVFS,
    NoBuildError,
    OverlayVFS,
    PakVFS,
    VfsAssets,
    WorkdirVFS,
    as_assets,
    detect_vfs,
)

__all__ = [
    "Artifact",
    "Asset",
    "AssetDatabase",
    "AssetRef",
    "AssetHandle",
    "AssetImportError",
    "AssetImportFailed",
    "AssetImported",
    "AssetImporter",
    "AssetLoaded",
    "AssetLoader",
    "AssetMeta",
    "AssetMoved",
    "AssetPreview",
    "AssetRemoved",
    "AssetType",
    "AssetUnloaded",
    "GlslShaderLoader",
    "ImageTextureLoader",
    "ImportBatchFinished",
    "ImportBatchProgress",
    "ImportBatchStarted",
    "ImportContext",
    "ImportManager",
    "ImportState",
    "ImporterNotFoundError",
    "ImporterRegistry",
    "InvalidGuidError",
    "LoadContext",
    "LoaderRegistry",
    "MeshAsset",
    "MetaFormatError",
    "MissingLoaderError",
    "ObjMeshLoader",
    "PakBuilder",
    "PakReader",
    "PakVFS",
    "LibraryVFS",
    "NoBuildError",
    "OverlayVFS",
    "VFS",
    "VfsAssets",
    "WorkdirVFS",
    "as_assets",
    "detect_vfs",
    "ResolvedMaterial",
    "ResourceError",
    "ResourceFormatError",
    "ResourceLoadError",
    "ResourceManager",
    "ResourceNotFoundError",
    "ResourceStats",
    "SettingSpec",
    "ShaderAsset",
    "ShaderStage",
    "SpriteAtlas",
    "TextureAsset",
    "bounds_of",
    "compute_normals",
    "cube_mesh",
    "cylinder_mesh",
    "default_importers",
    "default_loaders",
    "describe",
    "is_valid_guid",
    "line_grid_mesh",
    "load_material",
    "load_material_artifact",
    "load_shader",
    "load_sprite_atlas",
    "meta_path_for",
    "new_guid",
    "parse_guid",
    "plane_mesh",
    "read_meta",
    "settings_schema",
    "uv_sphere_mesh",
    "validate_settings",
    "wire_box_mesh",
    "write_meta",
]
