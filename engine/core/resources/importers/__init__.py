"""Встроенные импортёры ассетов и их набор по умолчанию."""

from __future__ import annotations

from engine.core.resources.importers.audio_importer import AudioImporter, probe_audio
from engine.core.resources.importers.audio_json_importer import (
    CueImporter,
    MusicImporter,
)
from engine.core.resources.importers.dds_importer import DdsImporter, parse_dds_header
from engine.core.resources.importers.fbx_importer import FbxImporter
from engine.core.resources.importers.gltf_importer import GltfImporter
from engine.core.resources.importers.image_importer import ImageImporter
from engine.core.resources.importers.material_importer import MaterialImporter
from engine.core.resources.importers.obj_importer import ObjImporter
from engine.core.resources.importers.prefab_importer import PrefabImporter
from engine.core.resources.importers.shader_importer import ShaderImporter
from engine.core.resources.importers.sprite_atlas_importer import SpriteAtlasImporter

__all__ = [
    "AudioImporter",
    "CueImporter",
    "DdsImporter",
    "FbxImporter",
    "GltfImporter",
    "ImageImporter",
    "MaterialImporter",
    "MusicImporter",
    "ObjImporter",
    "PrefabImporter",
    "ShaderImporter",
    "SpriteAtlasImporter",
    "default_importers",
    "parse_dds_header",
    "probe_audio",
]


def default_importers() -> tuple[
    ObjImporter,
    GltfImporter,
    ImageImporter,
    DdsImporter,
    AudioImporter,
    MaterialImporter,
    ShaderImporter,
    FbxImporter,
    SpriteAtlasImporter,
    PrefabImporter,
    CueImporter,
    MusicImporter,
]:
    """Собрать набор импортёров, который движок использует по умолчанию."""
    return (
        ObjImporter(),
        GltfImporter(),
        ImageImporter(),
        DdsImporter(),
        AudioImporter(),
        MaterialImporter(),
        ShaderImporter(),
        FbxImporter(),
        SpriteAtlasImporter(),
        PrefabImporter(),
        CueImporter(),
        MusicImporter(),
    )
