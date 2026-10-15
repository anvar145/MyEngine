"""Тесты cooker'а шейдеров: ключевые слова, варианты, импорт и загрузка."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.resources import (
    AssetDatabase,
    AssetImportError,
    ResourceNotFoundError,
    ShaderAsset,
    ShaderStage,
    load_shader,
)
from engine.core.resources.cookers import (
    ShaderCookSettings,
    ShaderVariant,
    cook_shader,
    scan_features,
    variant_key,
)
from engine.core.resources.importers import ShaderImporter, default_importers

ASSETS = Path(__file__).resolve().parents[2] / "assets"

COMMON = """\
#define PI 3.14159
float saturate(float x) { return clamp(x, 0.0, 1.0); }
"""

FRAGMENT = """\
#version 330 core
#include "common.glsl"
#ifndef QUALITY
#define QUALITY 2
#endif
// комментарий с #ifdef IGNORED
#ifdef ALPHA_MASK
uniform float u_cutoff;
#endif
#if QUALITY > 1 && defined(EXTRA)
float extra() { return PI; }
#endif
out vec4 color;
void main() { color = vec4(saturate(PI)); }
"""


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "shaders").mkdir(parents=True)
    (root / "shaders" / "common.glsl").write_text(COMMON, encoding="utf-8")
    (root / "shaders" / "a.frag").write_text(FRAGMENT, encoding="utf-8")
    return root


@pytest.fixture()
def db(project: Path):
    with AssetDatabase(project) as database:
        database.scan()
        yield database


class TestScan:
    def test_features_and_defaults(self) -> None:
        features, defaults = scan_features(FRAGMENT)
        assert features == ("ALPHA_MASK", "EXTRA", "QUALITY")
        assert defaults == {"QUALITY": "2"}

    def test_constants_are_not_features(self) -> None:
        source = "#define LIGHT 1\n#if LIGHT == 1\nint a;\n#endif\n"
        assert scan_features(source) == ((), {})

    def test_comments_do_not_count(self) -> None:
        source = "// #ifdef HIDDEN\n/* #ifdef HIDDEN2\n#endif */\nvoid main(){}\n"
        assert scan_features(source)[0] == ()

    def test_include_guards_are_not_features(self) -> None:
        source = "#ifndef LIB_H\n#define LIB_H\n#ifdef FAST\n#endif\n#endif\n"
        assert scan_features(source)[0] == ("FAST",)

    def test_default_value_is_recorded(self) -> None:
        source = "#ifndef MAX_LIGHTS\n#define MAX_LIGHTS 8\n#endif\n"
        assert scan_features(source) == (("MAX_LIGHTS",), {"MAX_LIGHTS": "8"})

    def test_driver_macros_ignored(self) -> None:
        assert scan_features("#ifdef GL_ES\n#endif\n")[0] == ()


class TestVariantKey:
    def test_order_independent(self) -> None:
        assert variant_key({"A": 1, "B": 2}) == variant_key({"B": 2, "A": 1})

    def test_base_and_values(self) -> None:
        assert variant_key({}) == variant_key(None) == "base"
        assert variant_key({"A": None}) != variant_key({"A": 1})
        assert variant_key({"A": None}) == "A"


class TestCook:
    def test_includes_resolved(self, project: Path) -> None:
        cooked = cook_shader(project / "shaders" / "a.frag")
        assert '#include' not in cooked.source
        assert "saturate" in cooked.source
        assert cooked.stage is ShaderStage.FRAGMENT
        assert cooked.glsl_version == 330
        assert cooked.has_main
        assert [Path(p).name for p in cooked.includes] == ["common.glsl"]
        assert len(cooked.digest) == 16

    def test_digest_is_stable_and_sensitive(self, project: Path) -> None:
        path = project / "shaders" / "a.frag"
        first = cook_shader(path).digest
        assert cook_shader(path).digest == first
        path.write_text(FRAGMENT + "// x\n", encoding="utf-8")
        assert cook_shader(path).digest != first

    def test_library_has_no_main(self, project: Path) -> None:
        cooked = cook_shader(project / "shaders" / "common.glsl")
        assert not cooked.has_main and cooked.glsl_version is None

    def test_strip_comments(self, project: Path) -> None:
        plain = cook_shader(project / "shaders" / "a.frag").source
        small = cook_shader(
            project / "shaders" / "a.frag", ShaderCookSettings(strip_comments=True)
        ).source
        assert len(small) < len(plain) and "//" not in small
        assert "\n\n" not in small

    def test_unbalanced_endif(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.frag"
        path.write_text("#version 330 core\n#endif\n", encoding="utf-8")
        with pytest.raises(AssetImportError, match="endif"):
            cook_shader(path)

    def test_unclosed_if(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.frag"
        path.write_text("#ifdef A\nvoid main(){}\n", encoding="utf-8")
        with pytest.raises(AssetImportError, match="не закрыто"):
            cook_shader(path)

    def test_missing_include(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.frag"
        path.write_text('#include "nope.glsl"\n', encoding="utf-8")
        with pytest.raises(AssetImportError, match=r"nope.glsl"):
            cook_shader(path)

    def test_unknown_variant_keyword(self, project: Path) -> None:
        settings = ShaderCookSettings(
            variants=(ShaderVariant("typo", {"ALPHA_MSK": 1}),)
        )
        with pytest.raises(AssetImportError, match="ALPHA_MSK"):
            cook_shader(project / "shaders" / "a.frag", settings)

    def test_engine_pbr_shader(self) -> None:
        cooked = cook_shader(ASSETS / "shaders" / "pbr.frag")
        assert {"ALPHA_MASK", "DYNAMIC_LIGHTING", "MAX_LIGHTS"} <= set(cooked.features)
        assert not {"MYENGINE_COMMON", "MYENGINE_LIGHTING"} & set(cooked.features)
        assert cooked.defaults["DYNAMIC_LIGHTING"] == "1"
        assert {Path(p).name for p in cooked.includes} >= {
            "common.glsl",
            "lighting.frag",
        }


class TestImport:
    def test_registered_by_default(self) -> None:
        assert any(isinstance(item, ShaderImporter) for item in default_importers())

    def test_artifact_and_info(self, db: AssetDatabase, project: Path) -> None:
        guid = db.guid_of(project / "shaders" / "a.frag")
        db.reimport(guid, wait=True)
        handle = db.resolve(guid)
        assert handle.imported, handle.error
        artifact = handle.artifact("shader")
        assert artifact is not None
        assert artifact.info["stage"] == "fragment"
        assert artifact.info["features"] == ["ALPHA_MASK", "EXTRA", "QUALITY"]
        assert artifact.info["includes"] == ["common.glsl"]
        assert artifact.info["library"] is False
        text = handle.artifact_path(artifact).read_text(encoding="utf-8")
        assert "#include" not in text

    def test_library_is_marked(self, db: AssetDatabase, project: Path) -> None:
        guid = db.guid_of(project / "shaders" / "common.glsl")
        db.reimport(guid, wait=True)
        assert db.resolve(guid).artifact("shader").info["library"] is True

    def test_include_change_makes_dirty(
        self, db: AssetDatabase, project: Path
    ) -> None:
        guid = db.guid_of(project / "shaders" / "a.frag")
        db.reimport(guid, wait=True)
        assert guid not in db.dirty()
        (project / "shaders" / "common.glsl").write_text(
            COMMON + "// правка\n", encoding="utf-8"
        )
        assert guid in db.dirty()

    def test_reimport_is_idempotent(self, db: AssetDatabase, project: Path) -> None:
        guid = db.guid_of(project / "shaders" / "a.frag")
        db.reimport(guid, wait=True)
        first = db.resolve(guid).artifact_path("shader")
        db.reimport(guid, wait=True)
        handle = db.resolve(guid)
        assert [a.kind for a in handle.artifacts] == ["shader"]
        assert sorted(p.name for p in first.parent.iterdir()) == [
            "artifacts.json",
            "shader.glsl",
        ]

    def test_variants_setting(self, db: AssetDatabase, project: Path) -> None:
        guid = db.guid_of(project / "shaders" / "a.frag")
        db.set_settings(
            guid,
            {"variants": [{"name": "masked", "defines": {"ALPHA_MASK": 1}}]},
        )
        db.reimport(guid, wait=True)
        variants = db.resolve(guid).artifact("shader").info["variants"]
        assert variants == [
            {"name": "masked", "defines": {"ALPHA_MASK": 1}, "key": "ALPHA_MASK=1"}
        ]

    def test_bad_variant_fails_import(self, db: AssetDatabase, project: Path) -> None:
        guid = db.guid_of(project / "shaders" / "a.frag")
        db.set_settings(guid, {"variants": [{"defines": {"NOPE": 1}}]})
        db.reimport(guid, wait=True)
        handle = db.resolve(guid)
        assert not handle.imported and "NOPE" in (handle.error or "")

    def test_include_dirs_setting(self, db: AssetDatabase, project: Path) -> None:
        lib = project / "lib"
        lib.mkdir()
        (lib / "extra.glsl").write_text("#define EXTRA_CONST 1\n", encoding="utf-8")
        path = project / "shaders" / "b.frag"
        path.write_text('#include "extra.glsl"\nvoid main(){}\n', encoding="utf-8")
        db.scan()
        guid = db.guid_of(path)
        db.set_settings(guid, {"include_dirs": ["../lib"]})
        db.reimport(guid, wait=True)
        assert db.resolve(guid).imported, db.resolve(guid).error

    def test_load_shader(self, db: AssetDatabase, project: Path) -> None:
        guid = db.guid_of(project / "shaders" / "a.frag")
        db.reimport(guid, wait=True)
        asset = load_shader(db, guid)
        assert isinstance(asset, ShaderAsset)
        assert asset.stage is ShaderStage.FRAGMENT
        assert "saturate" in asset.source and "#include" not in asset.source

    def test_load_shader_without_artifact(
        self, db: AssetDatabase, project: Path
    ) -> None:
        guid = db.guid_of(project / "shaders" / "a.frag")
        with pytest.raises(ResourceNotFoundError):
            load_shader(db, guid)
