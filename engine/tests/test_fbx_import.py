"""Тесты импорта FBX (нужен extra ``[fbx]``, иначе часть тестов пропускается)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from engine.core.resources import AssetDatabase, AssetImportError
from engine.core.resources.cookers import read_cooked_mesh
from engine.core.resources.importers import FbxImporter, default_importers
from engine.core.resources.importers.fbx_importer import (
    triangulate_faces,
    weld_corners,
)

FIXTURE = Path(__file__).parent / "fixtures" / "cube.fbx"


class TestHelpers:
    def test_triangulate_quad_and_triangle(self) -> None:
        corners, faces = triangulate_faces(np.array([0, 4]), np.array([4, 3]))
        assert corners.tolist() == [[0, 1, 2], [0, 2, 3], [4, 5, 6]]
        assert faces.tolist() == [0, 0, 1]

    def test_triangulate_skips_degenerate(self) -> None:
        corners, faces = triangulate_faces(np.array([0, 2]), np.array([2, 5]))
        assert len(corners) == 3 and set(faces.tolist()) == {1}

    def test_weld_merges_equal_attributes(self) -> None:
        corners = np.array([[0, 1, 2], [3, 4, 5]])
        position = np.array([0, 1, 2, 0, 2, 3])
        uv = np.array([0, 0, 0, 0, 0, 0])
        first, triangles = weld_corners(corners, position, uv)
        assert len(first) == 4  # позиции 0 и 2 повторяются
        assert triangles[0, 0] == triangles[1, 0]

    def test_weld_splits_on_different_attribute(self) -> None:
        corners = np.array([[0, 1, 2], [3, 4, 5]])
        position = np.array([0, 1, 2, 0, 1, 2])
        normal = np.array([0, 0, 0, 1, 1, 1])
        first, _ = weld_corners(corners, position, normal)
        assert len(first) == 6


class TestRegistration:
    def test_registered_by_default(self) -> None:
        assert any(isinstance(item, FbxImporter) for item in default_importers())

    def test_inspect_ascii(self) -> None:
        preview = FbxImporter().inspect(FIXTURE)
        assert preview.kind == "model" and preview.info["format"] == "ascii"

    def test_inspect_binary_header(self, tmp_path: Path) -> None:
        path = tmp_path / "b.fbx"
        path.write_bytes(b"Kaydara FBX Binary  \x00\x1a\x00" + (7400).to_bytes(4, "little"))
        assert FbxImporter().inspect(path).info == {"format": "binary", "version": 7400}

    def test_inspect_rejects_other_files(self, tmp_path: Path) -> None:
        path = tmp_path / "x.fbx"
        path.write_bytes(b"not an fbx at all")
        with pytest.raises(AssetImportError, match="не FBX"):
            FbxImporter().inspect(path)


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    shutil.copyfile(FIXTURE, root / "cube.fbx")
    return root


@pytest.fixture()
def db(project: Path):
    with AssetDatabase(project) as database:
        database.scan()
        yield database


class TestImport:
    @pytest.fixture(autouse=True)
    def _need_ufbx(self) -> None:
        pytest.importorskip("ufbx")

    def test_cube_is_cooked(self, db: AssetDatabase, project: Path) -> None:
        guid = db.guid_of(project / "cube.fbx")
        db.reimport(guid, wait=True)
        handle = db.resolve(guid)
        assert handle.imported, handle.error
        artifact = handle.artifact("mesh")
        assert artifact is not None
        info = artifact.info
        assert info["node"] == "Cube"
        assert info["triangles"] == 12  # 6 четырёхугольников
        assert info["vertices"] == 24  # нормали и UV разрезают углы куба
        assert info["has_normals"] and info["has_uvs"]
        mesh = read_cooked_mesh(handle.artifact_path(artifact).read_bytes())
        assert mesh.lods[0].indices.size == 36

    def test_winding_matches_normals_and_faces_outward(
        self, db: AssetDatabase, project: Path
    ) -> None:
        guid = db.guid_of(project / "cube.fbx")
        db.reimport(guid, wait=True)
        handle = db.resolve(guid)
        mesh = read_cooked_mesh(handle.artifact_path(handle.artifact("mesh")).read_bytes())
        positions, normals = mesh.positions(), mesh.normals()
        triangles = mesh.lods[0].indices.reshape(-1, 3)
        a, b, c = (positions[triangles[:, k]] for k in range(3))
        geometric = np.cross(b - a, c - a)
        stored = normals[triangles].mean(axis=1)
        outward = (a + b + c) / 3 - mesh.center
        assert (np.einsum("ij,ij->i", geometric, stored) > 0).all()
        assert (np.einsum("ij,ij->i", geometric, outward) > 0).all()

    def test_world_transform_and_units_are_baked(
        self, db: AssetDatabase, project: Path
    ) -> None:
        # В файле куб 1x1x1 и сдвиг 100 по Y при UnitScaleFactor=1 (сантиметры):
        # в метрах это куб 1 см на высоте 1 м.
        guid = db.guid_of(project / "cube.fbx")
        db.reimport(guid, wait=True)
        info = db.resolve(guid).artifact("mesh").info
        assert info["bounds_min"] == pytest.approx([-0.005, 0.995, -0.005], abs=1e-5)
        assert info["bounds_max"] == pytest.approx([0.005, 1.005, 0.005], abs=1e-5)

    def test_scale_setting(self, db: AssetDatabase, project: Path) -> None:
        guid = db.guid_of(project / "cube.fbx")
        db.set_settings(guid, {"scale": 2.0})
        db.reimport(guid, wait=True)
        info = db.resolve(guid).artifact("mesh").info
        assert info["bounds_max"] == pytest.approx([0.01, 2.01, 0.01], abs=1e-5)

    def test_material_is_extracted(self, db: AssetDatabase, project: Path) -> None:
        guid = db.guid_of(project / "cube.fbx")
        db.reimport(guid, wait=True)
        handle = db.resolve(guid)
        mesh = handle.artifact("mesh")
        assert mesh.info["material"]["name"] == "Red"
        assert mesh.info["material"]["base_color"][:3] == pytest.approx(
            [0.8, 0.1, 0.1], abs=1e-4
        )
        material = handle.artifact("material")
        assert material is not None and mesh.info["material_artifact"] == material.name

    def test_materials_can_be_disabled(self, db: AssetDatabase, project: Path) -> None:
        guid = db.guid_of(project / "cube.fbx")
        db.set_settings(guid, {"extract_materials": False})
        db.reimport(guid, wait=True)
        handle = db.resolve(guid)
        assert handle.artifact("material") is None
        assert handle.artifact("mesh").info["material_artifact"] is None

    def test_reimport_is_idempotent(self, db: AssetDatabase, project: Path) -> None:
        guid = db.guid_of(project / "cube.fbx")
        db.reimport(guid, wait=True)
        first = db.resolve(guid).artifact_path("mesh")
        db.reimport(guid, wait=True)
        names = sorted(p.name for p in first.parent.iterdir())
        assert names == ["artifacts.json", "material_000.json", "mesh_000.bin"]
        assert guid not in db.dirty()

    def test_broken_file_fails_import(self, db: AssetDatabase, project: Path) -> None:
        (project / "cube.fbx").write_text("; FBX 7.4.0\nObjects: {\n", encoding="utf-8")
        guid = db.guid_of(project / "cube.fbx")
        db.reimport(guid, wait=True)
        handle = db.resolve(guid)
        assert not handle.imported and handle.error

    def test_missing_ufbx_message(
        self, monkeypatch: pytest.MonkeyPatch, db: AssetDatabase, project: Path
    ) -> None:
        monkeypatch.setitem(__import__("sys").modules, "ufbx", None)
        guid = db.guid_of(project / "cube.fbx")
        db.reimport(guid, wait=True)
        assert "[fbx]" in (db.resolve(guid).error or "")


class TestIsolation:
    """Разбор FBX в дочернем процессе: краш не роняет вызывающего."""

    def _context(self, tmp_path: Path):
        from engine.core.resources.importer import ImportContext

        source = tmp_path / "cube.fbx"
        shutil.copyfile(FIXTURE, source)
        out = tmp_path / "out"
        out.mkdir()
        return ImportContext("0" * 32, source, FbxImporter().default_settings(), out)

    def test_native_crash_becomes_import_error(self, tmp_path, monkeypatch) -> None:
        import subprocess
        import sys

        from engine.core.resources.importers import fbx_worker

        real_run = subprocess.run

        def crashing(args, **kwargs):
            code = "import ctypes; ctypes.string_at(0)"
            return real_run([sys.executable, "-c", code], **kwargs)

        monkeypatch.setattr(fbx_worker.subprocess, "run", crashing)
        ctx = self._context(tmp_path)
        with pytest.raises(AssetImportError, match="аварийно"):
            fbx_worker.run_fbx_worker(ctx, 30.0)
        assert not (ctx.output_dir / ".fbx_request.json").exists()

    def test_timeout_becomes_import_error(self, tmp_path, monkeypatch) -> None:
        import subprocess
        import sys

        from engine.core.resources.importers import fbx_worker

        real_run = subprocess.run

        def hanging(args, **kwargs):
            return real_run(
                [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
            )

        monkeypatch.setattr(fbx_worker.subprocess, "run", hanging)
        with pytest.raises(AssetImportError, match="не уложился"):
            fbx_worker.run_fbx_worker(self._context(tmp_path), 0.5)

    def test_worker_matches_in_process(self, tmp_path) -> None:
        pytest.importorskip("ufbx")
        from engine.core.resources.importers.fbx_importer import FbxImporter

        results = []
        for isolate in (True, False):
            base = tmp_path / ("iso" if isolate else "local")
            base.mkdir()
            ctx = self._context(base)
            importer = FbxImporter()
            importer.isolated = isolate
            artifacts = importer.import_asset(ctx)
            results.append(
                [
                    (a.kind, a.name, a.info["vertices"])
                    for a in artifacts
                    if a.kind == "mesh"
                ]
            )
        assert results[0] == results[1] and results[0]
