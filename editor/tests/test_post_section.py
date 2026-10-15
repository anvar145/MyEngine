"""Секция «Пост-процесс»: правка документа, отправка в плеер, round-trip."""

from __future__ import annotations

from editor.panels.post_section import PostSection
from engine import scene_io
from engine.rendering.dynamic_resolution import read_hdr_hint


def test_post_section_shown_for_camera(window) -> None:
    window.create_camera()
    assert isinstance(window.inspector.post_section, PostSection)


def test_post_section_creates_and_edits_post(window) -> None:
    document = window.context.active_document
    assert document.post is None
    section = PostSection(window.context)
    section.exposure_spin.setValue(2.0)
    section.fxaa_box.setChecked(False)
    assert document.post is not None
    assert document.post.exposure == 2.0
    assert document.post.fxaa is False


def test_post_edit_is_sent_to_player(window) -> None:
    sent: list[tuple[str, object]] = []

    class FakePlayMode:
        is_active = True

        def send_setting(self, path, value) -> None:
            sent.append((path, value))

    window.context.play_mode = FakePlayMode()
    section = PostSection(window.context)
    section.exposure_spin.setValue(1.5)
    assert sent == [("post.exposure", 1.5)]


def test_post_round_trip_through_save(window, tmp_path) -> None:
    section = PostSection(window.context)
    section.exposure_spin.setValue(3.0)
    document = window.context.active_document
    path = scene_io.save_scene(
        document.scene, tmp_path / "p.scene.json", None, document.sky, document.post
    )
    _scene, _sky, post = scene_io.load_scene_full(path)
    assert post == document.post


def test_no_post_section_is_not_written(window, tmp_path) -> None:
    document = window.context.active_document
    path = scene_io.save_scene(document.scene, tmp_path / "q.scene.json")
    assert "post" not in scene_io.read_scene_data(path)
    assert scene_io.load_scene_full(path)[2] is None


def test_new_project_enables_hdr(tmp_path) -> None:
    import json

    from editor.project import create_project
    from engine.rendering.dynamic_resolution import read_render_hints

    project = create_project(tmp_path, "Demo")
    data = json.loads((project / "Demo.project.json").read_text("utf-8"))
    assert data["hdr"] is True
    assert read_render_hints(project) == (None, True, True)


def test_post_section_edits_bloom_vignette_lut(window) -> None:
    from engine.core.resources.asset_ref import AssetRef

    section = PostSection(window.context)
    section.bloom_box.setChecked(True)
    section.bloom_levels_spin.setValue(4)
    section.vignette_box.setChecked(True)
    section.vignette_intensity_spin.setValue(0.6)
    section.lut_intensity_spin.setValue(0.5)
    ref = AssetRef(guid="ab" * 16, path="Luts/warm.png")
    section._apply("lut", ref)
    post = window.context.active_document.post
    assert post.bloom and post.bloom_levels == 4
    assert post.vignette and post.vignette_intensity == 0.6
    assert post.lut_intensity == 0.5
    assert post.lut == ref


def test_post_lut_is_sent_to_player_as_dict(window) -> None:
    from engine.core.resources.asset_ref import AssetRef

    sent: list[tuple[str, object]] = []

    class FakePlayMode:
        is_active = True

        def send_setting(self, path, value) -> None:
            sent.append((path, value))

    window.context.play_mode = FakePlayMode()
    section = PostSection(window.context)
    ref = AssetRef(guid="cd" * 16, path="Luts/cold.png")
    section._apply("lut", ref)
    section._apply("lut", None)
    assert sent == [("post.lut", ref.to_dict()), ("post.lut", None)]


def test_post_round_trip_with_lut(window, tmp_path) -> None:
    from engine.core.resources.asset_ref import AssetRef

    section = PostSection(window.context)
    section.bloom_box.setChecked(True)
    section._apply("lut", AssetRef(guid="ef" * 16, path="Luts/x.png"))
    document = window.context.active_document
    path = scene_io.save_scene(
        document.scene, tmp_path / "l.scene.json", None, document.sky, document.post
    )
    assert scene_io.load_scene_full(path)[2] == document.post


def test_banner_shown_without_hdr_and_hidden_after_first_edit(window) -> None:
    document = window.context.active_document
    document.post = None
    section = PostSection(window.context)
    expected = read_hdr_hint(window.context.project_root) is None
    assert section.hdr_banner.isHidden() is not expected
    section._apply("exposure", 1.5)
    assert section.hdr_banner.isHidden()
    assert document.post is not None


def test_banner_hidden_when_scene_has_post(window) -> None:
    from engine.rendering.post import PostSettings

    window.context.active_document.post = PostSettings()
    assert PostSection(window.context).hdr_banner.isHidden()


def test_explicit_hdr_false_shows_user_off_banner(window, monkeypatch) -> None:
    from editor.panels import post_section
    from engine.rendering.post import PostSettings

    monkeypatch.setattr(post_section, "read_hdr_hint", lambda _root: False)
    window.context.active_document.post = PostSettings()
    section = PostSection(window.context)
    assert not section.hdr_user_off_banner.isHidden()
    assert section.hdr_banner.isHidden()
    section._apply("exposure", 1.5)
    assert not section.hdr_user_off_banner.isHidden()  # правка HDR не включает


def test_explicit_hdr_true_has_no_banners(window, monkeypatch) -> None:
    from editor.panels import post_section

    monkeypatch.setattr(post_section, "read_hdr_hint", lambda _root: True)
    window.context.active_document.post = None
    section = PostSection(window.context)
    assert section.hdr_banner.isHidden()
    assert section.hdr_user_off_banner.isHidden()
