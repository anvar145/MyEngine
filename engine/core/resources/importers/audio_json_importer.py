"""Импортёры JSON-ассетов звука: ``*.cue.json`` и ``*.music.json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine.core.resources.errors import AssetImportError
from engine.core.resources.importer import Artifact, AssetPreview, ImportContext

__all__ = ["CueImporter", "MusicImporter"]


class _JsonAudioImporter:
    """Проверить файл звукового ассета и скопировать его в ``Library``."""

    name = ""
    extensions: tuple[str, ...] = ()
    version = 1
    artifact_name = ""
    label = ""

    def default_settings(self) -> dict[str, Any]:
        """Настроек импорта нет."""
        return {}

    def _validate(self, data: dict[str, Any]) -> int:
        """Проверить содержимое; вернуть число элементов."""
        raise NotImplementedError

    def _load(self, path: Path) -> tuple[bytes, int]:
        try:
            raw = path.read_bytes()
            data = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise AssetImportError(
                f"{self.label} '{path.name}': не прочитан: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise AssetImportError(
                f"{self.label} '{path.name}': некорректный JSON: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise AssetImportError(
                f"{self.label} '{path.name}': корень должен быть объектом"
            )
        from engine.audio.asset_json import AudioAssetError

        try:
            count = self._validate(data)
        except AudioAssetError as exc:
            raise AssetImportError(f"{self.label} '{path.name}': {exc}") from exc
        return raw, count

    def inspect(self, path: Path) -> AssetPreview:
        """Краткая сводка: число элементов."""
        _, count = self._load(path)
        return AssetPreview(self.name, {"count": count})

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        """Проверить файл и записать копию как артефакт."""
        raw, count = self._load(ctx.source)
        (ctx.output_dir / self.artifact_name).write_bytes(raw)
        return [ctx.artifact(self.name, self.artifact_name, count=count)]


class CueImporter(_JsonAudioImporter):
    """``.cue.json`` -> артефакт ``sound_cue`` (``cue.json``)."""

    name = "sound_cue"
    extensions = (".cue.json",)
    artifact_name = "cue.json"
    label = "SoundCue"

    def _validate(self, data: dict[str, Any]) -> int:
        from engine.audio.cue import parse_cue

        return len(parse_cue(data).clips)


class MusicImporter(_JsonAudioImporter):
    """``.music.json`` -> артефакт ``music`` (``music.json``)."""

    name = "music"
    extensions = (".music.json",)
    artifact_name = "music.json"
    label = "Плейлист"

    def _validate(self, data: dict[str, Any]) -> int:
        from engine.audio.music import parse_playlist

        return len(parse_playlist(data).tracks)
