"""Тесты менеджера ресурсов: кэш, счётчики ссылок, выгрузка, события."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.event_bus import EventBus
from engine.core.resources import (
    Asset,
    AssetLoaded,
    AssetLoader,
    AssetType,
    AssetUnloaded,
    LoadContext,
    MissingLoaderError,
    ResourceLoadError,
    ResourceManager,
    ResourceNotFoundError,
)


class FakeAsset(Asset):
    """Ресурс-заглушка с дополнительным полем из параметров импорта."""

    __slots__ = ("payload",)


class FakeLoader(AssetLoader):
    """Загрузчик-заглушка: считает обращения и умеет падать по требованию."""

    extensions = (".fake",)
    asset_type = AssetType.TEXT

    def __init__(self, *, fail: bool = False) -> None:
        self.load_calls = 0
        self.unload_calls = 0
        self.fail = fail

    def load(self, context: LoadContext) -> Asset:
        self.load_calls += 1
        if self.fail:
            raise RuntimeError("сломанный загрузчик")
        asset = FakeAsset(context.path, AssetType.TEXT)
        asset.payload = context.option("payload", "default")
        return asset

    def unload(self, asset: Asset) -> None:
        self.unload_calls += 1
        super().unload(asset)


@pytest.fixture()
def asset_file(tmp_path: Path) -> Path:
    """Временный файл с неизвестным движку расширением ``.fake``."""
    path = tmp_path / "sample.fake"
    path.write_text("payload", encoding="utf-8")
    return path


@pytest.fixture()
def manager(asset_file: Path) -> ResourceManager:
    """Менеджер с единственным загрузчиком ``.fake`` и корнем ``tmp_path``."""
    loader = FakeLoader()
    return ResourceManager(
        roots=[asset_file.parent], loaders=[loader], name="TestResources"
    )


def loader_of(manager: ResourceManager) -> FakeLoader:
    """Достать загрузчик-заглушку из реестра менеджера."""
    loader = next(iter(manager.registry.loaders))
    assert isinstance(loader, FakeLoader)
    return loader


class TestCaching:
    def test_same_file_loaded_once(self, manager: ResourceManager) -> None:
        first = manager.load("sample.fake")
        second = manager.load("sample.fake")
        assert first is second
        assert loader_of(manager).load_calls == 1
        assert manager.stats.hits == 1
        assert manager.stats.misses == 1
        assert manager.stats.loaded_total == 1
        assert manager.stats.hit_ratio == pytest.approx(0.5)

    def test_absolute_and_relative_path_share_cache(
        self, manager: ResourceManager, asset_file: Path
    ) -> None:
        relative = manager.load("sample.fake")
        absolute = manager.load(asset_file)
        assert relative is absolute
        assert loader_of(manager).load_calls == 1

    def test_different_options_create_different_assets(
        self, manager: ResourceManager
    ) -> None:
        first = manager.load("sample.fake", options={"payload": "a"})
        second = manager.load("sample.fake", options={"payload": "b"})
        assert first is not second
        assert loader_of(manager).load_calls == 2
        assert first.payload == "a"

    def test_get_does_not_load(self, manager: ResourceManager) -> None:
        assert manager.get("sample.fake") is None
        loaded = manager.load("sample.fake")
        assert manager.get("sample.fake") is loaded
        assert manager.is_loaded("sample.fake")
        assert "sample.fake" in manager
        assert len(manager) == 1

    def test_preload_many(self, manager: ResourceManager, tmp_path: Path) -> None:
        second = tmp_path / "second.fake"
        second.write_text("x", encoding="utf-8")
        assets = manager.preload("sample.fake", "second.fake")
        assert len(assets) == 2
        assert loader_of(manager).load_calls == 2


class TestReferenceCounting:
    def test_refcount_grows_and_falls(self, manager: ResourceManager) -> None:
        asset = manager.load("sample.fake")
        manager.load("sample.fake")
        assert asset.ref_count == 2
        assert manager.release(asset) == 1
        assert manager.release(asset) == 0
        assert not asset.is_used

    def test_unload_unused_disposes_asset(self, manager: ResourceManager) -> None:
        asset = manager.load("sample.fake")
        manager.release(asset)
        assert manager.unload_unused() == 1
        assert loader_of(manager).unload_calls == 1
        assert len(manager) == 0
        assert manager.stats.unloaded_total == 1

    def test_unload_respects_references(self, manager: ResourceManager) -> None:
        asset = manager.load("sample.fake")
        assert manager.unload("sample.fake") is False
        assert manager.get("sample.fake") is asset
        assert manager.unload("sample.fake", force=True) is True
        assert manager.get("sample.fake") is None

    def test_auto_unload_on_zero_refs(self, asset_file: Path) -> None:
        manager = ResourceManager(
            roots=[asset_file.parent], loaders=[FakeLoader()], auto_unload=True
        )
        asset = manager.load("sample.fake")
        manager.release(asset)
        assert len(manager) == 0

    def test_reload_returns_new_instance(self, manager: ResourceManager) -> None:
        first = manager.load("sample.fake")
        second = manager.reload("sample.fake")
        assert first is not second
        assert loader_of(manager).load_calls == 2
        assert manager.stats.loaded_total == 2

    def test_clear_unloads_everything(self, manager: ResourceManager) -> None:
        manager.load("sample.fake")
        assert manager.clear() == 1
        assert len(manager) == 0
        assert loader_of(manager).unload_calls == 1


class TestResolution:
    def test_missing_file(self, manager: ResourceManager) -> None:
        with pytest.raises(ResourceNotFoundError):
            manager.load("nope.fake")

    def test_no_loader_for_extension(
        self, manager: ResourceManager, tmp_path: Path
    ) -> None:
        path = tmp_path / "model.unknown"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(MissingLoaderError):
            manager.load(path)

    def test_loader_failure_is_wrapped(self, asset_file: Path) -> None:
        manager = ResourceManager(
            roots=[asset_file.parent], loaders=[FakeLoader(fail=True)]
        )
        with pytest.raises(ResourceLoadError):
            manager.load("sample.fake")
        assert manager.stats.failed == 1

    def test_duplicate_extension_registration(self, manager: ResourceManager) -> None:
        with pytest.raises(ValueError):
            manager.register_loader(FakeLoader())

    def test_loader_without_extensions(self, manager: ResourceManager) -> None:
        class Broken(AssetLoader):
            def load(self, context: LoadContext) -> Asset:  # pragma: no cover
                return Asset(context.path, AssetType.UNKNOWN)

        with pytest.raises(ValueError):
            manager.register_loader(Broken())

    def test_supported_extensions(self, manager: ResourceManager) -> None:
        assert ".fake" in manager.supported_extensions
        assert ".obj" not in manager.supported_extensions  # загрузчики заданы явно
        assert ".obj" in ResourceManager().supported_extensions

    def test_add_root(self, asset_file: Path, tmp_path: Path) -> None:
        nested = tmp_path / "nested"
        nested.mkdir()
        manager = ResourceManager(loaders=[FakeLoader()])
        with pytest.raises(ResourceNotFoundError):
            manager.load(asset_file.name)
        manager.add_root(nested)
        manager.add_root(asset_file.parent)
        assert manager.load(asset_file.name) is not None
        assert manager.roots[-1] == asset_file.parent


class TestEvents:
    def test_load_and_unload_events(self, asset_file: Path) -> None:
        bus = EventBus()
        manager = ResourceManager(
            roots=[asset_file.parent], loaders=[FakeLoader()], event_bus=bus
        )
        loaded: list[Asset] = []
        unloaded: list[Path] = []
        bus.subscribe(AssetLoaded, lambda event: loaded.append(event.asset))
        bus.subscribe(AssetUnloaded, lambda event: unloaded.append(event.path))

        asset = manager.load("sample.fake")
        manager.release(asset)
        manager.unload_unused()

        assert loaded == [asset]
        assert unloaded == [asset.path]

    def test_repr_mentions_cache(self, manager: ResourceManager) -> None:
        manager.load("sample.fake")
        assert "cached=1" in repr(manager)
