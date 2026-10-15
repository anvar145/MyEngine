"""Секция «Пост-процесс» для инспектора.

Пост-обработка — свойство сцены (:class:`~engine.rendering.post.PostSettings`
в ``SceneDocument.post``). Пока секции нет в файле, действуют умолчания профиля
качества; первая правка создаёт её. Если запущен Play Mode, изменение
отправляется плееру командой ``set_setting``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from editor.widgets import FormSection
from engine.rendering.dynamic_resolution import read_hdr_hint
from engine.rendering.post import TONEMAP_MODES, PostSettings
from engine.rendering.settings import QualityTier

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from editor.context import EditorContext

__all__ = ["PostSection"]


class PostSection(QWidget):
    """Tonemap, экспозиция и FXAA активной сцены.

    Args:
        context: Контекст редактора.
        parent: Родительский виджет.

    Signals:
        changed: Параметр изменён (нужно перерисовать кадр).
    """

    changed = Signal()

    def __init__(self, context: EditorContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.context = context
        document = context.active_document
        current = document.post or PostSettings.for_quality(QualityTier.MEDIUM)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        section = FormSection(self.tr("post.title"), self)
        # Без ``hdr`` в проекте и без секции ``post`` в сцене эффекта нет:
        # первая правка создаёт секцию и тем самым включает HDR.
        self.hdr_banner = QLabel(self.tr("post.hdr_off"), self)
        self.hdr_banner.setWordWrap(True)
        self.hdr_banner.setStyleSheet(
            "background: #5c4a12; color: #ffd866; padding: 4px; border-radius: 3px;"
        )
        hdr_hint = read_hdr_hint(context.project_root)
        self.hdr_banner.setVisible(document.post is None and hdr_hint is None)
        section.form.addRow(self.hdr_banner)
        # Явный ``hdr: false`` сцена не перекрывает: объясняем, почему эффекта нет.
        self.hdr_user_off_banner = QLabel(self.tr("post.hdr_user_off"), self)
        self.hdr_user_off_banner.setWordWrap(True)
        self.hdr_user_off_banner.setStyleSheet(self.hdr_banner.styleSheet())
        self.hdr_user_off_banner.setVisible(hdr_hint is False)
        section.form.addRow(self.hdr_user_off_banner)
        self.tonemap_combo = section.add_combo(
            self.tr("post.tonemap"),
            [(self.tr(f"post.tonemap.{mode}"), mode) for mode in TONEMAP_MODES],
            current.tonemap,
            lambda value: self._apply("tonemap", value),
        )
        self.exposure_spin = section.add_spin(
            self.tr("post.exposure"),
            current.exposure,
            lambda value: self._apply("exposure", value),
            minimum=0.05,
            maximum=8.0,
        )
        self.fxaa_box = section.add_check(
            self.tr("post.fxaa"),
            current.fxaa,
            lambda value: self._apply("fxaa", value),
        )
        self.bloom_box = section.add_check(
            self.tr("post.bloom"),
            current.bloom,
            lambda value: self._apply("bloom", value),
        )
        self.bloom_threshold_spin = section.add_spin(
            self.tr("post.bloom_threshold"),
            current.bloom_threshold,
            lambda value: self._apply("bloom_threshold", value),
            minimum=0.0,
            maximum=16.0,
        )
        self.bloom_intensity_spin = section.add_spin(
            self.tr("post.bloom_intensity"),
            current.bloom_intensity,
            lambda value: self._apply("bloom_intensity", value),
            minimum=0.0,
            maximum=8.0,
        )
        self.bloom_levels_spin = section.add_int(
            self.tr("post.bloom_levels"),
            current.bloom_levels,
            lambda value: self._apply("bloom_levels", value),
            minimum=1,
            maximum=5,
        )
        self.vignette_box = section.add_check(
            self.tr("post.vignette"),
            current.vignette,
            lambda value: self._apply("vignette", value),
        )
        self.vignette_inner_spin = section.add_spin(
            self.tr("post.vignette_inner"),
            current.vignette_inner,
            lambda value: self._apply("vignette_inner", value),
            minimum=0.0,
            maximum=1.0,
        )
        self.vignette_outer_spin = section.add_spin(
            self.tr("post.vignette_outer"),
            current.vignette_outer,
            lambda value: self._apply("vignette_outer", value),
            minimum=0.0,
            maximum=1.5,
        )
        self.vignette_intensity_spin = section.add_spin(
            self.tr("post.vignette_intensity"),
            current.vignette_intensity,
            lambda value: self._apply("vignette_intensity", value),
            minimum=0.0,
            maximum=1.0,
        )
        self._build_lut_field(section, current)
        self.lut_intensity_spin = section.add_spin(
            self.tr("post.lut_intensity"),
            current.lut_intensity,
            lambda value: self._apply("lut_intensity", value),
            minimum=0.0,
            maximum=1.0,
        )
        layout.addWidget(section)

    def _build_lut_field(self, section: FormSection, current: PostSettings) -> None:
        """Поле «LUT»: drop PNG из панели ассетов, кнопка очистки."""
        from editor.panels.script_fields import ReferenceField, make_asset_ref

        db = self.context.asset_db
        resolver = (lambda ref: ref.resolve(db)) if db is not None else None
        field = ReferenceField("asset", self)
        field.setToolTip(self.tr("post.lut.tooltip"))
        field.set_asset(current.lut, resolver)  # type: ignore[arg-type]

        def dropped(path: str) -> None:
            if not path.lower().endswith(".png"):
                return
            ref = make_asset_ref(db, path)
            self._apply("lut", ref)
            field.set_asset(ref, resolver)

        def cleared() -> None:
            self._apply("lut", None)
            field.set_asset(None, resolver)

        field.dropped.connect(dropped)
        field.cleared.connect(cleared)
        section.form.addRow(self.tr("post.lut"), field)
        self.lut_field = field

    def tr(self, key: str, **kwargs: object) -> str:  # noqa: A003 - привычное имя Qt
        """Перевести строку через переводчик редактора."""
        return self.context.tr(key, **kwargs)

    def _apply(self, name: str, value: Any) -> None:
        """Записать параметр в секцию сцены и в запущенный плеер."""
        self.context.set_document_setting("post", name, value)
        self.hdr_banner.setVisible(False)
        play_mode = getattr(self.context, "play_mode", None)
        if play_mode is not None and play_mode.is_active:
            sent = value.to_dict() if hasattr(value, "to_dict") else value
            play_mode.send_setting(f"post.{name}", sent)
        self.changed.emit()
