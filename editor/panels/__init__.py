"""Панели редактора: иерархия, инспектор, ассеты, консоль, скрипты."""

from __future__ import annotations

from editor.panels.assets import AssetsPanel
from editor.panels.base import DockPanel, WindowPanel
from editor.panels.camera_section import CameraSection
from editor.panels.code_editor import CodeEditor, LineNumberArea, code_font
from editor.panels.console import ConsolePanel, QtLogHandler
from editor.panels.find_replace import FindReplaceDialog
from editor.panels.hierarchy import HierarchyPanel
from editor.panels.import_manager import ImportPanel
from editor.panels.import_settings import ImportSettingsSection
from editor.panels.input_settings import InputSettingsPanel
from editor.panels.inspector import InspectorPanel
from editor.panels.light_section import LightSection
from editor.panels.script_editor import (
    ScriptEditorPanel,
    ScriptRunner,
    ScriptSourceFilter,
)
from editor.panels.stats import StatsPanel

__all__ = [
    "AssetsPanel",
    "CameraSection",
    "CodeEditor",
    "ConsolePanel",
    "DockPanel",
    "FindReplaceDialog",
    "HierarchyPanel",
    "ImportPanel",
    "InputSettingsPanel",
    "ImportSettingsSection",
    "InspectorPanel",
    "LightSection",
    "LineNumberArea",
    "QtLogHandler",
    "ScriptEditorPanel",
    "ScriptRunner",
    "ScriptSourceFilter",
    "StatsPanel",
    "WindowPanel",
    "code_font",
]
