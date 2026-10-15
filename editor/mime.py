"""Типы данных перетаскивания внутри редактора."""

from __future__ import annotations

#: Объект сцены (в данных — его целочисленный id, текстом).
OBJECT_MIME = "application/x-myengine-object"

#: Объекты сцены для перестановки в иерархии (в данных - ``uid`` через ``\n``).
OBJECT_UID_MIME = "application/x-myengine-object-uid"

#: Файл префаба (``*.prefab.json``) из панели ассетов (в данных - путь, UTF-8).
PREFAB_MIME = "application/x-myengine-prefab+json"
