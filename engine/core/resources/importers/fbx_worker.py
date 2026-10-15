"""Дочерний процесс для разбора FBX: изоляция от нативных крашей ``ufbx``.

Родитель (``run_fbx_worker``) пишет запрос в JSON, запускает
``python -m engine.core.resources.importers.fbx_worker <запрос> <ответ>`` и читает
ответ. Если процесс упал (сегфо, ``0xC0000005``) или завис, импорт заканчивается
обычным ``AssetImportError``, а редактор продолжает работать.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from engine.core.resources.errors import AssetImportError
from engine.core.resources.importer import Artifact, ImportContext

__all__ = ["run_fbx_worker"]

_REQUEST = ".fbx_request.json"
_RESPONSE = ".fbx_response.json"

#: Сколько последних символов stderr дочернего процесса попадает в сообщение.
_STDERR_TAIL = 800


def run_fbx_worker(ctx: ImportContext, timeout: float) -> list[Artifact]:
    """Выполнить импорт FBX в дочернем процессе.

    Args:
        ctx: Контекст импорта (используются ``guid``, ``source``, ``settings``,
            ``output_dir``); зависимости и записи лога переносятся обратно.
        timeout: Предельное время работы процесса, секунды.

    Returns:
        Артефакты, записанные дочерним процессом в ``ctx.output_dir``.

    Raises:
        AssetImportError: Если процесс упал, завис или вернул ошибку.
    """
    request = ctx.output_dir / _REQUEST
    response = ctx.output_dir / _RESPONSE
    payload = {
        "guid": ctx.guid,
        "source": str(ctx.source),
        "settings": dict(ctx.settings),
        "output_dir": str(ctx.output_dir),
    }
    request.write_text(json.dumps(payload), encoding="utf-8")
    # Корень пакета в PYTHONPATH: дочернему процессу не нужна установка проекта.
    package_root = str(Path(__file__).resolve().parents[4])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [package_root, env.get("PYTHONPATH", "")])
    )
    env["PYTHONUTF8"] = "1"
    try:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "engine.core.resources.importers.fbx_worker",
                    str(request),
                    str(response),
                ],
                env=env,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AssetImportError(
                f"'{ctx.source.name}': разбор FBX не уложился в {timeout:g} с"
            ) from exc
        except OSError as exc:
            raise AssetImportError(
                f"'{ctx.source.name}': не удалось запустить разбор FBX: {exc}"
            ) from exc
        result = _read_response(response)
        if result is None:
            tail = completed.stderr.decode("utf-8", "replace").strip()[-_STDERR_TAIL:]
            code = completed.returncode & 0xFFFFFFFF
            raise AssetImportError(
                f"'{ctx.source.name}': процесс разбора FBX завершился аварийно "
                f"(код 0x{code:08X}). {tail}".rstrip()
            )
    finally:
        for leftover in (request, response):
            leftover.unlink(missing_ok=True)
    for level, message in result.get("logs", []):
        ctx.log.log(int(level), "%s", message)
    if "error" in result:
        raise AssetImportError(str(result["error"]))
    for dependency in result.get("dependencies", []):
        ctx.add_dependency(dependency)
    for baked in result.get("baked_inputs", []):
        ctx.add_baked_input(baked)
    return [Artifact.from_dict(item) for item in result["artifacts"]]


def _read_response(path: Path) -> dict[str, Any] | None:
    """Прочитать ответ дочернего процесса; ``None`` — ответа нет или он битый."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class _Collector(logging.Handler):
    """Складывает записи лога, чтобы родитель повторил их в своём логгере."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[tuple[int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelno, record.getMessage()))


def main(argv: list[str]) -> int:
    """Точка входа дочернего процесса: ``<запрос.json> <ответ.json>``."""
    from engine.core.resources.importers.fbx_importer import FbxImporter

    request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    collector = _Collector()
    log = logging.getLogger("engine.core.resources.import.fbx_worker")
    log.addHandler(collector)
    log.propagate = False
    log.setLevel(logging.INFO)
    ctx = ImportContext(
        guid=request["guid"],
        source=Path(request["source"]),
        settings=request["settings"],
        output_dir=Path(request["output_dir"]),
        log=log,
    )
    result: dict[str, Any]
    try:
        artifacts = FbxImporter().import_in_process(ctx)
        result = {
            "artifacts": [artifact.to_dict() for artifact in artifacts],
            "dependencies": [str(path) for path in ctx.dependencies],
            "baked_inputs": [str(path) for path in ctx.baked_inputs],
        }
    except AssetImportError as exc:
        result = {"error": str(exc)}
    result["logs"] = collector.records
    Path(argv[2]).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - запускается как процесс
    sys.exit(main(sys.argv))
