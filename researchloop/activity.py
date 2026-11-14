"""Append provenance events for research invocations and local tools."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from researchloop import storage

_LOCK = storage.LOCK


def log_path():
    override = os.environ.get("RESEARCHLOOP_LOG")
    return Path(override) if override else storage.RUNS / "activity.jsonl"


def record(kind, text, *, log_file=None, **extra):
    """Append an actual event; structured measurements remain available in details.

    Run writers supply their own root, so temporary/test runs cannot leak into
    the production activity stream. An explicit environment override wins.
    """
    entry = {"schema_version": 2, "id": uuid.uuid4().hex, "at": storage.now(),
             "kind": str(kind), "text": str(text)[:2000],
             "level": "error" if kind == "error" else "info", "status": "completed"}
    for key, value in extra.items():
        entry[key] = _bounded(value)
    path = log_path() if os.environ.get("RESEARCHLOOP_LOG") or log_file is None else Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with _LOCK:
        with path.open("a", encoding="utf-8") as output:
            output.write(line + "\n")
    return entry


def _bounded(value, depth=0):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth > 6:
        return "[嵌套过深]"
    if isinstance(value, dict):
        return {str(k)[:100]: _bounded(v, depth + 1) for k, v in list(value.items())[:80]}
    if isinstance(value, (list, tuple)):
        return [_bounded(v, depth + 1) for v in value[:80]]
    return str(value)[:12000]
