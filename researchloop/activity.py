"""工作台活动日志。拟定计划、加入任务和每次研究事件都追加到同一份记录。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from researchloop import storage

_LOCK = storage.LOCK


def log_path():
    override = os.environ.get("RESEARCHLOOP_LOG")
    return Path(override) if override else storage.ROOT / ".cache" / "activity.jsonl"


def record(kind, text, **extra):
    entry = {"at": storage.now(), "kind": str(kind), "text": str(text)[:800]}
    for key, value in extra.items():
        if value is None or isinstance(value, (dict, list)):
            continue
        entry[key] = value if isinstance(value, (int, float, bool)) else str(value)[:240]
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with _LOCK:
        with path.open("a", encoding="utf-8") as output:
            output.write(line + "\n")
    return entry


def recent(limit=80):
    path = log_path()
    if not path.exists():
        return []
    with _LOCK:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(entries))
