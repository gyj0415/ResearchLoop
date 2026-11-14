"""工作台活动日志。拟定计划、加入任务和每次研究事件都追加到同一份记录。"""

from __future__ import annotations

import json
import os
import hashlib
import heapq
import uuid
from collections import deque
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


def recent(limit=80):
    path = log_path()
    if not path.exists():
        return []
    with _LOCK:
        with path.open(encoding="utf-8") as handle:
            lines = list(deque(handle, maxlen=max(1, min(int(limit), 500))))
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(entries))


def query(*, limit=200, level="", source="", run_id="", text=""):
    """Read current events plus linked legacy events, without inventing history."""
    limit = max(1, min(int(limit), 500))
    entries = []
    matched = 0
    ignored_legacy = 0
    legacy = storage.ROOT / ".cache" / "activity.jsonl"
    paths = [(log_path(), False)]
    if not os.environ.get("RESEARCHLOOP_LOG") and legacy != log_path():
        paths.append((legacy, True))
    needle = text.casefold()
    for path, is_legacy in paths:
        if not path.exists():
            continue
        with _LOCK, path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if is_legacy:
                    identifier = item.get("run_id", "")
                    if identifier:
                        try:
                            storage.run_path(identifier)
                        except ValueError:
                            ignored_legacy += 1
                            continue
                    item = {**item, "id": "legacy-" + hashlib.sha256(line.encode()).hexdigest()[:20],
                            "schema_version": 1, "level": "info", "status": "legacy",
                            "legacy_note": "旧版事件，未记录的耗时、usage和来源不会补造"}
                if ((level and item.get("level") != level) or (source and item.get("source") != source)
                        or (run_id and item.get("run_id") != run_id)
                        or (needle and needle not in json.dumps(item, ensure_ascii=False).casefold())):
                    continue
                matched += 1
                heapq.heappush(entries, (str(item.get("at", "")), matched, item))
                if len(entries) > limit:
                    heapq.heappop(entries)
    return {"entries": [row[2] for row in sorted(entries, reverse=True)], "matched": matched, "limit": limit,
            "ignored_legacy": ignored_legacy, "log_file": "runs/activity.jsonl"}


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
