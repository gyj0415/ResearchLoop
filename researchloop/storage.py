import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "scifact"
RUNS = ROOT / "runs"
MEMORY = ROOT / "memory"
LOCK = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        with LOCK:
            return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows readers/AV can temporarily deny replacement of an open file.
    # Serialize in-process readers and retry transient external sharing locks.
    with LOCK:
        for attempt in range(12):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(min(.005 * 2 ** attempt, .1))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_hash():
    h = hashlib.sha256()
    for path in sorted((ROOT / "researchloop").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


CLOUD_DEFAULTS = {"enabled": False, "base_url": "", "model": "", "api_key": ""}
BUDGET_DEFAULTS = {"tokens_k": 60, "seconds": None, "steps": None}
WEB_DEFAULTS = {"enabled": False, "search_provider": "searxng", "search_endpoint": "",
                "search_api_key": "", "max_results": 5, "fetch_max_chars": 6000}


def settings():
    saved = dict(read_json(ROOT / "config.json", {}) or {})
    cloud = {**CLOUD_DEFAULTS, **(saved.pop("cloud", None) or {})}
    web = {**WEB_DEFAULTS, **(saved.pop("web", None) or {})}
    budgets = {**BUDGET_DEFAULTS, **(saved.pop("budgets", None) or {})}
    saved.pop("limits", None)
    value = {"ollama_url": "http://127.0.0.1:11434", "model": "deepseek-r1:8b",
             "model_source": "local", **saved}
    value["cloud"] = cloud
    value["web"] = web
    value["budgets"] = budgets
    return value


def masked_settings():
    """Public copy: API keys never leave the machine through the state endpoint."""
    value = settings()
    key = str(value["cloud"].get("api_key") or "")
    value["cloud"] = {**value["cloud"], "api_key": "", "api_key_set": bool(key)}
    web_key = str(value["web"].get("search_api_key") or "")
    value["web"] = {**value["web"], "search_api_key": "", "search_api_key_set": bool(web_key)}
    return value


def list_runs():
    runs = []
    for path in RUNS.glob("*/run.json"):
        if path.parent.name.startswith("."):
            continue
        run = read_json(path)
        if run:
            runs.append(run)
    return sorted(runs, key=lambda r: r["created_at"], reverse=True)


def run_path(run_id):
    if not isinstance(run_id, str) or not run_id.replace("-", "").isalnum():
        raise ValueError("无效的运行 ID")
    path = RUNS / run_id / "run.json"
    if not path.exists():
        raise ValueError("运行不存在")
    return path




def archive_run(run_id, archived=True):
    """Archive hides a session from the sidebar without touching its records."""
    path = run_path(run_id)
    run = read_json(path)
    run["archived"] = bool(archived)
    run["archived_at"] = now() if archived else None
    write_json(path, run)
    return run


def delete_run(run_id):
    """Move an archived session folder to runs/.trash instead of destroying evidence."""
    run = read_json(run_path(run_id))
    if not run.get("archived"):
        raise ValueError("只能删除已归档的会话；请先归档")
    trash = RUNS / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    target = trash / run_id
    if target.exists():
        raise ValueError("回收目录已有同名会话，未覆盖")
    os.replace(RUNS / run_id, target)
    return {"id": run_id, "trash": str(target)}
