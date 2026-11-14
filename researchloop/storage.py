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
LIMIT_DEFAULTS = {"max_tokens": 400000, "max_seconds": 7200, "max_steps": 40}
WEB_DEFAULTS = {"enabled": False, "search_provider": "searxng", "search_endpoint": "",
                "search_api_key": "", "max_results": 5, "fetch_max_chars": 6000}


def settings():
    saved = dict(read_json(ROOT / "config.json", {}) or {})
    cloud = {**CLOUD_DEFAULTS, **(saved.pop("cloud", None) or {})}
    web = {**WEB_DEFAULTS, **(saved.pop("web", None) or {})}
    limits = {**LIMIT_DEFAULTS, **(saved.pop("limits", None) or {})}
    value = {"ollama_url": "http://127.0.0.1:11434", "model": "deepseek-r1:8b",
             "model_source": "local", **saved}
    value["cloud"] = cloud
    value["web"] = web
    value["limits"] = limits
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


def memory_state():
    versions = sorted([read_json(p) for p in MEMORY.glob("v*.json")], key=lambda v: v["version"])
    return {"active": read_json(MEMORY / "active.json", {"version": 0})["version"],
            "versions": versions}


def get_memory(version=None):
    if version is None:
        version = memory_state()["active"]
    if not version:
        return None
    result = read_json(MEMORY / f"v{int(version)}.json")
    if result is None:
        raise ValueError("经验版本不存在")
    return result


def grounded_memory(run, model_commentary=None):
    """Only numeric measurements become experience facts; model prose is commentary."""
    baseline = run["experiments"][0]["result"]["metrics"]
    chosen = next(e for e in run["experiments"] if e["id"] == run["selected_experiment_id"])
    effects = []
    for experiment in run["experiments"][1:]:
        if experiment.get("result_scope") == "screen":
            continue
        effects.append({"experiment_id": experiment["id"],
                       "strategy": experiment["config"]["strategy"], "gate": experiment["config"]["gate"],
                       "normalization": experiment["config"].get("normalization", "legacy"),
                       "explore_delta_from_baseline": {k: experiment["result"]["metrics"][k] - baseline[k] for k in baseline},
                       "sample_count": experiment["result"]["count"]})
    diff = run["guard"]["delta"]
    strategy = chosen["config"]["strategy"]
    lesson = (f"来源任务探索 {chosen['result']['count']} 题，按预先固定指标选中 {strategy} / {chosen['config']['gate']}。"
              f"独立验收 {run['guard']['candidate']['count']} 题，完整证据命中率相对基线 {diff['complete_at_5'] * 100:+.1f} pp，"
              f"证据覆盖率 {diff['coverage_at_5'] * 100:+.1f} pp。下一任务可优先验证该干预，效果必须重新测量。")
    profile = run.get("query_profile", {})
    return {"schema_version": 3, "lesson": lesson, "limitations": "单个小样本科学摘要任务；仅通过来源验收，目标任务仍需检验；旧经验或特征不匹配时拒绝加载。",
            "scope": {"language": "en", "corpus_type": "scientific_abstracts"},
            "source_profile": profile,
            "applicability": {"feature_rates": profile.get("feature_rates", {}), "max_mean_distance": .2,
                              "rule": "Match query-only feature distribution AND require at least one triggered intervention. No target labels."},
            "priority": {"strategy": strategy, "gate": chosen["config"]["gate"], "normalization": chosen["config"].get("normalization", "legacy")},
            "measured_effects": effects,
            "evidence": [{"experiment_id": e["id"], "metrics": e["result"]["metrics"], "sample_count": e["result"].get("count"),
                          "data_scope": e.get("result_scope", "explore"),
                          "improved_count": len(e.get("improved_ids", [])), "regressed_count": len(e.get("regressed_ids", []))} for e in run["experiments"]],
            "guard_delta": diff, "status": "provisional", "origin": "ollama_agent",
            "model_commentary": model_commentary}


def promote(run_id):
    with LOCK:
        run = read_json(run_path(run_id))
        if run["status"] != "completed" or run["mode"] != "agent":
            raise ValueError("只有已完成的 Ollama Agent 研究可以发布经验")
        if not run.get("guard", {}).get("accepted") or not run.get("memory_candidate"):
            raise ValueError("候选未通过独立验收，无法发布")
        existing = [v for v in memory_state()["versions"] if v["source_run"] == run_id]
        if existing:
            return existing[0]
        version = max([v["version"] for v in memory_state()["versions"]] + [0]) + 1
        candidate = grounded_memory(run, run["memory_candidate"].get("model_commentary"))
        value = {**candidate, "version": version, "publisher_source_hash": source_hash(),
                 "parent_version": run["memory_version"], "source_run": run_id,
                 "created_at": now(), "best_config": run["best_config"]}
        write_json(MEMORY / f"v{version}.json", value)
        write_json(MEMORY / "active.json", {"version": version, "changed_at": now()})
        from . import db
        db.memory_add(session_id=None, version=version, scope="retrieval", lesson=value["lesson"],
                      applicability=value.get("applicability"), evidence=value.get("evidence"),
                      status=value.get("status", "provisional"), source_run=run_id,
                      parent_version=value.get("parent_version"), created_at=value.get("created_at"))
        return value


def rollback(version):
    version = int(version)
    get_memory(version)
    write_json(MEMORY / "active.json", {"version": version, "changed_at": now()})
    return memory_state()
