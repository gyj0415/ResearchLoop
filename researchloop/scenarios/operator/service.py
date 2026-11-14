"""把一次算子研究写成 ResearchLoop 的运行记录。

检索循环仍走 `researchloop.runner`。这里只负责算子任务：证据、假设、
实验和裁决分文件落下，超预算的结果不生成可发布经验。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from researchloop import activity, storage

TASK_DIR = Path(__file__).resolve().parent / "tasks"
TASKS = {
    "win_bias_mean": "win_bias_mean.py",
    "lose_gelu_pool": "lose_gelu_pool.py",
}


def user_task_dir():
    override = os.environ.get("OPERATOR_USER_TASKS")
    return Path(override) if override else storage.ROOT / "runs" / "user-tasks"


def task_search_path():
    return os.pathsep.join(str(path) for path in (TASK_DIR, user_task_dir()) if path.is_dir() or path == TASK_DIR)


def resolve_task_file(task_name):
    if task_name in TASKS:
        return TASK_DIR / TASKS[task_name]
    candidate = user_task_dir() / f"{task_name}.py"
    if candidate.is_file():
        return candidate
    raise ValueError("没有这个算子任务。先在对话里加入你的 reference，或从例子开始。")


def is_operator_task(task_name):
    if task_name in TASKS:
        return True
    return (user_task_dir() / f"{task_name}.py").is_file()
PUBLISHABLE = {"target_met", "blocked_by_transcendental", "blocked_by_bandwidth",
               "blocked_by_reduction", "blocked_by_overhead"}


def execute(task_name, *, seconds_limit=120, max_experiments=4, runner=None,
            runs_root=None):
    """同步跑完一个演示任务。测试可以注入 `runner(task_name, limit)`。"""
    run, run_dir = begin(task_name, seconds_limit=seconds_limit,
                         max_experiments=max_experiments, runs_root=runs_root)
    return complete(run, run_dir, runner=runner)


def begin(task_name, *, seconds_limit=120, max_experiments=4, runs_root=None, brief=None):
    """先落下「运行中」的记录，供页面立刻看到这次研究。"""
    resolve_task_file(task_name)
    if not (1 <= int(max_experiments) <= 8 and 10 <= int(seconds_limit) <= 3600):
        raise ValueError("预算参数超出允许范围")
    runs_root = Path(runs_root or storage.RUNS)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "id": run_id, "version": "0.2.0", "domain": "operator",
        "created_at": storage.now(), "status": "running", "task": task_name,
        "mode": "operator", "model": None, "cloud_enabled": False,
        "seed": 0, "max_experiments": int(max_experiments),
        "seconds_limit": int(seconds_limit), "experiments": [],
        "operator_experiments": [], "events": [],
        "source_hash": storage.source_hash(),
        "brief": _brief(brief),
    }
    _save(run_dir, run)
    _event(run_dir, run, "setup", f"开始加载算子任务 {task_name}", status="running",
           task_file=str(resolve_task_file(task_name)), max_experiments=int(max_experiments))
    return run, run_dir


def complete(run, run_dir, runner=None):
    """把研究做完并写上裁决。调用方保证 `run` 就是 `begin` 返回的那一份。"""
    run_dir = Path(run_dir)
    task_name = run["task"]
    started = time.perf_counter()
    previous = os.environ.get("RESEARCH_TASK_DIRS")
    os.environ["RESEARCH_TASK_DIRS"] = task_search_path()
    try:
        task_run = _run(task_name, int(run["max_experiments"]), runner,
                        on_event=lambda kind, text, **data: _event(run_dir, run, kind, text, **data))
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        run["finished_at"] = storage.now()
        _event(run_dir, run, "error", run["error"])
        _save(run_dir, run)
        raise
    finally:
        if previous is None:
            os.environ.pop("RESEARCH_TASK_DIRS", None)
        else:
            os.environ["RESEARCH_TASK_DIRS"] = previous
    elapsed = time.perf_counter() - started
    payload = task_run.to_json()
    verdict = payload.get("verdict") or {}
    _write(run_dir / "evidence.json", payload.get("evidence") or {})
    _write(run_dir / "hypothesis.json", {
        "skip_reason": payload.get("skip_reason") or "",
        "notes": payload.get("notes") or [],
        "experiments": [{"exp_id": item["exp_id"], "rule_chain": item["rule_chain"],
                         "family": item["family"]} for item in payload["experiments"]],
    })
    _write(run_dir / "experiment.json", payload["experiments"])
    _write(run_dir / "verdict.json", verdict)
    run["operator_experiments"] = payload["experiments"]
    run["verdict"] = verdict
    run["verdict_hint"] = payload.get("verdict_hint")
    run["skip_reason"] = payload.get("skip_reason") or ""
    run["budget"] = {"elapsed": round(elapsed, 3), "seconds_limit": int(run["seconds_limit"]),
                     "experiments": len(payload["experiments"]), "token_limit": 0, "used": 0}
    run["finished_at"] = storage.now()
    over_time = elapsed > int(run["seconds_limit"]) or float(payload.get("wallclock_s") or 0) > int(run["seconds_limit"])
    if over_time:
        run["status"] = "budget_stopped"
        run["stop_reason"] = "研究时间预算已耗尽"
        run.pop("memory_candidate", None)
    else:
        run["status"] = "completed"
        if verdict.get("reason_code") in PUBLISHABLE:
            run["memory_candidate"] = _candidate(task_name, verdict, payload)
    _event(run_dir, run, "completed", f"算子研究结束：{verdict.get('reason_code') or '未形成裁决'}",
           status=run["status"], level="warning" if over_time else "info",
           elapsed_ms=round(elapsed * 1000), verdict=verdict,
           experiment_count=len(payload["experiments"]), reused_case=payload.get("reused_case", False),
           artifacts=["evidence.json", "hypothesis.json", "experiment.json", "verdict.json"])
    _save(run_dir, run)
    return run


def _run(task_name, max_experiments, runner, on_event=None):
    if runner is not None:
        return runner(task_name, max_experiments)
    from researchloop.scenarios.operator.intake import load
    from researchloop.scenarios.operator.pipeline import RunConfig, run_task

    spec = load(resolve_task_file(task_name))
    return run_task(spec, RunConfig(budget=max_experiments, max_candidates=max_experiments),
                    memory_path=storage.RUNS / "operator-cases.jsonl", on_event=on_event)


def _brief(value):
    if not isinstance(value, dict):
        return None
    kept = {}
    for key in ("domain", "title", "question", "success", "falsify_if", "task_name", "source"):
        if isinstance(value.get(key), str):
            kept[key] = value[key][:500]
    for key in ("steps", "out_of_scope"):
        items = value.get(key)
        if isinstance(items, list):
            kept[key] = [str(item)[:200] for item in items[:6]]
    return kept or None


def _candidate(task_name, verdict, payload):
    chains = [item["rule_chain"] for item in payload["experiments"] if item.get("rule_chain")]
    return {
        "domain": "operator",
        "task": task_name,
        "decision": verdict.get("decision"),
        "reason_code": verdict.get("reason_code"),
        "best_speedup": verdict.get("best_speedup"),
        "rule_chains": chains,
        "lesson": verdict.get("detail") or verdict.get("next_action") or "",
        "limitations": "加速比是这次测量的历史值，下一题必须重测。结构不符时拒绝加载。",
    }


def _event(run_dir, run, kind, text, **data):
    item = {"at": storage.now(), "type": kind, "text": text, **data}
    run["events"].append(item)
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(item, ensure_ascii=False) + "\n")
    activity.record(kind, text, log_file=run_dir.parent / "activity.jsonl",
                    run_id=run.get("id", ""), task=run.get("task", ""), source="operator", actor="tool",
                    status=data.get("status", "failed" if kind == "error" else "completed"),
                    level=data.get("level", "error" if kind == "error" else "info"),
                    elapsed_ms=data.get("elapsed_ms"), data=data)
    run["updated_at"] = storage.now()
    _save(run_dir, run)


def _save(run_dir, run):
    storage.write_json(run_dir / "run.json", run)


def _write(path, value):
    storage.write_json(path, value)
