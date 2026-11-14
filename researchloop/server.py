import io
import json
import mimetypes
import threading
import time
import uuid
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .dataset import prepare
from . import __version__
from .ollama import probe
from . import activity
from .brief import draft
from .runner import ResearchService
from .scenarios.operator import service as operator_service
from .scenarios.operator.custom import save_task
from .engine import core as research_core
from .engine import registry as research_registry
from .engine import roles as research_roles
from .engine.tools import TOOLS as RESEARCH_TOOLS
from .storage import DATA, LOCK, ROOT, RUNS, get_memory, list_runs, masked_settings, memory_state, now, promote, read_json, rollback, run_path, settings, write_json
from . import cloudllm
from . import db
from . import webaccess

SERVICE = ResearchService()
RESEARCH_ENGINE = None
WEB = ROOT / "web"


def summary(run):
    keys = ("id", "version", "created_at", "finished_at", "status", "task", "mode", "model", "model_source", "provider", "memory_version", "requested_memory_version", "memory_selection", "seed", "best_config", "budget", "stop_reason", "error", "selected_experiment_id", "evaluation_counts", "source_hash")
    result = {k: run.get(k) for k in keys}
    result["domain"] = run.get("domain", "retrieval")
    if run.get("domain") == "research":
        result.update(question=run["question"], experiment_count=len(run["experiments"]),
                      verified_count=sum(bool(e.get("verification")) for e in run["experiments"]),
                      step_count=len(run["steps"]), baseline_metrics=None, guard=None,
                      has_memory_candidate=bool(run.get("report", {}).get("lesson") and run["status"] == "completed"))
        return result
    if run.get("domain") == "operator":
        result["experiment_count"] = len(run.get("operator_experiments") or [])
        result["baseline_metrics"] = None
        result["guard"] = None
        result["has_memory_candidate"] = bool(run.get("memory_candidate"))
        result["operator_reason"] = (run.get("verdict") or {}).get("reason_code")
        return result
    result["experiment_count"] = max(0, len(run["experiments"]) - 1)
    result["baseline_metrics"] = run["experiments"][0]["result"]["metrics"] if run["experiments"] else None
    guard = run.get("guard")
    result["guard"] = None if not guard else {"accepted": guard["accepted"], "delta": guard["delta"],
        "metrics": (guard["candidate"] if guard["accepted"] else guard["baseline"])["metrics"]}
    result["has_memory_candidate"] = bool(run.get("memory_candidate"))
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send(self, status, body, content_type="application/json; charset=utf-8", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        try:
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/state":
                manifest = read_json(DATA / "manifest.json")
                data = None if not manifest else {"name": manifest["name"], "corpus_count": manifest["corpus_count"],
                    "sha256": manifest["archive_sha256"], "licenses": manifest["licenses"],
                    "splits": {k: ({p: len(ids) for p, ids in v.items()} if isinstance(v, dict) else len(v)) for k, v in manifest["splits"].items()}}
                final = read_json(ROOT / "data" / "final_evaluation.json")
                registry = read_json(DATA / "research_tasks_v2.json")
                task_summary = {name: {part: len(ids) for part, ids in task.items()} for name, task in (registry or {}).get("tasks", {}).items()}
                return self.send(200, {"dataset": data, "active_run": SERVICE.active, "preparing": SERVICE.preparing,
                    "runs": [summary(r) for r in list_runs()], "memory": memory_state(), "settings": masked_settings(),
                    "research_tasks": task_summary, "task_registry_hash": registry.get("registry_hash") if registry else None,
                    "task_exposure": read_json(ROOT / "docs" / "experiments" / "v2-task-exposure.json", {}),
                    "final_test_revealed": bool(final), "version": __version__,
                    "research_tools": RESEARCH_TOOLS, "research_memory": research_core.memory_state(),
                    "sessions": db.sessions(20)})
            if path == "/api/research/tools":
                return self.send(200, {"tools": RESEARCH_TOOLS, "registered": research_registry.tools()})
            if path == "/api/sessions":
                return self.send(200, {"sessions": db.sessions(50)})
            if path == "/api/roles":
                return self.send(200, {"roles": research_roles.describe(),
                                       "models": settings().get("roles") or {}})
            if path.startswith("/api/sessions/"):
                session_id = path.split("/")[3]
                if not db.session(session_id):
                    return self.send(404, {"error": "会话不存在"})
                return self.send(200, {"session": db.session(session_id), "calls": db.calls(session_id),
                                       "steps": db.steps(session_id)})
            if path == "/api/memory/items":
                return self.send(200, {"items": db.memory_list()})
            if path == "/api/ollama":
                return self.send(200, probe(settings()["ollama_url"]))
            if path == "/api/cloud":
                cloud = settings()["cloud"]
                if not str(cloud.get("base_url") or "").strip():
                    return self.send(200, {"available": False, "models": [], "error": "尚未配置云端服务地址", "base_url": ""})
                return self.send(200, cloudllm.probe(cloud["base_url"], cloud.get("api_key", "")))
            if path == "/api/logs":
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                return self.send(200, activity.query(**{key: query[key][0] for key in
                    ("limit", "level", "source", "run_id", "text") if key in query}))
            if path == "/api/final":
                report = read_json(ROOT / "data" / "final_evaluation.json")
                if not report:
                    return self.send(200, {"sealed": True})
                return self.send(200, {k: v for k, v in report.items() if k != "results"} | {"results": {k: {"metrics": v["metrics"], "count": v["count"]} for k, v in report["results"].items()}})
            if path.startswith("/api/runs/"):
                run_id = path.split("/")[3]
                run = read_json(run_path(run_id))
                if path.endswith("/report") and run.get("domain") == "research":
                    report = RUNS / run_id / "report.md"
                    if not report.exists():
                        return self.send(404, {"error": "研究尚未结束，报告未生成"})
                    return self.send(200, report.read_bytes(), "text/markdown; charset=utf-8",
                                     {"Content-Disposition": 'attachment; filename="research-report.md"'})
                if path.endswith("/export") and run.get("domain") == "research":
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                        for file in (RUNS / run_id).rglob("*"):
                            if file.is_file() and "__pycache__" not in file.parts:
                                archive.write(file, file.relative_to(RUNS / run_id))
                    return self.send(200, buffer.getvalue(), "application/zip",
                                     {"Content-Disposition": f'attachment; filename="research-{run_id}.zip"'})
                if "/evidence/" in path:
                    claim_id = path.rsplit("/", 1)[-1]
                    allowed = set(run.get("split_ids", {}).get("explore", []))
                    if run.get("guard"):
                        allowed.update(run["split_ids"]["guard"])
                    if claim_id not in allowed:
                        raise ValueError("只可查看本次运行已开放的探索或验收证据")
                    dataset, _ = SERVICE.load()
                    claim = dataset.claims[claim_id]
                    by_doc = {str(d["doc_id"]): d for d in dataset.corpus}
                    gold = [{"id": f"{doc}:{index}", "text": by_doc[doc]["abstract"][index],
                             "title": by_doc[doc]["title"], "label": rationale["label"]}
                            for doc, rationales in claim["evidence"].items()
                            for rationale in rationales for index in rationale["sentences"]]
                    return self.send(200, {"claim": claim["claim"], "gold": list({g["id"]: g for g in gold}.values())})
                if path.endswith("/export"):
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                        archive.writestr("run.json", json.dumps(run, ensure_ascii=False, indent=2))
                        for filename in ("manifest.json", "LICENSE.md"):
                            archive.write(DATA / filename, f"dataset/{filename}")
                        if run.get("version") == "0.2.0":
                            archive.write(DATA / "research_tasks_v2.json", "dataset/research_tasks_v2.json")
                        snapshot = RUNS / run_id / "source" / "researchloop"
                        for file in (snapshot if snapshot.exists() else ROOT / "researchloop").glob("*.py"):
                            archive.write(file, "source/researchloop/" + file.name)
                        for file in ("run.py", "README.md", "reproduce.py"):
                            if (ROOT / file).exists():
                                archive.write(ROOT / file, "source/" + file)
                        events = RUNS / run_id / "events.jsonl"
                        if events.exists():
                            archive.write(events, "events.jsonl")
                        archive.writestr("REPRODUCE.txt", "Raw decisions, full usage, configs and per-query results are in run.json. Recompute locally with: python reproduce.py <path-to-run.json>. Use the matching SciFact archive SHA256 from dataset/manifest.json. Re-running the LLM can differ from replaying logged numeric experiments. No final-test labels or results are included in this pack.")
                    return self.send(200, buffer.getvalue(), "application/zip", {"Content-Disposition": f'attachment; filename="researchloop-{run_id}.zip"'})
                return self.send(200, run)
            relative = "index.html" if path == "/" else path.lstrip("/")
            target = (WEB / relative).resolve()
            if not target.is_relative_to(WEB.resolve()) or not target.is_file():
                return self.send(404, {"error": "Not found"})
            kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            return self.send(200, target.read_bytes(), kind + ("; charset=utf-8" if kind.startswith("text/") or kind == "application/javascript" else ""))
        except (ValueError, FileNotFoundError) as exc:
            self.send(400, {"error": str(exc)})
        except Exception as exc:
            self.send(500, {"error": str(exc)})

    def do_POST(self):
        global RESEARCH_ENGINE
        request_id = uuid.uuid4().hex
        request_started = time.perf_counter()
        try:
            origin = self.headers.get("Origin")
            if origin and urllib.parse.urlparse(origin).netloc != self.headers.get("Host"):
                return self.send(403, {"error": "Only same-origin requests are allowed"})
            if "application/json" not in self.headers.get("Content-Type", ""):
                return self.send(415, {"error": "Content-Type must be application/json"})
            length = int(self.headers.get("Content-Length", 0))
            limit = 40 * 1024 * 1024 if urllib.parse.urlparse(self.path).path == "/api/research" else 80000
            if length > limit:
                raise ValueError("请求内容过大")
            body = json.loads(self.rfile.read(length) or b"{}")
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/research":
                with LOCK:
                    if SERVICE.active or SERVICE.preparing:
                        raise ValueError("已有任务正在运行")
                    engine = RESEARCH_ENGINE = research_core.create(body)
                    SERVICE.active = engine.run["id"]
                    def research_work():
                        try:
                            engine.execute()
                        finally:
                            with LOCK:
                                if SERVICE.active == engine.run["id"]:
                                    SERVICE.active = None
                    threading.Thread(target=research_work, daemon=True).start()
                return self.send(202, summary(engine.run))
            if path == "/api/research/memory/activate":
                if SERVICE.active:
                    raise ValueError("运行中不能切换经验")
                return self.send(200, research_core.activate(body["version"]))
            if path == "/api/prepare":
                with LOCK:
                    if SERVICE.active or (SERVICE.preparing and SERVICE.preparing.get("status") == "running"):
                        raise ValueError("已有任务正在运行")
                    if (DATA / "manifest.json").exists():
                        return self.send(200, {"ready": True})
                    SERVICE.preparing = {"status": "running", "message": "开始准备数据"}
                    def work():
                        try:
                            prepare(lambda msg: SERVICE.preparing.update(message=msg))
                            SERVICE.preparing = None
                        except Exception as exc:
                            SERVICE.preparing = {"status": "failed", "message": str(exc)}
                    threading.Thread(target=work, daemon=True).start()
                return self.send(202, {"started": True})
            if path == "/api/web/test":
                config = settings()["web"]
                query = str(body.get("query") or "research agent").strip()[:200]
                result = webaccess.search(query, provider=config.get("search_provider"),
                                          endpoint=config.get("search_endpoint"),
                                          api_key=config.get("search_api_key"),
                                          max_results=3, recency="any", timeout=8)
                return self.send(200, {"provider": result["provider"], "query": result["query"],
                                       "results": result["results"][:3]})
            if path == "/api/cloud/test":
                given = body.get("cloud") if isinstance(body.get("cloud"), dict) else body
                current = settings()["cloud"]
                base_url = given.get("base_url") or current.get("base_url")
                api_key = given.get("api_key") or current.get("api_key")
                if not str(base_url or "").strip():
                    raise ValueError("请先填写云端服务地址")
                return self.send(200, cloudllm.probe(base_url, api_key))
            if path == "/api/settings":
                if SERVICE.active:
                    raise ValueError("研究运行期间不能修改模型设置")
                url = str(body.get("ollama_url", "http://127.0.0.1:11434")).rstrip("/")
                parsed = urllib.parse.urlparse(url)
                if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1") or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path:
                    raise ValueError("本地服务只接受 localhost / 127.0.0.1 / ::1 的 HTTP Ollama 地址")
                model = str(body.get("model", "")).strip()
                if not model or "cloud" in model.lower() or len(model) > 120:
                    raise ValueError("请选择本地模型")
                source = str(body.get("model_source") or "local").strip().lower()
                if source not in ("local", "cloud"):
                    raise ValueError("模型来源只能是 local 或 cloud")
                role_models = body.get("roles") if isinstance(body.get("roles"), dict) else {}
                cleaned_roles = {}
                for role, conf in role_models.items():
                    if role not in research_roles.ROLES:
                        raise ValueError(f"未知研究角色 {role}")
                    conf = conf if isinstance(conf, dict) else {}
                    role_source = str(conf.get("model_source") or "local").lower()
                    if role_source not in ("local", "cloud"):
                        raise ValueError(f"角色 {role} 的模型来源无效")
                    role_model = str(conf.get("model") or "").strip()
                    if len(role_model) > 160:
                        raise ValueError(f"角色 {role} 的模型名过长")
                    cleaned_roles[role] = {"model_source": role_source, "model": role_model}
                current = settings()["cloud"]
                given = body.get("cloud") if isinstance(body.get("cloud"), dict) else {}
                enabled = bool(given.get("enabled")) or source == "cloud"
                if enabled:
                    base_url = cloudllm.normalise(given.get("base_url") or current.get("base_url") or "")
                    cloud_model = str(given.get("model") or current.get("model") or "").strip()
                    if not cloud_model or len(cloud_model) > 160:
                        raise ValueError("请填写云端模型名称")
                    api_key = str(given.get("api_key") or "").strip() or str(current.get("api_key") or "")
                    if not 8 <= len(api_key) <= 400:
                        raise ValueError("启用云端模型需要 8–400 字符的 API Key")
                    value_cloud = {"enabled": True, "base_url": base_url, "model": cloud_model, "api_key": api_key}
                else:
                    value_cloud = {"enabled": False, "base_url": "", "model": "",
                                   "api_key": str(current.get("api_key") or "")}
                current_web = settings()["web"]
                web_given = body.get("web") if isinstance(body.get("web"), dict) else {}
                web_enabled = bool(web_given.get("enabled"))
                provider = str(web_given.get("search_provider") or current_web.get("search_provider") or "searxng").strip().lower()
                if provider not in ("searxng", "tavily"):
                    raise ValueError("联网搜索提供方只能是 searxng 或 tavily")
                endpoint = str(web_given.get("search_endpoint") or current_web.get("search_endpoint") or "").strip()
                if web_enabled:
                    if not endpoint:
                        raise ValueError("启用联网工具需要搜索服务地址")
                    parsed_endpoint = urllib.parse.urlparse(endpoint)
                    if parsed_endpoint.scheme not in ("http", "https") or not parsed_endpoint.hostname:
                        raise ValueError("搜索服务地址必须是 http(s) URL")
                web_key = str(web_given.get("search_api_key") or "").strip() or str(current_web.get("search_api_key") or "")
                if web_enabled and provider == "tavily" and not 8 <= len(web_key) <= 400:
                    raise ValueError("Tavily 搜索需要 8–400 字符的 API Key")
                web_value = {"enabled": web_enabled, "search_provider": provider, "search_endpoint": endpoint,
                             "search_api_key": web_key if web_enabled else "",
                             "max_results": max(1, min(int(web_given.get("max_results") or 5), 10)),
                             "fetch_max_chars": max(200, min(int(web_given.get("fetch_max_chars") or 6000), 20000))}
                value = {"ollama_url": url, "model": model, "model_source": source, "cloud": value_cloud,
                         "roles": cleaned_roles, "web": web_value}
                write_json(ROOT / "config.json", value)
                return self.send(200, masked_settings())
            if path == "/api/brief":
                messages = body.get("messages")
                if not isinstance(messages, list) or not 1 <= len(messages) <= 12:
                    raise ValueError("对话需要 1 到 12 条")
                activity.record("brief_started", "收到研究方向，开始拟定计划", request_id=request_id,
                                source="planner", actor="system", status="running", data={"messages": messages})
                model = None
                skip = "未调用模型"
                if body.get("use_model", True):
                    cfg = settings()
                    status = probe(cfg["ollama_url"])
                    if status["available"] and cfg.get("model"):
                        model = cfg
                        skip = None
                    elif not status["available"]:
                        skip = "Ollama 未连接"
                    else:
                        skip = "未选择模型"
                plan = draft(messages, task=body.get("task") if isinstance(body.get("task"), dict) else None,
                             model=model, skip_reason=skip)
                log = plan.get("log") or {}
                log["request_id"] = request_id
                fallback = bool(body.get("use_model", True)) and plan.get("source") != "ollama"
                activity.record("brief",
                                f"模型未生成可用计划，已改用规则草拟：{log.get('reason_label')}" if fallback else
                                "本地模型已生成计划" if plan.get("source") == "ollama" else "已使用规则草拟，未调用模型",
                                request_id=request_id, source="planner", actor=plan.get("source"),
                                status="fallback" if fallback else "completed", level="warning" if fallback else "info",
                                model=log.get("model") or "", elapsed_ms=round((time.perf_counter() - request_started) * 1000),
                                reason=log.get("reason_code") or "", usage=log.get("usage"),
                                data={"log": log, "plan": plan, "messages": messages})
                return self.send(200, plan)
            if path == "/api/tasks":
                try:
                    saved = save_task(body.get("name"), body.get("reference"), body.get("shapes"))
                except ValueError as exc:
                    activity.record("task", f"加入算子失败：{exc}", source="operator", actor="tool",
                                    status="failed", level="error", request_id=request_id)
                    raise
                activity.record("task", f"已加入算子 {saved['name']}：{saved['ir']}", source="operator",
                                actor="tool", task=saved["name"], request_id=request_id,
                                data={"task": saved, "reference": body.get("reference"), "shapes": body.get("shapes")})
                return self.send(200, saved)
            if path == "/api/runs":
                if operator_service.is_operator_task(str(body.get("task"))):
                    with LOCK:
                        if SERVICE.active or (SERVICE.preparing and SERVICE.preparing.get("status") == "running"):
                            raise ValueError("已有任务正在运行")
                        run, run_dir = operator_service.begin(
                            str(body.get("task")),
                            seconds_limit=int(body.get("seconds_limit", 180)),
                            max_experiments=int(body.get("max_experiments", 4)),
                            brief=body.get("brief") if isinstance(body.get("brief"), dict) else None)
                        SERVICE.active = run["id"]

                        def work(current=run, folder=run_dir):
                            try:
                                operator_service.complete(current, folder)
                            finally:
                                with LOCK:
                                    if SERVICE.active == current["id"]:
                                        SERVICE.active = None

                        threading.Thread(target=work, daemon=True).start()
                    return self.send(202, summary(run))
                return self.send(202, summary(SERVICE.start(body)))
            if path.startswith("/api/runs/") and path.endswith("/cancel"):
                if RESEARCH_ENGINE and RESEARCH_ENGINE.run["id"] == path.split("/")[3] and SERVICE.active == RESEARCH_ENGINE.run["id"]:
                    RESEARCH_ENGINE.cancelled.set()
                    return self.send(200, {"requested": True, "message": "当前工具调用结束后停止"})
                return self.send(200, SERVICE.cancel(path.split("/")[3]))
            if path == "/api/memory/promote":
                if SERVICE.active:
                    raise ValueError("请等待当前运行完成后发布经验")
                candidate = read_json(run_path(body["run_id"]))
                if candidate.get("domain") == "research":
                    return self.send(200, research_core.publish(candidate))
                return self.send(200, promote(body["run_id"]))
            if path == "/api/memory/activate":
                if SERVICE.active:
                    raise ValueError("请等待当前运行完成后切换经验")
                return self.send(200, rollback(body["version"]))
            if path == "/api/finalize":
                return self.send(200, SERVICE.finalize())
            self.send(404, {"error": "Unknown endpoint"})
        except (ValueError, KeyError, TypeError) as exc:
            activity.record("request_failed", str(exc), source="api", actor="system", request_id=request_id,
                            status="failed", level="error", endpoint=urllib.parse.urlparse(self.path).path,
                            elapsed_ms=round((time.perf_counter() - request_started) * 1000))
            self.send(400, {"error": str(exc)})
        except Exception as exc:
            activity.record("request_failed", str(exc), source="api", actor="system", request_id=request_id,
                            status="failed", level="error", endpoint=urllib.parse.urlparse(self.path).path,
                            elapsed_ms=round((time.perf_counter() - request_started) * 1000))
            self.send(500, {"error": str(exc)})


def serve(host="127.0.0.1", port=8767):
    try:
        db.migrate_memory_files()
    except Exception:
        pass
    for run in list_runs():
        if run["status"] in ("queued", "running"):
            run["status"] = "interrupted"
            run["stop_reason"] = "服务重启，保留此前已写入的实验记录；未继续调用模型"
            write_json(run_path(run["id"]), run)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"ResearchLoop: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
