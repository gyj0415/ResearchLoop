import io
import json
import mimetypes
import threading
import time
import uuid
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from . import __version__
from .ollama import probe
from . import activity
from .scenarios.operator import service as operator_service
from .scenarios.operator.custom import save_task
from .engine import core as research_core
from .engine import registry as research_registry
from .engine import roles as research_roles
from .storage import LOCK, ROOT, RUNS, archive_run, delete_run, list_runs, masked_settings, now, read_json, run_path, settings, write_json
from . import cloudllm
from . import db
from . import webaccess
from . import memory
from . import reporting
from .sessions import SessionManager

SERVICE = SimpleNamespace(active=None, preparing=None)
SESSIONS = SessionManager()
WEB = ROOT / "web"


def summary(run):
    keys = ("id", "version", "created_at", "finished_at", "status", "task", "mode", "model", "model_source", "provider", "archived", "archived_at", "updated_at", "queue_wait_seconds", "budget_enabled", "memory_version", "requested_memory_version", "memory_selection", "seed", "best_config", "budget", "stop_reason", "error", "selected_experiment_id", "evaluation_counts", "source_hash")
    result = {k: run.get(k) for k in keys}
    result["domain"] = run.get("domain", "unknown")
    result.update(subagent_count=len(run.get('subagents',[])),pending_approval_count=sum(r.get('status')=='pending' for r in run.get('approval_requests',[])))
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
    return result


def busy(run_id=None):
    return (SERVICE.active == run_id if run_id else bool(SERVICE.active)) or SESSIONS.busy(run_id)


def launch_research(engine):
    SESSIONS.submit(engine)


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def local_host(value):
    """DNS rebinding sends a foreign name in Host while reaching this loopback socket."""
    try:
        return urllib.parse.urlsplit("//" + str(value or "")).hostname in LOCAL_HOSTS
    except ValueError:
        return False


def stored_key_for(base_url, current):
    """A saved key is only reused for the endpoint it was saved with."""
    try:
        same = cloudllm.normalise(base_url) == cloudllm.normalise(current.get("base_url"))
    except ValueError:
        same = False
    return str(current.get("api_key") or "") if same else ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def parse_request(self):
        if not super().parse_request():
            return False
        if not local_host(self.headers.get("Host")):
            self.send(403, {"error": "Only localhost requests are allowed"})
            return False
        return True

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
                execution = SESSIONS.snapshot()
                if SERVICE.active:
                    execution["active_runs"].append(SERVICE.active)
                    execution["running_runs"].append(SERVICE.active)
                active = execution["running_runs"] or execution["queued_runs"]
                return self.send(200, {"active_run": active[0] if active else None, **execution, "runs": [summary(r) for r in list_runs()],
                    "settings": masked_settings(), "version": __version__, "research_tools": research_registry.tools(),
                    "research_memory": research_core.memory_state(), "memory_system": memory.state()})
            if path == "/api/research/tools":
                return self.send(200, {"tools": research_registry.tools(), "registered": research_registry.tools()})
            if path == "/api/memory":
                return self.send(200, memory.state())
            if path == "/api/roles":
                return self.send(200, {"roles": research_roles.describe(),
                                       "models": settings().get("roles") or {}})
            if path == "/api/memory/items":
                return self.send(200, {"items": db.memory_list()})
            if path == "/api/ollama":
                return self.send(200, probe(settings()["ollama_url"]))
            if path == "/api/cloud":
                cloud = settings()["cloud"]
                if not str(cloud.get("base_url") or "").strip():
                    return self.send(200, {"available": False, "models": [], "error": "尚未配置云端服务地址", "base_url": ""})
                return self.send(200, cloudllm.probe(cloud["base_url"], cloud.get("api_key", "")))
            if path.startswith("/api/runs/"):
                run_id = path.split("/")[3]
                run = read_json(run_path(run_id))
                parts=path.strip('/').split('/')
                if len(parts)==5 and parts[3]=='subagents':
                    child_id=parts[4]
                    if not child_id.startswith('sa') or not child_id[2:].isdigit():
                        raise ValueError('子 Agent ID 无效')
                    child=read_json(RUNS/run_id/'subagents'/child_id/'run.json')
                    if not child:
                        return self.send(404,{'error':'子 Agent 记录不存在'})
                    return self.send(200,child)
                if path.endswith("/report") and run.get("domain") == "research":
                    report = RUNS / run_id / "report.md"
                    if not report.exists():
                        return self.send(404, {"error": "研究尚未结束，报告未生成"})
                    export_format = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("format", ["md"])[0]
                    if export_format not in ("md", "tex", "pdf"):
                        raise ValueError("报告格式仅支持 md、tex 或 pdf")
                    if export_format in ("tex", "pdf"):
                        if not run.get("report") or run.get("status") in ("running", "queued", "pausing"):
                            return self.send(404, {"error": "本轮研究报告尚未生成"})
                        doc = reporting.document(run)
                        content = reporting.latex(doc) if export_format == "tex" else reporting.pdf(doc)
                        kind = "application/x-tex; charset=utf-8" if export_format == "tex" else "application/pdf"
                        return self.send(200, content, kind,
                                         {"Content-Disposition": f'attachment; filename="research-{run_id}.{export_format}"'})
                    return self.send(200, report.read_bytes(), "text/markdown; charset=utf-8",
                                     {"Content-Disposition": 'attachment; filename="research-report.md"'})
                if path.endswith("/export") and run.get("domain") == "research":
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                        ignored={'.git','.venv','node_modules','.cache','.tmp','.packages','__pycache__'}
                        for file in research_core.history._files(RUNS/run_id,ignored):
                            if file.is_file() and not file.is_symlink() and file.resolve().is_relative_to((RUNS/run_id).resolve()):
                                archive.write(file, file.relative_to(RUNS / run_id))
                        archive.writestr('EXPORT-NOTES.txt','Dependencies and caches (.venv, .packages, node_modules, .cache, .tmp, .git) are excluded. Recreate the matching toolchain before replay. Shell commands run with host user privileges, not in an OS sandbox.\n')
                    return self.send(200, buffer.getvalue(), "application/zip",
                                     {"Content-Disposition": f'attachment; filename="research-{run_id}.zip"'})
                run['subagent_live']=[]
                for child in run.get('subagents',[]):
                    saved=read_json(RUNS/run_id/'subagents'/child['id']/'run.json') or {}
                    run['subagent_live'].append({key:saved.get(key) for key in ('subagent_id','status','updated_at','budget','model','model_tier')})
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
            if length < 0:
                raise ValueError("Content-Length 无效")
            if length > limit:
                raise ValueError("请求内容过大")
            body = json.loads(self.rfile.read(length) or b"{}")
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/memory/add":
                return self.send(200, memory.add(body))
            if path == "/api/memory/toggle":
                return self.send(200, memory.toggle(body.get("id"), bool(body.get("enabled"))))
            if path == "/api/execution":
                parallel = body.get("max_parallel_sessions")
                if not isinstance(parallel, int) or isinstance(parallel, bool) or parallel < 1:
                    raise ValueError("并行研究数需要为正整数")
                with LOCK:
                    value = settings()
                    value["execution"] = {"max_parallel_sessions": parallel}
                    write_json(ROOT / "config.json", value)
                    execution = SESSIONS.reconfigure()
                return self.send(200, execution)
            if path == "/api/research":
                with LOCK:
                    if SERVICE.active:
                        raise ValueError("旧版算子任务正在运行，请等待结束")
                    engine = research_core.create(body)
                    launch_research(engine)
                return self.send(202, summary(engine.run))
            if path == "/api/research/memory/activate":
                if busy():
                    raise ValueError("请先暂停运行或排队的研究，再切换方法版本")
                return self.send(200, research_core.activate(body["version"]))
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
                api_key = given.get("api_key") or stored_key_for(base_url, current)
                if not str(base_url or "").strip():
                    raise ValueError("请先填写云端服务地址")
                return self.send(200, cloudllm.probe(base_url, api_key))
            if path == "/api/settings":
                if busy():
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
                role_models = (body.get("roles") if isinstance(body.get("roles"), dict)
                               else (settings().get("roles") or {}))
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
                    api_key = str(given.get("api_key") or "").strip() or stored_key_for(base_url, current)
                    if not 8 <= len(api_key) <= 400:
                        raise ValueError("启用云端模型需要 8–400 字符的 API Key；更换服务地址后需重新填写")
                    value_cloud = {"enabled": True, "base_url": base_url, "model": cloud_model, "api_key": api_key}
                else:
                    # Keep whatever the user typed/selected even while the provider is off,
                    # so a restart does not silently wipe the configuration.
                    disabled_url = cloudllm.normalise(given.get("base_url") or current.get("base_url") or "")
                    value_cloud = {"enabled": False, "base_url": disabled_url,
                                   "model": str(given.get("model") or current.get("model") or "").strip()[:160],
                                   "api_key": str(given.get("api_key") or "").strip() or stored_key_for(disabled_url, current)}
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
                web_key = str(web_given.get("search_api_key") or "").strip() or (
                    str(current_web.get("search_api_key") or "")
                    if endpoint.rstrip("/") == str(current_web.get("search_endpoint") or "").strip().rstrip("/") else "")
                if web_enabled and provider == "tavily" and not 8 <= len(web_key) <= 400:
                    raise ValueError("Tavily 搜索需要 8–400 字符的 API Key")
                web_value = {"enabled": web_enabled, "search_provider": provider, "search_endpoint": endpoint,
                             "search_api_key": web_key,
                             "max_results": max(1, min(int(web_given.get("max_results") or 5), 10)),
                             "fetch_max_chars": max(200, min(int(web_given.get("fetch_max_chars") or 6000), 20000))}
                budgets_given = body.get("budgets") if isinstance(body.get("budgets"), dict) else {}
                current_budgets = settings().get("budgets") or {}

                def optional_budget(key, label, default=None):
                    raw = budgets_given.get(key, current_budgets.get(key, default))
                    if raw in (None, ""):
                        return None
                    try:
                        number = int(raw)
                    except (TypeError, ValueError):
                        raise ValueError(f"{label}必须是整数，或留空表示不限")
                    if number <= 0:
                        raise ValueError(f"{label}必须为正数，或留空表示不限")
                    return number

                tokens_k = optional_budget("tokens_k", "默认 token（k）", 60) or 60
                budgets = {"tokens_k": tokens_k,
                           "seconds": optional_budget("seconds", "默认时间"),
                           "steps": optional_budget("steps", "默认步骤")}
                if web_value["enabled"]:
                    webaccess.validate_search_config(web_value["search_provider"], web_value["search_endpoint"])
                value = {"revision": uuid.uuid4().hex, "ollama_url": url, "model": model, "model_source": source, "cloud": value_cloud,
                         "roles": cleaned_roles, "web": web_value, "budgets": budgets,
                         "routing": research_core.routing.clean(body.get("routing") or settings().get("routing")),
                         "execution": settings()["execution"]}
                write_json(ROOT / "config.json", value)
                return self.send(200, masked_settings())
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
                        if busy():
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
                raise ValueError("旧版检索示例已移除；请通过 /api/research 提交研究问题")
            if path.startswith("/api/runs/") and path.endswith("/pause"):
                return self.send(202, SESSIONS.pause(path.split("/")[3]))
            if path.startswith('/api/runs/') and path.endswith('/approve'):
                return self.send(200,SESSIONS.approve(path.split('/')[3],body.get('request_id'),body.get('decision')))
            if path.startswith('/api/runs/') and path.endswith(('/retry','/recover')):
                with LOCK:
                    if SERVICE.active:
                        raise ValueError('旧版算子任务正在运行')
                    run_id=path.split('/')[3]
                    existing=read_json(run_path(run_id))
                    if existing.get('domain')!='research':
                        raise ValueError('仅支持研究会话')
                    folder=RUNS/run_id
                    is_retry=path.endswith('/retry')
                    if not is_retry:
                        SESSIONS.assert_idle(run_id)
                        if existing.get('status') in ('running','queued','pausing'):
                            raise ValueError('请先暂停研究')
                    enabled=body.get('budget_enabled',True)
                    if not isinstance(enabled,bool):
                        raise ValueError('budget_enabled 必须为布尔值')
                    def limit(key,default):
                        raw=body.get(key,default)
                        if raw in (None,''):
                            return None
                        if not isinstance(raw,int) or isinstance(raw,bool) or raw<=0:
                            raise ValueError('预算须为正整数或留空表示不限')
                        return raw
                    tokens=limit('token_limit',30000) if enabled else None
                    seconds=limit('seconds_limit',None) if enabled else None
                    steps=limit('max_steps',12) if enabled else None
                    if is_retry:
                        existing,folder=research_core.history.retry(existing,folder,str(body.get('node_id') or ''))
                        if body.get('question'):
                            existing['branch_note']+=' 用户补充：'+str(body['question'])[:2000]
                        existing.update(budget_enabled=enabled,token_limit=tokens,seconds_limit=seconds,max_steps=steps)
                        engine=research_core.resume_paused(existing,folder)
                    elif existing.get('status')=='paused':
                        # Increment the current turn's allowance without refunding any spend.
                        spent=existing.get('budget') or {}
                        existing.update(budget_enabled=enabled,
                            token_limit=(spent.get('total_tokens',0)+tokens) if tokens is not None else None,
                            seconds_limit=(spent.get('elapsed_seconds',0)+seconds) if seconds is not None else None,
                            max_steps=(len([s for s in existing.get('steps',[])[existing.get('branch_step_start',0):] if s.get('turn','t1')==(existing.get('turns') or [{'id':'t1'}])[-1]['id']])+steps) if steps is not None else None)
                        engine=research_core.resume_paused(existing,folder)
                    else:
                        existing['budget_enabled']=enabled
                        direction=str(body.get('question') or '').strip()
                        question=existing.get('current_question') or existing['question']
                        if direction:
                            question+='\n继续要求：'+direction
                        engine=research_core.resume(existing,folder,question,
                                                     token_limit=tokens,seconds_limit=seconds,max_steps=steps)
                        # Null means unlimited in this explicit UI action, not "keep old limit".
                        engine.run.update(token_limit=tokens,seconds_limit=seconds,max_steps=steps)
                        engine.budget.token_limit,engine.budget.seconds_limit=tokens,seconds
                    engine.run.setdefault('recovery_actions',[]).append({'at':now(),'action':'retry' if is_retry else 'continue',
                        'budget_enabled':enabled,'additional_tokens':tokens,'additional_seconds':seconds,'additional_steps':steps})
                    launch_research(engine)
                return self.send(202,summary(engine.run))
            if path.startswith("/api/runs/") and path.endswith(("/resume", "/fork")):
                with LOCK:
                    if SERVICE.active:
                        raise ValueError("旧版算子任务正在运行")
                    run_id = path.split("/")[3]
                    existing = read_json(run_path(run_id))
                    folder = RUNS / run_id
                    if existing.get("domain") != "research":
                        raise ValueError("仅支持研究会话")
                    if not path.endswith("/fork"):
                        SESSIONS.assert_idle(run_id)
                    if path.endswith("/fork"):
                        existing, folder = research_core.history.fork(existing, folder, str(body.get("node_id") or ""), body.get("question"))
                    if isinstance(body.get("routing"), dict):
                        existing["routing"] = research_core.routing.clean(body["routing"])
                    engine = research_core.resume_paused(existing, folder)
                    launch_research(engine)
                return self.send(202, summary(engine.run))
            if path.startswith("/api/runs/") and path.endswith("/continue"):
                run_id = path.split("/")[3]
                with LOCK:
                    if SERVICE.active:
                        raise ValueError("旧版算子任务正在运行")
                    SESSIONS.assert_idle(run_id)
                    existing = read_json(run_path(run_id))
                    if existing.get("domain") != "research":
                        raise ValueError("只有研究会话可以继续追问")
                    if existing.get("status") in ("running", "queued", "pausing", "paused"):
                        raise ValueError("请先恢复或结束当前研究，再开启新的追问")
                    engine = research_core.resume(existing, RUNS / run_id, body.get("question"),
                                                  token_limit=body.get("token_limit"),
                                                  seconds_limit=body.get("seconds_limit"),
                                                  max_steps=body.get("max_steps"))
                    launch_research(engine)
                return self.send(202, summary(engine.run))
            if path.startswith("/api/runs/") and path.endswith("/archive"):
                run_id = path.split("/")[3]
                if busy(run_id):
                    raise ValueError("运行中不能归档")
                archived = bool(body.get("archived", True))
                run = archive_run(run_id, archived)
                activity.record("archive", "会话已归档" if archived else "会话已恢复",
                                source="api", actor="user", run_id=run_id, status="completed",
                                data={"archived": archived})
                return self.send(200, summary(run))
            if path.startswith("/api/runs/") and path.endswith("/delete"):
                run_id = path.split("/")[3]
                if busy(run_id):
                    raise ValueError("运行或排队的研究不能删除")
                result = delete_run(run_id)
                activity.record("archive", "会话已移入回收目录", source="api", actor="user",
                                run_id=run_id, status="completed", data=result)
                return self.send(200, result)
            if path.startswith("/api/runs/") and path.endswith("/cancel"):
                return self.send(200, SESSIONS.cancel(path.split("/")[3]))
            if path == "/api/memory/promote":
                if busy():
                    raise ValueError("请等待当前运行完成后发布经验")
                candidate = read_json(run_path(body["run_id"]))
                if candidate.get("domain") == "research":
                    return self.send(200, research_core.publish(candidate))
                raise ValueError("仅支持发布科研 Agent 的已验证经验")
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
        if run["status"] == "queued":
            run.update(status="paused", restore_budget=True, stop_reason="服务重启，排队研究已暂停，未启动新调用")
            write_json(run_path(run["id"]), run)
        elif run["status"] in ("running", "pausing"):
            run["status"] = "interrupted"
            run["stop_reason"] = "服务重启，保留此前已写入的实验记录；未继续调用模型"
            for approval in run.get('approval_requests',[]):
                if approval.get('status')=='pending':
                    approval.update(status='cancelled',resolved_at=now(),reason='服务重启，未执行该操作')
            write_json(run_path(run["id"]), run)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"ResearchLoop: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
