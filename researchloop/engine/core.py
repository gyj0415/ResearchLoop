"""A model decides the next research action from observations, not task names."""
import hashlib
import json
import base64
import math
import secrets
import threading
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path

from researchloop import activity, storage, db, documents
from researchloop import cloudllm
from researchloop.engine import registry
from researchloop.engine import roles
from researchloop.engine import workflow as workflow_module
from researchloop.engine import extratools  # registers web_search/web_fetch/remember_fact/get_memory_context
from researchloop.engine import doctools  # registers read_docx/read_pptx/read_pdf
from researchloop import facts
from researchloop.ollama import Budget, BudgetError
from . import tools

SYSTEM = """你是执行科研的Agent。输入是开放研究问题，而不是预设算子任务。
你根据工具返回的真实证据决定下一步：检查材料/检索，注册可证伪假设，设计并运行Python实验，独立重跑验证，再根据结果修正或结束。
每次只调用一个工具；不预设结果，不编造文献和数据。工具失败时根据错误修复代码或换方法；已经成功的工具不要无意义重复。
没有用户数据时可以设计数值模拟，但必须明确模拟假设，不声称真实世界结论；不能以模拟替代原问题所需的真实实验或数据。条件不足则在plan中feasible=false。文献元数据不等于论文结论。
先inspect，再hypothesis，然后experiment、verify、finish是常见顺序，但你可按研究需要检索、读取、提出新假设或修正失败实验。
Python只能用np(numpy)、math、statistics；随机数必须用提供的rng。提供load_csv('a1')返回字符串字典列表、load_text('a1')返回文本。
实验必须实际计算并emit_result({'metrics': {'预注册指标名': float(实测值)}, 'notes':'方法说明'})。不能填写预想常数冒充实测值。
不导入os/sys/文件/网络，不使用np.random或自定固定随机种子；命名空间已提供 rng 变量，直接用 rng.normal/rng.uniform，不要为 rng 重新赋值或新建生成器；verify会用新的隐藏种子运行同一份代码。只写精简代码。
假设必须给出statement,prediction,controls,metric,relation(gt/ge/lt/le),threshold。实验必须关联hypothesis_id。
可以证伪假设；复核未通过阈值是有效负结果。你不能修改已经注册的阈值或冻结的复核代码。
finish写出结论、局限、引用IDs和lesson；只有引用已复核实验才形成实验报告。缺工具/材料时明确未完成。
当前动作由研究协议或基于结果的反思确定；只按本次指定的工具参数schema返回JSON。"""

ACTION_SCHEMA = {"type": "object", "properties": {
    "tool": {"type": "string", "enum": [t["name"] for t in tools.TOOLS]}, "reason": {"type": "string"},
    "args": {"type": "object", "properties": {
        **{key: {"type": "string"} for key in ("query", "asset_id", "statement", "prediction", "controls", "metric", "hypothesis_id", "experiment_id", "code", "conclusion", "lesson")},
        "relation": {"type": "string", "enum": ["gt", "ge", "lt", "le"]}, "threshold": {"type": "number"},
        "citations": {"type": "array", "items": {"type": "string"}}}}},
    "required": ["tool", "reason", "args"]}

ARGUMENTS = {
    "plan": ({"feasible": {"type": "boolean"}, **{k: {"type": "string"} for k in ("method", "limitations", "literature_query")}}, ["feasible", "method", "limitations", "literature_query"]),
    "inspect": ({}, []),
    "literature": ({"query": {"type": "string"}}, ["query"]),
    "read": ({"asset_id": {"type": "string"}}, ["asset_id"]),
    "hypothesis": ({**{k: {"type": "string"} for k in ("statement", "prediction", "controls", "metric")},
                    "relation": {"type": "string", "enum": ["gt", "ge", "lt", "le"]}, "threshold": {"type": "number"}},
                   ["statement", "prediction", "controls", "metric", "relation", "threshold"]),
    "experiment": ({"hypothesis_id": {"type": "string"}, "code": {"type": "string"}}, ["hypothesis_id", "code"]),
    "verify": ({"experiment_id": {"type": "string"}}, ["experiment_id"]),
    "reflect": ({"next_action": {"type": "string", "enum": ["revise", "finish"]}, "reason": {"type": "string"}}, ["next_action", "reason"]),
    "finish": ({"conclusion": {"type": "string"}, "citations": {"type": "array", "items": {"type": "string"}}, "lesson": {"type": "string"}}, ["conclusion", "citations", "lesson"]),
    "web_search": ({"query": {"type": "string"}, "reason": {"type": "string"},
                    "max_results": {"type": "integer"}, "recency": {"type": "string", "enum": ["any", "day", "week", "month", "year"]}},
                   ["query", "reason"]),
    "web_fetch": ({"url": {"type": "string"}, "reason": {"type": "string"}, "max_chars": {"type": "integer"}}, ["url", "reason"]),
    "remember_fact": ({"predicate": {"type": "string"}, "object": {"type": "string"}, "summary": {"type": "string"},
                       "tags": {"type": "array", "items": {"type": "string"}}, "confidence": {"type": "number"},
                       "source_message": {"type": "string"}}, ["predicate", "object", "summary"]),
    "get_memory_context": ({"query": {"type": "string"}}, []),
    "read_docx": ({"asset_id": {"type": "string"}, "max_chars": {"type": "integer"}}, ["asset_id"]),
    "read_pptx": ({"asset_id": {"type": "string"}, "max_chars": {"type": "integer"}}, ["asset_id"]),
    "read_pdf": ({"asset_id": {"type": "string"}, "max_chars": {"type": "integer"}}, ["asset_id"]),
}


def memory_state():
    root = storage.MEMORY / "research"
    versions = sorted([storage.read_json(p) for p in root.glob("v*.json")], key=lambda v: v["version"])
    return {"active": storage.read_json(root / "active.json", {"version": 0})["version"], "versions": versions}


def activate(version):
    version = int(version)
    if version and not (storage.MEMORY / "research" / f"v{version}.json").exists():
        raise ValueError("研究经验版本不存在")
    storage.write_json(storage.MEMORY / "research" / "active.json", {"version": version})
    return memory_state()


def publish(run):
    if run.get("domain") != "research" or run["status"] != "completed" or not run.get("report", {}).get("lesson"):
        raise ValueError("需要完成复核并形成经验的研究")
    state = memory_state()
    existing = next((v for v in state["versions"] if v["source_run"] == run["id"]), None)
    if existing:
        return existing
    version = max([v["version"] for v in state["versions"]] + [0]) + 1
    value = {"version": version, "parent_version": state["active"], "source_run": run["id"],
             "question": run["question"], "lesson": run["report"]["lesson"],
             "evidence_ids": [e["id"] for e in run["experiments"] if e.get("verification")],
             "created_at": storage.now(), "status": "provisional",
             "limitations": "仅为该问题的方法经验；下一项任务仍须验证，未证明净收益。"}
    storage.write_json(storage.MEMORY / "research" / f"v{version}.json", value)
    activate(version)
    db.memory_add(session_id=run["id"], version=version, scope="research", lesson=value["lesson"],
                  applicability={"question": run["question"], "limitations": value["limitations"]},
                  evidence=value["evidence_ids"], status="provisional", source_run=run["id"],
                  parent_version=value["parent_version"], created_at=value["created_at"])
    return value


class ResearchEngine:
    def __init__(self, run, folder, *, decide=None):
        self.run, self.folder = run, Path(folder)
        self.budget = Budget(run["token_limit"], run["seconds_limit"])
        self.cancelled = threading.Event()
        self.decide = decide or self.ask

    def save(self):
        self.run["updated_at"], self.run["budget"] = storage.now(), self.budget.snapshot()
        self.run["budget"]["tool_seconds"] = round(sum(s.get("elapsed_seconds", 0) for s in self.run["steps"]), 3)
        storage.write_json(self.folder / "run.json", self.run)
        db.sync_run(self.run)

    def event(self, kind, text, **data):
        entry = activity.record(kind, text, log_file=self.folder.parent / "activity.jsonl",
                                source="research", actor="ollama" if kind in ("planning", "model_call") else "tool",
                                run_id=self.run["id"], task=self.run["task"], data=data,
                                status=data.get("status", "completed"), level=data.get("level", "info"))
        self.run["events"].append(entry)
        with (self.folder / "events.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.save()

    def active_identity(self):
        agent = self.run.get("active_agent") or {}
        return (agent.get("model") or self.run["model"],
                agent.get("model_source") or self.run.get("model_source", "local"),
                agent.get("role"), agent.get("id"))

    def resolve_model(self, role):
        conf = (self.run.get("roles") or {}).get(role) or {}
        source = conf.get("model_source") or self.run.get("model_source", "local")
        if source == "cloud":
            model = conf.get("model") or storage.settings()["cloud"].get("model") or ""
        else:
            model = conf.get("model") or self.run["model"]
        if not model:
            raise ValueError(f"角色 {role} 没有可用模型")
        return str(model), source

    def complete(self, system, prompt, schema, purpose="research_decision", role=None):
        """One model call for the active sub-agent, with budget and usage accounting."""
        model, source, _, agent_id = self.active_identity()
        self.event("planning", f"{'云端' if source == 'cloud' else '本地'}模型正在生成 {purpose} 内容", status="running")
        text = json.dumps(prompt, ensure_ascii=False) if isinstance(prompt, (dict, list)) else str(prompt)
        if source == "cloud":
            client = cloudllm.Cloud(storage.settings()["cloud"], model, self.budget)
            parsed, item = client.call(system, text, schema, purpose, self.run.get("seed"))
            self.run["calls"].append(item)
            self.event("model_call", "云端科研模型调用完成" if item["status"] == "ok" else "云端科研模型调用失败",
                       status="completed" if item["status"] == "ok" else "failed", call=item)
        else:
            self.budget.reserve(6144 + 1500)
            body = {"model": model, "stream": False, "think": False, "format": schema,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                    "options": {"temperature": 0, "num_ctx": 6144, "num_predict": 1500}, "keep_alive": "20m"}
            started = time.perf_counter()
            item = {"purpose": purpose, "model": model, "request": body}
            try:
                request = urllib.request.Request(storage.settings()["ollama_url"] + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=max(1, min(150, self.run["seconds_limit"] - self.budget.elapsed))) as response:
                    result = json.load(response)
                if not all(isinstance(result.get(k), int) for k in ("prompt_eval_count", "eval_count")):
                    raise ValueError("模型未返回usage")
                item.update(input_tokens=result["prompt_eval_count"], output_tokens=result["eval_count"],
                            cached_tokens=result.get("prompt_eval_cached_count", 0),
                            context_tokens=result["prompt_eval_count"], response=result, status="ok")
                parsed = json.loads(result.get("message", {}).get("content", ""))
            except Exception as error:
                item.update(status="error", error=str(error), unknown_reserved=7644)
                raise
            finally:
                item["wall_seconds"] = time.perf_counter() - started
                item["source"] = "ollama"
                item["agent"], item["role"] = agent_id, role or self.run.get("active_role")
                self.budget.record(item)
                self.run["calls"].append(item)
                self.event("model_call", "科研模型调用完成" if item["status"] == "ok" else "科研模型调用失败",
                           status="completed" if item["status"] == "ok" else "failed", call=item)
        item["agent"] = agent_id
        item["role"] = role or self.run.get("active_role")
        return parsed, item

    def ask(self):
        tool, automatic_args = self.next_protocol_action()
        if automatic_args is not None:
            return {"tool": tool, "reason": "按研究协议执行确定性证据检查或独立验证，不额外消耗模型", "args": automatic_args, "source": "controller"}
        properties, required = ARGUMENTS[tool]
        schema = {"type": "object", "properties": properties, "required": required}
        observations = [{"tool": s["tool"], "status": s["status"], "observation": s["observation"],
                         "failed_code": s["args"].get("code", "") if s["status"] == "failed" else ""} for s in self.run["steps"][-3:]]
        prompt = {"question": self.run["question"], "current_action": tool,
                  "tool": {k: v for k, v in next(t for t in tools.TOOLS if t["name"] == tool).items() if k != "arguments"}, "plan": self.run.get("plan"), "assets": self.run["assets"],
                  "hypotheses": self.run["hypotheses"],
                  "experiments": [{k: e.get(k) for k in ("id", "hypothesis_id", "status", "result", "verification", "error")} for e in self.run["experiments"]],
                  "evidence": self.run["sources"], "observations": observations,
                  "documents": [{key: document.get(key) for key in
                                 ("asset_id", "name", "type", "chars", "note")} for document in self.run.get("documents", [])],
                  "memory": self.run.get("memory_context"),
                  "experience": self.run.get("experience"), "budget": self.budget.snapshot(),
                  "args_schema": schema,
                  "instruction": "本次只生成current_action的参数JSON，不要返回tool/reason/args外壳；必须只使用args_schema中的字段名。metric必须是一个简短英文标识符，实验代码须使用冻结的metric名称。"}
        system = SYSTEM + f"\n当前控制器已检查前置条件，进入{tool}。只返回该工具参数，不要选择其他工具。"
        if tool == "finish":
            verified = [e for e in self.run["experiments"] if e.get("verification")]
            if verified:
                ids = "、".join(e["id"] for e in verified)
                hypotheses = "、".join(sorted({e["hypothesis_id"] for e in verified}))
                system += f"\n本次citations必须包含已复核实验 {ids} 及其假设 {hypotheses}；只引用文献会使报告判为未完成。"
        parsed, item = self.complete(system, prompt, schema)
        return {"tool": tool, "reason": f"按通用研究协议进入{tool}，研究内容由{item.get('source') == 'cloud' and '云端' or '本地'}模型生成",
                "args": parsed, "source": item.get("source", "ollama")}

    def next_protocol_action(self):
        run = self.run
        if not run.get("evidence_collected"):
            return "inspect", {}
        read_ids = {item.get("asset_id") for item in run.get("documents", [])}
        pending = next((asset for asset in run.get("assets", [])
                        if asset.get("kind") == "document" and asset["id"] not in read_ids), None)
        if pending:
            tool = {".pdf": "read_pdf", ".docx": "read_docx", ".pptx": "read_pptx"}.get(
                Path(pending["name"]).suffix.lower())
            if tool:
                return tool, {"asset_id": pending["id"]}
        if not run.get("plan"):
            return "plan", None
        web = storage.settings()["web"]
        if web.get("enabled") and not any(s["tool"] in ("web_search", "literature") for s in run["steps"]):
            return "web_search", None
        if web.get("enabled") and any(s["tool"] == "web_search" and s["status"] == "completed" for s in run["steps"]) \
                and not any(s["tool"] == "web_fetch" for s in run["steps"]):
            return "web_fetch", None
        query = run["plan"].get("literature_query", "").strip()
        if query and not any(s["tool"] == "literature" for s in run["steps"]):
            return "literature", {"query": query}
        if not run["hypotheses"]:
            return "hypothesis", None
        reflections = run.get("reflections", [])
        if reflections and reflections[-1]["next_action"] == "revise" and len(run["hypotheses"]) == reflections[-1]["hypothesis_count"]:
            return "hypothesis", None
        current = run["hypotheses"][-1]
        candidates = [e for e in run["experiments"] if e["hypothesis_id"] == current["id"] and e["status"] == "completed"]
        if not candidates:
            return "experiment", None
        if not candidates[-1].get("verification"):
            return "verify", {"experiment_id": candidates[-1]["id"]}
        if not reflections or reflections[-1]["after_experiment"] != candidates[-1]["id"]:
            return "reflect", None
        return "finish", None

    def available_tools(self):
        run = self.run
        evidence = (["read"] if run["assets"] else []) + ["literature"]
        if not run.get("evidence_collected"):
            return ["inspect", *evidence, "finish"]
        pending = [h for h in run["hypotheses"] if not any(e["hypothesis_id"] == h["id"] and e["status"] == "completed" for e in run["experiments"])]
        if pending:
            return ["experiment", *evidence, "finish"]
        if any(e["status"] == "completed" and not e.get("verification") for e in run["experiments"]):
            return ["verify", *evidence, "finish"]
        return ["hypothesis", *evidence, "finish"]

    def allowed_tools(self):
        agent = self.run.get("active_agent") or {}
        role = agent.get("role")
        return set(roles.tools_for(role)) if role else None

    def execute(self):
        self.run["status"] = "running"
        self.event("setup", "开始研究开放问题，按工具反馈决定下一步", question=self.run["question"])
        try:
            if self.run.get("workflow_mode") == "plan":
                self.orchestrate()
            else:
                self._loop(self.run["max_steps"])
        except BudgetError as error:
            self.run["status"], self.run["stop_reason"] = "budget_stopped", str(error)
        except Exception as error:
            self.run["status"], self.run["error"] = "failed", str(error)
        finally:
            if self.run["status"] == "running":
                self.run["status"] = "incomplete"
            self.run["finished_at"] = storage.now()
            self.write_report()
            self.event("completed", f"研究结束：{self.run['status']}", status=self.run["status"])
        return self.run

    def _loop(self, limit, mark_incomplete=True):
        allowed = self.allowed_tools()
        for _ in range(int(limit)):
            index = len(self.run["steps"])
            if self.cancelled.is_set():
                self.run["status"] = "cancelled"
                return
            if self.budget.elapsed >= self.run["seconds_limit"]:
                raise BudgetError("研究时间预算耗尽")
            self.event("progress", f"研究协议正在调度第 {index + 1} 步", status="running")
            try:
                decision = self.decide()
            except BudgetError:
                raise
            except Exception as error:
                decision = {"tool": "invalid", "reason": "模型调用或JSON解析失败", "args": {"error": str(error)}}
            tool, args = decision.get("tool"), decision.get("args", {})
            if self.budget.elapsed >= self.run["seconds_limit"]:
                raise BudgetError("模型调用后已达到时间上限，不再启动工具")
            agent = self.run.get("active_agent") or {}
            step = {"id": f"s{index + 1}", "at": storage.now(), "tool": tool, "reason": str(decision.get("reason", "")),
                    "args": args, "source": decision.get("source", "injected"),
                    "agent": agent.get("id"), "role": agent.get("role")}
            if allowed is not None and tool not in allowed:
                self.event("stage", f"阶段 {agent.get('id')}（{agent.get('role')}）结束：协议请求 {tool} 不属于该角色工具",
                           status="completed")
                return
            self.event("action", f"选择 {tool}：{step['reason']}", status="running", step=step)
            tool_started = time.perf_counter()
            try:
                observation = self.apply(tool, args)
                step.update(status="completed", observation=observation)
            except Exception as error:
                step.update(status="failed", observation={"error": str(error), "instruction": "根据错误修正方法；不要捏造成功结果。"})
            step["elapsed_seconds"] = round(time.perf_counter() - tool_started, 3)
            self.run["steps"].append(step)
            self.event("tool_result", f"{tool} {'执行完成' if step['status'] == 'completed' else '执行失败'}",
                       status=step["status"], level="info" if step["status"] == "completed" else "warning", step=step)
            if self.run["status"] in ("completed", "incomplete"):
                return
        if mark_incomplete and self.run["status"] == "running":
            self.run["status"], self.run["stop_reason"] = "incomplete", "达到研究步骤上限，保留已有证据"

    def controller_step(self, tool, args):
        """Deterministic controller action that does not consume a model call."""
        index = len(self.run["steps"])
        agent = self.run.get("active_agent") or {}
        step = {"id": f"s{index + 1}", "at": storage.now(), "tool": tool,
                "reason": f"控制器执行确定性检查：{tool}", "args": args, "source": "controller",
                "agent": agent.get("id"), "role": agent.get("role")}
        started = time.perf_counter()
        try:
            observation = self.apply(tool, args)
            step.update(status="completed", observation=observation)
        except Exception as error:
            step.update(status="failed", observation={"error": str(error), "instruction": "按错误修正或跳过"})
        step["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        self.run["steps"].append(step)
        self.event("tool_result", f"{tool} 控制器执行完成" if step["status"] == "completed" else f"{tool} 控制器执行失败",
                   status=step["status"], level="info" if step["status"] == "completed" else "warning", step=step)
        return step

    def orchestrate(self):
        workflow_state = self.run.get("workflow")
        if not workflow_state:
            model, source = self.resolve_model("planner")
            agent = {"id": "g0", "role": "planner", "goal": "规划工作流", "status": "running",
                     "model": model, "model_source": source, "steps": []}
            self.run["agents"].append(agent)
            self.run["active_agent"] = agent
            self.controller_step("get_memory_context", {})
            workflow_state = workflow_module.plan(self)
            agent["status"] = "completed"
            agent["steps"] = [step["id"] for step in self.run["steps"]]
            self.run["active_agent"] = None
            self.save()
        for index, stage in enumerate(workflow_state["stages"]):
            if self.cancelled.is_set():
                self.run["status"] = "cancelled"
                break
            if self.run["status"] in ("completed", "incomplete", "failed", "budget_stopped", "cancelled"):
                break
            model, source = self.resolve_model(stage["role"])
            agent = {"id": f"g{index + 1}", "role": stage["role"], "goal": stage["goal"], "status": "running",
                     "model": model, "model_source": source, "steps": []}
            self.run["agents"].append(agent)
            self.run["active_agent"] = agent
            self.event("stage", f"阶段 {agent['id']}（{stage['role']}）：{stage['goal']}", status="running")
            self._loop(int(stage.get("max_steps", 3)), mark_incomplete=False)
            agent["steps"] = [step["id"] for step in self.run["steps"] if step.get("agent") == agent["id"]]
            agent["status"] = ("completed" if self.run["status"] == "completed" else
                               "failed" if self.run["status"] in ("failed", "budget_stopped") else "done")
            self.event("stage", f"阶段 {agent['id']} 结束：{agent['status']}", status=agent["status"])
            self.save()
        self.run["active_agent"] = None
        if self.run["status"] == "running":
            self.run["status"], self.run["stop_reason"] = "incomplete", "工作流已执行但未形成已复核报告"

    def apply(self, tool, args):
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是对象")
        return registry.dispatch(self, tool, args)

    def _apply_builtin(self, tool, args):
        run = self.run
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是对象")
        if tool == "plan":
            if not isinstance(args.get("feasible"), bool) or not str(args.get("method", "")).strip():
                raise ValueError("计划需要feasible判断和method研究设计")
            run["plan"] = args
            if not args["feasible"]:
                run["status"] = "incomplete"
                run["report"] = {"conclusion": f"现有条件不足以执行：{args.get('limitations','')}。研究设计：{args['method']}", "citations": [], "lesson": ""}
            return args
        if tool == "reflect":
            verified = [e for e in run["experiments"] if e.get("verification")]
            if not verified or args.get("next_action") not in ("revise", "finish"):
                raise ValueError("需要根据已复核实验决定revise或finish")
            item = {**args, "after_experiment": verified[-1]["id"], "hypothesis_count": len(run["hypotheses"])}
            run.setdefault("reflections", []).append(item)
            return item
        if tool == "inspect":
            result = tools.inspect_assets(run, self.folder)
            run["evidence_collected"] = True
            return result
        if tool == "literature":
            rows = tools.literature(args.get("query"), timeout=max(1, min(15, run["seconds_limit"] - self.budget.elapsed)))
            for row in rows:
                row["id"] = f"p{len(run['sources'])+1}"
                run["sources"].append(row)
            run["evidence_collected"] = True
            return rows
        if tool == "read":
            asset = next((a for a in run["assets"] if a["id"] == args.get("asset_id")), None)
            if asset is None:
                raise ValueError("材料不存在，先inspect")
            run["evidence_collected"] = True
            if asset.get("kind") == "document":
                payload = (self.folder / "inputs" / asset["id"]).read_bytes()
                extracted = documents.extract(asset["name"], payload, max_chars=12000)
                return {"id": asset["id"], "name": asset["name"], "text": extracted["text"],
                        "note": extracted["note"], "truncated": extracted["truncated"]}
            return {"id": asset["id"], "text": (self.folder / "inputs" / asset["id"]).read_text(encoding="utf-8-sig")[:12000]}
        if tool == "hypothesis":
            if not run.get("evidence_collected"):
                raise ValueError("先检查材料或收集证据")
            if not all(isinstance(args.get(k), str) and args[k].strip() for k in ("statement", "prediction", "controls", "metric")):
                raise ValueError("假设需要statement、prediction、controls、metric")
            threshold = args.get("threshold")
            if args.get("relation") not in ("gt", "ge", "lt", "le") or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
                raise ValueError("需要合法的比较关系和有限数值阈值")
            item = {k: args[k] for k in ("statement", "prediction", "controls", "metric", "relation", "threshold")}
            item.update(id=f"h{len(run['hypotheses'])+1}", registered_at=storage.now())
            run["hypotheses"].append(item)
            return item
        if tool == "experiment":
            hypothesis = next((h for h in run["hypotheses"] if h["id"] == args.get("hypothesis_id")), None)
            if hypothesis is None:
                raise ValueError("先注册有效假设，再引用其hypothesis_id")
            item = {"id": f"x{len(run['experiments'])+1}", "hypothesis_id": hypothesis["id"], "code": args.get("code"), "status": "running"}
            run["experiments"].append(item)
            try:
                item["result"] = tools.experiment(item["code"], run, self.folder, item["id"], "explore", run["seconds_limit"] - self.budget.elapsed, run["seed"] + len(run["experiments"]))
                item["check"] = tools.check(hypothesis, item["result"])
                item["status"] = "completed"
            except Exception as error:
                item.update(status="failed", error=str(error))
                raise
            return {"id": item["id"], "result": item["result"], "check": item["check"]}
        if tool == "verify":
            item = next((e for e in run["experiments"] if e["id"] == args.get("experiment_id") and e["status"] == "completed"), None)
            if item is None or item.get("verification"):
                raise ValueError("需要一个尚未复核的成功实验ID")
            hypothesis = next(h for h in run["hypotheses"] if h["id"] == item["hypothesis_id"])
            result = tools.experiment(item["code"], run, self.folder, item["id"], "verify", run["seconds_limit"] - self.budget.elapsed, secrets.randbelow(2**31))
            item["verification"] = {"result": result, "check": tools.check(hypothesis, result),
                                     "protocol": "同一冻结代码，新隐藏种子，预注册数值判据；不等于统计显著性或跨领域有效性"}
            return {"id": item["id"], **item["verification"]}
        if tool == "finish":
            valid = {a["id"] for a in run["assets"]} | {p["id"] for p in run["sources"]} | {h["id"] for h in run["hypotheses"]} | {e["id"] for e in run["experiments"]}
            citations = args.get("citations", [])
            if not isinstance(citations, list) or any(c not in valid for c in citations):
                raise ValueError("报告只能引用本轮已存在的证据ID")
            verified = [e for e in run["experiments"] if e.get("verification") and e["id"] in citations]
            run["report"] = {"conclusion": str(args.get("conclusion", "")), "citations": citations, "lesson": str(args.get("lesson", ""))}
            run["status"] = "completed" if verified else "incomplete"
            if verified and str(args.get("lesson", "")).strip():
                fact = facts.remember(run["id"], "method_lesson", run["question"][:80],
                                      str(args["lesson"])[:400], tags=["research", run.get("model_source", "local")],
                                      confidence=0.7, source_message=run["question"])
                run["report"]["fact_id"] = fact["id"]
            return {"status": run["status"], "verified_experiments": len(verified), "note": "有复核记录不代表假设成立，须查看固定判据结果。"}
        raise ValueError(f"未开放工具 {tool}；请从可用工具选择")

    def write_report(self):
        report = self.run.get("report", {})
        lines = [f"# {self.run['question']}", f"\n状态：{self.run['status']}。", "\n## Agent分析", report.get("conclusion") or self.run.get("stop_reason") or self.run.get("error") or "尚未形成完整结论。",
                 "\n## 预注册假设与固定判据"]
        for h in self.run["hypotheses"]:
            lines.append(f"- {h['id']}: {h['statement']}；{h['metric']} {h['relation']} {h['threshold']}；控制：{h['controls']}")
        lines.append("\n## 实际实验与复核")
        for e in self.run["experiments"]:
            lines.append(f"- {e['id']} / {e['hypothesis_id']}：{json.dumps({k: e.get(k) for k in ('status','result','check','verification','error')}, ensure_ascii=False)}")
        lines.append("\n## 文献与材料")
        lines.extend(f"- {p['id']}: {p['title']} {p.get('url','')}（{p['evidence_scope']}）" for p in self.run["sources"])
        lines.extend(f"- {a['id']}: {a['name']} SHA256={a['sha256']}" for a in self.run["assets"])
        lines.extend(["\n## 成本", json.dumps(self.budget.snapshot(), ensure_ascii=False),
                      "\n## 局限", "本地可计算任务原型；模拟结果受生成机制限制。独立随机重跑不等于统计显著性，也不能证明任意真实世界结论。文本输出为模型解释，数字以实验记录为准。",
                      "\n## 复现", "experiments/<id>/explore/含冻结code.py、input.json和result.json；source/为当时源码。使用匹配环境运行：python source/researchloop/engine/worker.py experiments/<id>/explore/input.json。复制到新路径时先将input.json中的材料路径改到本包inputs/。"])
        (self.folder / "report.md").write_text("\n\n".join(lines), encoding="utf-8")


def create(body):
    question = str(body.get("question", "")).strip()
    if not 8 <= len(question) <= 3000:
        raise ValueError("请写8至3000字符的研究问题")
    tokens, seconds, steps = int(body.get("token_limit", 40000)), int(body.get("seconds_limit", 600)), int(body.get("max_steps", 12))
    if not (8000 <= tokens <= 150000 and 30 <= seconds <= 1800 and 4 <= steps <= 20):
        raise ValueError("研究预算超出允许范围")
    cfg = storage.settings()
    source = str(body.get("model_source") or cfg.get("model_source") or "local").strip().lower()
    if source not in ("local", "cloud"):
        raise ValueError("模型来源只能是 local 或 cloud")
    provider = None
    if source == "cloud":
        cloud = cfg["cloud"]
        missing = [label for key, label in (("base_url", "服务地址"), ("model", "模型名称"), ("api_key", "API Key"))
                   if not str(cloud.get(key) or "").strip()]
        if not cloud.get("enabled") or missing:
            raise ValueError("云端模型未配置完整：" + "、".join(missing or ["启用开关"]) + "；请先在模型设置中填写并保存")
        model = str(cloud["model"]).strip()
        provider = urllib.parse.urlparse(cloudllm.normalise(cloud["base_url"])).hostname
    else:
        model = str(body.get("model") or cfg["model"])
        if not model or "cloud" in model.lower() or len(model) > 120:
            raise ValueError("只允许已有的本地Ollama模型")
    mode = str(body.get("workflow_mode") or "react").strip().lower()
    if mode not in ("react", "plan"):
        raise ValueError("workflow_mode 只能是 react 或 plan")
    role_models = body.get("roles") if isinstance(body.get("roles"), dict) else {}
    identifier = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    folder = storage.RUNS / identifier
    assets = body.get("assets", [])
    if not isinstance(assets, list) or len(assets) > 8:
        raise ValueError("最多附加 8 份材料")
    prepared = []
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise ValueError("材料格式无效")
        name = Path(str(asset.get("name", ""))).name
        suffix = Path(name).suffix.lower()
        encoded = asset.get("content_base64")
        if encoded:
            try:
                payload = base64.b64decode(str(encoded), validate=True)
            except Exception as error:
                raise ValueError(f"{name} 的 base64 内容无效") from error
            kind = "document"
            if suffix not in documents.SUFFIXES:
                raise ValueError(f"不支持的文档类型：{suffix or name}")
            limit = 20 * 1024 * 1024
        else:
            payload = str(asset.get("content", "")).encode()
            kind = "text"
            if suffix not in (".txt", ".md", ".csv", ".json"):
                raise ValueError("仅接受 TXT/MD/CSV/JSON 文本材料，或 PDF/DOCX/PPTX 文档")
            limit = 500000
        if len(payload) > limit:
            raise ValueError(f"{name} 超过 {limit // 1024}KB")
        prepared.append({"id": f"a{index+1}", "name": name, "kind": kind, "bytes": len(payload),
                         "sha256": hashlib.sha256(payload).hexdigest(), "payload": payload})
    (folder / "inputs").mkdir(parents=True)
    for asset in prepared:
        (folder / "inputs" / asset["id"]).write_bytes(asset.pop("payload"))
    state = memory_state()
    experience = next((v for v in state["versions"] if v["version"] == state["active"]), None) if body.get("use_memory") else None
    manifest = {}
    for module in sorted((storage.ROOT / "researchloop").rglob("*.py")):
        relative = module.relative_to(storage.ROOT)
        target = folder / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(module.read_bytes())
        manifest[str(relative)] = hashlib.sha256(module.read_bytes()).hexdigest()
    storage.write_json(folder / "source-manifest.json", manifest)
    run = {"id": identifier, "session_id": identifier, "version": "0.4.0", "domain": "research", "task": question[:32], "question": question,
           "mode": "agent", "model": model, "created_at": storage.now(), "status": "queued",
           "token_limit": tokens, "seconds_limit": seconds, "max_steps": steps, "seed": int(body.get("seed", 42)),
           "assets": prepared, "sources": [], "hypotheses": [], "experiments": [], "steps": [], "events": [], "calls": [],
           "experience": experience, "memory_version": experience["version"] if experience else 0,
           "model_source": source, "provider": provider, "cloud_enabled": source == "cloud",
           "workflow_mode": mode, "roles": role_models, "workflow": None, "agents": [], "active_agent": None,
           "source_hash": storage.fingerprint(manifest)}
    storage.write_json(folder / "run.json", run)
    return ResearchEngine(run, folder)


def _builtin_handler(name):
    """Built-in tools register through the same interface as third-party tools."""
    def handler(engine, args):
        return engine._apply_builtin(name, args)
    handler.__qualname__ = f"builtin:{name}"
    return handler


for _name in ("plan", "inspect", "literature", "read", "hypothesis", "experiment", "verify", "reflect", "finish"):
    registry.register_tool(_name, _builtin_handler(_name))
