"""A model decides the next research action from observations, not task names."""
import hashlib
import json
import base64
import copy
import math
import re
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
from researchloop import facts, memory, reliability, llmjson
from researchloop.ollama import Budget, BudgetError
from . import tools
from . import protocol, routing, history

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
自主选择下一项工具；严格返回 {"tool":"工具名","reason":"选择依据","args":{工具参数}}，不要把参数直接放在顶层。定性和文献问题可以采用证据分析，无须强行运行数值实验。
异常或玩笑式提问也应先核实前提、语境或出处，不能在取得证据前直接认定它是梗或无研究价值。工作流目标也是待检验设想，不是已知事实。
记忆和错误日志不是研究证据；finish.citations只能从allowed_citation_ids选择，没有证据时用[]明确报告未完成。401/403应换来源或修复配置，不能只改关键词重复访问同一服务。"""

ACTION_SCHEMA = {"type": "object", "properties": {
    "tool": {"type": "string", "enum": [t["name"] for t in tools.TOOLS]}, "reason": {"type": "string"},
    "args": {"type": "object", "properties": {
        **{key: {"type": "string"} for key in ("query", "asset_id", "statement", "prediction", "controls", "metric", "hypothesis_id", "experiment_id", "code", "conclusion", "lesson")},
        "relation": {"type": "string", "enum": ["gt", "ge", "lt", "le"]}, "threshold": {"type": "number"},
        "citations": {"type": "array", "items": {"type": "string"}}}}},
    "required": ["tool", "reason", "args"]}

ARGUMENTS = {
    "plan": ({"feasible": {"type": "boolean"}, **{k: {"type": "string"} for k in ("method", "limitations", "literature_query")},
              "research_kind": {"type": "string", "enum": ["quantitative", "simulation", "qualitative", "literature", "mixed"]}},
             ["feasible", "method", "limitations", "literature_query"]),
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
    "platform_search": ({"platform": {"type": "string", "enum": ["bilibili", "xiaohongshu", "tieba",
                                                                    "weibo", "douyin", "zhihu"]},
                         "query": {"type": "string"}, "reason": {"type": "string"},
                         "limit": {"type": "integer"}}, ["platform", "query", "reason"]),
    "get_memory_context": ({"query": {"type": "string"}}, []),
    "design_workflow": ({"reason": {"type": "string"},
                          "stages": {"type": "array", "items": {
                              "type": "object",
                              "properties": {"role": {"type": "string", "enum": list(roles.ROLES)},
                                             "goal": {"type": "string"}, "why": {"type": "string"},
                                             "max_steps": {"type": "integer"}},
                              "required": ["role", "goal", "why"]}}},
                         ["reason", "stages"]),
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
        self.pause_requested = threading.Event()
        if run.get("restore_budget"):
            self.budget.restore(run.get("budget") or {}, run.get("calls", [])[run.get("budget_call_start", 0):])
            run.pop("restore_budget", None)
        self.decide = decide or self.agent_step
        self.turn_id = ((run.get("turns") or [{}])[-1] or {}).get("id") or "t1"

    def save(self):
        run = self.run
        run["updated_at"], run["budget"] = storage.now(), self.budget.snapshot()
        run["budget"]["tool_seconds"] = round(sum(s.get("elapsed_seconds", 0)
            for s in run["steps"][run.get("branch_step_start", 0):] if s.get("turn", "t1") == self.turn_id), 3)
        turn = next((item for item in run.get("turns", []) if item["id"] == self.turn_id), None)
        if turn is not None:
            turn["status"], turn["budget"] = run["status"], dict(run["budget"])
        storage.write_json(self.folder / "run.json", run)
        db.sync_run(run)

    def question_text(self):
        return self.run.get("current_question") or self.run.get("question", "")

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

    def guard_pause(self):
        if self.pause_requested.is_set():
            raise history.Paused()

    def reserve_for(self, system, text, max_output=1500):
        """Conservative reservation: CJK prompts cost roughly one token per 2 characters."""
        chars = len(system or "") + len(text or "")
        return max(2048, chars // 2) + int(max_output)

    def complete(self, system, prompt, schema, purpose="research_decision", role=None):
        """One admitted request, one trace record, including failed attempts."""
        cached = self.run.pop("pending_response", None)
        replay = self.run.pop("replay_response", False)
        if cached and replay and (cached.get("purpose") == purpose or
                                  cached.get("purpose", "").startswith("agent_step") and purpose.startswith("agent_step")):
            return cached["parsed"], cached["call"]
        self.guard_pause()
        route = routing.choose(self, purpose, prompt)
        model, source = route["model"], route["model_source"]
        text = json.dumps(prompt, ensure_ascii=False) if isinstance(prompt, (dict, list)) else str(prompt)
        reserve = self.reserve_for(system, text)
        self.last_reserve = reserve
        self.budget.reserve(reserve)
        node = history.begin(self, "model_call", model=model, source=source, route=route,
                             role=role or route["role"], agent=route["agent"], purpose=purpose,
                             input={"system": system, "prompt": prompt, "schema": schema})
        self.event("planning", f"{'云端' if source == 'cloud' else '本地'}模型正在生成 {purpose} 内容", status="running")
        started = time.perf_counter()
        item = {"purpose": purpose, "model": model, "source": "cloud" if source == "cloud" else "ollama",
                "agent": route["agent"], "role": role or route["role"], "turn": self.turn_id,
                "node_id": node["id"], "route": route}
        parsed = None
        try:
            if source == "cloud":
                def capture(value):
                    item.update(value)
                client = cloudllm.Cloud(storage.settings()["cloud"], model, self.budget, on_call=capture)
                parsed, _ = client.call(system, text, schema, purpose, self.run.get("seed"), reserve=reserve)
            else:
                body = {"model": model, "stream": False, "think": False, "format": schema,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                        "options": {"temperature": 0, "num_ctx": 6144, "num_predict": 1500}, "keep_alive": "20m"}
                item["request"] = body
                request = urllib.request.Request(storage.settings()["ollama_url"] + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=max(1, min(150, self.run["seconds_limit"] - self.budget.elapsed))) as response:
                    result = json.load(response)
                item["response"] = result
                if not all(isinstance(result.get(k), int) for k in ("prompt_eval_count", "eval_count")):
                    raise ValueError("模型未返回usage")
                item.update(input_tokens=result["prompt_eval_count"], output_tokens=result["eval_count"],
                            cached_tokens=result.get("prompt_eval_cached_count", 0),
                            context_tokens=result["prompt_eval_count"], status="ok")
                parsed = llmjson.parse(result.get("message", {}).get("content", ""))
            self.run["pending_response"] = {"purpose": purpose, "parsed": parsed, "call": dict(item)}
            return parsed, item
        except Exception as error:
            item.update(status="error", error=str(error))
            failure = reliability.classify(error, "llm:" + source)
            reliability.record(self.run, failure, storage.settings(), self.turn_id)
            self.run["decision_feedback"] = {"model_error": str(error), "purpose": purpose}
            if source != "cloud" and "input_tokens" not in item:
                item["unknown_reserved"] = 7644
            raise
        finally:
            item["wall_seconds"] = round(time.perf_counter() - started, 3)
            if source != "cloud":
                self.budget.record(item)
            self.run["calls"].append(item)
            history.settle(self, node, status="completed" if item.get("status") == "ok" else "failed",
                           output=parsed if item.get("status") == "ok" else {"error": item.get("error"), "response": item.get("response")},
                           usage={key: item.get(key, 0) for key in ("input_tokens", "output_tokens", "unknown_reserved", "cached_tokens")},
                           elapsed_seconds=item["wall_seconds"])
            self.event("model_call", "模型调用完成" if item.get("status") == "ok" else "模型调用失败",
                       status="completed" if item.get("status") == "ok" else "failed", call=item)

    def ask(self):
        tool, automatic_args = self.next_protocol_action()
        if automatic_args is not None:
            return {"tool": tool, "reason": "按研究协议执行确定性证据检查或独立验证，不额外消耗模型", "args": automatic_args, "source": "controller"}
        properties, required = ARGUMENTS[tool]
        schema = {"type": "object", "properties": properties, "required": required}
        observations = [{"tool": s["tool"], "status": s["status"], "observation": s["observation"],
                         "failed_code": s["args"].get("code", "") if s["status"] == "failed" else ""} for s in self.run["steps"][-3:]]
        prompt = {"question": self.question_text(), "current_action": tool,
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
        system = SYSTEM.replace('自主选择下一项工具；严格返回 {"tool":"工具名","reason":"选择依据","args":{工具参数}}，不要把参数直接放在顶层。', "") + f"\n当前控制器已检查前置条件，进入{tool}。只返回该工具参数，不要选择其他工具。"
        if tool == "finish":
            verified = [e for e in self.run["experiments"] if e.get("verification")]
            if verified:
                ids = "、".join(e["id"] for e in verified)
                hypotheses = "、".join(sorted({e["hypothesis_id"] for e in verified}))
                system += f"\n本次citations必须包含已复核实验 {ids} 及其假设 {hypotheses}；只引用文献会使报告判为未完成。"
        parsed, item = self.complete(system, prompt, schema)
        self.run.pop("pending_response", None)
        return {"tool": tool, "reason": f"按通用研究协议进入{tool}，研究内容由{item.get('source') == 'cloud' and '云端' or '本地'}模型生成",
                "args": parsed, "source": item.get("source", "ollama")}

    def tool_catalog(self, names):
        catalog = []
        for name in names:
            entry = registry.get(name) or {}
            properties, required = ARGUMENTS.get(name, ({}, []))
            properties = copy.deepcopy(properties)
            if name == "platform_search":
                properties["platform"]["enum"] = self.available_platforms()
            if name in ("finish", "remember_fact"):
                field = "citations" if name == "finish" else "evidence_ids"
                properties[field] = {"type": "array", "items": {"type": "string"}}
                identifiers = self.evidence_ids() if name == "finish" else self.memory_evidence_ids()
                if identifiers:
                    properties[field]["items"]["enum"] = identifiers
                else:
                    properties[field]["maxItems"] = 0
                if name == "remember_fact":
                    properties[field]["minItems"] = 1
                    required = [*required, "evidence_ids"]
            catalog.append({"tool": name, "description": entry.get("description", ""),
                            "args_schema": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}})
        return catalog

    def evidence_ids(self):
        return sorted({row["id"] for group in ("assets", "sources", "hypotheses", "experiments")
                       for row in self.run.get(group, [])})

    def memory_evidence_ids(self):
        return sorted({row["id"] for group in ("assets", "sources") for row in self.run.get(group, [])} |
                      {row["id"] for row in self.run.get("experiments", []) if row.get("verification")})

    def agent_step(self):
        """The model chooses the next tool itself; the controller only gates validity."""
        run = self.run
        options = self.available_tools()
        schema = protocol.action_schema(self.tool_catalog(options))
        def compact(value, limit=600):
            raw = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value or "")
            return raw[:limit]

        observations = [{"tool": step["tool"], "status": step["status"],
                         "observation": compact(step.get("observation"), 500)}
                        for step in run["steps"][-3:]]
        prompt = {"question": self.question_text(), "branch_note": run.get("branch_note"),
                  "branch_context": run.get("branch_context"), "decision_feedback": run.get("decision_feedback"),
                  "recovery": run.get("recovery_feedback"), "allowed_citation_ids": self.evidence_ids(),
                  "citation_policy": "记忆ID不能引用。无证据可用时必须 citations=[]，说明尚未核实并以未完成状态收尾。", "plan": run.get("plan"),
                  "research_kind": run.get("research_kind"), "assets": run["assets"][:4],
                  "documents": run.get("documents", [])[:6], "hypotheses": run["hypotheses"][:6],
                  "experiments": [{key: experiment.get(key) for key in
                                   ("id", "hypothesis_id", "status", "check", "verification", "error")}
                                  for experiment in run["experiments"][-6:]],
                  "evidence": [{key: source.get(key) for key in ("id", "title", "evidence_scope")}
                               for source in run["sources"][-12:]],
                  "evidence_count": len(run["sources"]),
                  "memory": {"text": str((run.get("memory_context") or {}).get("text") or "")[:1200]}
                  if run.get("memory_context") else None,
                  "experience": compact(run.get("experience"), 1200) if run.get("experience") else None,
                  "observations": observations, "budget": self.budget.snapshot(),
                  "allowed_tools": self.tool_catalog(options),
                  "instruction": "你决定下一步：从allowed_tools选一个tool并给出符合其args_schema的args。"
                                 "证据不足可继续取证；实验完成后应复核。有证据时引用合法ID形成结论。"
                                 "若证据或服务不足，允许finish(citations=[])如实报告尚未核实和未完成；不要用模拟或计数替代真实证据。"
                                 "如果需要多阶段编排，先调用design_workflow并写明每个阶段的设计理由。"}
        correction = ""
        for attempt in range(2):
            try:
                parsed, item = self.complete(SYSTEM + correction, prompt, schema,
                                             purpose="agent_step" if attempt == 0 else "agent_step_retry")
            except (BudgetError, history.Paused):
                raise
            except Exception as error:
                if attempt == 1:
                    raise RuntimeError(f"模型未返回可解析的决策：{error}") from error
                failure = reliability.classify(error)
                if failure.get("retryable"):
                    self.wait_for_recovery(failure["retry_after"])
                prompt["repair"] = {"error": str(error), "instruction": "重新返回满足schema的完整JSON对象"}
                correction = "\n上次输出不是可解析的 JSON；只返回一个 JSON 对象，字段为 tool、reason、args。"
                continue
            self.run.pop("pending_response", None)
            issues = []
            tool = str(parsed.get("tool") or "").strip() if isinstance(parsed, dict) else ""
            args = parsed.get("args") if isinstance(parsed, dict) else None
            try:
                if tool not in options:
                    raise ValueError(f"tool={tool!r} 不在可用集合 {options}")
                if not isinstance(parsed.get("reason"), str):
                    raise ValueError("reason 必须是字符串")
                entry = next(row for row in self.tool_catalog(options) if row["tool"] == tool)
                protocol.validate(args, entry["args_schema"])
            except ValueError as error:
                issues.append(str(error))
            if not issues:
                self.run.pop("decision_feedback", None)
                break
            self.run["decision_feedback"] = {"invalid_response": parsed, "errors": issues}
            if tool == "finish":
                self.run["draft_report"] = {"args": args, "errors": issues, "verified": False}
            check = history.begin(self, "decision_error", tool="validate_decision", source="controller",
                                  input=parsed, parent_id=item.get("node_id"), reason="校验模型选择与工具参数")
            history.settle(self, check, status="failed", output={"error": "；".join(issues)}, elapsed_seconds=0)
            self.guard_pause()
            if attempt == 1:
                raise ValueError(f"模型两次返回无效决策：{'；'.join(issues)}")
            prompt["repair"] = {"invalid_response": parsed, "errors": issues,
                                "instruction": "根据工具schema修复整个tool/reason/args对象，不得编造缺失研究证据"}
            correction = "\n修复上次无效决策。必须返回 tool/reason/args 外壳及所选工具的完整参数。"
        return {"tool": tool, "reason": str(parsed.get("reason") or "模型自主选择"),
                "args": args, "source": item.get("source", "ollama")}

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
        reliability.check_search_config(run, storage.settings(), self.turn_id)
        agent = run.get("active_agent") or {}
        if run.get("workflow_mode") == "plan" and not run.get("workflow") and agent.get("id") == "g0":
            return ["get_memory_context", "inspect", "design_workflow"]
        if self.in_wrap_up():
            return ["finish"]
        names = []

        def add(*items):
            for item in items:
                if item not in names:
                    names.append(item)

        add("get_memory_context")
        if not run.get("evidence_collected"):
            add("inspect")
        add("literature", "platform_search")
        if storage.settings()["web"].get("enabled"):
            add("web_search", "web_fetch")
        assets = run.get("assets", [])
        if assets:
            suffixes = {Path(asset["name"]).suffix.lower() for asset in assets}
            if ".pdf" in suffixes:
                add("read_pdf")
            if ".docx" in suffixes:
                add("read_docx")
            if ".pptx" in suffixes:
                add("read_pptx")
            if any(asset.get("kind") == "text" for asset in assets):
                add("read")
        if run.get("workflow_mode") == "plan" and not run.get("workflow"):
            add("design_workflow")
        if not run.get("plan"):
            add("plan")
        else:
            add("hypothesis", "reflect")
            if run.get("hypotheses"):
                add("experiment")
            if any(e.get("status") == "completed" and not e.get("verification") for e in run.get("experiments", [])):
                add("verify")
        if run.get("sources") or run.get("assets") or any(e.get("verification") for e in run.get("experiments", [])):
            add("remember_fact")
        if run.get("plan") or not (run.get("sources") or run.get("assets") or run.get("experiments")):
            add("finish")
        role = (agent.get("role") or "") if isinstance(agent, dict) else ""
        if role:
            permitted = set(roles.tools_for(role))
            names = [name for name in names if name in permitted]
            if not names:
                names = ["plan"] if not run.get("plan") else ["finish"]
        cfg = storage.settings()
        names = [name for name in names if name not in ("web_search", "literature") or
                 not reliability.blocked(run, reliability.resource(name, {}, cfg), cfg)]
        if "platform_search" in names and not self.available_platforms():
            names.remove("platform_search")
        return names

    def wrap_up_budget(self):
        """Reserve enough room to write one evidence-gap report, not for exploration."""
        reserve_tokens = max(9000, int(self.run["token_limit"] * 0.15))
        reserve_seconds = max(30, min(90, int(self.run["seconds_limit"] * 0.15)))
        return reserve_tokens, reserve_seconds

    def in_wrap_up(self):
        if not self.run.get("steps"):
            return False
        tokens, seconds = self.wrap_up_budget()
        remaining = self.budget.token_limit - self.budget.used
        # The next agent_step prompt is about as large as the previous one.
        needed = max(tokens, getattr(self, "last_reserve", 0) + 2000)
        return remaining <= needed or self.budget.seconds_limit - self.budget.elapsed <= seconds

    def infer_kind(self):
        run = self.run
        scopes = [str(source.get("evidence_scope") or "") for source in run.get("sources", [])]
        if run.get("documents") or any("网页正文" in scope or "网页搜索摘要" in scope for scope in scopes):
            return "mixed"
        if run.get("sources"):
            return "literature"
        if run.get("experiments"):
            return "quantitative"
        return "quantitative"  # nothing collected: keep the strict "incomplete" grade

    def available_platforms(self):
        cfg = storage.settings()
        reliability.check_search_config(self.run, cfg, self.turn_id)
        return [name for name in ARGUMENTS["platform_search"][0]["platform"]["enum"]
                if (cfg["web"].get("enabled") or name in ("bilibili", "tieba")) and
                not reliability.blocked(self.run, reliability.resource("platform_search", {"platform": name}, cfg), cfg)]

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
        except history.Paused:
            self.run["status"] = "paused"
            self.run["paused_at"] = storage.now()
        except BudgetError as error:
            self.run["status"], self.run["stop_reason"] = "budget_stopped", str(error)
        except Exception as error:
            self.run["status"], self.run["error"] = "failed", str(error)
        finally:
            if self.run["status"] == "running":
                self.run["status"] = "incomplete"
            if self.run["status"] == "paused":
                self.event("paused", "研究已暂停，可从本页继续", status="paused")
                self.run["restore_budget"] = True
                self.save()
                return self.run
            self.run["finished_at"] = storage.now()
            turn = next((item for item in self.run.get("turns", []) if item["id"] == self.turn_id), None)
            report_now = self.run.get("report") or {}
            if not str(report_now.get("conclusion") or "").strip():
                report_now = {**report_now, **self.failure_report()}
                self.run["report"] = report_now
                if self.run["status"] in ("running", "incomplete") and (
                        self.run.get("sources") or self.run.get("documents") or self.run.get("experiments")):
                    self.run["status"] = "inconclusive"
                    self.run["stop_reason"] = self.run.get("stop_reason") or "达到步骤或预算上限，已生成证据缺口报告"
            if turn is not None:
                turn["status"], turn["finished_at"] = self.run["status"], self.run["finished_at"]
                turn["conclusion"] = str(report_now.get("conclusion", ""))[:1200]
            self.write_report()
            if self.run["status"] == "completed" and self.run.get("memory_mode") != "disabled" and report_now.get("lesson"):
                try:
                    saved = publish(self.run)
                    self.run["memory_saved_version"] = saved["version"]
                except Exception as error:
                    self.run["memory_save_error"] = str(error)
            self.event("completed", f"研究结束：{self.run['status']}", status=self.run["status"])
        return self.run

    def wait_for_recovery(self, delay):
        if delay + 2 >= self.budget.seconds_limit - self.budget.elapsed:
            raise BudgetError("剩余时间不足以等待服务重试，保留已有结果")
        until = time.monotonic() + delay
        while time.monotonic() < until:
            self.guard_pause()
            if self.cancelled.is_set():
                self.run["status"] = "cancelled"
                return
            self.pause_requested.wait(min(.1, max(0, until - time.monotonic())))
        self.guard_pause()

    def _turn_steps(self):
        return [step for step in self.run["steps"][self.run.get("branch_step_start", 0):]
                if step.get("turn", "t1") == self.turn_id]

    def _perform_tool(self, decision, retry=None):
        """Only retry read-only transient failures; settle each physical call."""
        tool, args = decision["tool"], decision.get("args", {})
        retry = retry or {}
        attempt = retry.get("attempt", 1)
        retry_of = retry.get("retry_of")
        if retry:
            self.run["pending_tool_retry"] = retry
            delay = max(0, retry.get("retry_at", 0) - time.time())
            if delay:
                self.wait_for_recovery(delay)
        while True:
            self.guard_pause()
            if self.cancelled.is_set():
                self.run["status"] = "cancelled"
                return
            if len(self._turn_steps()) >= self.run["max_steps"]:
                self.run["status"], self.run["stop_reason"] = "incomplete", "达到研究步骤上限，保留已有证据"
                return
            if self.budget.elapsed >= self.budget.seconds_limit - 1:
                raise BudgetError("时间预算耗尽，不再启动工具")
            self.run.pop("pending_tool_retry", None)
            self.run.pop("pending_decision", None)
            self.run.pop("pending_response", None)
            agent = self.run.get("active_agent") or {}
            step = {"id": f"s{len(self.run['steps']) + 1}", "at": storage.now(), "tool": tool,
                    "reason": decision.get("reason", ""), "args": args, "source": decision.get("source", "injected"),
                    "agent": agent.get("id"), "role": agent.get("role"), "turn": self.turn_id,
                    "attempt": attempt, "retry_of": retry_of}
            node = history.begin(self, "tool_call", tool=tool, input=args, reason=step["reason"], source="local",
                                 role=step["role"], agent=step["agent"], attempt=attempt, retry_of=retry_of,
                                 parent_id=next((n["id"] for n in reversed(self.run.get("trace", [])) if n["kind"] == "model_call"), None))
            step["node_id"] = node["id"]
            started = time.perf_counter()
            next_retry = None
            try:
                observation = self.apply(tool, args)
                step.update(status="completed", observation=observation)
                self.run.pop("recovery_feedback", None)
            except Exception as error:
                info = reliability.classify(error, reliability.resource(tool, args, storage.settings()))
                delay = info.get("retry_after", .25)
                can_retry = (tool in reliability.READ_TOOLS and info.get("retryable") and attempt < 2 and
                             delay <= 30 and self.budget.seconds_limit - self.budget.elapsed > delay + 2 and
                             len(self._turn_steps()) + 1 < self.run["max_steps"])
                if info.get("retryable") and attempt >= 2:
                    info["blocked"] = True
                    info["message"] += " 已达到两次尝试上限，本轮改用其他来源。"
                info["action"] = "retry" if can_retry else "switch_source" if info.get("blocked") else "repair_decision"
                if info.get("retryable") and not can_retry and attempt < 2:
                    info["action"] = "defer"
                    info["message"] += " 当前预算或等待上限不允许立即重试。"
                reliability.record(self.run, info, storage.settings(), self.turn_id)
                step.update(status="failed", observation={"error": str(error), "recovery": info,
                            "instruction": info["message"] + "；只能使用真实证据，必要时明确未完成。"})
                if can_retry:
                    next_retry = {"decision": decision, "attempt": attempt + 1, "retry_of": node["id"],
                                  "retry_at": time.time() + delay}
                    self.run["pending_tool_retry"] = next_retry
            step["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            self.run["steps"].append(step)
            history.settle(self, node, status=step["status"], output=step["observation"], elapsed_seconds=step["elapsed_seconds"])
            self.event("tool_result", f"{tool} 第 {attempt} 次尝试：{step['status']}", status=step["status"], step=step)
            if not next_retry:
                return step
            self.event("recovery", f"{tool} 暂时失败，等待后进行第 2 次尝试", status="running", retry=next_retry)
            self.wait_for_recovery(max(0, next_retry["retry_at"] - time.time()))
            if self.run["status"] == "cancelled":
                return step
            attempt, retry_of = next_retry["attempt"], next_retry["retry_of"]

    def _loop(self, limit, mark_incomplete=True):
        allowed = self.allowed_tools()
        agent_id = (self.run.get("active_agent") or {}).get("id")
        extension_key = f"{self.turn_id}:{agent_id or 'react'}"
        while True:
            self.guard_pause()
            if self.cancelled.is_set():
                self.run["status"] = "cancelled"
                return
            steps = self._turn_steps()
            if len(steps) >= self.run["max_steps"]:
                self.run["status"], self.run["stop_reason"] = "incomplete", "达到研究步骤上限，保留已有证据"
                return
            scoped = [step for step in steps if step.get("agent") == agent_id]
            extensions = self.run.get("recovery_extensions", {}).get(extension_key, 0)
            pending_retry = self.run.get("pending_tool_retry")
            if len(scoped) >= int(limit) + extensions and not pending_retry:
                last = scoped[-1] if scoped else {}
                observation = last.get("observation")
                recovery = observation.get("recovery", {}) if isinstance(observation, dict) else {}
                if (last.get("status") == "failed" and recovery.get("repairable") and extensions < 1 and
                        self.budget.used + 7644 <= self.budget.token_limit and self.budget.elapsed + 2 < self.budget.seconds_limit):
                    self.run.setdefault("recovery_extensions", {})[extension_key] = extensions + 1
                    self.event("recovery", "阶段最后一步失败，使用剩余总预算给予一次修正机会", status="running", agent=agent_id)
                else:
                    break
            if self.budget.elapsed >= self.budget.seconds_limit:
                raise BudgetError("研究时间预算耗尽")
            try:
                retry = self.run.get("pending_tool_retry")
                decision = retry["decision"] if retry else self.run.pop("pending_decision", None) or self.decide()
                self.run["pending_decision"] = decision
                self.guard_pause()
            except (BudgetError, history.Paused):
                raise
            except Exception as error:
                info = reliability.classify(error)
                failed = {"id": f"s{len(self.run['steps']) + 1}", "at": storage.now(), "tool": "invalid",
                          "reason": "模型调用或工具选择失败", "args": {}, "source": "controller",
                          "agent": agent_id, "role": (self.run.get("active_agent") or {}).get("role"),
                          "turn": self.turn_id, "status": "failed", "elapsed_seconds": 0,
                          "observation": {"error": str(error), "recovery": info, "instruction": "按错误修正后重新选择工具。"}}
                self.run["steps"].append(failed)
                self.event("tool_result", f"决策失败：{error}", status="failed", step=failed)
                if not info.get("repairable", True):
                    self.run["status"], self.run["stop_reason"] = "incomplete", info["message"]
                    return
                continue
            tool = decision.get("tool")
            if allowed is not None and tool not in allowed:
                self.run.pop("pending_decision", None)
                failed = {"id": f"s{len(self.run['steps'])+1}", "tool": tool, "status": "failed", "turn": self.turn_id,
                          "agent": agent_id, "role": (self.run.get("active_agent") or {}).get("role"), "args": decision.get("args", {}),
                          "elapsed_seconds": 0, "observation": {"error": f"角色不可使用工具 {tool}"}}
                self.run["steps"].append(failed)
                self.event("tool_result", "工具被角色边界拒绝", status="failed", step=failed)
                continue
            self._perform_tool(decision, retry=retry)
            if self.run["status"] in ("completed", "incomplete", "inconclusive", "cancelled"):
                return
            self.guard_pause()
        if mark_incomplete and self.run["status"] == "running":
            self.run["status"], self.run["stop_reason"] = "incomplete", "达到研究步骤上限，保留已有证据"

    def controller_step(self, tool, args):
        """Deterministic controller action that does not consume a model call."""
        index = len(self.run["steps"])
        agent = self.run.get("active_agent") or {}
        step = {"id": f"s{index + 1}", "at": storage.now(), "tool": tool,
                "reason": f"控制器执行确定性检查：{tool}", "args": args, "source": "controller",
                "agent": agent.get("id"), "role": agent.get("role"), "turn": self.turn_id}
        self.guard_pause()
        node = history.begin(self, "tool_call", tool=tool, input=args, reason=step["reason"],
                             source="controller", role=step["role"], agent=step["agent"])
        step["node_id"] = node["id"]
        started = time.perf_counter()
        try:
            observation = self.apply(tool, args)
            step.update(status="completed", observation=observation)
        except Exception as error:
            step.update(status="failed", observation={"error": str(error), "instruction": "按错误修正或跳过"})
        step["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        self.run["steps"].append(step)
        history.settle(self, node, status=step["status"], output=step["observation"], elapsed_seconds=step["elapsed_seconds"])
        self.event("tool_result", f"{tool} 控制器执行完成" if step["status"] == "completed" else f"{tool} 控制器执行失败",
                   status=step["status"], level="info" if step["status"] == "completed" else "warning", step=step)
        return step

    def orchestrate(self):
        def activate(identifier, role, goal, why=""):
            agent = next((a for a in self.run["agents"] if a["id"] == identifier and a.get("turn") == self.turn_id), None)
            if agent is None:
                model, source = self.resolve_model(role)
                agent = {"id": identifier, "role": role, "goal": goal, "why": why, "status": "running",
                         "model": model, "model_source": source, "steps": [], "turn": self.turn_id}
                self.run["agents"].append(agent)
            self.run["active_agent"] = agent
            return agent

        if not self.run.get("workflow"):
            agent = activate("g0", "planner", "设计工作流并说明设计理由")
            if not any(s.get("agent") == "g0" and s.get("turn") == self.turn_id and s["tool"] == "get_memory_context" for s in self.run["steps"]):
                self.controller_step("get_memory_context", {})
            self._loop(4, mark_incomplete=False)
            self.guard_pause()
            if self.run["status"] != "running":
                return
            agent["status"] = "done"
            if not self.run.get("workflow"):
                workflow_state = workflow_module.plan(self)
                self.controller_step("design_workflow", {"reason": workflow_state.get("reason") or "规划器补齐工作流",
                    "stages": workflow_state["stages"]})
        for index, stage in enumerate(self.run["workflow"]["stages"]):
            self.guard_pause()
            if self.cancelled.is_set():
                self.run["status"] = "cancelled"
                break
            if self.run["status"] != "running":
                break
            agent = activate(f"g{index + 1}", stage["role"], stage["goal"], stage.get("why", ""))
            if agent["status"] in ("done", "completed"):
                continue
            skip_reason = ""
            if stage["role"] == "experimenter" and not self.run.get("hypotheses") and not self.run.get("plan"):
                skip_reason = "尚无计划或假设，实验阶段无事可做"
            elif stage["role"] == "verifier" and not any(
                    experiment.get("status") == "completed" and not experiment.get("verification")
                    for experiment in self.run.get("experiments", [])):
                skip_reason = "没有待复核的实验"
            if skip_reason:
                agent["status"] = "skipped"
                self.event("stage", f"阶段 {agent['id']}（{stage['role']}）跳过：{skip_reason}",
                           status="skipped", level="warning")
                self.save()
                continue
            self.event("stage", f"阶段 {agent['id']}（{stage['role']}）：{stage['goal']}", status="running")
            self._loop(int(stage.get("max_steps", 3)), mark_incomplete=False)
            self.guard_pause()
            agent["steps"] = [s["id"] for s in self.run["steps"] if s.get("agent") == agent["id"] and s.get("turn") == self.turn_id]
            agent["status"] = "completed" if self.run["status"] == "completed" else "done"
            self.event("stage", f"阶段 {agent['id']} 结束：{agent['status']}", status=agent["status"])
        self.run["active_agent"] = None
        if self.run["status"] == "running":
            self.run["status"], self.run["stop_reason"] = "incomplete", "研究阶段已结束，仍未形成通过校验的报告"

    def apply(self, tool, args):
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是对象")
        cfg = storage.settings()
        resource = reliability.resource(tool, args, cfg)
        if reliability.blocked(self.run, resource, cfg):
            raise reliability.Failure("该服务当前配置已被停用，未再次发起请求。请换来源或修复配置。",
                                      resource=resource, kind="circuit_open", blocked=True)
        return registry.dispatch(self, tool, args)

    def _apply_builtin(self, tool, args):
        run = self.run
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是对象")
        if tool == "plan":
            if not isinstance(args.get("feasible"), bool) or not str(args.get("method", "")).strip():
                raise ValueError("计划需要feasible判断和method研究设计")
            kind = str(args.get("research_kind") or "quantitative").strip().lower()
            if kind not in ("quantitative", "simulation", "qualitative", "literature", "mixed"):
                raise ValueError("plan的research_kind必须是 quantitative/simulation/qualitative/literature/mixed")
            run["research_kind"] = kind
            run["plan"] = args
            if not args["feasible"]:
                run["status"] = "inconclusive"
                run["report"] = {"conclusion": f"现有条件不足以执行：{args.get('limitations','')}。研究设计：{args['method']}",
                                 "citations": [], "lesson": "",
                                 "evidence_gaps": ["缺少可直接分析的真实数据或材料",
                                                   "问题本身未被实证回答，只完成可行性判断"]}
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
            metric_name = str(args["metric"]).strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,40}", metric_name):
                raise ValueError("metric 必须是字母开头的短英文标识符（≤40字符，可含数字与下划线），不能使用中文句子或标点")
            args["metric"] = metric_name
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
            role = (run.get("active_agent") or {}).get("role")
            if role and role != "reporter" and not self.in_wrap_up():
                raise ValueError("只有报告角色可以结束研究；当前阶段请继续其职责，或让流程进入报告阶段")
            valid = {a["id"] for a in run["assets"]} | {p["id"] for p in run["sources"]} | {h["id"] for h in run["hypotheses"]} | {e["id"] for e in run["experiments"]}
            conclusion_text = str(args.get("conclusion") or "").strip()
            if len(conclusion_text) < 10:
                raise ValueError("finish需要至少10字符的研究结论；未完成时也要写明证据缺口，不能提交空报告")
            citations = args.get("citations", [])
            if not isinstance(citations, list) or any(c not in valid for c in citations):
                run["draft_report"] = {"args": copy.deepcopy(args), "verified": False,
                                       "errors": ["引用不在本轮证据集合"]}
                raise reliability.Failure(f"报告只能引用本轮已存在的证据ID：{sorted(valid)}；记忆不是证据，无可用证据时使用 citations=[]。",
                                          kind="invalid_citations")
            verified = [e for e in run["experiments"] if e.get("verification") and e["id"] in citations]
            kind = run.get("research_kind") or self.infer_kind()
            run["research_kind"] = kind
            substantive = bool(run.get("documents")) or any(
                "正文" in scope or "网页搜索摘要" in scope
                for scope in (source.get("evidence_scope") or "" for source in run["sources"]))
            report = {"conclusion": str(args.get("conclusion", "")), "citations": citations,
                      "lesson": str(args.get("lesson", ""))}

            if kind in ("qualitative", "literature", "mixed"):
                if substantive:
                    report["evidence_scope_note"] = "定性/文献类结论只依据本轮检索到的摘要或正文；未获取平台数据时不做外推。"
                    run["status"] = "completed"
                else:
                    report["evidence_gaps"] = ["检索仅返回元数据或摘要级线索",
                                               "缺少可直接归因于研究对象的正文、平台或受众数据"]
                    run["status"] = "inconclusive"
            else:
                run["status"] = "completed" if verified else "incomplete"
            run["report"] = report
            return {"status": run["status"], "verified_experiments": len(verified),
                    "evidence_gaps": report.get("evidence_gaps", []),
                    "note": "有复核记录不代表假设成立，须查看固定判据结果。"}
        raise ValueError(f"未开放工具 {tool}；请从可用工具选择")

    def failure_report(self):
        errors = []
        for step in self._turn_steps():
            if step.get("status") == "failed":
                info = step.get("observation", {}).get("recovery", {})
                text = info.get("message") or step.get("observation", {}).get("error")
                if text and text not in errors:
                    errors.append(text)
        gaps = errors[-4:]
        if not self.run.get("sources") and not self.run.get("experiments"):
            gaps.append("未取得外部证据或可验证实验结果；记忆和常识推断不作为研究证据。")
        elif not gaps:
            gaps.append(f"已收集 {len(self.run.get('sources', []))} 条来源、"
                        f"{len(self.run.get('experiments', []))} 项实验，但没有形成可引用结论。")
        reason = "；".join(gaps) or self.run.get("error") or self.run.get("stop_reason") or "本轮未形成可引用结论"
        return {"conclusion": "未完成：" + reason, "citations": [], "lesson": "", "evidence_gaps": gaps,
                "recovery": {"reason": reason, "next_steps": ["修复不可用服务的配置，或从调用节点分支改用其他来源", "依据真实证据重新核验问题前提与结论"]}}

    def write_report(self):
        report = self.run.get("report", {})
        run = self.run
        lines = [f"# {self.question_text()}", f"\n状态：{run['status']}。",
                 "\n## Agent分析",
                 report.get("conclusion") or run.get("stop_reason") or run.get("error") or "尚未形成完整结论。"]
        if run.get("workflow"):
            workflow = run["workflow"]
            lines.append(f"\n## 工作流（{workflow.get('source', 'default')}）")
            if workflow.get("reason"):
                lines.append(f"设计理由：{workflow['reason']}")
            for stage in workflow.get("stages", []):
                why = f"；理由：{stage['why']}" if stage.get("why") else ""
                lines.append(f"- {stage['role']}（上限 {stage.get('max_steps', 3)} 步）：{stage['goal']}{why}")
        if run.get("agents"):
            lines.append("\n## 子 Agent")
            for agent in run["agents"]:
                lines.append(f"- {agent.get('id')} / {agent.get('role')}：{agent.get('model')}"
                             f"（{agent.get('model_source')}），{len(agent.get('steps') or [])} 步，{agent.get('status')}")
        lines.append("\n## 预注册假设与固定判据")
        for hypothesis in run["hypotheses"]:
            lines.append(f"- {hypothesis['id']}: {hypothesis['statement']}；{hypothesis['metric']} "
                         f"{hypothesis['relation']} {hypothesis['threshold']}；控制：{hypothesis['controls']}")
        if run["experiments"]:
            lines.append("\n## 实际实验与复核")
            for experiment in run["experiments"]:
                lines.append(f"- {experiment['id']} / {experiment['hypothesis_id']}："
                             + json.dumps({key: experiment.get(key) for key in
                                           ('status', 'result', 'check', 'verification', 'error')},
                                          ensure_ascii=False))
        lines.append("\n## 文献与材料")
        lines.extend(f"- {source['id']}: {source['title']} {source.get('url', '')}（{source['evidence_scope']}）"
                     for source in run["sources"])
        lines.extend(f"- {asset['id']}: {asset['name']} SHA256={asset['sha256']}" for asset in run["assets"])
        gaps = report.get("evidence_gaps") or []
        if gaps:
            lines.append("\n## 证据缺口")
            lines.extend(f"- {gap}" for gap in gaps)
        if report.get("recovery"):
            lines.append("\n## 可继续的操作")
            lines.extend(f"- {step}" for step in report["recovery"].get("next_steps", []))
        if run.get("draft_report"):
            lines.extend(["\n## 未通过校验的草稿（不作为研究结论）",
                          json.dumps(run["draft_report"], ensure_ascii=False)])
        kind = run.get("research_kind") or "quantitative"
        limitations = {
            "qualitative": "定性研究：结论受检索覆盖与正文可获取性限制；元数据不等于结论；没有平台或受众数据时不做外推。",
            "literature": "文献研究：只覆盖可检索引文；标题与摘要不是全文结论；相关记录比例是保守下界。",
            "simulation": "模拟研究：结论只适用于声明的生成机制；独立重跑不等于统计显著性。",
            "mixed": "混合研究：不同证据层级不能相加；缺少真实数据时只报告证据缺口。",
            "quantitative": "本地可计算任务原型；结论受数据与实验设计限制；独立重跑不等于统计显著性。",
        }[kind]
        lines.extend(["\n## 成本", json.dumps(self.budget.snapshot(), ensure_ascii=False),
                      "\n## 局限", limitations,
                      "\n## 复现",
                      "experiments/<id>/explore/含冻结code.py、input.json和result.json；source/为当时源码。"
                      "使用匹配环境运行：python source/researchloop/engine/worker.py experiments/<id>/explore/input.json。"])
        current = "\n\n".join(lines)
        (self.folder / f"report-{self.turn_id}.md").write_text(current, encoding="utf-8")
        sections = []
        for turn in run.get("turns", []):
            file = self.folder / f"report-{turn['id']}.md"
            if file.exists():
                sections.append(file.read_text(encoding="utf-8"))
        (self.folder / "report.md").write_text("\n\n---\n\n".join(sections) or current, encoding="utf-8")


def create(body):
    question = str(body.get("question", "")).strip()
    if not 8 <= len(question) <= 3000:
        raise ValueError("请写8至3000字符的研究问题")
    tokens, seconds, steps = int(body.get("token_limit", 40000)), int(body.get("seconds_limit", 600)), int(body.get("max_steps", 12))
    limits = storage.settings().get("limits") or {}
    max_tokens = int(limits.get("max_tokens", 400000))
    max_seconds = int(limits.get("max_seconds", 7200))
    max_steps = int(limits.get("max_steps", 40))
    if not (1000 <= tokens <= max_tokens and 10 <= seconds <= max_seconds and 1 <= steps <= max_steps):
        raise ValueError(f"研究预算超出允许范围：token ≤ {max_tokens}、时间 ≤ {max_seconds}s、步骤 ≤ {max_steps}；"
                         f"可在“模型设置 → 预算上限”里调整")
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
    context = memory.recall(question) if body.get("use_memory", True) else {"facts": [], "method": None, "text": ""}
    experience = context.get("method")
    manifest = {}
    for module in sorted((storage.ROOT / "researchloop").rglob("*.py")):
        relative = module.relative_to(storage.ROOT)
        target = folder / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(module.read_bytes())
        manifest[str(relative)] = hashlib.sha256(module.read_bytes()).hexdigest()
    storage.write_json(folder / "source-manifest.json", manifest)
    run = {"id": identifier, "session_id": identifier, "version": "0.4.0", "domain": "research",
           "task": question[:32], "question": question, "current_question": question,
           "turns": [{"id": "t1", "question": question, "at": storage.now(), "status": "queued"}],
           "mode": "agent", "model": model, "created_at": storage.now(), "status": "queued",
           "token_limit": tokens, "seconds_limit": seconds, "max_steps": steps, "seed": int(body.get("seed", 42)),
           "assets": prepared, "sources": [], "hypotheses": [], "experiments": [], "steps": [], "events": [], "calls": [],
           "experience": experience, "memory_version": experience["version"] if experience else 0,
           "memory_context": context, "memory_mode": "auto" if body.get("use_memory", True) else "disabled",
           "model_source": source, "provider": provider, "cloud_enabled": source == "cloud",
           "workflow_mode": mode, "roles": role_models, "routing": routing.clean(body.get("routing") or cfg.get("routing")), "workflow": None, "agents": [], "active_agent": None,
           "source_hash": storage.fingerprint(manifest)}
    storage.write_json(folder / "run.json", run)
    return ResearchEngine(run, folder)


def resume(run, folder, question, *, token_limit=None, seconds_limit=None, max_steps=None):
    """Continue an existing session with a follow-up question."""
    text = str(question or "").strip()
    if not 8 <= len(text) <= 3000:
        raise ValueError("追问需要 8 至 3000 个字符")
    folder = Path(folder)
    turns = run.setdefault("turns", [])
    if not turns:
        turns.append({"id": "t1", "question": run.get("question", ""),
                      "at": run.get("created_at"), "status": run.get("status", "completed")})
    turns.append({"id": f"t{len(turns) + 1}", "question": text, "at": storage.now(), "status": "queued"})
    run["current_question"] = text
    context = memory.recall(text)
    run["memory_context"], run["experience"] = context, context.get("method")
    run["memory_version"] = (context.get("method") or {}).get("version", 0)
    run["status"], run["finished_at"], run["stop_reason"], run["error"] = "queued", None, None, None
    for key in ("pending_response", "pending_decision", "pending_tool_retry", "decision_feedback", "recovery_feedback", "draft_report", "restore_budget", "report"):
        run.pop(key, None)
    run["budget_call_start"] = len(run.get("calls", []))
    run["plan"], run["research_kind"], run["workflow"] = None, None, None
    run["agents"], run["active_agent"], run["reflections"] = [], None, []
    run["token_limit"] = int(token_limit or run.get("token_limit") or 40000)
    run["seconds_limit"] = int(seconds_limit or run.get("seconds_limit") or 600)
    run["max_steps"] = int(max_steps or run.get("max_steps") or 12)
    limits = storage.settings().get("limits") or {}
    max_tokens = int(limits.get("max_tokens", 400000))
    max_seconds = int(limits.get("max_seconds", 7200))
    max_steps = int(limits.get("max_steps", 40))
    if not (1000 <= run["token_limit"] <= max_tokens
            and 10 <= run["seconds_limit"] <= max_seconds
            and 1 <= run["max_steps"] <= max_steps):
        raise ValueError(f"追问预算超出允许范围：token ≤ {max_tokens}、时间 ≤ {max_seconds}s、步骤 ≤ {max_steps}")
    storage.write_json(folder / "run.json", run)
    return ResearchEngine(run, folder)


def resume_paused(run, folder):
    if run.get("domain") != "research" or run.get("status") != "paused":
        raise ValueError("只有已暂停的研究可以原位继续")
    run["status"], run["restore_budget"] = "queued", True
    run["replay_response"] = bool(run.get("pending_response") and not run.get("pending_decision"))
    run.pop("finished_at", None)
    return ResearchEngine(run, folder)


def _builtin_handler(name):
    """Built-in tools register through the same interface as third-party tools."""
    def handler(engine, args):
        return engine._apply_builtin(name, args)
    handler.__qualname__ = f"builtin:{name}"
    return handler


for _name in ("plan", "inspect", "literature", "read", "hypothesis", "experiment", "verify", "reflect", "finish"):
    registry.register_tool(_name, _builtin_handler(_name))
