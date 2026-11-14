import itertools
import json
import random
import shutil
import threading
import time
import uuid

from .dataset import Dataset
from .diagnostics import diagnose, query_profile, screen_accepts, select_memory
from .ollama import Budget, BudgetError, Ollama, probe
from .policy import BASELINE, CONFIG_SCHEMA, FEATURES, config_changes, gate_matches, normalize_config, signature
from .retrieval import Retriever, evaluate, guard_accepts, subset_result
from .storage import ROOT, RUNS, LOCK, get_memory, grounded_memory, list_runs, now, read_json, run_path, settings, source_hash, write_json

DECISION_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "choice": {"type": "string"},
    "diagnosis": {"type": "string", "enum": ["recall", "partial_recall", "ranking", "constraints", "alias_ambiguity", "none"]},
    "expected_effect": {"type": "string", "enum": ["increase", "no_worse"]},
    **{k: {"type": "string"} for k in ("hypothesis", "mechanism", "scope", "prediction", "falsify_if", "reason")},
    "proposal": {"type": "object", "additionalProperties": False, "properties": {
        "parent_id": {"type": "string"}, "config": CONFIG_SCHEMA}, "required": ["parent_id", "config"]}},
    "required": ["choice", "diagnosis", "expected_effect", "hypothesis", "mechanism", "scope", "prediction", "falsify_if", "reason", "proposal"]}


def rank(result):
    m = result["metrics"]
    return (m["complete_at_5"], m["coverage_at_5"], m["ndcg_at_5"])


def behavior_key(config, features=None):
    """Equivalence on all exploration query features, with NO gold labels."""
    cfg = normalize_config(config)
    if features is None:
        return signature(cfg)
    behavior = []
    for f in features:
        active = gate_matches(cfg["gate"], f)
        protected = active and cfg["normalization"] == "preserve_constraints" and (f["has_negation"] or f["has_number"])
        strategy = cfg["strategy"] if active else "baseline"
        if strategy == "acronym" and not f["has_acronym"]:
            strategy = "baseline"
        behavior.append((strategy, cfg["prf_terms"] if strategy == "prf" else 0, bool(protected)))
    return json.dumps(behavior)


def frontier(experiments, features=None, stats=None):
    seen = {behavior_key(e["config"], features) for e in experiments}
    proposals = []
    gate_options = ["all", *FEATURES]
    for size in (2, 3):
        for group in itertools.combinations(FEATURES, size):
            gate_options.extend(({"all": list(group)}, {"any": list(group)}))
    generated = duplicates = 0
    for parent in experiments:
        if parent.get("result_scope") == "screen":
            continue
        original = normalize_config(parent["config"])
        for field, values in (("normalization", ["legacy", "preserve_constraints"]),
                              ("strategy", ["acronym", "prf"]), ("gate", gate_options), ("prf_terms", [1, 3, 5])):
            for value in values:
                cfg = normalize_config({**original, field: value})
                if len(config_changes(original, cfg)) != 1:
                    continue
                generated += 1
                key = behavior_key(cfg, features)
                if key in seen:
                    duplicates += 1
                    continue
                proposals.append({"parent_id": parent["id"], "config": cfg})
                seen.add(key)
    if stats is not None:
        stats.update(generated=generated, equivalent_or_inactive_removed=duplicates, remaining=len(proposals))
    return proposals


class ResearchService:
    def __init__(self):
        self.dataset = self.retriever = None
        self.cancel_events, self.active, self.preparing = {}, None, None
        self._load_lock = threading.Lock()
        self.loaded_source_hash = source_hash()

    def load(self):
        with self._load_lock:
            if self.dataset is None:
                self.dataset = Dataset()
                self.retriever = Retriever(self.dataset.corpus)
        return self.dataset, self.retriever

    def start(self, options):
        with LOCK:
            if self.loaded_source_hash != source_hash():
                raise ValueError("后端代码已更新，请重新启动应用以确保实验版本一致")
            if self.active or (self.preparing and self.preparing.get("status") == "running"):
                raise ValueError("已有任务正在运行")
            mode, task = options.get("mode", "agent"), options.get("task", "C")
            if mode not in ("agent", "search", "carry") or task not in ("A", "B", "C", "D", "E", "F"):
                raise ValueError("无效模式或任务")
            limit, token_limit = int(options.get("max_experiments", 4)), int(options.get("token_limit", 16000))
            seconds, seed = int(options.get("seconds_limit", 900)), int(options.get("seed", 42))
            if not (1 <= limit <= 8 and 5000 <= token_limit <= 100000 and 10 <= seconds <= 3600 and 0 <= seed <= 2147483647):
                raise ValueError("预算参数超出允许范围")
            cfg = settings()
            if "model" in options:
                cfg["model"] = str(options["model"])
            if mode == "agent" and (not cfg["model"] or "cloud" in cfg["model"].lower()):
                raise ValueError("请选择本地 Ollama 模型")
            memory = get_memory(options.get("memory_version")) if options.get("use_memory") or mode == "carry" else None
            if mode == "carry" and not memory:
                raise ValueError("请先发布通过验收的来源经验")
            run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            run = {"id": run_id, "version": "0.2.0", "created_at": now(), "status": "queued", "task": task,
                   "mode": mode, "model": cfg["model"] if mode == "agent" else None,
                   "ollama_url": cfg["ollama_url"] if mode == "agent" else None,
                   "memory_version": memory["version"] if mode == "carry" and memory else 0,
                   "requested_memory_version": memory["version"] if memory else 0,
                   "seed": seed, "max_experiments": limit, "token_limit": token_limit, "seconds_limit": seconds,
                   "experiments": [], "events": [], "calls": [], "best_config": dict(BASELINE), "cloud_enabled": False,
                   "source_hash": self.loaded_source_hash, "memory_snapshot": memory,
                   "evaluation_counts": {"screen": 0, "full": 0, "guard": 0, "computed_rows": 0, "reused_rows": 0},
                   "shared_stop_rule": "Stop after 2 successive candidates without full-exploration improvement; identical for cold, warm and random.",
                   "evaluation_protocol": "Fixed explore-only screen; select on full explore; freeze before one guard check. Final test sealed.",
                   "cache_protocol": "Reuse candidate screen rows only inside the same run, with corpus/config/claim hashes. No cross-run result cache."}
            write_json(RUNS / run_id / "run.json", run)
            snapshot = RUNS / run_id / "source" / "researchloop"
            snapshot.mkdir(parents=True, exist_ok=True)
            for file in (ROOT / "researchloop").glob("*.py"):
                shutil.copyfile(file, snapshot / file.name)
            self.active = run_id
            cancel = self.cancel_events[run_id] = threading.Event()
            threading.Thread(target=self._run, args=(run, cfg, memory, cancel), daemon=True).start()
            return run

    def _run(self, run, cfg, requested_memory, cancel):
        budget = Budget(run["token_limit"], run["seconds_limit"])
        def save():
            run["budget"], run["updated_at"] = budget.snapshot(), now()
            write_json(RUNS / run["id"] / "run.json", run)
        def event(kind, text, **extra):
            item = {"at": now(), "type": kind, "text": text, **extra}
            run["events"].append(item)
            with (RUNS / run["id"] / "events.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
            save()
        def on_call(item):
            run["calls"].append(item)
            save()
        def stopped():
            if cancel.is_set():
                raise InterruptedError("用户停止运行")
            if budget.elapsed >= budget.seconds_limit:
                raise BudgetError("研究时间预算已耗尽")
        def measure(claims, config, kind, previous=None):
            result = evaluate(self.retriever, claims, config, previous=previous)
            counts = run["evaluation_counts"]
            counts[kind] += 1
            counts["computed_rows"] += result.get("computed_rows", result["count"])
            counts["reused_rows"] += result.get("reused_rows", 0)
            return result
        try:
            run["status"] = "running"
            event("setup", "加载固定语料、冻结任务与策略工具；评估器不接受 Agent 修改")
            dataset, retriever = self.load()
            tasks = getattr(dataset, "tasks", dataset.manifest["splits"])
            run["dataset_sha256"], run["split_ids"] = dataset.manifest["archive_sha256"], tasks[run["task"]]
            claims = dataset.task(run["task"], "explore")
            screen_ids = run["split_ids"].get("screen", sorted(run["split_ids"]["explore"])[:8])
            run["screen_ids"] = screen_ids
            screen_claims = [c for c in claims if str(c["id"]) in screen_ids]
            profile = run["query_profile"] = query_profile(retriever, claims)
            features = [retriever.features(c["claim"]) for c in claims]
            if run["task"] in ("C", "D", "E", "F"):
                ledger_path = ROOT / "docs" / "experiments" / "v2-task-exposure.json"
                ledger = read_json(ledger_path, {})
                ledger.setdefault(run["task"], {"first_run": run["id"], "first_started_at": now(), "source_hash": run["source_hash"]})
                write_json(ledger_path, ledger)
            if run["mode"] == "agent":
                memory, selection = select_memory(requested_memory, profile, retriever, claims)
                run["memory_selection"] = selection
                run["memory_version"] = memory["version"] if memory else 0
                run["loaded_memory_rule"] = memory
                event("memory_match", selection["reason"], applied=selection["applied"])
            else:
                memory = None
            event("baseline", f"建立 {len(claims)} 道完整探索基线，固定开发筛选 {len(screen_claims)} 道")
            base_result = measure(claims, BASELINE, "full")
            baseline = {"id": "e0", "parent_id": None, "config": dict(BASELINE), "result_scope": "explore",
                        "hypothesis": "建立固定句子级 BM25 基线", "result": base_result,
                        "accepted_on_explore": True, "delta_complete": 0, "cumulative_tokens": budget.used}
            run["experiments"].append(baseline)
            run["diagnostics"] = diagnose(base_result, retriever)
            event("diagnosis", "已区分初始召回缺失、部分召回和排序失败；诊断金标仅用于开发", failure_counts=run["diagnostics"]["failure_counts"])
            champion, stagnation = baseline, 0
            client = Ollama(cfg["ollama_url"], cfg["model"], budget, on_call) if run["mode"] == "agent" else None
            if client:
                status = probe(cfg["ollama_url"])
                selected = next((m for m in status["models"] if m["name"] == cfg["model"]), None)
                if not status["available"] or selected is None:
                    raise RuntimeError("Ollama 不可用或模型未安装；不会切换云端或伪造 Agent 决策")
                run["model_digest"] = selected["digest"]
            rng = random.Random(run["seed"])
            for trial in range(run["max_experiments"]):
                stopped()
                if stagnation >= 2:
                    run["stop_reason"] = "连续两项候选未改善完整探索结果，触发所有比较组共用的停止规则"
                    event("shared_stop", run["stop_reason"])
                    break
                filter_stats = {}
                choices = frontier(run["experiments"], features, filter_stats)
                run["candidate_filter"] = filter_stats
                if memory:
                    priority = memory["priority"]
                    choices.sort(key=lambda p: (p["config"]["strategy"] != priority["strategy"], p["config"]["normalization"] != priority.get("normalization", "legacy"), p["config"]["gate"] != priority["gate"]))
                if not choices:
                    run["stop_reason"] = "去重及不触发干预排除后，没有可检验的单因素候选"
                    break
                if run["mode"] == "carry":
                    carried = normalize_config(requested_memory["best_config"])
                    if trial or carried == BASELINE:
                        run["stop_reason"] = "配置沿用对照完成"
                        break
                    decision = {"parent_id": "e0", "config": carried, "action": "experiment", "expected_effect": "increase",
                                "hypothesis": "验证直接沿用来源任务配置的收益", "reason": "配置沿用对照，无研究规划"}
                elif run["mode"] == "search":
                    decision = {**rng.choice(choices), "action": "experiment", "expected_effect": "increase",
                                "hypothesis": "从相同合法策略空间随机选择一个单因素候选", "reason": "确定性搜索对照"}
                else:
                    decision, validation_error = None, None
                    for attempt in range(2):
                        try:
                            event("planning", "本地模型结合失败类型、短经验规则与预算提出实验")
                            shown = choices[:10]
                            schema = {**DECISION_SCHEMA, "properties": {**DECISION_SCHEMA["properties"],
                                      "choice": {"type": "string", "enum": ["stop", "custom"] + [f"c{i+1}" for i in range(len(shown))]}}}
                            proposed = client.call(self._prompt(run, champion, shown, memory, retriever, budget, validation_error), schema,
                                                   "planner" if not attempt else "planner_retry", run["seed"] + trial + 1000 * attempt)
                            if proposed.get("choice") == "stop":
                                decision = {**proposed, "action": "stop"}
                                run["stop_reason"] = proposed.get("reason", "模型决定停止")
                                break
                            if proposed.get("choice") == "custom":
                                candidate = proposed.get("proposal")
                                if not isinstance(candidate, dict) or set(candidate) != {"parent_id", "config"}:
                                    raise ValueError("custom 必须提供 proposal.parent_id 和完整策略 config")
                            else:
                                candidate = next((p for i, p in enumerate(shown) if f"c{i+1}" == proposed.get("choice")), None)
                            if candidate is None:
                                raise ValueError("请选择列出的 ID，或 custom 生成受约束条件策略")
                            proposed = {**proposed, **candidate, "action": "experiment"}
                            self._validate(proposed, run["experiments"], features)
                            decision = proposed
                            break
                        except ValueError as exc:
                            validation_error = str(exc)
                            event("invalid_decision", validation_error, retry=attempt)
                        except BudgetError as exc:
                            decision, run["stop_reason"] = {"action": "stop"}, str(exc)
                            break
                    if decision is None:
                        run["stop_reason"] = "模型连续返回无效策略，保留已测配置并停止"
                        break
                    if decision["action"] == "stop":
                        event("stop", run["stop_reason"])
                        break
                stopped()
                config = normalize_config(decision["config"])
                parent = next(e for e in run["experiments"] if e["id"] == decision["parent_id"])
                event("hypothesis", decision["hypothesis"], decision=decision)
                screened = measure(screen_claims, config, "screen")
                parent_screen = subset_result(parent["result"], screen_ids)
                passed = screen_accepts(parent_screen, screened)
                event("screening", "开发筛选通过，进入完整探索评估" if passed else "开发筛选退化，否决候选；不接触验收集",
                      passed=passed, count=screened["count"], metrics=screened["metrics"])
                result = measure(claims, config, "full", previous=screened) if passed else screened
                scope = "explore" if passed else "screen"
                accepted = passed and rank(result) > rank(champion["result"])
                parent_result = parent["result"] if passed else parent_screen
                prior = {r["claim_id"]: r for r in parent_result["rows"]}
                experiment = {"id": f"e{len(run['experiments'])}", "parent_id": parent["id"], "config": config,
                    **{k: decision.get(k, "") for k in ("hypothesis", "mechanism", "scope", "prediction", "falsify_if", "reason", "diagnosis", "expected_effect")},
                    "diff": config_changes(parent["config"], config), "result": result, "result_scope": scope,
                    "screen_result": screened, "screen_passed": passed, "accepted_on_explore": accepted,
                    "delta_complete": result["metrics"]["complete_at_5"] - parent_result["metrics"]["complete_at_5"],
                    "improved_ids": [r["claim_id"] for r in result["rows"] if r["complete_at_5"] > prior[r["claim_id"]]["complete_at_5"]],
                    "regressed_ids": [r["claim_id"] for r in result["rows"] if r["complete_at_5"] < prior[r["claim_id"]]["complete_at_5"]],
                    "cumulative_tokens": budget.used}
                expected = decision.get("expected_effect", "increase")
                consistent = experiment["delta_complete"] > 0 if expected == "increase" else experiment["delta_complete"] >= -1e-12
                experiment["hypothesis_check"] = {"expected_effect": expected, "direction_consistent": consistent,
                    "measurement_scope": scope, "note": "Descriptive direction check; not significance or proof of the proposed mechanism."}
                run["experiments"].append(experiment)
                if accepted:
                    champion, stagnation = experiment, 0
                else:
                    stagnation += 1
                run["current_diagnostics"] = diagnose(champion["result"], retriever)
                event("measurement", "完整探索测量完成" if passed else "仅保存开发筛选结果，不参与全量最优选择",
                      experiment_id=experiment["id"], result_scope=scope, metrics=result["metrics"])
            run.setdefault("stop_reason", "已达到候选实验次数上限")
            run["frozen_candidate"] = {"experiment_id": champion["id"], "config": champion["config"], "at": now()}
            stopped()
            event("validation", "候选已冻结，仅做一次独立验收；验收后不再改配置")
            guard_claims = dataset.task(run["task"], "guard")
            guard_base = measure(guard_claims, BASELINE, "guard")
            guard_candidate = guard_base if champion["config"] == BASELINE else measure(guard_claims, champion["config"], "guard")
            delta = {k: guard_candidate["metrics"][k] - guard_base["metrics"][k] for k in guard_base["metrics"]}
            accepted = guard_accepts(guard_base["metrics"], guard_candidate["metrics"])
            run["guard"] = {"accepted": accepted, "baseline": guard_base, "candidate": guard_candidate, "delta": delta,
                            "rule": "Complete@5 and Coverage@5 must not regress. Small-sample engineering gate; not significance."}
            run["best_config"] = champion["config"] if accepted else dict(BASELINE)
            run["selected_experiment_id"] = champion["id"] if accepted else "e0"
            event("verification", "独立验收通过" if accepted else "独立验收退化，保留原基线", accepted=accepted, delta=delta)
            if client and len(run["experiments"]) > 1 and accepted:
                run["memory_candidate"] = grounded_memory(run)
                event("experience", "根据数值证据生成短经验；不额外调用模型编写未经验证的结论")
            if budget.used > budget.token_limit:
                raise BudgetError("实际模型消耗超出预算，禁止发布经验")
            run["status"] = "completed"
            event("completed", "研究完成，保存已验收配置、诊断与所有开发/完整评估记录")
        except InterruptedError as exc:
            run["status"], run["stop_reason"] = "cancelled", str(exc)
            event("cancelled", str(exc))
        except BudgetError as exc:
            run["status"], run["stop_reason"] = "budget_stopped", str(exc)
            event("budget_stop", str(exc) + "；未验收候选不发布")
        except Exception as exc:
            run["status"], run["error"] = "failed", str(exc)
            event("error", str(exc))
        finally:
            run["finished_at"] = now()
            try:
                save()
            finally:
                with LOCK:
                    self.active = None
                    self.cancel_events.pop(run["id"], None)

    @staticmethod
    def _validate(decision, experiments, features=None):
        if decision.get("action") != "experiment":
            raise ValueError("action 必须为 experiment")
        if any(not isinstance(decision.get(k), str) or not decision[k].strip() for k in ("hypothesis", "mechanism", "scope", "prediction", "falsify_if", "reason")):
            raise ValueError("必须提供机制、范围、预测与否证条件")
        parent = next((e for e in experiments if e["id"] == decision.get("parent_id") and e.get("result_scope") != "screen"), None)
        if parent is None:
            raise ValueError("父实验不存在，或只有开发筛选结果")
        cfg = normalize_config(decision.get("config"))
        if len(config_changes(parent["config"], cfg)) != 1:
            raise ValueError("每次只能相对父实验修改一个策略字段")
        key = behavior_key(cfg, features)
        if any(behavior_key(e["config"], features) == key for e in experiments):
            raise ValueError("配置与已测策略语义等价，或在整个探索任务不触发干预")

    @staticmethod
    def _prompt(run, champion, choices, memory, retriever, budget, validation_error=None):
        diagnosis = diagnose(champion["result"], retriever, max_examples=2)
        history = [{"id": e["id"], "config": e["config"], "metrics": e["result"]["metrics"],
                    "data_scope": e.get("result_scope", "explore"), "n": e["result"]["count"],
                    "improved": len(e.get("improved_ids", [])), "regressed": len(e.get("regressed_ids", []))} for e in run["experiments"]]
        return json.dumps({"instruction": "你研究检索算法，不是生物学。根据失败类型提出一项可否证实验，或说明为何停止。可选择候选ID；也可 choice=custom 并给 proposal={parent_id,config}，自行组合最多3个可观测特征的all/any门控。不能修改指标、标注或代码。一次只改一个配置字段。筛选结果不能冒充完整探索结果。",
            "tools": {"preserve_constraints": "仅在含否定词或数字且门控触发时，使用保留否定词、小数、单个数字的分词与索引；其余查询保持旧算法。",
                      "acronym": "仅扩展语料内有出处的缩写，可能存在歧义", "prf": "从初始前3句选1/3/5个高IDF词，可能漂移；保留原查询并固定RRF融合"},
            "features": list(FEATURES), "objective": "完整探索 Complete@5 优先，其次 Coverage@5、nDCG@5；减少不必要实验。",
            "diagnosis": {k: diagnosis[k] for k in ("failure_counts", "feature_groups", "examples")},
            "memory_prior": memory, "history": history,
            "budget": {"remaining_tokens": budget.token_limit - budget.used, "remaining_seconds": round(budget.seconds_limit - budget.elapsed), "remaining_trials": run["max_experiments"] - len(run["experiments"]) + 1},
            "allowed_candidates": [{"id": f"c{i+1}", **c} for i, c in enumerate(choices)],
            "proposal_format": {"parent_id": champion["id"], "config": champion["config"]},
            "previous_validation_error": validation_error,
            "response": "必须输出proposal对象：parent_id与完整config。custom时该对象就是要测的策略；选择候选ID或stop时仍按格式回传父配置，系统会忽略它。用短中文写依据。increase表示主指标必须提高，no_worse表示不退化。不要编造显著性，没有信息价值可stop。"}, ensure_ascii=False, separators=(",", ":"))

    def cancel(self, run_id):
        flag = self.cancel_events.get(run_id)
        if flag:
            flag.set()
        return {"requested": bool(flag), "note": "当前模型或工具调用结束后停止"}

    def finalize(self):
        with LOCK:
            previous = read_json(ROOT / "data" / "final_evaluation.json")
            if previous:
                return previous
            if self.active:
                raise ValueError("请等待研究完成再冻结最终验收")
            completed = [r for r in list_runs() if r["status"] == "completed" and r.get("source_hash") == self.loaded_source_hash]
            if not completed:
                raise ValueError("当前版本尚无可验收运行")
            dataset, retriever = self.load()
            claims = [dataset.claims[cid] for cid in dataset.manifest["splits"]["test"]]
            results = {"baseline": evaluate(retriever, claims, BASELINE)}
            for run in completed:
                results[run["id"]] = evaluate(retriever, claims, run["best_config"])
            report = {"created_at": now(), "frozen_run_ids": [r["id"] for r in completed],
                "dataset_sha256": dataset.manifest["archive_sha256"], "source_hash": self.loaded_source_hash,
                "test_count": len(claims), "results": results,
                "warning": "Final test revealed for frozen runs only; do not tune and call later runs unseen."}
            write_json(ROOT / "data" / "final_evaluation.json", report)
            return report
