# -*- coding: utf-8 -*-
"""端到端研究管线：把 C1–C7 按 docs/05 §2 的顺序串起来。

一次 `run_task()` 做的事（与文档里的 ①–⑧ 一一对应）：

  ① intake      任务文件 → TaskSpec
  ② lift        参考实现 → IR（不支持的写法降级，不猜）
  ③ evidence    微基准 + 访存下界 → 值不值得试（可关：no-evidence 消融）
  ④ search      规则组合 → 结构不同的候选（可关：no-rewrite 消融 = 只做直译）
  ⑤ experiment  族级预算分配（bandit）+ 同窗锚点 A/B
  ⑥ verify      复用 gates 的 G0–G4，语义零改动
  ⑦ verdict     reason_code 枚举归因（可核查 evidence_refs）
  ⑧ memory      写 case base；下一次同签名任务可复用结论（可关：no-memory 消融）

三个消融开关就是 `RunConfig` 的三个布尔量，其余路径完全相同——这样消融表
比的是「模块」，不是「两份不同的代码」。

诚实边界：
  - 全流程离线，**零真实云端调用**；`cloud_calls=0`、`cloud_cost_usd=0.0` 是实测值，
    不是「模拟成本包装成真实成本」。
  - 经验库只复用**结论**，不复用历史加速比：同签名 adopt case 仍要重新计时，
    只有 blocked case 才跳过实验（因为「赢不了」不需要重测）。
  - 任何门控/计时都走 `src/gates.py` 原函数；本模块不复制判定逻辑。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import bandit, evidence, gates_adapter, intake, lifter, memory
from .ir import Node, UnsupportedPattern
from . import search as search_mod
from .verdict import decide

__all__ = ["RunConfig", "Experiment", "TaskRun", "Evaluator", "run_task",
           "ARMS"]


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    arm: str = "full"
    use_evidence: bool = True
    use_rewrite: bool = True
    use_memory: bool = True
    budget: int = 4                # 单任务实验名额上限（族级分配）
    max_candidates: int = 4        # 搜索最多返回几条
    target: float = 1.10           # 达标线（与 docs/05 §7.2 SR 定义一致）
    passes: int = 1                # G4 分趟数；repeats 仍由 gates 固定 ≥5

    @classmethod
    def of(cls, arm: str, **kw) -> "RunConfig":
        preset = {
            "full": dict(),
            "no-rewrite": dict(use_rewrite=False),
            "no-evidence": dict(use_evidence=False),
            "no-memory": dict(use_memory=False),
        }
        if arm not in preset:
            raise KeyError(f"未知消融版本 {arm!r}；已有 {sorted(preset)}")
        cfg = dict(preset[arm])
        cfg.update(kw)
        return cls(arm=arm, **cfg)


ARMS = ("full", "no-rewrite", "no-evidence", "no-memory")


# ---------------------------------------------------------------------------
# 记录结构
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    exp_id: str
    task: str
    arm: str
    family: str
    rule_chain: Tuple[str, ...]
    origin: str
    source_sha1: str
    gates: Dict[str, Any]
    anchor_before_ms: Optional[float] = None
    anchor_after_ms: Optional[float] = None
    anchor_ms_used: Optional[float] = None
    median_ms: Optional[float] = None
    speedup_same_window: Optional[float] = None
    target_met: bool = False
    cloud_calls: int = 0
    cloud_cost_usd: float = 0.0
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "exp_id": self.exp_id, "task": self.task, "arm": self.arm,
            "family": self.family, "rule_chain": list(self.rule_chain),
            "origin": self.origin, "source_sha1": self.source_sha1,
            "gates": {k: {"passed": v.get("passed"),
                          "evidence": (v.get("evidence") or "")[:300]}
                      for k, v in self.gates.items()},
            "anchor_before_ms": _r(self.anchor_before_ms),
            "anchor_after_ms": _r(self.anchor_after_ms),
            "anchor_ms_used": _r(self.anchor_ms_used),
            "median_ms": _r(self.median_ms),
            "speedup_same_window": _r(self.speedup_same_window),
            "target_met": bool(self.target_met),
            "cloud_calls": int(self.cloud_calls),
            "cloud_cost_usd": round(float(self.cloud_cost_usd), 9),
            "note": self.note,
        }


@dataclass
class TaskRun:
    task: str
    arm: str
    verdict_hint: str
    dominant: str
    budget_alloc: Dict[str, int]
    experiments: List[Experiment]
    verdict: Any
    wallclock_s: float
    reused_case: bool = False
    skip_reason: str = ""
    notes: List[str] = field(default_factory=list)
    evidence_json: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_speedup(self) -> float:
        vals = [e.speedup_same_window for e in self.experiments
                if e.speedup_same_window is not None]
        return max(vals) if vals else 0.0

    @property
    def timed_experiments(self) -> int:
        return sum(1 for e in self.experiments if e.median_ms is not None)

    def to_json(self) -> Dict[str, Any]:
        return {
            "task": self.task, "arm": self.arm,
            "verdict_hint": self.verdict_hint, "dominant": self.dominant,
            "budget_alloc": dict(self.budget_alloc),
            "n_experiments": len(self.experiments),
            "n_timed": self.timed_experiments,
            "best_speedup": _r(self.best_speedup),
            "wallclock_s": round(float(self.wallclock_s), 4),
            "reused_case": bool(self.reused_case),
            "skip_reason": self.skip_reason,
            "notes": list(self.notes),
            "evidence": self.evidence_json,
            "verdict": self.verdict.to_json() if hasattr(self.verdict, "to_json") else {},
            "experiments": [e.to_json() for e in self.experiments],
        }


def _r(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 6)


# ---------------------------------------------------------------------------
# 评测器：复用门控，缓存「正确性」但不缓存「计时」
# ---------------------------------------------------------------------------
#
# 为什么只缓存 G0–G3：正确性与机器负载无关，同一份源码重复验证是纯浪费；
# 而计时会随机器状态漂移（实测同协议同机器可差 4–5 倍），所以每次都在
# **当前窗口**重新计时，速度比只和本窗口的 anchor 相比。

class Evaluator:
    def __init__(self, passes: int = 1) -> None:
        self.passes = int(passes)
        self._verify_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.calls = {"reference": 0, "candidate": 0, "verify": 0}

    # ---------------------------------------------------------------- verify
    def verify(self, task: Any, source: str) -> Dict[str, Any]:
        key = (getattr(task, "name", "?"), _sha1(source))
        if key in self._verify_cache:
            return dict(self._verify_cache[key])
        self.calls["verify"] += 1
        out = gates_adapter.verify_g0_to_g3(task, source)
        self._verify_cache[key] = dict(out)
        return dict(out)

    # ---------------------------------------------------------------- timing
    def time_candidate(self, task: Any, source: str) -> Optional[float]:
        self.calls["candidate"] += 1
        return gates_adapter.time_candidate(task, source, passes=self.passes)

    def reference(self, task: Any) -> float:
        self.calls["reference"] += 1
        return gates_adapter.time_reference(task, passes=self.passes)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_task(spec: Any, cfg: RunConfig, *,
             evaluator: Optional[Evaluator] = None,
             memory_path: Optional[Any] = None,
             states: Optional[Dict[str, bandit.FamilyState]] = None,
             exp_prefix: str = "") -> TaskRun:
    """跑一个任务块。`states` 是跨任务共享的族级统计（C5 用）。"""
    started = time.perf_counter()
    ev_obj = evaluator or Evaluator(passes=cfg.passes)
    notes: List[str] = []
    exp_seq = {"n": 0}

    # ② lift
    ir_node: Optional[Node] = None
    try:
        ir_node = lifter.lift(spec)
    except UnsupportedPattern as exc:      # noqa: PERF203
        notes.append(f"lift 失败，降级为「无结构候选」：{exc}")

    # ③ evidence
    ev = None
    if cfg.use_evidence and ir_node is not None:
        ev = evidence.profile_task(spec, ir_node)
        hint, factor = ev.verdict_hint, float(ev.budget_factor)
    elif ir_node is not None:
        hint, factor = "winnable", 1.0
        if not cfg.use_evidence:
            notes.append("no-evidence：跳过瓶颈判断，所有族都允许实验")
    else:
        hint, factor = "blocked_by_bandwidth", 0.0

    # ⑧ memory（读）
    cases: List[memory.Case] = []
    hints: Dict[str, Any] = {"prefer_rules": [], "avoid_families": [],
                             "known_ceiling": 0.0, "notes": []}
    exact = None
    if cfg.use_memory and ir_node is not None:
        cases = memory.recall(spec, ir_node, memory_path)
        hints = memory.transfer_hints(cases)
        sig = memory.signature(ir_node)
        exact = next((c for c in cases if c.signature == sig), None)
        if hints["notes"]:
            notes.extend(hints["notes"][:2])

    # 是否整块跳过实验
    skip_reason = ""
    reused = False
    if ir_node is None:
        skip_reason = "lifter 不支持该写法（无 IR 则无结构候选）"
    elif exact is not None and exact.decision == "keep_baseline" \
            and exact.reason_code.startswith("blocked_by"):
        skip_reason = f"经验复用：同签名已判定 {exact.reason_code}"
        reused = True
        notes.append("同签名 blocked case 命中：不重复跑实验（结论复用，不冒充本次测量）")
    elif cfg.use_evidence and hint.startswith("blocked_by"):
        skip_reason = f"证据层实验前拦截：{hint}"

    experiments: List[Experiment] = []
    alloc: Dict[str, int] = {}
    if states is None:
        states = {}

    if not skip_reason and ir_node is not None:
        candidates = _build_hypotheses(ir_node, spec, cfg, hints)
        alloc = _allocate(candidates, cfg, factor, states, notes)
        chosen = _pick(candidates, alloc, cfg.budget)

        if not chosen:
            notes.append("假设空间没有产出候选：规则覆盖不足（不是任务不可解）")
        else:
            # ⑤ 同窗锚点：一个任务块前后各一次
            anchor_before = ev_obj.reference(spec.task)
            medians: List[Tuple[Any, Optional[float], Dict[str, Any]]] = []
            for cand in chosen:
                exp_seq["n"] += 1
                eid = f"{exp_prefix}{getattr(spec, 'name', 'task')}-{exp_seq['n']:02d}"
                verified = ev_obj.verify(spec.task, cand["source"])
                median = None
                note = ""
                if verified["passed"]:
                    median = ev_obj.time_candidate(spec.task, cand["source"])
                    if median is None:
                        note = "G4 计时失败（完整性/子进程）"
                else:
                    note = f"门控拦截：{verified['blocked_by']}"
                medians.append((cand, median, {"exp_id": eid, "note": note,
                                               "gates": verified["gates"],
                                               "blocked_by": verified["blocked_by"]}))
            anchor_after = ev_obj.reference(spec.task)
            anchor_used = min(anchor_before, anchor_after)

            for cand, median, meta in medians:
                speedup = (anchor_used / median) if median else None
                if speedup is not None:
                    st = states.setdefault(
                        cand["family"], bandit.FamilyState(cand["family"]))
                    st.record(speedup)
                experiments.append(Experiment(
                    exp_id=meta["exp_id"], task=getattr(spec, "name", "?"),
                    arm=cfg.arm, family=cand["family"],
                    rule_chain=tuple(cand["rule_chain"]), origin=cand["origin"],
                    source_sha1=_sha1(cand["source"]),
                    gates=meta["gates"],
                    anchor_before_ms=anchor_before,
                    anchor_after_ms=anchor_after,
                    anchor_ms_used=anchor_used,
                    median_ms=median,
                    speedup_same_window=speedup,
                    target_met=bool(speedup is not None and speedup >= cfg.target),
                    note=meta["note"],
                ))

    # ⑦ 裁决
    exp_json = [e.to_json() for e in experiments]
    verdict = decide(spec, ev, exp_json, target=cfg.target,
                     prior_reason=(exact.reason_code if reused else None))

    # ⑧ memory（写）：no-memory 只关「读」，写入仍做（诚实：写不影响本 arm 的结果）
    if memory_path is not None and ir_node is not None:
        best = max(experiments, key=lambda e: e.speedup_same_window or 0.0,
                   default=None)
        case = memory.case_from_verdict(
            spec, ir_node, verdict,
            family=(best.family if best else ""),
            rule_chain=(best.rule_chain if best else ()),
            note=f"arm={cfg.arm}",
            meta={"arm": cfg.arm, "reused": reused},
        )
        try:
            memory.remember(case, memory_path)
        except OSError as exc:              # 落盘失败不该毁掉整个 run
            notes.append(f"case 写入失败：{type(exc).__name__}: {exc}")

    return TaskRun(
        task=getattr(spec, "name", "?"), arm=cfg.arm,
        verdict_hint=hint, dominant=(getattr(ev, "dominant", "") or ""),
        budget_alloc=dict(alloc), experiments=experiments,
        verdict=verdict, wallclock_s=time.perf_counter() - started,
        reused_case=reused, skip_reason=skip_reason, notes=notes,
        evidence_json=(ev.to_json() if ev is not None else {}),
    )


# ---------------------------------------------------------------------------
# 候选与预算
# ---------------------------------------------------------------------------

def _build_hypotheses(ir_node: Node, spec: Any, cfg: RunConfig,
                      hints: Dict[str, Any]) -> List[Dict[str, Any]]:
    """返回候选描述列表；no-rewrite 时只有「同一 IR 的直译」。"""
    out: List[Dict[str, Any]] = []
    if cfg.use_rewrite:
        from .codegen import CodegenError, generate

        for h in search_mod.search(ir_node, spec, cap=cfg.max_candidates):
            try:
                src = generate(spec, h.node)
            except CodegenError:
                continue
            out.append({"family": h.family, "rule_chain": list(h.rule_chain),
                        "origin": h.origin, "node": h.node, "source": src})

    if not out:
        from .codegen import CodegenError, generate

        try:
            src = generate(spec, ir_node)
        except CodegenError:
            return []
        out.append({"family": "identity", "rule_chain": [],
                    "origin": "identity:no_structural_rewrite",
                    "node": ir_node, "source": src})

    # 经验提示：把历史上成功的规则链排前面（不改变候选集合，只改顺序）
    prefer = [tuple(x) for x in hints.get("prefer_rules", [])]

    def rank(item: Dict[str, Any]) -> Tuple[int, int, str]:
        chain = tuple(item["rule_chain"])
        pos = prefer.index(chain) if chain in prefer else len(prefer)
        return (pos, len(chain), item["family"])

    out.sort(key=rank)
    return out


def _allocate(candidates: Sequence[Dict[str, Any]], cfg: RunConfig, factor: float,
              states: Optional[Dict[str, bandit.FamilyState]],
              notes: List[str]) -> Dict[str, int]:
    fams: List[str] = []
    for c in candidates:
        if c["family"] not in fams:
            fams.append(c["family"])
    budget = max(1, int(round(cfg.budget * max(0.0, float(factor)))))
    if states is None:
        states = {}
    global_best = max((s.best for s in states.values()), default=0.0)
    sub = {f: states.get(f) or bandit.FamilyState(f) for f in fams}
    quotas = dict(bandit.allocate(sub, budget, global_best))
    if budget > 0 and sum(quotas.values()) == 0 and fams:
        quotas[fams[0]] = 1
        notes.append("冻结兜底：该任务唯一可用族，保留 1 次验证机会")
    return quotas


def _pick(candidates: Sequence[Dict[str, Any]], quotas: Dict[str, int],
          budget: int) -> List[Dict[str, Any]]:
    left = {f: int(q) for f, q in quotas.items()}
    out: List[Dict[str, Any]] = []
    for c in candidates:
        if len(out) >= max(0, int(budget)):
            break
        f = c["family"]
        if left.get(f, 0) <= 0:
            continue
        left[f] -= 1
        out.append(c)
    return out


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()