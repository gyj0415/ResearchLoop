# -*- coding: utf-8 -*-
"""C6 · 裁决与归因：输了要说得清「为什么输」，且理由可核查。

归因是**枚举**（`REASON_CODES`），不是散文；每条裁决都带 `evidence_refs`，
可以指回 `evidence.py` 的微基准或某条实验记录。

判定顺序（写死，docs/05 §5 M7）：
  1) 有实验且 best_same_window ≥ target(1.10) → adopt / target_met
  2) 没有可计时的实验：
     - evidence 给出 blocked_by_* → 用它
     - 有实验但全被门控拦 → incorrect_candidate
     - 一条实验都没有 → no_candidate（说明规则覆盖不足）
  3) evidence 命中 blocked_by_* → 用它的归因
  4) best ≥ 0.80 → inconclusive / near_miss
  5) 其余 → blocked_by_bandwidth（默认归因，必须带倍数证据）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["Verdict", "REASON_CODES", "NEXT_ACTIONS", "decide", "TARGET_SPEEDUP"]

TARGET_SPEEDUP = 1.10
NEAR_MISS = 0.80

REASON_CODES = (
    "target_met",                  # ≥1.10×
    "blocked_by_transcendental",   # 算子里有 exp/tanh/erf，numpy 结构性落后
    "blocked_by_bandwidth",        # 单线程访存打不过多线程 eager
    "blocked_by_reduction",        # 归约本身慢
    "blocked_by_overhead",         # 小张量，调用开销主导
    "incorrect_candidate",         # 写错了，被门控拦下
    "no_candidate",                # 假设空间没产出（应触发「该扩规则」）
    "near_miss",                   # 方向对但没到门槛
)

NEXT_ACTIONS = {
    "target_met": "登记 case；同类结构优先尝试同一规则链",
    "blocked_by_transcendental": "不再为该任务生成候选；把超越函数上限写入 case base",
    "blocked_by_bandwidth": "默认不信；只有出现可消元/可融合结构时才重开",
    "blocked_by_reduction": "先确认归约能否换成更快的等价形式（例：sum+scale 替代 mean）",
    "blocked_by_overhead": "把小张量任务排除出本场景，或改测大 shape",
    "incorrect_candidate": "先修正确性（G0–G3），不要进入计时",
    "no_candidate": "规则覆盖不足：记录缺失的结构模式，扩规则后重跑",
    "near_miss": "同结构加大预算/试邻近规则；不要换任务刷分",
}


@dataclass(frozen=True)
class Verdict:
    task: str
    decision: str                 # adopt / keep_baseline / inconclusive
    reason_code: str
    evidence_refs: List[str]
    best_speedup: float
    next_action: str
    detail: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "task": self.task, "decision": self.decision,
            "reason_code": self.reason_code,
            "evidence_refs": list(self.evidence_refs),
            "best_speedup": round(float(self.best_speedup), 6),
            "next_action": self.next_action, "detail": self.detail,
            "extra": dict(self.extra),
        }


def decide(task: Any, ev: Any, experiments: Sequence[Dict[str, Any]] = (), *,
           target: float = TARGET_SPEEDUP,
           prior_reason: Optional[str] = None) -> Verdict:
    """按固定顺序给出裁决。`task` 可以是 TaskSpec 或任务名字符串。

    `prior_reason` 只用于 C7 经验复用：同签名历史结论已经给出 blocked_by_*，
    本轮又没有重新实验时，沿用该归因（而不是退化成默认带宽归因）。
    """
    name = getattr(task, "name", None) or str(task)
    exps = list(experiments or ())
    timed = [e for e in exps if _num(e.get("speedup_same_window")) is not None]
    best = max((_num(e.get("speedup_same_window")) or 0.0 for e in timed), default=0.0)
    refs = _refs(ev, timed)

    if best >= target:
        e = max(timed, key=lambda x: _num(x.get("speedup_same_window")) or 0.0)
        return _make(name, "adopt", "target_met", refs, best,
                     detail=(f"同窗加速比 {best:.3f}× ≥ {target:.2f}×；"
                             f"exp_id={e.get('exp_id')} family={e.get('family')}"))

    hint = getattr(ev, "verdict_hint", None) if ev is not None else None
    if not timed:
        if isinstance(prior_reason, str) and prior_reason in REASON_CODES \
                and prior_reason.startswith("blocked_by"):
            return _make(name, "keep_baseline", prior_reason, refs, best,
                         detail="经验复用：同签名历史结论直接沿用（本轮未重复实验）")
        if isinstance(hint, str) and hint.startswith("blocked_by"):
            return _make(name, "keep_baseline", hint, refs, best,
                         detail="证据层在实验前就判定该结构赢不了（未浪费实验预算）")
        if exps:
            return _make(name, "keep_baseline", "incorrect_candidate", refs, best,
                         detail="有实验记录但没有可计时的候选（G0–G3 全被拦）")
        return _make(name, "keep_baseline", "no_candidate", refs, best,
                     detail="假设空间没有产出候选：规则覆盖不足，不是任务不可解")

    if isinstance(hint, str) and hint.startswith("blocked_by"):
        return _make(name, "keep_baseline", hint, refs, best,
                     detail="实测未达标，且证据层归因指向结构性上限")

    if best >= NEAR_MISS:
        return _make(name, "inconclusive", "near_miss", refs, best,
                     detail=f"最好 {best:.3f}×，方向对但没到 {target:.2f}×")

    return _make(name, "keep_baseline", "blocked_by_bandwidth", refs, best,
                 detail="未达标且无更强的结构性解释，按默认访存上限归因")


# ---------------------------------------------------------------------------

def _make(name: str, decision: str, reason: str, refs: List[str],
          best: float, detail: str = "") -> Verdict:
    if reason not in REASON_CODES:
        raise ValueError(f"未知 reason_code: {reason!r}")
    return Verdict(task=name, decision=decision, reason_code=reason,
                   evidence_refs=refs, best_speedup=float(best),
                   next_action=NEXT_ACTIONS[reason], detail=detail)


def _num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _refs(ev: Any, timed: List[Dict[str, Any]]) -> List[str]:
    refs: List[str] = []
    if ev is not None:
        dom = getattr(ev, "dominant", None)
        if dom:
            refs.append(f"evidence.dominant={dom}")
        for p in getattr(ev, "profiles", ()) or ():
            if getattr(p, "ratio", 0) and p.ratio >= 2.5:
                refs.append(
                    f"micro:{p.op} numpy={p.ms_numpy:.2f}ms torch={p.ms_torch:.2f}ms "
                    f"({p.ratio:.1f}x)")
        for n in getattr(ev, "notes", ()) or ():
            if "io_bytes" in n or "bandwidth" in n:
                refs.append(n)
            if len(refs) >= 4:
                break
    for e in timed[:4]:
        refs.append(f"exp_id={e.get('exp_id')} family={e.get('family')} "
                    f"speedup_same_window={_num(e.get('speedup_same_window')):.4f} "
                    f"anchor_ms={e.get('anchor_ms_used')} median_ms={e.get('median_ms')}")
    return refs