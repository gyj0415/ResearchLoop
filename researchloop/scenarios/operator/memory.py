# -*- coding: utf-8 -*-
"""C7 · 经验库：把一次裁决沉淀成下一次能直接用的提示。

为什么需要（docs/05 §4.4）：同一类结构（例如 `mean(x + b)` 且 b 沿归约轴恒定）
在不同任务里反复出现；如果每次都从零推一遍，等于把已经付过学费的结论再付一遍。
经验库只记**可迁移的结构结论**，不记任务名黑名单——所以它不是「刷过的题」，
而是「结构 → 该用哪条规则链 / 这个结构的上限在哪」。

三个能力：
  1. `signature(ir_node)`：结构指纹（`to_source()` 的 sha1），精确命中用；
  2. `recall(task, ir_node)`：没有精确命中时按算子多重集 Jaccard 找最近邻，
     返回带相似度排序的 Case 列表；
  3. `transfer_hints(cases)`：把 Case 翻成搜索阶段能直接消费的提示
     （建议规则链、建议规避的族、已知上限）。

诚实边界：
  - 经验只提供**先验**（先试哪条规则、要不要直接放弃），任何命中都必须重新
    过一遍门控与同窗计时；经验库里的加速比是历史测量值，不得当作当前任务的成绩。
  - 默认落盘路径 `artifacts/cases.jsonl` 是**追加写**的本地文件（不是官方冻结
    产物 experiments.jsonl / budget.json）；测试一律显式传 tmp 路径。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import ir
from .ir import Node, op_names

__all__ = ["Case", "signature", "remember", "recall", "load_cases",
           "transfer_hints", "case_from_verdict", "DEFAULT_PATH",
           "SIM_THRESHOLD"]

# 经验文件落在 ResearchLoop 已忽略的 runs/ 下，避免写到 8 的 artifacts/。
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "runs" / "operator-cases.jsonl"

# 低于这个 Jaccard 相似度的历史 case 不作为「同类」推荐（但仍然保留在文件里）
SIM_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    """一次「结构 → 结论」的经验条目。"""

    signature: str                       # IR 规范形式的 sha1
    task: str
    ir: str                              # 当时的 IR 规范形式
    ops: Tuple[str, ...]                 # 算子多重集（相似度用）
    family: str = ""                     # 命中的假设族
    rule_chain: Tuple[str, ...] = ()     # 生效的规则链
    decision: str = ""                   # adopt / keep_baseline / inconclusive
    reason_code: str = ""
    best_speedup: float = 0.0            # 同窗实测（历史值，不可当现成绩）
    shape_rank: int = 0
    note: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "task": self.task,
            "ir": self.ir,
            "ops": list(self.ops),
            "family": self.family,
            "rule_chain": list(self.rule_chain),
            "decision": self.decision,
            "reason_code": self.reason_code,
            "best_speedup": round(float(self.best_speedup), 6),
            "shape_rank": int(self.shape_rank),
            "note": self.note,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_json(cls, raw: Dict[str, Any]) -> "Case":
        """从一行 JSON 恢复。字段缺失用默认值，坏行由调用方跳过。"""
        return cls(
            signature=str(raw["signature"]),
            task=str(raw.get("task", "")),
            ir=str(raw.get("ir", "")),
            ops=tuple(str(x) for x in raw.get("ops", ())),
            family=str(raw.get("family", "")),
            rule_chain=tuple(str(x) for x in raw.get("rule_chain", ())),
            decision=str(raw.get("decision", "")),
            reason_code=str(raw.get("reason_code", "")),
            best_speedup=float(raw.get("best_speedup", 0.0) or 0.0),
            shape_rank=int(raw.get("shape_rank", 0) or 0),
            note=str(raw.get("note", "")),
            meta=dict(raw.get("meta", {}) or {}),
        )


# ---------------------------------------------------------------------------
# 指纹与相似度
# ---------------------------------------------------------------------------

def signature(ir_node: Node) -> str:
    """结构指纹：IR 规范形式（`to_source()`）的 sha1。"""
    return hashlib.sha1(ir_node.to_source().encode("utf-8")).hexdigest()


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """算子多重集 Jaccard（按多重集计数，不是集合）。"""
    if not a and not b:
        return 1.0
    ca: Dict[str, int] = {}
    cb: Dict[str, int] = {}
    for x in a:
        ca[x] = ca.get(x, 0) + 1
    for x in b:
        cb[x] = cb.get(x, 0) + 1
    keys = set(ca) | set(cb)
    inter = sum(min(ca.get(k, 0), cb.get(k, 0)) for k in keys)
    union = sum(max(ca.get(k, 0), cb.get(k, 0)) for k in keys)
    return (inter / union) if union else 1.0


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------

def remember(case: Case, path: Optional[os.PathLike] = None,
             *, dedup: bool = True) -> Path:
    """追加写入经验库；默认去重（同 signature + 同结论不重复记）。"""
    target = Path(path) if path is not None else DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if dedup:
        for old in load_cases(target):
            if old.signature == case.signature and old.decision == case.decision \
                    and old.rule_chain == case.rule_chain:
                return target
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(case.to_json(), ensure_ascii=False) + "\n")
    return target


def load_cases(path: Optional[os.PathLike] = None) -> List[Case]:
    """读整库。文件不存在 → 空表；坏行跳过（不让一行脏数据毁掉整库）。"""
    target = Path(path) if path is not None else DEFAULT_PATH
    if not target.exists():
        return []
    out: List[Case] = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict) or "signature" not in raw:
                    continue
                out.append(Case.from_json(raw))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return out


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

def recall(task: Any, ir_node: Node, path: Optional[os.PathLike] = None,
           *, top_k: int = 3, threshold: float = SIM_THRESHOLD) -> List[Case]:
    """按「精确指纹 → 算子集相似度」检索同类 case。

    `task` 只用于标注与排同任务优先，不作为过滤条件——否则就变成「刷过的题」。
    """
    name = getattr(task, "name", None) or str(task)
    cases = load_cases(path)
    if not cases:
        return []

    sig = signature(ir_node)
    ops = tuple(op_names(ir_node))
    scored: List[Tuple[float, int, int, Case]] = []
    for c in cases:
        if c.signature == sig:
            sim, kind = 1.0, 0
        else:
            sim = jaccard(ops, c.ops)
            if sim < threshold:
                continue
            kind = 1
        same_task = 0 if c.task == name else 1
        scored.append((sim, same_task, kind, c))

    scored.sort(key=lambda t: (-t[0], t[1], t[2], t[3].signature))
    out: List[Case] = []
    seen: set = set()
    for _, _, _, c in scored:
        if c.signature in seen:
            continue
        seen.add(c.signature)
        out.append(c)
        if len(out) >= max(0, int(top_k)):
            break
    return out


# ---------------------------------------------------------------------------
# 迁移提示
# ---------------------------------------------------------------------------

def transfer_hints(cases: Iterable[Case]) -> Dict[str, Any]:
    """把检索到的 case 翻成搜索/预算阶段能直接用的提示。

    返回字段：
      prefer_rules   —— 值得优先尝试的规则链（按历史成败加权，成功优先）
      avoid_families —— 历史上明确打不过的族（配 best_speedup 上限）
      known_ceiling  —— 同类结构的最高历史同窗加速比（0 表示没有可用数字）
      notes          —— 给人看的解释，可直接贴进报告
    """
    prefer: List[Tuple[float, Tuple[str, ...]]] = []
    avoid: List[Tuple[float, str]] = []
    ceiling = 0.0
    notes: List[str] = []

    for c in cases:
        if c.best_speedup > 0:
            ceiling = max(ceiling, float(c.best_speedup))
        if c.decision == "adopt" and c.rule_chain:
            prefer.append((float(c.best_speedup), tuple(c.rule_chain)))
            notes.append(
                f"同类结构在 {c.task} 上用过 {'+'.join(c.rule_chain)}，"
                f"同窗 {c.best_speedup:.3f}x")
        elif c.decision == "keep_baseline" and c.reason_code.startswith("blocked_by"):
            avoid.append((float(c.best_speedup), c.reason_code))
            notes.append(
                f"同类结构在 {c.task} 上被判定 {c.reason_code}"
                f"（最好 {c.best_speedup:.3f}x），不必重复付学费")
        elif c.reason_code == "no_candidate":
            notes.append(f"{c.task} 曾因规则覆盖不足而无候选（该扩规则，不是任务不可解）")

    # 成功链按 (speedup 降序, 链长度升序, 字典序) 稳定排序；失败族按上限升序
    prefer.sort(key=lambda t: (-t[0], len(t[1]), t[1]))
    avoid.sort(key=lambda t: (t[0], t[1]))

    return {
        "prefer_rules": [list(chain) for _, chain in prefer][:3],
        "avoid_families": [{"reason_code": r, "best_speedup": s}
                           for s, r in avoid][:5],
        "known_ceiling": round(ceiling, 6),
        "notes": notes[:8],
    }


# ---------------------------------------------------------------------------
# 从裁决生成 case
# ---------------------------------------------------------------------------

def case_from_verdict(task: Any, ir_node: Node, verdict: Any, *,
                      family: str = "",
                      rule_chain: Sequence[str] = (),
                      note: str = "",
                      meta: Optional[Dict[str, Any]] = None) -> Case:
    """把 (任务, IR, 裁决) 合成一条可入库的经验。"""
    name = getattr(task, "name", None) or str(task)
    rank = 0
    for n in ir_node.walk():
        if isinstance(n, ir.Input):
            fn = getattr(task, "shape_rank", None)
            if callable(fn):
                rank = max(rank, int(fn(n.name)))
    return Case(
        signature=signature(ir_node),
        task=name,
        ir=ir_node.to_source(),
        ops=tuple(op_names(ir_node)),
        family=str(family),
        rule_chain=tuple(str(x) for x in rule_chain),
        decision=str(getattr(verdict, "decision", "")),
        reason_code=str(getattr(verdict, "reason_code", "")),
        best_speedup=float(getattr(verdict, "best_speedup", 0.0) or 0.0),
        shape_rank=rank,
        note=note,
        meta=dict(meta or {}),
    )