# -*- coding: utf-8 -*-
"""C4 · 假设搜索：规则组合 + 去重 + 按代价排序。

输入一个 IR，输出若干**结构不同**的候选假设（每条带 rule_chain）。
搜索是 BFS，深度默认 3；去重用 `to_source()` 规范形式；排序用「张量算子个数」
作为访存趟数代理（越少越可能快），不引入训练/打分模型（几十条样本训不出东西）。

`expected_gain` 只是**代价模型给出的先验**（before/after 算子数），不是性能承诺；
真实快慢一律由 G4 实测，二者在报告里分开写。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from . import codegen, ir
from .ir import Node, UnsupportedPattern
from .rules import RULES

__all__ = ["Hypothesis", "Candidate", "search", "build_candidates",
           "FAMILY_OF_RULE"]

# 规则 → 假设族（报告里按族看收益，而不是按单条规则）
FAMILY_OF_RULE: Dict[str, str] = {
    "R1_reduce_of_add_mean": "alg-eliminate",
    "R1b_reduce_of_add_sum": "alg-eliminate",
    "R2_scale_through_reduce": "alg-refactor",
    "R2b_mean_of_scaled_add": "alg-eliminate",
    "R3_scale_fold": "alg-refactor",
    "R5_softmax_shift": "softmax-shift",
}


@dataclass(frozen=True)
class Hypothesis:
    hid: str
    family: str
    origin: str                       # "rule:R1_reduce_of_add_mean+R4_prealloc"
    ir_before: str
    ir_after: str
    rule_chain: Tuple[str, ...]
    expected_gain: float
    risk: str                          # exact / approx / layout_dependent
    cost_before: int
    cost_after: int
    node: Node = field(compare=False, repr=False)

    def to_json(self) -> Dict[str, object]:
        return {
            "hid": self.hid, "family": self.family, "origin": self.origin,
            "ir_before": self.ir_before, "ir_after": self.ir_after,
            "rule_chain": list(self.rule_chain),
            "expected_gain": round(self.expected_gain, 3),
            "risk": self.risk,
            "cost_before": self.cost_before, "cost_after": self.cost_after,
        }


@dataclass
class Candidate:
    hypothesis: Hypothesis
    source: str
    family: str
    est_bytes: int

    def to_json(self) -> Dict[str, object]:
        return {
            "hid": self.hypothesis.hid, "family": self.family,
            "rule_chain": list(self.hypothesis.rule_chain),
            "est_bytes": self.est_bytes,
            "source_chars": len(self.source),
        }


# ---------------------------------------------------------------------------

def _structured_cost(node: Node, spec: object) -> float:
    """结构性访存代价：满秩算子算 1 遍，降秩算子算 0.25 遍。

    目的不是精确预测毫秒，而是把「用一个小张量的逐元素后续替换掉一个满尺寸
    临时量」这件事在排序里体现出来（R1 的核心收益）。
    """
    from .rules import rank_of

    max_rank = max((rank_of(n, spec) for n in node.walk()
                    if isinstance(n, ir.Input)), default=0)
    cost = 0.0
    for n in node.walk():
        if isinstance(n, ir.Softmax):
            cost += 3.0
        elif isinstance(n, ir.LayerNorm):
            cost += 3.0
        elif isinstance(n, (ir.TensorOp, ir.Unary, ir.Act, ir.Reduce)):
            r = rank_of(n, spec)
            cost += 1.0 if (max_rank and r >= max_rank) else 0.25
    return cost


def search(ir_node: Node, spec: object, cap: int = 12, depth: int = 3
           ) -> List[Hypothesis]:
    """BFS 搜索可改写空间；返回按代价升序的前 cap 条。"""
    base_key = ir_node.to_source()
    seen = {base_key}
    results: List[Hypothesis] = []
    frontier: List[Tuple[Node, Tuple[str, ...]]] = [(ir_node, ())]
    cost_before = _structured_cost(ir_node, spec)

    for _ in range(max(1, int(depth))):
        nxt: List[Tuple[Node, Tuple[str, ...]]] = []
        for node, chain in frontier:
            for rid, fn in RULES:
                for target in node.walk():
                    try:
                        applied = fn(target, spec)
                    except UnsupportedPattern:
                        continue
                    if applied is None or applied == target:
                        continue
                    new_node = ir.replace_subtree(node, target, applied)
                    key = new_node.to_source()
                    if key in seen:
                        continue
                    seen.add(key)
                    new_chain = chain + (rid,)
                    family = FAMILY_OF_RULE.get(new_chain[0], "alg-refactor")
                    cost_after = _structured_cost(new_node, spec)
                    h = Hypothesis(
                        hid=f"H-auto-{len(results) + 1:02d}",
                        family=family,
                        origin="rule:" + "+".join(new_chain) + "+R4_prealloc",
                        ir_before=base_key,
                        ir_after=key,
                        rule_chain=new_chain,
                        expected_gain=(cost_before / cost_after) if cost_after else 1.0,
                        risk="exact",
                        cost_before=cost_before,
                        cost_after=cost_after,
                        node=new_node,
                    )
                    results.append(h)
                    nxt.append((new_node, new_chain))
        frontier = nxt
        if len(results) >= cap * 4 or not frontier:
            break

    results.sort(key=lambda h: (h.cost_after, len(h.rule_chain), h.ir_after))
    return results[:cap]


def build_candidates(hypotheses: Sequence[Hypothesis], spec: object) -> List[Candidate]:
    """把假设落成候选源码（codegen 失败的单条跳过并记录在异常里由调用方处理）。"""
    out: List[Candidate] = []
    for h in hypotheses:
        source = codegen.generate(spec, h.node)
        out.append(Candidate(
            hypothesis=h,
            source=source,
            family=h.family,
            est_bytes=_est_bytes(h.node, spec),
        ))
    return out


def _est_bytes(node: Node, spec: object) -> int:
    """代价代理：张量算子个数 × 最大 shape 的 float32 字节数。"""
    shapes = list(getattr(spec, "shapes", ()) or ())
    numel = 1
    for s in shapes:
        n = 1
        for v in s:
            n *= int(v)
        numel = max(numel, n)
    return int(ir.count_nodes(node) * numel * 4)