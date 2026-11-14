# -*- coding: utf-8 -*-
"""C4 · 重写规则：结构不同的候选（不是参数变异）。

每条规则都是**精确等式**（不引入近似），并且带形状/轴前提；前提不满足就返回
None，宁可不生成也不猜。规则的依据写在 `_R1_NOTE` 等常量里，报告可直接引用。

已实现（对应 docs/05 §5 M4 表）：
  R1_reduce_of_add_mean   mean(x+b) → sum(x)·(1/T) + b      前提：b 沿归约轴恒定
  R1b_reduce_of_add_sum   sum(x+b)  → sum(x) + b·T          前提：同上
  R2_scale_through_reduce mean/sum(s·x) → s·mean/sum(x)     前提：s 为标量
  R2b_mean_of_scaled_add  mean(s·x+b) → s·mean(x)+b         前提：s 标量且 b 沿轴恒定
  R3_scale_fold           (x·a)·b → x·(a·b)                 前提：a,b 标量
  R5_softmax_shift        softmax(x+c) → softmax(x)         前提：c 为标量常量
  R6_identity_guarded     无规则命中时不生成（由 search 统一处理）

R4（预分配/就地）不是结构改写，由 codegen 落实（归约用 out= 写入小缓冲区）。
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from . import ir
from .ir import (
    Act, Const, Dim, Input, LayerNorm, Node, Reduce, ScalarOp, Softmax,
    TensorOp, Unary, UnsupportedPattern, is_scalar,
)

__all__ = ["RULES", "Rule", "apply_rule", "apply_first", "rewritable_reduce"]

Note = str
Rule = Callable[[Node, object], Optional[Node]]

_R1_NOTE = "mean(x+b)=sum(x)/T+b（b 沿归约轴恒定，精确）"
_R1B_NOTE = "sum(x+b)=sum(x)+T·b（b 沿归约轴恒定，精确）"
_R2_NOTE = "mean/sum(s·x)=s·mean/sum(x)（标量线性，精确）"
_R2B_NOTE = "mean/sum(s·x+b)=s·mean/sum(x)+b（标量线性 + 轴恒定，精确）"
_R3_NOTE = "(x·a)·b=x·(a·b)（标量结合律，精确）"
_R5_NOTE = "softmax(x+c)=softmax(x)（c 为标量常量，精确）"

_NOTES = {
    "R1_reduce_of_add_mean": _R1_NOTE,
    "R1b_reduce_of_add_sum": _R1B_NOTE,
    "R2_scale_through_reduce": _R2_NOTE,
    "R2b_mean_of_scaled_add": _R2B_NOTE,
    "R3_scale_fold": _R3_NOTE,
    "R5_softmax_shift": _R5_NOTE,
}


# ---------------------------------------------------------------------------
# 前提判断
# ---------------------------------------------------------------------------

def rank_of(node: Node, spec: object) -> int:
    """符号秩。标量为 0；未知返回 0。"""
    if isinstance(node, Input):
        fn = getattr(spec, "shape_rank", None)
        return int(fn(node.name)) if callable(fn) else 0
    if isinstance(node, Reduce):
        r = rank_of(node.x, spec)
        return r if node.keepdims else max(0, r - 1)
    if isinstance(node, (TensorOp, Unary, Act)):
        kids = node.args if isinstance(node, TensorOp) else (node.x,)
        return max((rank_of(k, spec) for k in kids), default=0)
    if isinstance(node, (Softmax, LayerNorm)):
        return rank_of(node.x, spec)
    return 0


def invariant_along(op: Node, axis: int, src_rank: int, spec: object) -> bool:
    """算子结果沿 axis 是否恒定（可移出归约）。"""
    if is_scalar(op):
        return True
    if isinstance(op, Input):
        r = rank_of(op, spec)
        # 广播项只覆盖尾部 r 维；归约轴落在其左侧 → 该轴取值不变
        return r < src_rank and 0 <= axis < src_rank - r
    return False


# ---------------------------------------------------------------------------
# 规则
# ---------------------------------------------------------------------------

def r1_reduce_of_add_mean(node: Node, spec: object) -> Optional[Node]:
    if not (isinstance(node, Reduce) and node.kind == "mean" and node.axis is not None):
        return None
    add = node.x
    if not (isinstance(add, TensorOp) and add.op == "add"):
        return None
    for src, other in (add.args, add.args[::-1]):
        if not isinstance(src, Input):
            continue
        rank = rank_of(src, spec)
        try:
            axis = ir.norm_axis(node.axis, rank)
        except UnsupportedPattern:
            continue
        if not invariant_along(other, axis, rank, spec):
            continue
        inv = ScalarOp("div", (Const(1.0), Dim(src.name, node.axis)))
        summed = Reduce("sum", src, axis=node.axis, keepdims=node.keepdims)
        return TensorOp("add", (TensorOp("mul", (summed, inv)), other))
    return None


def r1b_reduce_of_add_sum(node: Node, spec: object) -> Optional[Node]:
    if not (isinstance(node, Reduce) and node.kind == "sum" and node.axis is not None):
        return None
    add = node.x
    if not (isinstance(add, TensorOp) and add.op == "add"):
        return None
    for src, other in (add.args, add.args[::-1]):
        if not isinstance(src, Input):
            continue
        rank = rank_of(src, spec)
        try:
            axis = ir.norm_axis(node.axis, rank)
        except UnsupportedPattern:
            continue
        if not invariant_along(other, axis, rank, spec):
            continue
        total = Dim(src.name, node.axis)
        summed = Reduce("sum", src, axis=node.axis, keepdims=node.keepdims)
        scaled = TensorOp("mul", (other, total)) if not is_scalar(other) else \
            ScalarOp("mul", (other, total))
        return TensorOp("add", (summed, scaled))
    return None


def r2_scale_through_reduce(node: Node, spec: object) -> Optional[Node]:
    if not (isinstance(node, Reduce) and node.kind in ("mean", "sum")):
        return None
    mul = node.x
    if not (isinstance(mul, TensorOp) and mul.op == "mul"):
        return None
    for x, s in (mul.args, mul.args[::-1]):
        if not is_scalar(s) or is_scalar(x):
            continue
        red = Reduce(node.kind, x, axis=node.axis, keepdims=node.keepdims)
        return TensorOp("mul", (red, s))
    return None


def r2b_mean_of_scaled_add(node: Node, spec: object) -> Optional[Node]:
    if not (isinstance(node, Reduce) and node.kind in ("mean", "sum")
            and node.axis is not None):
        return None
    add = node.x
    if not (isinstance(add, TensorOp) and add.op == "add"):
        return None
    for scaled, other in (add.args, add.args[::-1]):
        if not (isinstance(scaled, TensorOp) and scaled.op == "mul"):
            continue
        for x, s in (scaled.args, scaled.args[::-1]):
            if not is_scalar(s) or is_scalar(x):
                continue
            src_rank = rank_of(x, spec)
            try:
                axis = ir.norm_axis(node.axis, src_rank)
            except UnsupportedPattern:
                continue
            if not invariant_along(other, axis, src_rank, spec):
                continue
            red = Reduce(node.kind, x, axis=node.axis, keepdims=node.keepdims)
            return TensorOp("add", (TensorOp("mul", (red, s)), other))
    return None


def r3_scale_fold(node: Node, spec: object) -> Optional[Node]:
    if not (isinstance(node, TensorOp) and node.op == "mul"):
        return None
    left, right = node.args
    if isinstance(left, TensorOp) and left.op == "mul" and is_scalar(right):
        inner, outer = left, right
    elif isinstance(right, TensorOp) and right.op == "mul" and is_scalar(left):
        inner, outer = right, left
    else:
        return None
    a, b = inner.args
    if is_scalar(a) and is_scalar(b):
        return ScalarOp("mul", (ScalarOp("mul", (a, b)), outer))
    if is_scalar(b) and not is_scalar(a):
        return TensorOp("mul", (a, ScalarOp("mul", (b, outer))))
    if is_scalar(a) and not is_scalar(b):
        return TensorOp("mul", (b, ScalarOp("mul", (a, outer))))
    return None


def r5_softmax_shift(node: Node, spec: object) -> Optional[Node]:
    if not isinstance(node, Softmax):
        return None
    add = node.x
    if not (isinstance(add, TensorOp) and add.op == "add"):
        return None
    for shifted, other in (add.args, add.args[::-1]):
        if isinstance(other, Const):
            return Softmax(shifted, node.axis)
    return None


RULES: Tuple[Tuple[str, Rule], ...] = (
    ("R1_reduce_of_add_mean", r1_reduce_of_add_mean),
    ("R1b_reduce_of_add_sum", r1b_reduce_of_add_sum),
    ("R2_scale_through_reduce", r2_scale_through_reduce),
    ("R2b_mean_of_scaled_add", r2b_mean_of_scaled_add),
    ("R3_scale_fold", r3_scale_fold),
    ("R5_softmax_shift", r5_softmax_shift),
)

# 只有这四条会真正「把归约变便宜」：消掉一遍满尺寸访存或把标量提出归约。
_REDUCTION_RULES = ("R1_reduce_of_add_mean", "R1b_reduce_of_add_sum",
                    "R2_scale_through_reduce", "R2b_mean_of_scaled_add")


def rewritable_reduce(node: Node, spec: object) -> Optional[str]:
    """`node` 若是能被归约规则精确改写的 Reduce，返回规则名；否则 None。

    单一事实来源：C3 证据层用它判断「归约是不是硬上限」。以前证据层自己维护了
    一份更窄的判断（只认 `Reduce(Add(Input, Input))`），结果 `mean(s·x+b)`（R2b
    能赢）偶尔被单算子微基准的噪声判成 blocked_by_reduction —— 规则能改写就不该
    在实验前被拦。任何新增归约规则只改这里，证据层自动跟上。
    """
    if not (isinstance(node, Reduce) and node.axis is not None):
        return None
    for name, fn in RULES:
        if name in _REDUCTION_RULES and fn(node, spec) is not None:
            return name
    return None


def apply_rule(rid: str, node: Node, spec: object) -> Optional[Node]:
    for name, fn in RULES:
        if name == rid:
            return fn(node, spec)
    raise KeyError(f"未知规则 {rid!r}；已有 {[n for n, _ in RULES]}")


def apply_first(root: Node, spec: object
                ) -> Optional[Tuple[Node, str, str]]:
    """先序找到第一处可改写的子树，返回 (新根, 规则名, 说明)。"""
    for name, fn in RULES:
        for target in root.walk():
            new = fn(target, spec)
            if new is None or new == target:
                continue
            return ir.replace_subtree(root, target, new), name, _NOTES[name]
    return None