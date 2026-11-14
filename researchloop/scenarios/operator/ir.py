# -*- coding: utf-8 -*-
"""C2 · 表示层：memory-bound 算子图的受限 IR。

设计目标（docs/05 §5 M2）：
  1) 只表示「能精确改写」的算子组合：逐元素算术、广播、归约、激活、softmax、
     归一化——覆盖 KernelBench L1–L2 的 memory-bound 融合场景；
  2) 不可变 + 可哈希：同一结构永远得到同一个对象/同一个 `to_source()`，
     去重与 golden 测试都直接用它；
  3) `to_source()` 是**规范形式**（不是 Python 源码），用于 golden 比对与日志，
     真正的代码生成在 `codegen.py`（Day 3）里做。

约定：标量代数（形状维、常量）与张量代数分开表示。`_is_scalar(node)` 为真时
该子树只含 `Const` / `Dim` / `External` / `ScalarOp`，可以参与形状推导；
张量节点（`Input` 及以上）不可参与标量运算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

__all__ = [
    "UnsupportedPattern", "Node", "Input", "Const", "Dim", "External",
    "ScalarOp", "TensorOp", "Unary", "Reduce", "Act", "Softmax", "LayerNorm",
    "is_scalar", "op_names", "walk",
]


class UnsupportedPattern(Exception):
    """参考实现里出现了 IR 不认识的写法。调用方应降级（只用变异族），不得猜。"""


# ---------------------------------------------------------------------------
# 节点基类
# ---------------------------------------------------------------------------

class Node:
    """所有节点的基类：不可变、可哈希、可比较（子类均为 frozen dataclass）。"""

    __slots__ = ()

    def children(self) -> Tuple["Node", ...]:
        return ()

    def walk(self) -> Iterator["Node"]:
        """先序遍历：自身 → 子节点（深度优先）。"""
        yield self
        for c in self.children():
            yield from c.walk()

    def to_source(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.to_source()


# ---------------------------------------------------------------------------
# 叶子：输入 / 常量 / 形状维 / 外部标量
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Input(Node):
    """参考实现入参字典里的一个张量，例如 inputs["x"] → Input("x")。"""
    name: str

    def to_source(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const(Node):
    """数值常量（标量）。"""
    value: float

    def to_source(self) -> str:
        return _fmt_num(self.value)


@dataclass(frozen=True)
class Dim(Node):
    """某个输入张量的某一维长度，例如 x.shape[-1] → Dim("x", -1)。"""
    input_name: str
    axis: int

    def to_source(self) -> str:
        return f"dim({self.input_name},{self.axis})"


@dataclass(frozen=True)
class External(Node):
    """能识别名字但取不到值的标量（如未解析的闭包变量）。只用于保真打印。"""
    name: str

    def to_source(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# 标量 / 张量运算
# ---------------------------------------------------------------------------

_SCALAR_BINOPS = ("add", "sub", "mul", "div", "pow", "maximum", "minimum")


@dataclass(frozen=True)
class ScalarOp(Node):
    """纯标量运算（形状维 / 常量），不会产生张量。"""
    op: str
    args: Tuple[Node, ...]

    def children(self) -> Tuple[Node, ...]:
        return tuple(self.args)

    def to_source(self) -> str:
        return f"{self.op}({', '.join(a.to_source() for a in self.args)})"


@dataclass(frozen=True)
class TensorOp(Node):
    """逐元素张量运算（含广播）。op ∈ add/sub/mul/div/pow/maximum/minimum。"""
    op: str
    args: Tuple[Node, ...]

    def children(self) -> Tuple[Node, ...]:
        return tuple(self.args)

    def to_source(self) -> str:
        return f"{self.op}({', '.join(a.to_source() for a in self.args)})"


@dataclass(frozen=True)
class Unary(Node):
    """逐元素一元函数：exp / log / tanh / sqrt / abs / erf / sigmoid / neg。"""
    op: str
    x: Node

    def children(self) -> Tuple[Node, ...]:
        return (self.x,)

    def to_source(self) -> str:
        return f"{self.op}({self.x.to_source()})"


@dataclass(frozen=True)
class Reduce(Node):
    """归约：kind ∈ mean/sum/max/min/prod；axis=None 表示全归约。"""
    kind: str
    x: Node
    axis: Optional[int] = None
    keepdims: bool = False

    def children(self) -> Tuple[Node, ...]:
        return (self.x,)

    def to_source(self) -> str:
        tail = f"axis={self.axis}"
        if self.keepdims:
            tail += ", keepdims=True"
        return f"reduce({self.kind}, {self.x.to_source()}, {tail})"


@dataclass(frozen=True)
class Act(Node):
    """激活函数。exact=False 仅用于显式标记近似实现（tanh-GELU 等）。"""
    kind: str
    x: Node
    exact: bool = True

    def children(self) -> Tuple[Node, ...]:
        return (self.x,)

    def to_source(self) -> str:
        flag = "" if self.exact else ", approx"
        return f"act({self.kind}, {self.x.to_source()}{flag})"


@dataclass(frozen=True)
class Softmax(Node):
    """沿 axis 的 softmax（内部含 max/sum/exp，不展开）。"""
    x: Node
    axis: int = -1

    def children(self) -> Tuple[Node, ...]:
        return (self.x,)

    def to_source(self) -> str:
        return f"softmax({self.x.to_source()}, axis={self.axis})"


@dataclass(frozen=True)
class LayerNorm(Node):
    """归一化层。normalized_shape 为形状节点元组（通常是 (Dim(x,-1),)）。"""
    x: Node
    normalized_shape: Tuple[Node, ...]
    weight: Optional[Node] = None
    bias: Optional[Node] = None
    eps: Node = Const(1e-5)

    def children(self) -> Tuple[Node, ...]:
        out: List[Node] = [self.x, *self.normalized_shape, self.eps]
        if self.weight is not None:
            out.append(self.weight)
        if self.bias is not None:
            out.append(self.bias)
        return tuple(out)

    def to_source(self) -> str:
        shape = ", ".join(n.to_source() for n in self.normalized_shape)
        tail = ""
        if self.weight is not None:
            tail += f", weight={self.weight.to_source()}"
        if self.bias is not None:
            tail += f", bias={self.bias.to_source()}"
        return (f"layer_norm({self.x.to_source()}, norm=({shape},)"
                f"{tail}, eps={self.eps.to_source()})")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def is_scalar(node: Node) -> bool:
    """该子树是否只含标量节点（可参与形状推导）。"""
    if isinstance(node, (Const, Dim, External)):
        return True
    if isinstance(node, ScalarOp):
        return all(is_scalar(a) for a in node.args)
    return False


def walk(node: Node) -> Iterator[Node]:
    return node.walk()


def op_names(node: Node) -> List[str]:
    """算子多重集（用于 C7 的结构指纹与相似度）。顺序为先序遍历。"""
    names: List[str] = []
    for n in node.walk():
        if isinstance(n, Input):
            names.append("input")
        elif isinstance(n, TensorOp):
            names.append(n.op)
        elif isinstance(n, ScalarOp):
            names.append(n.op)
        elif isinstance(n, Unary):
            names.append(n.op)
        elif isinstance(n, Reduce):
            names.append(f"reduce:{n.kind}")
        elif isinstance(n, Act):
            names.append(f"act:{n.kind}")
        elif isinstance(n, Softmax):
            names.append("softmax")
        elif isinstance(n, LayerNorm):
            names.append("layer_norm")
    return names


def _fmt_num(v: float) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return f"{float(v):.6g}"


def norm_axis(axis: int, rank: int) -> int:
    """把负轴规格化到 [0, rank)；越界抛 UnsupportedPattern。"""
    a = int(axis)
    if a < 0:
        a += rank
    if not (0 <= a < rank):
        raise UnsupportedPattern(f"归约轴 {axis} 超出秩 {rank} 的范围")
    return a


def map_children(node: Node, fn) -> Node:
    """重建节点：把每个直接子节点替换为 fn(child)。叶子原样返回。"""
    if isinstance(node, (Input, Const, Dim, External)):
        return node
    if isinstance(node, (ScalarOp, TensorOp)):
        return type(node)(node.op, tuple(fn(a) for a in node.args))
    if isinstance(node, Unary):
        return Unary(node.op, fn(node.x))
    if isinstance(node, Reduce):
        return Reduce(node.kind, fn(node.x), node.axis, node.keepdims)
    if isinstance(node, Act):
        return Act(node.kind, fn(node.x), node.exact)
    if isinstance(node, Softmax):
        return Softmax(fn(node.x), node.axis)
    if isinstance(node, LayerNorm):
        return LayerNorm(
            fn(node.x),
            tuple(fn(s) for s in node.normalized_shape),
            weight=None if node.weight is None else fn(node.weight),
            bias=None if node.bias is None else fn(node.bias),
            eps=fn(node.eps),
        )
    raise TypeError(f"map_children: 未支持的节点类型 {type(node).__name__}")


def replace_subtree(root: Node, target: Node, new: Node) -> Node:
    """把 root 里第一处（先序）与 target 相等的子树换成 new。"""
    if root == target:
        return new
    return map_children(root, lambda c: replace_subtree(c, target, new))


def count_nodes(node: Node) -> int:
    """张量级算子个数（代价代理：越少代表访存趟数越少）。"""
    n = 0
    for x in node.walk():
        if isinstance(x, (TensorOp, Unary, Reduce, Act, Softmax, LayerNorm)):
            n += 1
    return n