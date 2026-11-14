# -*- coding: utf-8 -*-
"""C4 · IR → 可执行 numpy 源码。

三条硬要求（docs/05 §5 M4）：
  1. 中间结果预分配，能就地就地：**归约**用 `np.empty` + `out=` 直接写进小缓冲区，
     逐元素算子在前一个临时量只被用一次时用 `out=` 就地更新；
  2. 形状从运行时张量推导（`x.shape` / `_rshape(shape, axis)`），源码里**禁止**
     硬编码 shape，边界 shape（batch=1、999/127 等不整除长度）必须能跑；
  3. 生成后 `ast.parse` + `compile` 自检，失败即抛 `CodegenError`，不交给门控去撞。

生成的候选只依赖 numpy，不含 torch（G0 的禁用导入根：torch/src/eval/ctypes/subprocess）。
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import Dict, List, Optional

from . import ir
from .ir import (
    Act, Const, Dim, External, Input, LayerNorm, Node, Reduce, ScalarOp,
    Softmax, TensorOp, Unary, UnsupportedPattern, is_scalar,
)
from .rules import rank_of

__all__ = ["CodegenError", "generate", "generate_with_meta"]

# 与 src/hypotheses.py ERF_POLY_SRC 同一近似（|eps| <= 1.5e-7），仅 gelu/erf 用到
_ERF_HELPER = '''\
_ERF_P = 0.3275911
_ERF_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def _erf(x):
    ax = np.abs(x)
    t = 1.0 / (1.0 + _ERF_P * ax)
    poly = t * (_ERF_A[0] + t * (_ERF_A[1] + t * (_ERF_A[2] + t * (
        _ERF_A[3] + t * _ERF_A[4]))))
    return np.sign(x) * (1.0 - poly * np.exp(-ax * ax))
'''

_RSHAPE_HELPER = '''\
def _rshape(shape, axis):
    axis = axis % len(shape)
    return shape[:axis] + shape[axis + 1:]
'''

_BINOP = {
    "add": "add", "sub": "subtract", "mul": "multiply", "div": "divide",
    "pow": "power", "maximum": "maximum", "minimum": "minimum",
}

_UNARY = {
    "exp": "exp", "log": "log", "tanh": "tanh", "sqrt": "sqrt",
    "abs": "abs", "neg": "negative", "sigmoid": "sigmoid", "erf": "erf",
    "rsqrt": "rsqrt",
}


class CodegenError(Exception):
    """IR 无法安全地翻译成 numpy 源码。"""


def generate(spec: object, node: Node) -> str:
    """IR → 候选源码（`def run(inputs)` 契约与 gates 一致）。"""
    return generate_with_meta(spec, node)["source"]


def generate_with_meta(spec: object, node: Node) -> Dict[str, object]:
    gen = _Emitter(spec, node)
    source = gen.build()
    try:
        ast.parse(source)
        compile(source, "<research-codegen>", "exec")
    except SyntaxError as e:
        raise CodegenError(f"生成源码无法编译：{e}") from e
    if "torch" in source:
        raise CodegenError("生成源码里出现 torch —— 违反禁用导入约定")
    return {
        "source": source,
        "uses_erf": gen.uses_erf,
        "n_ops": ir.count_nodes(node),
        "inplace_ops": gen.inplace_ops,
        "numpy_imports": ["numpy"],
    }


# ---------------------------------------------------------------------------

class _Emitter:
    def __init__(self, spec: object, root: Node) -> None:
        self.spec = spec
        self.root = root
        self.lines: List[str] = []
        self.uses: Counter = Counter()
        for n in root.walk():
            self.uses[n] += 1
        self.owned: set = set()
        self.input_vars: Dict[str, str] = {}
        self.uses_erf = False
        self.needs_rshape = False
        self.inplace_ops = 0
        self._tmp_no = 0

    # ---------------------------------------------------------------- build
    def build(self) -> str:
        body_var = self._emit(self.root)
        used_inputs = self._used_inputs()
        head = ["import numpy as np", "", ""]
        if self.uses_erf:
            head.append(_ERF_HELPER)
        if self.needs_rshape:
            head.append(_RSHAPE_HELPER)
        head.append("")
        head.append("def run(inputs):")
        for name in used_inputs:
            head.append(f'    {self._input_var(name)} = inputs["{name}"]')
        for line in self.lines:
            head.append("    " + line)
        head.append(f"    return {body_var}")
        head.append("")
        return "\n".join(head)

    def _used_inputs(self) -> List[str]:
        seen: List[str] = []
        for n in self.root.walk():
            if isinstance(n, Input) and n.name not in seen:
                seen.append(n.name)
            elif isinstance(n, (Dim,)) and n.input_name not in seen:
                seen.append(n.input_name)
        return seen

    # ------------------------------------------------------------- helpers
    def _tmp(self) -> str:
        self._tmp_no += 1
        return f"t{self._tmp_no}"

    def _input_var(self, name: str) -> str:
        if name not in self.input_vars:
            self.input_vars[name] = name
        return name

    def _consume(self, node: Node) -> int:
        self.uses[node] -= 1
        return self.uses[node]

    # --------------------------------------------------------------- emit
    def _emit(self, node: Node) -> str:
        """返回持有该节点值的变量名（标量返回字面量表达式）。"""
        if is_scalar(node):
            return self._scalar(node)

        if isinstance(node, Input):
            self._consume(node)
            return self._input_var(node.name)

        self._consume(node)

        if isinstance(node, Reduce):
            return self._emit_reduce(node)
        if isinstance(node, TensorOp):
            return self._emit_tensor_op(node)
        if isinstance(node, Unary):
            return self._emit_unary(node)
        if isinstance(node, Act):
            return self._emit_act(node)
        if isinstance(node, Softmax):
            return self._emit_softmax(node)
        if isinstance(node, LayerNorm):
            return self._emit_layer_norm(node)
        raise CodegenError(f"不支持的节点类型：{type(node).__name__}")

    # ---------------------------------------------------------- reductions
    def _emit_reduce(self, node: Reduce) -> str:
        xv = self._emit(node.x)
        fn = node.kind
        if node.axis is None:
            t = self._tmp()
            keep = ", keepdims=True" if node.keepdims else ""
            self.lines.append(f"{t} = np.{fn}({xv}{keep})")
            return t
        if node.keepdims:
            t = self._tmp()
            self.lines.append(f"{t} = np.{fn}({xv}, axis={node.axis}, keepdims=True)")
            return t
        t = self._tmp()
        self.lines.append(
            f"{t} = np.empty(_rshape({xv}.shape, {node.axis}), dtype={xv}.dtype)")
        self.needs_rshape = True
        self.lines.append(f"np.{fn}({xv}, axis={node.axis}, out={t})")
        self.owned.add(t)
        return t

    # --------------------------------------------------------- elementwise
    def _emit_tensor_op(self, node: TensorOp) -> str:
        if node.op not in _BINOP:
            raise CodegenError(f"不支持的逐元素算子：{node.op}")
        if len(node.args) != 2:
            raise CodegenError(f"{node.op} 只支持二元形式，实得 {len(node.args)} 元")
        left, right = node.args
        lv = self._emit(left)
        rv = self._emit(right)
        fn = _BINOP[node.op]

        if self._can_inplace(lv, left, right):
            self.lines.append(f"np.{fn}({lv}, {rv}, out={lv})")
            self.inplace_ops += 1
            return lv
        if self._can_inplace(rv, right, left):
            self.lines.append(f"np.{fn}({rv}, {lv}, out={rv})")
            self.inplace_ops += 1
            return rv
        t = self._tmp()
        self.lines.append(f"{t} = np.{fn}({lv}, {rv})")
        return t

    def _can_inplace(self, var: str, node: Node, other: Node) -> bool:
        if var not in self.owned:
            return False
        if self.uses[node] != 0:
            return False
        if is_scalar(other):
            return True
        if isinstance(other, Input):
            return rank_of(other, self.spec) <= rank_of(node, self.spec)
        return False

    def _emit_unary(self, node: Unary) -> str:
        xv = self._emit(node.x)
        if node.op == "erf":
            self.uses_erf = True
            t = self._tmp()
            self.lines.append(f"{t} = _erf({xv})")
            return t
        if node.op == "rsqrt":
            t = self._tmp()
            self.lines.append(f"{t} = 1.0 / np.sqrt({xv})")
            return t
        if node.op not in _UNARY:
            raise CodegenError(f"不支持的一元算子：{node.op}")
        t = self._tmp()
        self.lines.append(f"{t} = np.{_UNARY[node.op]}({xv})")
        return t

    def _emit_act(self, node: Act) -> str:
        xv = self._emit(node.x)
        t = self._tmp()
        if node.kind == "gelu":
            if not node.exact:
                self.lines.append(
                    f"{t} = 0.5 * {xv} * (1.0 + np.tanh(0.7978845608028654 * "
                    f"({xv} + 0.044715 * {xv} * {xv} * {xv})))")
            else:
                self.uses_erf = True
                self.lines.append(
                    f"{t} = 0.5 * {xv} * (1.0 + _erf({xv} * 0.7071067811865476))")
        elif node.kind == "silu":
            self.lines.append(f"{t} = {xv} / (1.0 + np.exp(-{xv}))")
        elif node.kind == "relu":
            self.lines.append(f"{t} = np.maximum({xv}, 0.0)")
        else:
            raise CodegenError(f"不支持的激活：{node.kind}")
        return t

    def _emit_softmax(self, node: Softmax) -> str:
        xv = self._emit(node.x)
        t1 = self._tmp()
        t2 = self._tmp()
        t3 = self._tmp()
        ax = node.axis
        self.lines.append(f"{t1} = {xv} - np.max({xv}, axis={ax}, keepdims=True)")
        self.lines.append(f"{t2} = np.exp({t1})")
        self.lines.append(f"{t3} = {t2} / np.sum({t2}, axis={ax}, keepdims=True)")
        return t3

    def _emit_layer_norm(self, node: LayerNorm) -> str:
        xv = self._emit(node.x)
        k = len(node.normalized_shape)
        axis = "-1" if k == 1 else "(" + ", ".join(str(-i) for i in range(k, 0, -1)) + ")"
        eps = self._scalar(node.eps)
        m = self._tmp()
        v = self._tmp()
        y = self._tmp()
        self.lines.append(f"{m} = np.mean({xv}, axis={axis}, keepdims=True)")
        self.lines.append(f"{v} = np.mean(({xv} - {m}) ** 2, axis={axis}, keepdims=True)")
        self.lines.append(f"{y} = ({xv} - {m}) / np.sqrt({v} + {eps})")
        cur = y
        if node.weight is not None:
            wv = self._emit(node.weight)
            nxt = self._tmp()
            self.lines.append(f"{nxt} = {cur} * {wv}")
            cur = nxt
        if node.bias is not None:
            bv = self._emit(node.bias)
            nxt = self._tmp()
            self.lines.append(f"{nxt} = {cur} + {bv}")
            cur = nxt
        return cur

    # -------------------------------------------------------------- scalar
    def _scalar(self, node: Node) -> str:
        if isinstance(node, Const):
            return repr(float(node.value))
        if isinstance(node, Dim):
            return f"{self._input_var(node.input_name)}.shape[{node.axis}]"
        if isinstance(node, External):
            raise CodegenError(
                f"标量 {node.name!r} 没有静态值（需要在 intake 的 consts 里提供）")
        if isinstance(node, ScalarOp):
            return f"float({self._scalar_expr(node)})"
        raise CodegenError(f"不是标量节点：{type(node).__name__}")

    def _scalar_expr(self, node: Node) -> str:
        if isinstance(node, Const):
            return repr(float(node.value))
        if isinstance(node, Dim):
            return f"{self._input_var(node.input_name)}.shape[{node.axis}]"
        if isinstance(node, External):
            raise CodegenError(f"标量 {node.name!r} 没有静态值")
        if isinstance(node, ScalarOp):
            a = [self._scalar_expr(x) for x in node.args]
            if node.op == "div":
                return f"({a[0]} / {a[1]})"
            if node.op == "mul":
                return "(" + " * ".join(a) + ")"
            if node.op == "add":
                return f"({a[0]} + {a[1]})"
            if node.op == "sub":
                return f"({a[0]} - {a[1]})"
            if node.op == "pow":
                return f"({a[0]} ** {a[1]})"
            if node.op == "sqrt":
                return f"np.sqrt({a[0]})"
            if node.op == "neg":
                return f"(-{a[0]})"
            if node.op == "maximum":
                return f"np.maximum({a[0]}, {a[1]})"
            if node.op == "minimum":
                return f"np.minimum({a[0]}, {a[1]})"
        raise CodegenError(f"不支持的标量表达式：{type(node).__name__}")