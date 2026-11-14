# -*- coding: utf-8 -*-
"""C2 · 源码 → IR：把参考实现 lift 成可改写的结构。

只认模式，不做通用符号执行（docs/05 §5 M2）：
  - 函数体允许「简单赋值 + 单条 return」（现有 5 个任务都是这个形态）；
  - 认识 torch / F / math / numpy 的固定算子名，其余一律 `UnsupportedPattern`；
  - 拿不准就抛异常，**不猜**——猜错会污染整条假设链。

`UnsupportedPattern` 由假设层捕获并降级为「只用变异族」，不会中断 run。
"""

from __future__ import annotations

import ast
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from .ir import (
    Act, Const, Dim, External, Input, LayerNorm, Node, Reduce, ScalarOp,
    Softmax, TensorOp, Unary, UnsupportedPattern, is_scalar,
)

__all__ = ["lift", "lift_source", "LiftError", "UnsupportedPattern"]

LiftError = UnsupportedPattern

_MODULE_ROOTS = {"torch", "np", "numpy", "F", "math", "operator", "builtins"}

_BINOPS = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul",
    ast.Div: "div", ast.Pow: "pow", ast.Mod: "mod",
}

_UNARY_FUNCS = {
    "exp": "exp", "log": "log", "tanh": "tanh", "sqrt": "sqrt",
    "abs": "abs", "absolute": "abs", "erf": "erf", "sigmoid": "sigmoid",
    "rsqrt": "rsqrt", "negative": "neg", "neg": "neg",
}

_REDUCE_FUNCS = {
    "mean": "mean", "sum": "sum", "amax": "max", "amin": "min",
    "max": "max", "min": "min", "prod": "prod",
}

_ACT_FUNCS = {"gelu": "gelu", "silu": "silu", "relu": "relu"}

_ELEMENTWISE_FUNCS = {"add", "sub", "mul", "div", "pow", "maximum", "minimum"}

# math 模块的函数一律当标量运算（形状推导用）
_MATH_SCALAR = {"sqrt", "exp", "log", "pow", "fabs", "floor", "ceil"}


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def lift(spec: Any) -> Node:
    """TaskSpec → IR，并校验 IR 用到的输入都在 TaskSpec 声明里。"""
    node = lift_source(spec.reference_source, consts=dict(getattr(spec, "consts", {})))
    declared = set(getattr(spec, "inputs", {}))
    used = {n.name for n in node.walk() if isinstance(n, Input)}
    extra = used - declared
    if extra:
        raise UnsupportedPattern(
            f"{getattr(spec, 'name', '?')}: IR 引用了未声明的输入 {sorted(extra)}；"
            f"已声明 {sorted(declared)}")
    return node


def lift_source(source: str, *, consts: Optional[Dict[str, Any]] = None) -> Node:
    """源码文本 → IR。`consts` 提供闭包/全局里已求值的标量。"""
    text = textwrap.dedent(source)
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise UnsupportedPattern(f"参考实现源码无法解析：{e}") from e

    func = _find_reference(tree)
    params = [a.arg for a in func.args.args]
    if not params:
        raise UnsupportedPattern("reference() 没有任何参数，无法确定输入字典名")
    inputs_arg = params[0]

    env: Dict[str, Node] = {}
    ret: Optional[Node] = None
    for stmt in func.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                and isinstance(stmt.value.value, str):
            continue                                   # docstring
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                raise UnsupportedPattern("reference() 返回 None")
            ret = _lift(stmt.value, env, consts or {}, inputs_arg)
            break
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
            if not isinstance(target, ast.Name):
                raise UnsupportedPattern(f"第 {stmt.lineno} 行：只支持单名字赋值")
            if stmt.value is None:
                continue
            env[target.id] = _lift(stmt.value, env, consts or {}, inputs_arg)
            continue
        raise UnsupportedPattern(
            f"第 {getattr(stmt, 'lineno', '?')} 行出现不支持的控制流/语句："
            f"{type(stmt).__name__}")

    if ret is None:
        raise UnsupportedPattern("reference() 没有 return 语句")
    return ret


# ---------------------------------------------------------------------------
# 定位 reference 函数
# ---------------------------------------------------------------------------

def _find_reference(tree: ast.Module) -> ast.FunctionDef:
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for f in funcs:
        if f.name == "reference":
            return f
    if len(funcs) == 1:
        return funcs[0]
    raise UnsupportedPattern(
        f"源码里找到 {len(funcs)} 个顶层函数，无法确定哪个是 reference()")


# ---------------------------------------------------------------------------
# 表达式
# ---------------------------------------------------------------------------

def _lift(expr: ast.AST, env: Dict[str, Node], consts: Dict[str, Any],
          inputs_arg: str) -> Node:
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, (int, float, bool)):
            return Const(float(expr.value))
        raise UnsupportedPattern(f"第 {expr.lineno} 行：不支持常量 {expr.value!r}")

    if isinstance(expr, ast.Name):
        if expr.id in env:
            return env[expr.id]
        if expr.id in consts and isinstance(consts[expr.id], (int, float, bool)):
            return Const(float(consts[expr.id]))
        if expr.id in ("None", "True", "False"):
            raise UnsupportedPattern(f"第 {expr.lineno} 行：{expr.id} 不能作为张量使用")
        raise UnsupportedPattern(
            f"第 {expr.lineno} 行：未知名字 {expr.id!r}（既不在源码赋值里，"
            "也不在闭包常量里）")

    if isinstance(expr, ast.BinOp):
        op = _BINOPS.get(type(expr.op))
        if op is None:
            raise UnsupportedPattern(f"第 {expr.lineno} 行：不支持二元运算 "
                                     f"{type(expr.op).__name__}")
        left = _lift(expr.left, env, consts, inputs_arg)
        right = _lift(expr.right, env, consts, inputs_arg)
        return _binary(op, left, right)

    if isinstance(expr, ast.UnaryOp):
        if isinstance(expr.op, ast.USub):
            inner = _lift(expr.operand, env, consts, inputs_arg)
            return ScalarOp("neg", (inner,)) if is_scalar(inner) else Unary("neg", inner)
        if isinstance(expr.op, ast.UAdd):
            return _lift(expr.operand, env, consts, inputs_arg)
        raise UnsupportedPattern(f"第 {expr.lineno} 行：不支持一元运算 "
                                 f"{type(expr.op).__name__}")

    if isinstance(expr, ast.Subscript):
        return _lift_subscript(expr, env, consts, inputs_arg)

    if isinstance(expr, ast.Call):
        return _lift_call(expr, env, consts, inputs_arg)

    if isinstance(expr, ast.Tuple):
        raise UnsupportedPattern(
            f"第 {expr.lineno} 行：元组只能作为 layer_norm 的 normalized_shape")

    if isinstance(expr, ast.Attribute):
        raise UnsupportedPattern(
            f"第 {expr.lineno} 行：属性访问 .{expr.attr} 只能出现在 "
            "shape[...] 或算子调用里")

    raise UnsupportedPattern(
        f"第 {getattr(expr, 'lineno', '?')} 行：不支持的表达式 {type(expr).__name__}")


def _lift_subscript(expr: ast.Subscript, env: Dict[str, Node],
                    consts: Dict[str, Any], inputs_arg: str) -> Node:
    # inputs["x"] → Input("x")
    if isinstance(expr.value, ast.Name) and expr.value.id == inputs_arg:
        key = _const_str(expr.slice)
        if key is None:
            raise UnsupportedPattern(
                f"第 {expr.lineno} 行：inputs[...] 的键必须是字符串常量")
        return Input(key)

    # x.shape[-1] → Dim("x", -1)
    if isinstance(expr.value, ast.Attribute) and expr.value.attr == "shape":
        base = _lift(expr.value.value, env, consts, inputs_arg)
        if not isinstance(base, Input):
            raise UnsupportedPattern(
                f"第 {expr.lineno} 行：只支持对输入张量直接取 .shape[i]")
        axis = _const_int(expr.slice)
        if axis is None:
            raise UnsupportedPattern(
                f"第 {expr.lineno} 行：.shape[...] 的下标必须是整数常量")
        return Dim(base.name, axis)

    raise UnsupportedPattern(
        f"第 {expr.lineno} 行：不支持的下标访问（只认 inputs[\"key\"] 和 x.shape[i]）")


def _lift_call(expr: ast.Call, env: Dict[str, Node], consts: Dict[str, Any],
               inputs_arg: str) -> Node:
    kind, module, fname = _call_target(expr.func)
    if fname is None:
        raise UnsupportedPattern(f"第 {expr.lineno} 行：无法识别的调用")

    kwargs = {kw.arg: kw.value for kw in expr.keywords if kw.arg is not None}

    if kind == "method":
        receiver = _lift(expr.func.value, env, consts, inputs_arg)
        pos = list(expr.args)
    else:
        receiver = None
        pos = list(expr.args)

    # ---- 标量数学（math.sqrt 等）----
    if module == "math" and fname in _MATH_SCALAR:
        args = tuple(_lift(a, env, consts, inputs_arg) for a in pos)
        if not all(is_scalar(a) for a in args):
            raise UnsupportedPattern(
                f"第 {expr.lineno} 行：math.{fname} 的参数必须是标量"
                "（形状维或常量）")
        return ScalarOp(fname, args)

    # ---- 张量一元函数 ----
    if fname in _UNARY_FUNCS and kind != "method":
        if not pos:
            raise UnsupportedPattern(f"第 {expr.lineno} 行：{fname} 缺少参数")
        return Unary(_UNARY_FUNCS[fname], _lift(pos[0], env, consts, inputs_arg))

    # ---- 逐元素张量二元函数：torch.add(x, y) 等 ----
    if fname in _ELEMENTWISE_FUNCS and kind != "method":
        if len(pos) < 2:
            raise UnsupportedPattern(
                f"第 {expr.lineno} 行：{fname} 需要两个参数")
        left = _lift(pos[0], env, consts, inputs_arg)
        right = _lift(pos[1], env, consts, inputs_arg)
        return _binary(fname, left, right)

    # ---- 激活 ----
    if fname in _ACT_FUNCS:
        x = receiver if kind == "method" else (
            _lift(pos[0], env, consts, inputs_arg) if pos else None)
        if x is None:
            raise UnsupportedPattern(f"第 {expr.lineno} 行：{fname} 缺少输入")
        exact = _kw_str(kwargs.get("approximate")) not in ("tanh", "None")
        return Act(_ACT_FUNCS[fname], x, exact=exact)

    # ---- softmax ----
    if fname == "softmax":
        x = receiver if kind == "method" else (
            _lift(pos[0], env, consts, inputs_arg) if pos else None)
        if x is None:
            raise UnsupportedPattern(f"第 {expr.lineno} 行：softmax 缺少输入")
        axis = _axis_from(kwargs, pos, method=(kind == "method")) or -1
        return Softmax(x, axis=int(axis))

    # ---- layer_norm ----
    if fname == "layer_norm":
        return _lift_layer_norm(expr, receiver, pos, kwargs, env, consts, inputs_arg)

    # ---- 归约 ----
    if fname in _REDUCE_FUNCS:
        x = receiver if kind == "method" else (
            _lift(pos[0], env, consts, inputs_arg) if pos else None)
        if x is None:
            raise UnsupportedPattern(f"第 {expr.lineno} 行：{fname} 缺少输入")
        axis = _axis_from(kwargs, pos, method=(kind == "method"))
        keep = _const_bool(kwargs.get("keepdims"))
        return Reduce(_REDUCE_FUNCS[fname], x, axis=axis, keepdims=keep)

    raise UnsupportedPattern(
        f"第 {expr.lineno} 行：不认识的算子 "
        f"{'.'.join(x for x in (module, fname) if x)}")


def _lift_layer_norm(expr: ast.Call, receiver: Optional[Node], pos: List[ast.AST],
                     kwargs: Dict[str, ast.AST], env: Dict[str, Node],
                     consts: Dict[str, Any], inputs_arg: str) -> Node:
    if receiver is not None:
        x = receiver
        rest = pos
    else:
        if not pos:
            raise UnsupportedPattern(f"第 {expr.lineno} 行：layer_norm 缺少输入")
        x = _lift(pos[0], env, consts, inputs_arg)
        rest = pos[1:]

    shape_expr = kwargs.get("normalized_shape") or (rest[0] if rest else None)
    if shape_expr is None:
        raise UnsupportedPattern(
            f"第 {expr.lineno} 行：layer_norm 缺少 normalized_shape")
    if isinstance(shape_expr, (ast.Tuple, ast.List)):
        norm = tuple(_lift(e, env, consts, inputs_arg) for e in shape_expr.elts)
    else:
        norm = (_lift(shape_expr, env, consts, inputs_arg),)

    weight = _lift_optional(kwargs.get("weight") or (rest[1] if len(rest) > 1 else None),
                            env, consts, inputs_arg)
    bias = _lift_optional(kwargs.get("bias") or (rest[2] if len(rest) > 2 else None),
                          env, consts, inputs_arg)
    eps_expr = kwargs.get("eps") or (rest[3] if len(rest) > 3 else None)
    eps = (_lift(eps_expr, env, consts, inputs_arg) if eps_expr is not None
           else Const(1e-5))
    return LayerNorm(x, norm, weight=weight, bias=bias, eps=eps)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _binary(op: str, left: Node, right: Node) -> Node:
    if is_scalar(left) and is_scalar(right):
        return ScalarOp(op, (left, right))
    return TensorOp(op, (left, right))


def _call_target(func: ast.AST) -> Tuple[str, Optional[str], Optional[str]]:
    """返回 (kind, module, name)：kind ∈ module/name/method。"""
    if isinstance(func, ast.Attribute):
        module = _module_of(func.value)
        if module is not None:
            return "module", module, func.attr
        return "method", None, func.attr
    if isinstance(func, ast.Name):
        return "name", None, func.id
    return "?", None, None


def _module_of(expr: ast.AST) -> Optional[str]:
    dotted = _dotted(expr)
    if dotted and dotted.split(".")[0] in _MODULE_ROOTS:
        return dotted
    return None


def _dotted(expr: ast.AST) -> Optional[str]:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _dotted(expr.value)
        return None if base is None else f"{base}.{expr.attr}"
    return None


def _lift_optional(expr: Optional[ast.AST], env: Dict[str, Node],
                   consts: Dict[str, Any], inputs_arg: str) -> Optional[Node]:
    if expr is None:
        return None
    if isinstance(expr, ast.Constant) and expr.value is None:
        return None
    return _lift(expr, env, consts, inputs_arg)


def _axis_from(kwargs: Dict[str, ast.AST], pos: List[ast.AST],
               method: bool) -> Optional[int]:
    expr = kwargs.get("dim") or kwargs.get("axis")
    if expr is None:
        idx = 0 if method else 1
        expr = pos[idx] if len(pos) > idx else None
    if expr is None:
        return None
    axis = _const_int(expr)
    if axis is None:
        if isinstance(expr, (ast.Tuple, ast.List)) and len(expr.elts) == 1:
            axis = _const_int(expr.elts[0])
        if axis is None:
            raise UnsupportedPattern("归约轴必须是整数常量（多轴归约暂不支持）")
    return axis


def _const_int(expr: ast.AST) -> Optional[int]:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int) \
            and not isinstance(expr.value, bool):
        return int(expr.value)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        inner = _const_int(expr.operand)
        return None if inner is None else -inner
    return None


def _const_str(expr: ast.AST) -> Optional[str]:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return str(expr.value)
    return None


def _const_bool(expr: Optional[ast.AST]) -> bool:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, bool):
        return bool(expr.value)
    return False


def _kw_str(expr: Optional[ast.AST]) -> str:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return str(expr.value)
    return ""