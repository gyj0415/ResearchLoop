# -*- coding: utf-8 -*-
"""C3 · 证据层：先判断「值不值得试」，再决定要不要花实验预算。

产出三样东西（docs/05 §4.2）：
  1. 算子级微基准：任务里出现的每个算子，numpy 单线程 vs torch（钉 4 线程）各测一次；
  2. 访存量与理论下界：io_bytes / 实测带宽 → 融合后最多能快多少；
  3. `verdict_hint`：winnable / blocked_by_transcendental / blocked_by_reduction /
     blocked_by_bandwidth —— 被判定为 blocked 的任务**不生成候选**，直接省掉整轮实验。

口径（与 G4 一致，便于横向比较）：
  - torch 线程钉 min(4, cpu)、interop=1；
  - 每个测量点：预热 3 次 + 2 趟 × 每趟 7 次取中位数；
  - 计时用 `time.perf_counter`（系统时钟），CPU 上 torch 同步执行，无需 synchronize；
  - `ratio = ms_numpy / ms_torch`，>1 表示 numpy 更慢。**这个方向不能写反**：
    numpy 慢才是「我们赢不了」的原因，不是 torch 慢。

诚实边界：微基准只代表「单算子」能力，不等于任务级上限；所以它只用于
`verdict_hint`（是否值得试），最终结论仍由 G0–G4 实测给出，两者不得互相替代。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .ir import (
    Act, Dim, Input, LayerNorm, Node, Reduce, ScalarOp, Softmax, TensorOp, Unary,
)

__all__ = ["OpProfile", "Evidence", "profile_ops", "estimate", "profile_task"]

# 与 g4_worker.py 完全一致的线程预算（4 线程；torch interop=1）
TORCH_THREADS = 4
WARMUP_ITERS = 3
REPEATS = 7
PASSES = 2
N_DEFAULT = 1 << 20

# numpy 单线程结构性落后超过这个倍数 → 判定「融合也补不回来」
TRANSCENDENTAL_RATIO = 4.0
REDUCTION_RATIO = 2.5
BANDWIDTH_RATIO = 3.0

_TRANSCENDENTAL = {"exp", "tanh", "erf", "log", "sigmoid", "act:gelu", "act:silu",
                   "softmax"}

_warmed = False


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpProfile:
    """单算子微基准（1M 元素，中位耗时）。"""
    op: str
    ms_numpy: float
    ms_torch: float
    bound: str                 # memory | compute | transcendental
    ratio: float               # ms_numpy / ms_torch（>1 = numpy 更慢)
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"op": self.op, "ms_numpy": round(self.ms_numpy, 6),
                "ms_torch": round(self.ms_torch, 6), "bound": self.bound,
                "ratio_numpy_over_torch": round(self.ratio, 4), "note": self.note}


@dataclass
class Evidence:
    """C3 的产物：决定「该往哪想」以及「要不要想」。"""
    profiles: List[OpProfile]
    dominant: str
    io_bytes: int
    theoretical_floor_ms: float
    overhead_ms: float
    bandwidth_bytes_per_ms: float
    verdict_hint: str
    budget_factor: float
    notes: List[str] = field(default_factory=list)
    shape_used: Tuple[int, ...] = ()

    def to_json(self) -> Dict[str, Any]:
        return {
            "profiles": [p.to_json() for p in self.profiles],
            "dominant": self.dominant,
            "io_bytes": self.io_bytes,
            "theoretical_floor_ms": round(self.theoretical_floor_ms, 6),
            "overhead_ms": round(self.overhead_ms, 6),
            "bandwidth_bytes_per_ms": round(self.bandwidth_bytes_per_ms, 2),
            "verdict_hint": self.verdict_hint,
            "budget_factor": self.budget_factor,
            "shape_used": list(self.shape_used),
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        parts = [f"verdict_hint={self.verdict_hint}",
                 f"budget_factor={self.budget_factor}",
                 f"dominant={self.dominant}",
                 f"io_bytes={self.io_bytes}",
                 f"floor={self.theoretical_floor_ms:.3f}ms",
                 f"overhead={self.overhead_ms:.4f}ms"]
        if self.profiles:
            parts.append("profiles=[" + ", ".join(
                f"{p.op}:{p.ms_numpy:.2f}/{p.ms_torch:.2f}ms({p.ratio:.1f}x)"
                for p in self.profiles) + "]")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def profile_ops(spec: Any, ir_node: Node, n: int = N_DEFAULT) -> List[OpProfile]:
    """对 IR 里出现的每个算子做微基准（去重、保持出现顺序）。"""
    _ensure_env()
    out: List[OpProfile] = []
    seen: set = set()
    for label in _tensor_op_labels(ir_node):
        if label in seen:
            continue
        seen.add(label)
        prof = _profile_one(label, n)
        if prof is not None:
            out.append(prof)
    return out


def estimate(spec: Any, ir_node: Node, profiles: List[OpProfile], *,
             io_bytes: Optional[int] = None,
             bandwidth_bytes_per_ms: Optional[float] = None,
             overhead_ms: Optional[float] = None) -> Evidence:
    """由 profile + 访存量给出 Evidence。参数可注入（测试用）。"""
    io = _measure_io_bytes(spec) if io_bytes is None else int(io_bytes)
    bw = _bandwidth() if bandwidth_bytes_per_ms is None else float(bandwidth_bytes_per_ms)
    ov = _overhead() if overhead_ms is None else float(overhead_ms)

    floor_ms = (io / bw) if bw > 0 else float("nan")
    dominant = max(profiles, key=lambda p: p.ms_numpy).op if profiles else "unknown"

    hint, factor, notes = _decide(spec, ir_node, profiles, dominant)
    notes.append(f"io_bytes={io}（一遍读全部输入 + 写输出，float32）")
    notes.append(f"bandwidth={bw:.0f} bytes/ms（np.copyto 实测，1M float32）")
    notes.append(f"np/torch 比值 >1 表示 numpy 更慢；阈值：超越函数 {TRANSCENDENTAL_RATIO}x、"
                 f"归约 {REDUCTION_RATIO}x")
    return Evidence(profiles=list(profiles), dominant=dominant, io_bytes=io,
                    theoretical_floor_ms=floor_ms, overhead_ms=ov,
                    bandwidth_bytes_per_ms=bw, verdict_hint=hint,
                    budget_factor=factor, notes=notes,
                    shape_used=_largest_shape(spec))


def profile_task(spec: Any, ir_node: Optional[Node] = None,
                 n: int = N_DEFAULT) -> Evidence:
    """一步到位：lift（可选）→ 算子 profile → Evidence。"""
    if ir_node is None:
        from .lifter import lift
        ir_node = lift(spec)
    profiles = profile_ops(spec, ir_node, n=n)
    return estimate(spec, ir_node, profiles)


# ---------------------------------------------------------------------------
# 判定（写死，可核查）
# ---------------------------------------------------------------------------

def _decide(spec: Any, ir_node: Node, profiles: List[OpProfile],
            dominant: str) -> Tuple[str, float, List[str]]:
    notes: List[str] = []

    trans = [p for p in profiles if p.bound == "transcendental"
             and p.ratio == p.ratio]
    worst = max(trans, key=lambda p: p.ratio) if trans else None
    if worst is not None and worst.ratio >= TRANSCENDENTAL_RATIO:
        notes.append(
            f"超越函数上限：{worst.op} numpy {worst.ms_numpy:.2f}ms vs torch "
            f"{worst.ms_torch:.2f}ms（{worst.ratio:.1f}x）→ 融合补不回来")
        return "blocked_by_transcendental", 0.0, notes

    rule = _reduce_rule(ir_node, spec)
    if rule is not None:
        notes.append(f"归约可被 {rule} 精确改写 → 归约不是硬上限（先试结构改写）")
        return "winnable", 1.0, notes

    mem = [p for p in profiles if p.bound != "transcendental"
           and not p.op.startswith("reduce:") and p.ratio == p.ratio]
    worst_mem = max(mem, key=lambda p: p.ratio) if mem else None
    if worst_mem is not None and worst_mem.ratio >= BANDWIDTH_RATIO:
        notes.append(
            f"访存/多趟结构性落后：{worst_mem.op} numpy {worst_mem.ms_numpy:.2f}ms "
            f"vs torch {worst_mem.ms_torch:.2f}ms（{worst_mem.ratio:.1f}x）；"
            "且没有可消元的归约 → 单线程 numpy 打不过多线程 eager")
        return "blocked_by_bandwidth", 0.0, notes

    red = max((p for p in profiles if p.op.startswith("reduce:")),
              key=lambda p: p.ratio, default=None)
    if red is not None and red.ratio >= REDUCTION_RATIO:
        notes.append(
            f"归约本身慢：{red.op} numpy {red.ms_numpy:.2f}ms vs torch "
            f"{red.ms_torch:.2f}ms（{red.ratio:.1f}x）")
        return "blocked_by_reduction", 0.0, notes

    notes.append("没有命中已知的硬上限，给一次机会但预算减半（默认保守）")
    return "winnable", 0.5, notes


def _pure_arithmetic(node: Node) -> bool:
    for n in node.walk():
        if isinstance(n, (Reduce, Softmax, LayerNorm, Act)):
            return False
        if isinstance(n, Unary) and n.op not in ("neg",):
            return False
    return True


def _reduce_eliminable(node: Node, spec: Any) -> bool:
    """是否存在「能被归约规则精确改写」的归约（有 → 归约不是硬上限）。

    判断委托给 `rules.rewritable_reduce`（C4 的精确前提），不在这里维护第二份：
    规则能改写的结构，证据层不得在实验前拦掉。
    """
    return _reduce_rule(node, spec) is not None


def _reduce_rule(node: Node, spec: Any) -> Optional[str]:
    from .rules import rewritable_reduce

    for n in node.walk():
        if not isinstance(n, Reduce) or n.axis is None:
            continue
        rid = rewritable_reduce(n, spec)
        if rid is not None:
            return rid
    return None


# ---------------------------------------------------------------------------
# 微基准实现
# ---------------------------------------------------------------------------

def _tensor_op_labels(node: Node) -> List[str]:
    """按先序给出张量级算子标签（跳过形状/标量运算）。"""
    labels: List[str] = []
    for n in node.walk():
        if isinstance(n, TensorOp):
            labels.append(n.op)
        elif isinstance(n, Unary):
            labels.append(n.op)
        elif isinstance(n, Reduce):
            labels.append(f"reduce:{n.kind}")
        elif isinstance(n, Act):
            labels.append(f"act:{n.kind}")
        elif isinstance(n, Softmax):
            labels.append("softmax")
        elif isinstance(n, LayerNorm):
            labels.append("layer_norm")
    return labels


def _profile_one(label: str, n: int) -> Optional[OpProfile]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    try:
        if label in ("add", "sub", "mul", "div", "pow", "maximum", "minimum"):
            x = _rnd(n)
            y = _rnd(n)
            tx, ty = _t(x), _t(y)
            out = np.empty_like(x)
            tout = torch.empty_like(tx)
            np_fn = {
                "add": lambda: np.add(x, y, out=out),
                "sub": lambda: np.subtract(x, y, out=out),
                "mul": lambda: np.multiply(x, y, out=out),
                "div": lambda: np.divide(x, y, out=out),
                "pow": lambda: np.power(x, np.float32(2.0), out=out),
                "maximum": lambda: np.maximum(x, y, out=out),
                "minimum": lambda: np.minimum(x, y, out=out),
            }[label]
            torch_fn = {
                "add": lambda: torch.add(tx, ty, out=tout),
                "sub": lambda: torch.sub(tx, ty, out=tout),
                "mul": lambda: torch.mul(tx, ty, out=tout),
                "div": lambda: torch.div(tx, ty, out=tout),
                "pow": lambda: torch.pow(tx, 2.0, out=tout),
                "maximum": lambda: torch.maximum(tx, ty, out=tout),
                "minimum": lambda: torch.minimum(tx, ty, out=tout),
            }[label]
            return _make(label, np_fn, torch_fn, "memory")

        if label in ("exp", "tanh", "log", "sqrt", "abs", "erf", "sigmoid",
                     "neg", "rsqrt"):
            x = _rnd(n, positive=(label in ("log", "sqrt", "rsqrt")))
            tx = _t(x)
            out = np.empty_like(x)
            tout = torch.empty_like(tx)
            if label == "erf":
                np_fn = lambda: _np_erf(x, out=out)
                torch_fn = lambda: torch.erf(tx, out=tout)
            else:
                np_fn = {
                    "exp": lambda: np.exp(x, out=out),
                    "tanh": lambda: np.tanh(x, out=out),
                    "log": lambda: np.log(x, out=out),
                    "sqrt": lambda: np.sqrt(x, out=out),
                    "abs": lambda: np.abs(x, out=out),
                    "sigmoid": lambda: np.divide(
                        1.0, 1.0 + np.exp(-x, out=out), out=out),
                    "neg": lambda: np.negative(x, out=out),
                    "rsqrt": lambda: np.divide(
                        1.0, np.sqrt(x, out=out), out=out),
                }[label]
                torch_fn = {
                    "exp": lambda: torch.exp(tx, out=tout),
                    "tanh": lambda: torch.tanh(tx, out=tout),
                    "log": lambda: torch.log(tx, out=tout),
                    "sqrt": lambda: torch.sqrt(tx, out=tout),
                    "abs": lambda: torch.abs(tx, out=tout),
                    "sigmoid": lambda: torch.sigmoid(tx, out=tout),
                    "neg": lambda: torch.neg(tx, out=tout),
                    "rsqrt": lambda: torch.rsqrt(tx, out=tout),
                }[label]
            bound = "transcendental" if label in _TRANSCENDENTAL else "memory"
            return _make(label, np_fn, torch_fn, bound)

        if label.startswith("reduce:"):
            kind = label.split(":", 1)[1]
            side = int(round(n ** 0.5))
            x = _rnd2(side, side)
            tx = _t(x)
            np_fn = {"mean": lambda: np.mean(x, axis=-1),
                     "sum": lambda: np.sum(x, axis=-1),
                     "max": lambda: np.max(x, axis=-1),
                     "min": lambda: np.min(x, axis=-1),
                     "prod": lambda: np.prod(x, axis=-1)}[kind]
            torch_fn = {"mean": lambda: tx.mean(dim=-1),
                        "sum": lambda: tx.sum(dim=-1),
                        "max": lambda: tx.amax(dim=-1),
                        "min": lambda: tx.amin(dim=-1),
                        "prod": lambda: tx.prod(dim=-1)}[kind]
            return _make(label, np_fn, torch_fn, "memory")

        if label == "softmax":
            side = int(round(n ** 0.5))
            x = _rnd2(side, side)
            tx = _t(x)
            np_fn = lambda: _np_softmax(x)
            torch_fn = lambda: torch.softmax(tx, dim=-1)
            return _make(label, np_fn, torch_fn, "transcendental")

        if label == "layer_norm":
            side = int(round(n ** 0.5))
            x = _rnd2(side, side)
            tx = _t(x)
            np_fn = lambda: _np_layer_norm(x)
            torch_fn = lambda: F.layer_norm(tx, (tx.shape[-1],))
            return _make(label, np_fn, torch_fn, "compute")

        if label == "act:gelu":
            x = _rnd(n)
            tx = _t(x)
            np_fn = lambda: 0.5 * x * (1.0 + _np_erf(x * np.float32(0.7071067811865476)))
            torch_fn = lambda: F.gelu(tx)
            return _make(label, np_fn, torch_fn, "transcendental")

        if label == "act:silu":
            x = _rnd(n)
            tx = _t(x)
            np_fn = lambda: x / (1.0 + np.exp(-x))
            torch_fn = lambda: F.silu(tx)
            return _make(label, np_fn, torch_fn, "transcendental")

        if label == "act:relu":
            x = _rnd(n)
            tx = _t(x)
            np_fn = lambda: np.maximum(x, np.float32(0.0))
            torch_fn = lambda: F.relu(tx)
            return _make(label, np_fn, torch_fn, "memory")
    except Exception as e:  # noqa: BLE001 —— 单算子失败不该中断整轮证据
        return OpProfile(label, float("nan"), float("nan"), "unknown", float("nan"),
                         note=f"profile 失败：{type(e).__name__}: {e}")
    return OpProfile(label, float("nan"), float("nan"), "unknown", float("nan"),
                     note="未实现该算子的微基准")


def _make(op: str, np_fn: Callable[[], Any], torch_fn: Callable[[], Any],
          bound: str) -> OpProfile:
    ms_np = _bench(np_fn)
    ms_t = _bench(torch_fn)
    ratio = (ms_np / ms_t) if ms_t > 0 else float("inf")
    return OpProfile(op, ms_np, ms_t, bound, ratio)


def _bench(fn: Callable[[], Any]) -> float:
    for _ in range(WARMUP_ITERS):
        fn()
    samples: List[float] = []
    for _ in range(PASSES):
        pass_samples: List[float] = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fn()
            pass_samples.append((time.perf_counter() - t0) * 1000.0)
        pass_samples.sort()
        samples.append(pass_samples[len(pass_samples) // 2])
    samples.sort()
    return samples[len(samples) // 2]


def _ensure_env() -> None:
    global _warmed
    import os

    import numpy as np
    import torch

    torch.set_num_threads(min(TORCH_THREADS, os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if _warmed:
        return
    x = np.zeros(2_000_000, dtype=np.float32)
    tx = torch.zeros(2_000_000, dtype=torch.float32)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.5:
        _ = x * 0.5 + 1.0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.5:
        _ = tx * 0.5 + 1.0
    _warmed = True


def _rnd(n: int, positive: bool = False):
    import numpy as np
    rng = np.random.default_rng(20261114)
    x = rng.standard_normal(n, dtype=np.float32)
    return np.abs(x) + 0.5 if positive else x


def _rnd2(rows: int, cols: int):
    import numpy as np
    rng = np.random.default_rng(20261114)
    return rng.standard_normal((rows, cols), dtype=np.float32)


def _t(x):
    import torch
    return torch.from_numpy(x)


def _np_erf(x, out=None):
    """Abramowitz-Stegun 7.1.26 有理近似（仅用于微基准，不参与正确性判定）。"""
    import numpy as np
    p = 0.3275911
    a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741,
                          -1.453152027, 1.061405429)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    res = np.sign(x) * (1.0 - poly * np.exp(-ax * ax))
    if out is None:
        return res
    np.copyto(out, res)
    return out


def _np_softmax(x):
    import numpy as np
    s = x - x.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def _np_layer_norm(x, eps: float = 1e-5):
    import numpy as np
    m = x.mean(axis=-1, keepdims=True)
    v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)


def _bandwidth() -> float:
    """用 np.copyto（读+写 2×4MB）实测可用带宽，返回 bytes/ms。"""
    import numpy as np
    n = N_DEFAULT
    src = np.arange(n, dtype=np.float32)
    dst = np.empty_like(src)
    ms = _bench(lambda: np.copyto(dst, src))
    return (2 * n * 4) / ms if ms > 0 else float("inf")


def _overhead() -> float:
    """numpy 单次调用开销（8 元素加法），单位 ms。"""
    import numpy as np
    a = np.zeros(8, dtype=np.float32)
    b = np.ones(8, dtype=np.float32)
    return _bench(lambda: np.add(a, b))


def _largest_shape(spec: Any) -> Tuple[int, ...]:
    shapes = list(getattr(spec, "shapes", ()) or ())
    if not shapes:
        return ()
    return tuple(max(shapes, key=lambda s: _prod(s)))


def _measure_io_bytes(spec: Any) -> int:
    """一遍完整读入 + 写出的字节数（float32）。失败则按 shape 估算。"""
    shape = _largest_shape(spec)
    task = getattr(spec, "task", None)
    if task is not None and shape:
        try:
            seed = int(spec.seeds[0]) if spec.seeds else 0
            inputs = task.make_inputs(shape, seed)
            total = sum(int(v.numel()) * int(v.element_size())
                        for v in inputs.values())
            out = task.run_reference(inputs)
            return total + int(out.numel()) * int(out.element_size())
        except Exception:  # noqa: BLE001
            pass
    n = _prod(shape) if shape else N_DEFAULT
    return int(n * 4 * (len(getattr(spec, "inputs", {}) or {}) + 1))


def _prod(shape: Sequence[int]) -> int:
    n = 1
    for v in shape:
        n *= int(v)
    return n