# -*- coding: utf-8 -*-
"""C1 · 任务接入：把外部任务文件变成统一的 `TaskSpec`。

验收口径（docs/05 §5 M1）：把一个新的 `tasks/*.py` 放进来，**不改 src/ 任何代码**
就能 `load()` 出一个 TaskSpec，且 `reference_source` 是可直接喂给 lifter 的源码文本。

设计取舍：
  - 不发明 yaml/json 配置——任务本身就是 Python（现有 `OperatorTask` 契约）；
  - 符号形状优先读任务声明的 `input_shapes`，没有就按秩推断，并如实标注
    `shape_symbols_declared=False`（推断值只用于规则匹配，不参与数值计算）；
  - `consts` 从闭包/全局变量里取**数值常量**（如 layer_norm 的 eps），
    这样 lifter 不需要猜；取不到的留给 `External` 节点。
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]

__all__ = ["TaskSpec", "load", "load_dir", "from_task", "infer_reduce_axis"]

_REDUCE_ATTRS = ("mean", "sum", "amax", "amin", "max", "min", "prod")


@dataclass(frozen=True)
class TaskSpec:
    """任务进来时的唯一入口对象（docs/05 §4.1）。"""

    name: str
    description: str
    category: str
    reference_source: str                 # reference() 的源码文本（lifter 的输入）
    inputs: Dict[str, str]                # {"x": "(B,T,C)", "bias": "(C,)"}
    shapes: Tuple[Tuple[int, ...], ...]
    seeds: Tuple[int, ...]
    rtol: float
    atol: float
    reduce_axis: Optional[int]            # 语法层推断的归约轴；lifter 会再确认
    consts: Dict[str, Any] = field(default_factory=dict)      # 可静态求值的标量
    modules: Tuple[str, ...] = ()          # 源码引用的模块名（torch/F/math/...）
    shape_symbols_declared: bool = False   # False = 按秩推断的占位名
    source_path: str = ""
    task: Any = field(default=None, compare=False, repr=False)  # 原始 OperatorTask

    def shape_rank(self, key: str) -> int:
        """输入 key 的秩（用于广播/归约轴合法性判断）。"""
        s = str(self.inputs.get(key, "")).strip()
        if not (s.startswith("(") and s.endswith(")")):
            return 0
        inner = s[1:-1].strip()
        if not inner:
            return 0
        return len([p for p in inner.split(",") if p.strip()])


# ---------------------------------------------------------------------------
# 模块加载
# ---------------------------------------------------------------------------

def _module_name(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"_research_task_{digest}"


def _import_path(path: Path) -> ModuleType:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"任务文件不存在：{path}")
    name = _module_name(path)
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {path} 构造 import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _tasks_in_module(module: ModuleType) -> List[Any]:
    """按契约找任务：ALL_TASKS / TASKS 列表，或 TASK 单例。"""
    found: List[Any] = []
    for attr in ("ALL_TASKS", "TASKS"):
        value = getattr(module, attr, None)
        if isinstance(value, (list, tuple)):
            found.extend(value)
    single = getattr(module, "TASK", None)
    if single is not None and not any(single is t for t in found):
        found.append(single)
    if not found:
        raise ValueError(
            f"{module.__name__}: 未找到任务定义（需要 ALL_TASKS / TASKS / TASK 之一）")
    for t in found:
        for attr in ("name", "reference", "make_inputs", "shapes"):
            if not hasattr(t, attr):
                raise ValueError(f"{module.__name__}: 任务缺少字段 {attr!r}：{t!r}")
    return found


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def load(task_path: Path, name: Optional[str] = None) -> TaskSpec:
    """加载单个任务文件。文件里有多个任务时必须给 name。"""
    module = _import_path(Path(task_path))
    tasks = _tasks_in_module(module)
    if name is not None:
        for t in tasks:
            if getattr(t, "name", None) == name:
                return from_task(t, source_path=task_path)
        raise KeyError(f"{task_path}: 没有名为 {name!r} 的任务；已有 "
                       f"{[getattr(t, 'name', '?') for t in tasks]}")
    if len(tasks) != 1:
        raise ValueError(f"{task_path}: 含 {len(tasks)} 个任务，请显式指定 name="
                         f"{[getattr(t, 'name', '?') for t in tasks]}")
    return from_task(tasks[0], source_path=task_path)


def load_dir(dir_path: Path) -> List[TaskSpec]:
    """加载目录下所有任务文件（跳过 `_` 开头的私有文件），按任务名排序。"""
    directory = Path(dir_path)
    if not directory.is_dir():
        raise NotADirectoryError(f"不是目录：{directory}")
    specs: List[TaskSpec] = []
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = _import_path(path)
        try:
            tasks = _tasks_in_module(module)
        except ValueError:
            continue          # 普通辅助模块（没有任务定义）不当作错误
        for t in tasks:
            specs.append(from_task(t, source_path=path))
    specs.sort(key=lambda s: s.name)
    return specs


def from_task(task: Any, source_path: Optional[Path] = None) -> TaskSpec:
    """把现有 `OperatorTask` 变成 TaskSpec（不落盘、不改任务定义）。"""
    if not hasattr(task, "reference"):
        raise TypeError(f"不是 OperatorTask：{task!r}")
    # reference 常定义在工厂函数里，getsource 会带外层缩进；统一去缩进后再存储，
    # 这样 lifter 与 golden 测试拿到的是稳定文本。
    source = textwrap.dedent(inspect.getsource(task.reference))
    consts, modules = _static_env(task.reference)
    inputs = _symbolic_inputs(task)
    declared = isinstance(getattr(task, "input_shapes", None), dict)
    return TaskSpec(
        name=str(task.name),
        description=str(getattr(task, "description", "")),
        category=str(getattr(task, "category", "")),
        reference_source=source,
        inputs=inputs,
        shapes=tuple(tuple(int(v) for v in s) for s in task.shapes),
        seeds=tuple(int(s) for s in getattr(task, "seeds", ())),
        rtol=float(getattr(task, "rtol", 1e-4)),
        atol=float(getattr(task, "atol", 1e-5)),
        reduce_axis=infer_reduce_axis(source),
        consts=consts,
        modules=modules,
        shape_symbols_declared=declared,
        source_path=str(source_path or ""),
        task=task,
    )


# ---------------------------------------------------------------------------
# 静态环境 / 符号形状
# ---------------------------------------------------------------------------

_NUM = (int, float, bool)


def _static_env(fn: Any) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    """从闭包/全局取数值常量与模块名（拿不到的不猜）。"""
    consts: Dict[str, Any] = {}
    modules: List[str] = []
    try:
        cv = inspect.getclosurevars(fn)
    except (TypeError, ValueError):
        return consts, ()
    for table in (cv.nonlocals, cv.globals):
        for k, v in table.items():
            if isinstance(v, _NUM) and not isinstance(v, complex):
                consts[k] = v
    for k, v in cv.globals.items():
        if isinstance(v, ModuleType):
            modules.append(k)
    return consts, tuple(sorted(set(modules)))


_RANK_SYMBOLS: Dict[int, Tuple[str, ...]] = {
    0: (),
    1: ("C",),
    2: ("B", "C"),
    3: ("B", "T", "C"),
    4: ("B", "H", "W", "C"),
}


def _symbolic_inputs(task: Any) -> Dict[str, str]:
    declared = getattr(task, "input_shapes", None)
    if isinstance(declared, dict) and declared:
        return {str(k): str(v) for k, v in declared.items()}
    # 用最小 shape 调一次 make_inputs 只为拿秩（不参与后续计算）
    smallest: Sequence[int] = min(task.shapes, key=lambda s: _prod(s))
    sample = task.make_inputs(smallest, int(getattr(task, "seeds", [0])[0]))
    out: Dict[str, str] = {}
    for key, value in sample.items():
        rank = len(getattr(value, "shape", ()))
        syms = _RANK_SYMBOLS.get(rank)
        if syms is None:
            syms = tuple(f"S{i}" for i in range(rank - 1)) + ("C",)
        out[str(key)] = "(" + ",".join(syms) + ("," if rank == 1 else "") + ")"
    return out


def _prod(shape: Sequence[int]) -> int:
    n = 1
    for v in shape:
        n *= int(v)
    return n


def infer_reduce_axis(source: str) -> Optional[int]:
    """从源码语法层推断归约轴（只是提示；lifter 会给出权威 IR）。"""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        attr = fn.attr if isinstance(fn, ast.Attribute) else (
            fn.id if isinstance(fn, ast.Name) else None)
        if attr not in _REDUCE_ATTRS:
            continue
        axis: Optional[int] = None
        for kw in node.keywords:
            if kw.arg in ("dim", "axis"):
                axis = _const_int_ast(kw.value)
        if axis is None:
            # 位置参数：torch.mean(x, 1) / x.mean(1)
            positional = node.args[1:]
            if positional:
                axis = _const_int_ast(positional[0])
        if axis is not None:
            return axis
    return None


def _const_int_ast(expr: ast.AST) -> Optional[int]:
    """AST 里的整数常量（负数在 AST 里是 UnaryOp(USub, Constant)）。"""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int) \
            and not isinstance(expr.value, bool):
        return int(expr.value)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        inner = _const_int_ast(expr.operand)
        return None if inner is None else -inner
    return None