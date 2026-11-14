"""把用户贴上的 reference 收成一个算子任务文件。

读不懂的写法不会落盘。文件只含参考实现和输入构造，不包含计时或评估代码。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from researchloop.scenarios.operator.lifter import UnsupportedPattern, lift_source
from researchloop.scenarios.operator.service import user_task_dir

_NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_BANNED = {"exec", "eval", "open", "compile", "__import__", "input", "breakpoint"}
_ALLOWED_MODULES = {"torch", "torch.nn.functional", "math"}


def save_task(name, source, shapes=None, directory=None):
    task_name = str(name or "").strip()
    if not _NAME.match(task_name) or task_name in ("win_bias_mean", "lose_gelu_pool"):
        raise ValueError("任务名要用小写字母、数字或下划线，且不能覆盖两个例子")
    reference, inputs = _reference(str(source or ""))
    try:
        ir = lift_source(reference).to_source()
    except UnsupportedPattern as exc:
        raise ValueError(f"研究层读不懂这个写法，任务没有加入：{exc}") from exc
    parsed = _shapes(shapes)
    directory = Path(directory or user_task_dir())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task_name}.py"
    path.write_text(_render(task_name, reference, parsed, inputs), encoding="utf-8")
    return {"name": task_name, "ir": ir, "path": str(path), "runnable": True,
            "note": "结构已读懂。计划确认后才会做微基准、门控和同窗计时。"}


def _reference(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"reference 不是合法 Python：{exc.msg}") from exc
    function = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node)
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "reference":
            if function is not None:
                raise ValueError("只能有一个 reference")
            function = node
            continue
        raise ValueError("只接受 import 和一个名为 reference 的函数")
    if function is None or len(function.args.args) != 1:
        raise ValueError("需要 def reference(inputs):")
    argument = function.args.args[0].arg
    inputs = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            continue
        if node.value.id != argument or not isinstance(node.slice, ast.Constant):
            continue
        if isinstance(node.slice.value, str) and node.slice.value not in inputs:
            inputs.append(node.slice.value)
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id in _BANNED:
            raise ValueError("reference 不能读文件、执行字符串或再导入模块")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _BANNED:
            raise ValueError("reference 不能读文件、执行字符串或再导入模块")
    if "x" not in inputs:
        raise ValueError("reference 需要读取 inputs[\"x\"]")
    return ast.get_source_segment(source, function) or ast.unparse(function), inputs


def _check_import(node):
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    else:
        names = [node.module or ""]
    if any(name not in _ALLOWED_MODULES for name in names):
        raise ValueError("只能导入 torch、torch.nn.functional 或 math")


def _shapes(value):
    text = str(value or "4,64,32; 1,31,16")
    shapes = []
    for chunk in text.split(";"):
        if not chunk.strip():
            continue
        dims = [int(part) for part in chunk.split(",") if part.strip()]
        if not dims or any(dim <= 0 or dim > 4096 for dim in dims):
            raise ValueError("shape 的每一维要在 1 到 4096 之间")
        shapes.append(dims)
    if not shapes or len(shapes) > 4:
        raise ValueError("至少给一个 shape，最多四个")
    return shapes


def _render(name, reference, shapes, inputs):
    shape_text = ", ".join("[" + ", ".join(str(dim) for dim in shape) + "]" for shape in shapes)
    lines = ["    values = {\"x\": torch.randn(*shape, generator=generator)}"]
    for key in inputs:
        if key == "x":
            continue
        lines.append(f"    values[{key!r}] = torch.randn(int(shape[-1]), generator=generator)")
    body = "\n".join(lines)
    return f'''# -*- coding: utf-8 -*-
"""用户加入的算子任务。参考实现只负责对照，计时在门控子进程里。"""

from __future__ import annotations

import torch

from src.tasks import OperatorTask


{reference}


def make_inputs(shape, seed):
    generator = torch.Generator().manual_seed(int(seed))
{body}
    return values


TASK = OperatorTask(
    name="{name}",
    description="用户从对话加入的参考实现",
    category="user",
    make_inputs=make_inputs,
    reference=reference,
    shapes=[{shape_text}],
    seeds=[0, 1, 2],
    rtol=1e-4,
    atol=1e-4,
)

ALL_TASKS = [TASK]
'''
