# -*- coding: utf-8 -*-
"""算子任务注册表。

G4 计时子进程按任务名调用 `get_task`。本文件只保留任务契约和外部目录加载，
不带入 8 里已经冻结的 5 个官方任务。演示任务放在
`researchloop/scenarios/operator/tasks/`，由 `RESEARCH_TASK_DIRS` 注册。
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch

TensorDict = Dict[str, torch.Tensor]

DEFAULT_SEEDS: List[int] = [0, 1, 2]


@dataclass(frozen=True)
class OperatorTask:
    """单个算子任务。参考实现同时充当正确性对照和计时分母。"""

    name: str
    description: str
    category: str
    make_inputs: Callable[[Sequence[int], int], TensorDict]
    reference: Callable[[TensorDict], torch.Tensor]
    shapes: List[List[int]]
    seeds: List[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    rtol: float = 1e-4
    atol: float = 1e-5
    shape_key: str = "x"

    def run_reference(self, inputs: TensorDict) -> torch.Tensor:
        return self.reference(inputs)


ALL_TASKS: List[OperatorTask] = []

_EXTRA_TASKS: Dict[str, OperatorTask] = {}
_EXTRA_LOADED = False


def register_task(task: OperatorTask) -> None:
    """登记一个外部任务。同名时后登记的覆盖外部表。"""
    _EXTRA_TASKS[str(task.name)] = task


def _load_extra_tasks() -> None:
    global _EXTRA_LOADED
    if _EXTRA_LOADED:
        return
    _EXTRA_LOADED = True
    spec = os.environ.get("RESEARCH_TASK_DIRS", "")
    if not spec:
        return
    for entry in spec.split(os.pathsep):
        if not entry:
            continue
        directory = Path(entry)
        if not directory.is_dir():
            print(f"[tasks] 跳过不存在的外部任务目录: {directory}", file=sys.stderr)
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
            mod_name = f"_extra_task_{digest}"
            if mod_name in sys.modules:
                module = sys.modules[mod_name]
            else:
                try:
                    loaded = importlib.util.spec_from_file_location(mod_name, path)
                    if loaded is None or loaded.loader is None:
                        continue
                    module = importlib.util.module_from_spec(loaded)
                    sys.modules[mod_name] = module
                    loaded.loader.exec_module(module)
                except Exception as exc:  # noqa: BLE001 — 坏的外部文件不拖垮注册表
                    sys.modules.pop(mod_name, None)
                    print(f"[tasks] 加载外部任务失败 {path}: "
                          f"{type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
            found: List[OperatorTask] = []
            for attr in ("ALL_TASKS", "TASKS"):
                value = getattr(module, attr, None)
                if isinstance(value, (list, tuple)):
                    found.extend(value)
            single = getattr(module, "TASK", None)
            if single is not None and not any(single is item for item in found):
                found.append(single)
            for item in found:
                if isinstance(item, OperatorTask):
                    register_task(item)


def get_task(name: str) -> OperatorTask:
    """按任务名取任务。官方表为空，演示任务来自 RESEARCH_TASK_DIRS。"""
    for task in ALL_TASKS:
        if task.name == name:
            return task
    _load_extra_tasks()
    if name in _EXTRA_TASKS:
        return _EXTRA_TASKS[name]
    raise KeyError(f"未知任务: {name!r}；现有任务: "
                   f"{[task.name for task in ALL_TASKS] + sorted(_EXTRA_TASKS)}")
