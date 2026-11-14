# -*- coding: utf-8 -*-
"""M1 · G4 计时【隔离子进程】worker（评测设施，不是候选代码，候选无法修改本文件）。

由 src/gates.py 以 `python src/g4_worker.py` 派生：spec 经 stdin JSON 传入，
结果以 stdout JSON 返回（诊断信息走 stderr；stdout 只输出一个 JSON 对象）。

为什么必须独立子进程（防的是 §4.3 手法 3「篡改计时」——Sakana monkey-patch
torch.cuda.synchronize 让计时恒 0 的教训；M0 报告遗留问题 1 的落地）：

  1. 候选代码只在本进程 exec，门控主进程（对比/判定逻辑）永不 import 候选——
     候选就算 patch 了时钟，也污染不到主进程。
  2. 执行候选【之前】快照原始时钟对象（perf_counter/monotonic/time/...），
     计时一律用快照里的原始 C 函数对象——即使候选事后替换 time 模块属性
     （甚至用后台线程替换），也不影响本进程的计时时钟。
  3. 执行候选【之后】做时钟完整性校验：time 模块各时钟属性必须与快照是同一
     内建对象（BuiltinFunctionType）。rebinding monkey-patch → integrity=fail →
     G4 直接拒绝候选（取证信息照样输出）。
  4. 已知时长 canary：time.sleep(0.02) 用原始时钟实测，必须落在合理区间
     [0.015, 1.0]s——防「缩放/平移时钟」等非 rebinding 的高级手法。
  5. 计时协议与 M0 eval/baseline.py 完全同口径：线程钉 min(4,cpu)、interop=1、
     全局预热 1.5s（torch dummy 计算）、每配置预热 warmup 次 + 重复 repeats 次
     取中位数（repeats ≥ 5，任务书要求）。计时器 time.perf_counter；CPU 上
     torch eager 同步执行，无 CUDA 异步提前返回问题（本环境无 GPU）。
  6. 运行时禁用模块扫描：候选命名空间里若绑定了 torch 族模块对象（import torch /
     import torch.nn.functional as F / from torch import nn 都会在 ns 里留下
     __name__ 以 torch 开头的模块），即「绕过算子库」的运行时证据（§4.3 手法 1；
     G0 静态检查的兜底层）。
  7. 候选模式下计时结束后做一次「未见输入探针」（seed=probe_seed，不在公开/
     隐藏正确性 seeds 里）：候选若按值记忆化公开输入，探针耗时会暴露真实成本
     （证据级信号，不参与通过判定）。
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 快照并监视的时钟属性（rebinding 任一即 integrity=fail）
CLOCKS = ("perf_counter", "monotonic", "time", "process_time", "thread_time")
CANARY_REQUESTED = 0.02     # 已知时长操作：time.sleep 秒数
CANARY_MIN, CANARY_MAX = 0.015, 1.0   # 实测必须落在的区间（sleep 不会少于请求值）
BANNED_MODULE_ROOTS = ("torch",)   # 候选命名空间绑定 torch 族模块 = eager 绕过证据


def _emit(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _err_exit(msg: str) -> None:
    print(msg, file=sys.stderr)
    _emit({"integrity": "error", "error": msg})


def _pin_threads_torch(torch) -> None:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _global_warmup_torch(seconds: float = 1.5) -> None:
    """与 M0 eval/baseline.py 相同的全局预热（CPU 提频 + torch 线程池进入稳态）。"""
    import torch
    import torch.nn.functional as F
    x = torch.randn(4, 256, 1024)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        y = F.gelu(x + 1.0)
        _ = float(y.mean())


def _global_warmup_numpy(seconds: float, pc) -> None:
    import numpy as np
    x = np.zeros(4_000_000, dtype=np.float32)
    t0 = pc()
    while pc() - t0 < seconds:
        y = x * 0.5 + 1.0
        _ = float(y[::65537].sum())


def _candidate_torch_bindings(ns: dict) -> list:
    """扫候选命名空间：绑定了 __name__ 以禁用根开头的模块对象 → eager 绕过证据。"""
    found = []
    for name, val in ns.items():
        if name.startswith("__"):
            continue
        if isinstance(val, types.ModuleType):
            root = val.__name__.split(".")[0]
            if root in BANNED_MODULE_ROOTS:
                found.append(f"{name} -> {val.__name__}")
    return sorted(set(found))


def main() -> int:
    try:
        spec = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        _err_exit(f"worker: spec 不是合法 JSON: {e}")
        return 0

    mode = spec.get("mode", "candidate")
    tasks_names = spec.get("tasks", [])
    passes = int(spec.get("passes", 1))
    warmup = int(spec.get("warmup", 3))
    repeats = int(spec.get("repeats", 7))
    probe_seed = spec.get("probe_seed")

    # --- 1. 执行候选【之前】快照原始时钟对象；计时一律用快照对象 -------------
    pristine = {c: getattr(time, c) for c in CLOCKS}
    pc = pristine["perf_counter"]

    from src.tasks import get_task   # 任务定义（含 torch）；worker 属评测设施，允许导入

    run_fn = None
    tampered: list = []
    torch_bindings: list = []
    canary_s = None
    canary_bad = False

    if mode == "candidate":
        source = spec.get("source")
        if not source:
            _err_exit("worker: mode=candidate 但未提供候选源码")
            return 0
        try:
            ns = {"__name__": "candidate_g4"}
            exec(compile(source, "<candidate>", "exec"), ns)  # noqa: S102 - 隔离进程内运行候选
        except Exception as e:  # noqa: BLE001
            _err_exit(f"worker: 候选在执行顶层代码时抛出 {type(e).__name__}: {e}")
            return 0
        fn = ns.get("run")
        if not callable(fn):
            _err_exit("worker: 候选未定义可调用 run(inputs)")
            return 0
        run_fn = fn

        # --- 2. 执行候选【之后】的时钟完整性校验（rebinding 即拒）-------------
        for c in CLOCKS:
            cur = getattr(time, c)
            if (cur is not pristine[c]) or (type(cur) is not types.BuiltinFunctionType):
                tampered.append(c)
        # --- 2b. 候选命名空间里的 torch 族模块绑定（eager 绕过运行时证据）-----
        torch_bindings = _candidate_torch_bindings(ns)

        # --- 3. 已知时长 canary：防缩放/平移时钟 ------------------------------
        t0 = pc()
        time.sleep(CANARY_REQUESTED)
        canary_s = pc() - t0
        canary_bad = not (CANARY_MIN <= canary_s <= CANARY_MAX)

        _global_warmup_numpy(seconds=1.0, pc=pc)
    else:  # reference：被测对象是可信的参考实现（M0 同款），无候选代码进入
        import torch
        _pin_threads_torch(torch)
        _global_warmup_torch(seconds=1.5)
        for c in CLOCKS:  # 参考模式无候选；仍做一致性快照记录（应恒过）
            if getattr(time, c) is not pristine[c]:
                tampered.append(c)
        t0 = pc()
        time.sleep(CANARY_REQUESTED)
        canary_s = pc() - t0
        canary_bad = not (CANARY_MIN <= canary_s <= CANARY_MAX)

    integrity = "ok" if not (tampered or torch_bindings or canary_bad) else "fail"

    # --- 4. 计时循环（与 M0 同口径）----------------------------------------
    tasks = [get_task(n) for n in tasks_names]
    samples: dict = {}
    probe: dict = {}

    for p_i in range(passes):
        for task in tasks:
            for shape in task.shapes:
                shape_t = tuple(int(v) for v in shape)
                for seed in task.seeds:
                    inputs_t = task.make_inputs(shape_t, seed)   # 输入构造不计时（确定性）
                    if mode == "candidate":
                        # 候选拿输入的 numpy 拷贝（与 G2 相同口径）；构造在计时外
                        args = {k: v.numpy().copy() for k, v in inputs_t.items()}
                        call = lambda a=args: run_fn(a)          # noqa: E731
                    else:
                        args = inputs_t
                        call = lambda t=task, a=args: t.run_reference(a)  # noqa: E731

                    key = f"{task.name}|{'x'.join(map(str, shape_t))}|{seed}"
                    bucket = samples.setdefault(key, [])

                    if p_i == 0:
                        for _ in range(warmup):                  # 预热，不计入统计
                            call()

                    for _ in range(repeats):
                        t0 = pc()
                        call()
                        bucket.append((pc() - t0) * 1000.0)

    # --- 5. 未见输入探针（候选模式；证据级）---------------------------------
    if mode == "candidate" and probe_seed is not None and tasks:
        for task in tasks:
            shape_t = tuple(int(v) for v in task.shapes[0])
            inputs_t = task.make_inputs(shape_t, int(probe_seed))
            p_args = {k: v.numpy().copy() for k, v in inputs_t.items()}
            p_call = lambda a=p_args: run_fn(a)                  # noqa: E731
            p_call()                                             # 探针输入的预热
            t0 = pc()
            p_call()
            ms = (pc() - t0) * 1000.0
            pooled = []
            for key, s in samples.items():
                if key.startswith(task.name + "|"):
                    pooled.extend(s)
            pooled.sort()
            med = pooled[len(pooled) // 2] if pooled else 0.0
            probe[task.name] = {"ms": round(ms, 6),
                                "median_ms": round(med, 6),
                                "ratio_to_median": round(ms / med, 4) if med else None}

    payload = {
        "integrity": integrity,
        "tampered": tampered,
        "torch_bindings": torch_bindings,
        "canary_requested": CANARY_REQUESTED,
        "canary_s": canary_s,
        "canary_bad": canary_bad,
        "protocol": {"warmup": warmup, "repeats": repeats, "passes": passes,
                     "global_warmup_seconds": 1.5 if mode == "reference" else 1.0,
                     "timer": "time.perf_counter（候选执行前快照的原始内建对象）",
                     "aggregation": "median"},
        "env": {"python": platform.python_version(),
                "torch": sys.modules["torch"].__version__ if "torch" in sys.modules else None,
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "mode": mode},
        "samples": {k: [round(x, 6) for x in v] for k, v in samples.items()},
        "probe_seed": probe_seed,
        "probe": probe,
    }
    _emit(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
