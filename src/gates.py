# -*- coding: utf-8 -*-
"""M1 · 验证门控 G0–G4（零 token、纯程序判定，无任何 LLM API 调用）。

依据：`Research-Agent-参考文档.md` §4.2（四道硬门控 + G5 元规则）、§4.3（反 reward
hacking 的三类已发生作弊 + 三条补充措施）、§11.8（修订架构 v2 在 Verification Gate
中新增 G0 反作弊静态检查）。任务书：.loop/phases/M1.md。

【每道门防的是哪一类已发生过的作弊】（§4.3 表格；三类手法被 METR 2025-02 与
Sakana robust-kbench 独立记录过，Sakana 宣称的 10–100× 独立复现后修正为平均 1.49×）：

  1. 绕过算子库（eager bypass）——不写 kernel，直接调 torch/cuBLAS/cuDNN 参考算子
     冒充自研实现。
       → G0 静态禁止 torch 族/ctypes/subprocess 导入与动态导入变体；
         G2 运行时兜底：输出必须 np.ndarray（返回 torch 张量即违约拦截）。
  2. 显存别名（memory aliasing）——no-op kernel，输出显存与输入/参考输出重叠，
     「结果正确」纯属共享内存。
       → G0 静态标记可疑视图构造（broadcast_to / as_strided）；
         G2 运行时硬检查：输出不得是任何输入对象/与其共享内存（np.shares_memory）、
         「改输入探针」：候选返回后原地改写全部输入，已返回的输出必须逐字节不变；
         另查「候选不得改写输入」（非函数式实现一律拦截）。
  3. 篡改计时——monkey-patch torch.cuda.synchronize / time.perf_counter 让计时恒 0。
       → G0 静态禁止对 time 模块属性赋值 / setattr(time, ...)；
         G4 计时在候选无法触及的独立子进程进行（src/g4_worker.py），且：
         (a) 执行候选【前】快照原始时钟 C 函数对象，计时一律用快照对象——
             候选事后替换 time 模块属性也无法影响计时时钟；
         (b) 执行候选【后】校验 time 模块各时钟属性与快照同一（rebinding 即拒）；
         (c) 已知时长 canary（time.sleep(0.02) 实测须在合理区间）防缩放/平移时钟。

G5 元规则（§4.2）：性能（G4）只能在 G1–G4 全过的沙箱里测，「先测速再补正确性」
的顺序是错的。run_all_gates 按固定顺序 G0→G1→G2→G3→G4 短路执行。

候选契约（M1 定稿，M2/M3 沿用）：
  候选 = 一段 Python 源码，必须定义 `run(inputs: dict[str, np.ndarray]) -> np.ndarray`。
  输入由 src/tasks.py 的 make_inputs(shape, seed)（torch 张量）转 numpy 拷贝交付；
  候选必须返回新分配的 numpy 数组，不得返回输入对象/视图，不得改写输入。

G2 补充措施（§4.3 第 1/2 条）：正确性种子 = 公开 seeds（tasks.py）+ 隐藏 seeds
（本文件 HIDDEN_SEEDS，候选作者协议上不可见）。诚实声明：本仓库开源，候选作者
理论上能读到 HIDDEN_SEEDS 的字面值；真正不可见的隐藏集需要运行时才解密的测试
文件，M1 先把机制（双种子层）跑通，M3 收尾时可再把种子值移出仓库。

诚实边界（红线 5/8）：
  - G0 是 AST 静态检查，可被嵌套 exec/动态字符串拼接等手法规避；此类残余风险由
    G2 输出类型契约与 G4 子进程时钟校验运行时兜底，但 G0–G4 不是 OS 级沙箱
    （候选仍可读文件/占内存）；完整沙箱超出本阶段范围，如实记录。
  - 本文件与 src/budget.py 均为纯本地确定性代码：无网络请求、无 LLM API。
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G4_WORKER_PATH = Path(__file__).with_name("g4_worker.py")

# ---------------------------------------------------------------------------
# 门控常量（显式写进代码，不许「看起来差不多」——G3 精神）
# ---------------------------------------------------------------------------

# G2 隐藏测试集种子（§4.3 第 2 条；公开 seeds 在 src/tasks.py 的 task.seeds）
HIDDEN_SEEDS: List[int] = [40407, 81283]
# G4 计时后的「未见输入探针」种子：用候选没被计时过的输入测一次，暴露按值记忆化
HIDDEN_PROBE_SEED: int = 999331

# G4 计时协议：与 M0 eval/baseline.py 完全同口径（M0 报告遗留问题 1 的落地），
# 只有同口径，候选耗时 / 基线耗时算出的加速比才可比。
TIMING_PROTOCOL: Dict[str, Any] = {
    "warmup": 3,               # 每配置预热次数（不计入统计）
    "repeats": 7,              # 每趟重复次数（任务书要求 ≥5）
    "passes": 1,               # 分趟数（worker 内循环；对基线对照脚本用 3）
    "global_warmup_seconds": 1.5,
    "timer": "time.perf_counter（候选执行前快照的原始内建对象）",
    "aggregation": "median（同一任务全部样本合并取中位数，与 M0 主口径一致）",
}
MIN_REPEATS = 5   # 任务书：重复 ≥5 次取中位数

# G0 静态检查规则表。每条注明防的是哪一类已发生过的作弊（§4.3）。
FORBIDDEN_IMPORT_ROOTS: Dict[str, str] = {
    "torch": ("绕过算子库（§4.3 手法 1，METR 2025-02 / Sakana robust-kbench 记录）："
              "候选直接 import torch 调参考算子冒充自研实现"),
    "src":    ("偷取评测内部：导入 src.tasks 可直接调参考实现，"
               "等于把「对照答案」搬进候选"),
    "eval":   ("偷取评测内部：导入 eval.baseline 等评测脚本可摸到计时/对比逻辑"),
    "ctypes": ("绕过算子库的变体（§4.3 手法 1）：经 ctypes 直接调 cublas/cudnn "
               "等外部算子库，绕开符号级静态检查"),
    "subprocess": ("逃逸门控：候选不得派生子进程（可被用于篡改计时进程/外呼）"),
}
# 允许的动态导入目标之外一律拦截：__import__("torch") / importlib.import_module("torch")
# 的字符串字面量在 AST 里可查。
WATCH_PATCH_MODULES: Dict[str, str] = {
    "time": ("篡改计时（§4.3 手法 3，Sakana monkey-patch torch.cuda.synchronize 的 "
             "CPU 版）：候选不得替换 time 模块的计时函数"),
}
# 可疑视图构造（输出与输入共享内存的常见写法）：只作静态警告，硬判定在 G2 运行时。
ALIAS_SUSPECT_ATTRS: Dict[str, str] = {
    "broadcast_to": "可疑显存别名：np.broadcast_to 返回与输入共享内存的只读视图",
    "as_strided":   "可疑显存别名：as_strided 可构造与输入重叠的视图",
}


# ---------------------------------------------------------------------------
# 门控结果结构
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """单道门控的结构化结果：通过与否 + 证据字符串 + 结构化细节。"""
    gate: str                       # "G0".."G4"
    passed: Optional[bool]          # True/False；None = 被 G5 元规则跳过未运行
    evidence: str                   # 人读证据（引用具体数字/行号/规则）
    details: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.details.items() if not k.startswith("_")}
        return {"gate": self.gate, "passed": self.passed,
                "evidence": self.evidence, "details": d}


def _gate(gate: str, passed: Optional[bool], evidence: str,
          details: Optional[Dict[str, Any]] = None) -> GateResult:
    return GateResult(gate=gate, passed=passed, evidence=evidence,
                      details=details or {})


# ---------------------------------------------------------------------------
# G0 · 反作弊静态检查（AST）
# ---------------------------------------------------------------------------

def _dotted_name(node: ast.AST) -> Optional[str]:
    """把 Name/Attribute 链展平成 'np.lib.stride_tricks' 形式（解析不了返回 None）。"""
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def run_g0(source: str) -> GateResult:
    """G0 反作弊静态检查：AST 逐节点扫描候选源码。

    防的三类已发生作弊（§4.3）：① 绕过算子库（import torch/ctypes 直调参考算子）；
    ② 显存别名（broadcast_to/as_strided 可疑构造，静态警告 + G2 运行时硬拦）；
    ③ 篡改计时（对 time 模块属性赋值 / setattr(time, ...)）。
    诚实边界：静态检查可被嵌套 exec/动态拼接规避（见模块 docstring），
    残余风险由 G2/G4 运行时兜底。
    """
    hits: List[str] = []
    warns: List[str] = []
    details: Dict[str, Any] = {"hits": hits, "warnings": warns,
                               "forbidden_roots": dict(FORBIDDEN_IMPORT_ROOTS)}

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return _gate("G0", False,
                     f"G0 拒绝：源码不是合法 Python（静态检查无法进行；G1 亦无法通过）："
                     f"{type(e).__name__}: {e} (line {e.lineno})", details)

    aliases: Dict[str, str] = {}     # 本地名 -> 顶层模块名
    watch: set = set()               # 被 monkey-patch 监视的模块别名

    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", 0)

        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                bound = a.asname or root
                aliases[bound] = root
                if root in FORBIDDEN_IMPORT_ROOTS:
                    hits.append(f"L{lineno} [禁用导入] `import {a.name}`："
                                f"{FORBIDDEN_IMPORT_ROOTS[root]}")
                if root in WATCH_PATCH_MODULES:
                    watch.add(bound)

        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                names = ", ".join(a.name for a in node.names)
                hits.append(f"L{lineno} [禁用导入] `from {node.module} import {names}`："
                            f"{FORBIDDEN_IMPORT_ROOTS[root]}")
            for a in node.names:
                if a.name == "*":
                    continue
                bound = a.asname or a.name
                aliases[bound] = root
                if root in WATCH_PATCH_MODULES:
                    watch.add(bound)

        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                        and t.value.id in watch):
                    hits.append(
                        f"L{lineno} [篡改计时] `{t.value.id}.{t.attr} = ...`："
                        f"{WATCH_PATCH_MODULES[t.value.id]}")

        elif isinstance(node, ast.Call):
            f = node.func
            # setattr(time, "perf_counter", ...) 形式的 monkey-patch
            if (isinstance(f, ast.Name) and f.id == "setattr"
                    and node.args and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in watch):
                attr = (node.args[1].value if len(node.args) > 1
                        and isinstance(node.args[1], ast.Constant) else "?")
                hits.append(f"L{lineno} [篡改计时] `setattr({node.args[0].id}, "
                            f"'{attr}', ...)`：{WATCH_PATCH_MODULES.get(node.args[0].id, '')}")
            # __import__("torch") 动态导入
            if (isinstance(f, ast.Name) and f.id == "__import__" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and str(node.args[0].value).split(".")[0] in FORBIDDEN_IMPORT_ROOTS):
                hits.append(f"L{lineno} [禁用动态导入] `__import__('{node.args[0].value}')`："
                            f"{FORBIDDEN_IMPORT_ROOTS[str(node.args[0].value).split('.')[0]]}")
            # importlib.import_module("torch")（importlib 本身若未导入则此处够不到）
            if (isinstance(f, ast.Attribute) and f.attr == "import_module"
                    and isinstance(f.value, ast.Name)
                    and aliases.get(f.value.id) == "importlib" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and str(node.args[0].value).split(".")[0] in FORBIDDEN_IMPORT_ROOTS):
                hits.append(f"L{lineno} [禁用动态导入] `import_module('{node.args[0].value}')`："
                            "绕过算子库的动态导入变体（§4.3 手法 1）")
            # 可疑别名构造（软警告；硬判定在 G2 运行时）
            dn = _dotted_name(f)
            if dn:
                head = dn.split(".")[0]
                tail = dn.split(".")[-1]
                if (tail in ALIAS_SUSPECT_ATTRS
                        and aliases.get(head) in ("numpy", "np")):
                    warns.append(f"L{lineno} [可疑显存别名] `{dn}(...)`："
                                 f"{ALIAS_SUSPECT_ATTRS[tail]}（硬判定在 G2 运行时："
                                 "输出与输入共享内存即拒）")

    if hits:
        ev = ("G0 拒绝（反作弊静态检查，§4.3/§11.8）：\n  " + "\n  ".join(hits))
        if warns:
            ev += "\n另有静态可疑点：\n  " + "\n  ".join(warns)
        return _gate("G0", False, ev, details)

    ev = ("G0 通过：未发现禁用导入（torch/src/eval/ctypes/subprocess）、未发现 "
          "time 模块 monkey-patch、未发现动态导入 torch 的字面量。")
    if warns:
        ev += " 静态可疑点（不拦截，G2 运行时硬判）：" + "；".join(warns)
    return _gate("G0", True, ev, details)


# ---------------------------------------------------------------------------
# G1 · 编译 / 导入
# ---------------------------------------------------------------------------

def run_g1(source: str) -> GateResult:
    """G1 编译/导入：候选源码能 compile + exec，且暴露可调用 run(inputs)。

    防的是：交付了语法错误/不可运行/缺入口的候选（确定性、零成本，§4.2 G1 行）。
    注意：G1 在门控主进程 exec 候选，须在 G0 通过之后（run_all_gates 保证顺序）。
    """
    try:
        code = compile(source, "<candidate>", "exec")
    except SyntaxError as e:
        return _gate("G1", False,
                     f"G1 拒绝：语法错误 {type(e).__name__}: {e} (line {e.lineno})")
    ns: Dict[str, Any] = {"__name__": "candidate_g1"}
    try:
        exec(code, ns)  # noqa: S102 - 门控须实际导入候选；仅在本文件控制下运行
    except Exception as e:  # noqa: BLE001 - 候选导入期任何异常都算 G1 失败
        return _gate("G1", False,
                     f"G1 拒绝：候选在导入/执行顶层代码时抛出 {type(e).__name__}: {e}")
    fn = ns.get("run")
    if not callable(fn):
        return _gate("G1", False,
                     "G1 拒绝：候选未暴露可调用 `run(inputs)`（契约见 src/gates.py 模块注释）")
    try:
        import inspect
        n_params = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n_params = -1
    return _gate("G1", True,
                 f"G1 通过：compile + 顶层 exec 成功；暴露可调用 run（参数 {n_params} 个）",
                 {"_run_fn": fn})


# ---------------------------------------------------------------------------
# G2/G3 共用的候选评估（一次执行，两道门分别取所需证据）
# ---------------------------------------------------------------------------

def evaluate_candidate(task, run_fn) -> Dict[str, Any]:
    """按 (shape, seed) 全组合运行候选并与参考实现对比，产出 G2/G3 所需证据。

    种子 = 公开 seeds（task.seeds）+ 隐藏 seeds（HIDDEN_SEEDS，§4.3 第 2 条），
    输入 = task.make_inputs(shape, seed)（确定性，跨进程可复现，M0 已验证）。
    参考实现先算（防候选改写输入污染对照）；候选拿到输入的 numpy 拷贝。

    结构检查（G2）：可运行 / 返回 np.ndarray（torch 张量 = eager 绕过，违约即拒）/
    shape 一致 / 输出有限 / 输出与输入零别名（is + np.shares_memory）/ 不改写输入 /
    同输入两次调用结果逐字节一致（确定性）/ 改输入探针（输出不被后续输入原地改写影响）。
    数值检查（G3）：|c-r| ≤ atol + rtol·|r| 逐元素，记录最大绝对/相对误差与最差配置。
    """
    seeds = list(task.seeds) + list(HIDDEN_SEEDS)
    configs = [(tuple(int(v) for v in shape), int(seed))
               for shape in task.shapes for seed in seeds]

    records: List[Dict[str, Any]] = []
    fail: Optional[str] = None

    for idx, (shape, seed) in enumerate(configs):
        inputs_t = task.make_inputs(shape, seed)
        ref = task.run_reference(inputs_t).detach()
        ref64 = ref.numpy().astype(np.float64)          # 拷贝，防后续污染
        cand_in = {k: v.numpy().copy() for k, v in inputs_t.items()}
        before = {k: a.tobytes() for k, a in cand_in.items()}

        try:
            out_raw = run_fn(dict(cand_in))
        except Exception as e:  # noqa: BLE001
            fail = (f"候选在 {task.name} shape={list(shape)} seed={seed} 抛出 "
                    f"{type(e).__name__}: {e}")
            break

        # —— 契约：必须返回 numpy 数组（返回 torch 张量 = eager 绕过，G0 运行时兜底）——
        if not isinstance(out_raw, np.ndarray):
            fail = (f"{task.name} shape={list(shape)} seed={seed}：run 返回 "
                    f"{type(out_raw).__name__} 而非 np.ndarray（违反候选契约；"
                    "若为 torch 张量即「绕过算子库」作弊，§4.3 手法 1）")
            break
        out = out_raw

        # —— shape ——
        if tuple(out.shape) != tuple(ref64.shape):
            fail = (f"{task.name} shape={list(shape)} seed={seed}：输出 shape "
                    f"{list(out.shape)} != 参考 shape {list(ref64.shape)}")
            break

        # —— 有限值 ——
        if not np.isfinite(out).all():
            fail = (f"{task.name} shape={list(shape)} seed={seed}：输出含 NaN/Inf")
            break

        # —— 显存别名硬检查（§4.3 手法 2）——
        alias = None
        for k, a in cand_in.items():
            if out is a:
                alias = f"输出对象与输入 '{k}' 是同一对象（no-op 直接返回输入）"
                break
            if np.shares_memory(out, a):
                alias = f"输出与输入 '{k}' 共享内存（np.shares_memory=True，视图/no-op 别名）"
                break
        if alias:
            fail = f"{task.name} shape={list(shape)} seed={seed}：{alias}（§4.3 手法 2）"
            break

        # —— 候选不得改写输入 ——
        mutated = [k for k, a in cand_in.items() if a.tobytes() != before[k]]
        if mutated:
            fail = (f"{task.name} shape={list(shape)} seed={seed}：候选原地改写了输入 "
                    f"{mutated}（非函数式实现，破坏对照/计时假设）")
            break

        # —— 数值误差统计（供 G3 判定）——
        c64 = out.astype(np.float64)
        denom = task.atol + task.rtol * np.abs(ref64)
        err = np.abs(c64 - ref64)
        tol_ratio = float((err / denom).max())
        records.append({
            "shape": list(shape), "seed": seed,
            "max_abs_err": float(err.max()),
            "max_rel_err": float((err / np.maximum(np.abs(ref64), 1e-300)).max()),
            "max_tol_ratio": tol_ratio,   # ≤1.0 即在 rtol/atol 内
        })

        # —— 首个配置追加两项运行时探针 ——
        if idx == 0:
            # (a) 确定性：同值输入再跑一次，输出必须逐字节一致
            redet_in = {k: v.copy() for k, v in cand_in.items()}
            try:
                out2 = run_fn(dict(redet_in))
            except Exception as e:  # noqa: BLE001
                fail = f"确定性复跑抛出 {type(e).__name__}: {e}"
                break
            if not isinstance(out2, np.ndarray) or out2.tobytes() != out.tobytes():
                fail = (f"{task.name} shape={list(shape)} seed={seed}：同输入两次调用"
                        "输出不一致（候选含未初始化读/随机性，结果不可信）")
                break
            # (b) 改输入探针：候选返回后原地改写全部输入，已返回的输出必须不变
            snap = out.copy()
            for a in cand_in.values():
                a += np.float32(1.0)
            if not np.array_equal(out, snap):
                fail = (f"{task.name} shape={list(shape)} seed={seed}：改写输入后已返回"
                        "的输出随之变化 —— 输出与输入显存别名（§4.3 手法 2，探针捕获）")
                break

    return {
        "task": task.name,
        "structural_ok": fail is None,
        "fail_reason": fail,
        "records": records,
        "n_configs": len(configs),
        "n_completed": len(records),
        "seeds_used": {"public": list(task.seeds), "hidden": list(HIDDEN_SEEDS)},
    }


def run_g2(task, eval_art: Dict[str, Any]) -> GateResult:
    """G2 正确性：候选在全部 (shape, seed) 组合（含边界 shape + 隐藏 seed）上
    可运行、结构正确、无别名、不改输入、确定。

    防的是：单一输入被「特判」骗过（§4.2 G2 行：多组随机输入 ≥3 组含边界 shape）、
    显存别名/no-op（§4.3 手法 2）、输出不确定。数值容差判定归 G3。
    """
    if not eval_art["structural_ok"]:
        return _gate("G2", False,
                     f"G2 拒绝：{eval_art['fail_reason']}",
                     {"n_configs": eval_art["n_configs"],
                      "n_completed": eval_art["n_completed"]})
    ev = (f"G2 通过：{eval_art['task']} 共 {eval_art['n_configs']} 组 "
          f"(shape × seed) 全部通过——shape/有限值/确定性复跑一致/无显存别名"
          f"（is + np.shares_memory + 改输入探针）/未改写输入；"
          f"seeds：公开 {eval_art['seeds_used']['public']} + 隐藏 "
          f"{eval_art['seeds_used']['hidden']}（§4.3 第 2 条）。")
    return _gate("G2", True, ev,
                 {"records": eval_art["records"],
                  "seeds_used": eval_art["seeds_used"],
                  "n_configs": eval_art["n_configs"]})


def run_g3(task, eval_art: Dict[str, Any]) -> GateResult:
    """G3 数值容差：显式 rtol/atol 阈值逐元素判定，不许「看起来差不多」。

    阈值来自 src/tasks.py 的 task.rtol/task.atol（TOL_ELEMENTWISE=1e-4/1e-5、
    TOL_REDUCE=1e-4/1e-4，M0 定稿常量）；判定式 |c-r| ≤ atol + rtol·|r|。
    防的是：误差贴着容差边缘蒙混 / 用近似的输出冒充（§4.2 G3 行「必须显式设定」）。
    """
    if not eval_art["structural_ok"]:
        return _gate("G3", None,
                     f"G3 未运行（G5 元规则）：G2 结构检查未过——{eval_art['fail_reason']}",
                     {"skipped": True})
    if eval_art["n_completed"] != eval_art["n_configs"]:
        return _gate("G3", None,
                     "G3 未运行（G5 元规则）：对比未覆盖全部配置", {"skipped": True})

    worst = max(eval_art["records"], key=lambda r: r["max_tol_ratio"])
    passed = worst["max_tol_ratio"] <= 1.0
    ev = (f"G3 {'通过' if passed else '拒绝'}：阈值显式取自 tasks.py 常量 "
          f"rtol={task.rtol} atol={task.atol}（判式 |c-r| ≤ atol + rtol·|r|）；"
          f"{eval_art['n_configs']} 组对比，最差配置 shape={worst['shape']} "
          f"seed={worst['seed']}：max_abs_err={worst['max_abs_err']:.3e}, "
          f"max_rel_err={worst['max_rel_err']:.3e}, "
          f"误差/容差上限比={worst['max_tol_ratio']:.4f}（≤1 为过）。")
    return _gate("G3", passed, ev,
                 {"rtol": task.rtol, "atol": task.atol,
                  "worst": worst, "records": eval_art["records"]})


# ---------------------------------------------------------------------------
# G4 · 计时隔离（独立子进程）
# ---------------------------------------------------------------------------

def run_g4(tasks: List, source: Optional[str] = None, mode: str = "candidate",
           passes: Optional[int] = None, warmup: Optional[int] = None,
           repeats: Optional[int] = None, timeout_s: float = 900.0) -> GateResult:
    """G4 计时隔离：在候选无法篡改的独立子进程里计时（src/g4_worker.py）。

    防的是：篡改计时（§4.3 手法 3：monkey-patch 计时函数让耗时归零）。机制：
      1) 候选代码只进 worker 子进程；门控主进程永不 import 候选；
      2) worker 在执行候选前快照原始时钟 C 对象并用它计时，候选替换 time 属性
         无效；执行后校验身份，rebinding 即拒（integrity=fail）；
      3) 已知时长 canary（time.sleep(0.02)）防缩放/平移时钟；
      4) 协议与 M0 基线同口径：全局预热 1.5s + 每配置预热 3 + 重复 7（≥5）取中位数。
    额外信号：计时结束后用未见输入（HIDDEN_PROBE_SEED）探针一次，暴露
    「按值记忆化把计时缩水」类作弊（证据级，不作硬判）。
    """
    if mode not in ("candidate", "reference"):
        raise ValueError(f"未知 mode: {mode}")
    if mode == "candidate" and not source:
        raise ValueError("mode=candidate 需要候选源码")
    p = dict(TIMING_PROTOCOL)
    passes = int(passes if passes is not None else p["passes"])
    warmup = int(warmup if warmup is not None else p["warmup"])
    repeats = int(repeats if repeats is not None else p["repeats"])
    if repeats < MIN_REPEATS:
        raise ValueError(f"G4 要求 repeats ≥ {MIN_REPEATS}（任务书计时协议），实得 {repeats}")

    spec = {
        "mode": mode,
        "source": source,
        "tasks": [t.name for t in tasks],
        "passes": passes, "warmup": warmup, "repeats": repeats,
        "probe_seed": HIDDEN_PROBE_SEED if mode == "candidate" else None,
    }
    try:
        proc = subprocess.run(
            [sys.executable, str(G4_WORKER_PATH)],
            input=json.dumps(spec), capture_output=True, text=True,
            cwd=str(ROOT), timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return _gate("G4", False, f"G4 拒绝：计时子进程超时（>{timeout_s:.0f}s）")

    if proc.returncode != 0:
        tail = (proc.stderr or "")[-800:]
        return _gate("G4", False,
                     f"G4 拒绝：计时子进程异常退出（code={proc.returncode}）——"
                     f"stderr 末段：{tail!r}")

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return _gate("G4", False,
                     f"G4 拒绝：子进程输出不是合法 JSON（{e}）——stdout 末段：{proc.stdout[-400:]!r}")

    env = payload.get("env", {})
    if payload.get("integrity") != "ok":
        reasons = []
        if payload.get("tampered"):
            reasons.append(f"时钟完整性校验失败：候选替换了 time.{', time.'.join(payload['tampered'])}"
                           "（monkey-patch 计时函数，§4.3 手法 3）")
        if payload.get("torch_bindings"):
            reasons.append(f"候选命名空间绑定了 torch 族模块 {payload['torch_bindings']}"
                           "（eager 绕过的运行时证据，§4.3 手法 1）")
        if payload.get("canary_bad"):
            reasons.append(f"已知时长 canary 异常：time.sleep({payload['canary_requested']}s) "
                           f"被测得 {payload['canary_s']:.4f}s（时钟被缩放/平移的迹象）")
        if payload.get("error"):
            reasons.append(f"worker 错误：{payload['error']}")
        return _gate("G4", False,
                     "G4 拒绝（计时完整性不通过）：\n  - " + "\n  - ".join(reasons),
                     {"integrity": payload, "env": env,
                      # 取证：即便拒绝，worker 仍用「候选执行前快照的原始时钟」测了
                      # 全部样本——可据此证明计时未被候选污染（样本量级正常）。
                      "samples": payload.get("samples", {}),
                      "protocol": payload.get("protocol", {})})

    # 按任务汇总：全部样本合并中位数（与 M0 主口径一致）
    per_task: Dict[str, Dict[str, Any]] = {}
    for t in tasks:
        pooled: List[float] = []
        for key, s in payload["samples"].items():
            if key.startswith(t.name + "|"):
                pooled.extend(s)
        if pooled:
            pooled.sort()
            per_task[t.name] = {
                "median_ms": round(_median(pooled), 6),
                "n_samples": len(pooled),
                "min_ms": round(min(pooled), 6),
                "max_ms": round(max(pooled), 6),
            }
    probe = payload.get("probe", {})
    probe_txt = ""
    if probe:
        probe_txt = ("；未见输入探针（seed=%s，证据级，非硬判）：" % payload.get("probe_seed"))
        parts = []
        for k, v in probe.items():
            ratio = v.get("ratio_to_median")
            parts.append(f"{k}={v['ms']:.3f}ms" +
                         (f"(中位数的 {ratio:.2f}×)" if ratio is not None else ""))
        probe_txt += ", ".join(parts)

    ev = (f"G4 通过：计时在独立子进程（候选无法触及主进程时钟）；时钟完整性校验 ok"
          f"（执行候选后 time.perf_counter/monotonic/... 与快照同一内建对象）；"
          f"已知时长 canary：sleep({payload['canary_requested']}s) 实测 "
          f"{payload['canary_s']:.4f}s（合理区间内）；协议=全局预热 "
          f"{payload['protocol']['global_warmup_seconds']}s + 每配置预热 "
          f"{payload['protocol']['warmup']} 次 + 重复 {payload['protocol']['repeats']} 次"
          f" × {payload['protocol']['passes']} 趟取中位数（与 M0 同口径，repeats≥5 满足任务书）。"
          + "各任务中位耗时：" + ", ".join(
              f"{k}={v['median_ms']}ms(n={v['n_samples']})"
              for k, v in per_task.items()) + probe_txt)
    return _gate("G4", True, ev,
                 {"per_task": per_task, "samples": payload["samples"],
                  "protocol": payload["protocol"], "integrity": payload.get("integrity"),
                  "canary_s": payload["canary_s"], "env": env, "probe": probe})


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ---------------------------------------------------------------------------
# G5 元规则：固定顺序短路执行
# ---------------------------------------------------------------------------

def run_all_gates(task, source: str, candidate_name: str = "candidate",
                  g4_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """按 G0→G1→G2→G3→G4 固定顺序执行，前序未全过则后续门不运行（§4.2 G5：
    「任何先测速再补正确性的顺序都是错的」）。性能只在 G0–G3 全过的沙箱里测。
    """
    g4_kwargs = dict(g4_kwargs or {})
    gates: Dict[str, GateResult] = {}

    g0 = run_g0(source)
    gates["G0"] = g0
    if g0.passed is False:
        return _assemble(candidate_name, task.name, gates, "fail")

    g1 = run_g1(source)
    gates["G1"] = g1
    if g1.passed is False:
        return _assemble(candidate_name, task.name, gates, "fail")

    eval_art = evaluate_candidate(task, g1.details["_run_fn"])
    gates["G2"] = run_g2(task, eval_art)
    if gates["G2"].passed is False:
        gates["G3"] = _gate("G3", None, "G3 未运行（G5 元规则）：G2 未通过",
                            {"skipped": True})
    else:
        gates["G3"] = run_g3(task, eval_art)

    if gates["G2"].passed is True and gates["G3"].passed is True:
        gates["G4"] = run_g4([task], source=source, mode="candidate", **g4_kwargs)
    else:
        gates["G4"] = _gate("G4", None,
                            "G4 未运行（G5 元规则）：G0–G3 未全部通过——性能不在未验证的"
                            "候选上测量（§4.2：任何「先测速再补正确性」的顺序都是错的）",
                            {"skipped": True})

    verdict = "pass" if all(g.passed is True for g in gates.values()) else "fail"
    return _assemble(candidate_name, task.name, gates, verdict)


def _assemble(candidate: str, task_name: str,
              gates: Dict[str, GateResult], verdict: str) -> Dict[str, Any]:
    blocked_by = next((g.gate for g in gates.values() if g.passed is False), None)
    return {
        "candidate": candidate,
        "task": task_name,
        "order": list(gates.keys()),
        "verdict": verdict,
        "blocked_by": blocked_by,
        "gates": {k: v.to_json() for k, v in gates.items()},
    }
