# -*- coding: utf-8 -*-
"""C6 的验证适配层：把 `src/gates.py` 的 G0–G4 拆成「验证」与「计时」两半。

为什么需要薄薄一层适配（而不是直接改 gates）：
  - 评测器要缓存「正确性」但**不能**缓存「计时」（计时随机器状态漂移）；
  - `run_all_gates` 把 G4 也包在一起，无法把候选计时夹在同一窗口的两个
    anchor 之间；
  - `src/gates.py` 是已验证资产，一行不改（docs/05 §3 原则）。

本模块只做调用与字段搬运，不复制、不重写任何判定逻辑。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 门控文件保持原样，仍以 `src.gates` 被计时子进程所在的项目根导入。
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src import gates as _g

__all__ = ["verify_g0_to_g3", "time_candidate", "time_reference"]


def _rec(gate: str, passed: Any, evidence: str,
         details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"gate": gate, "passed": passed, "evidence": evidence,
            "details": dict(details or {})}


def _skip(gate: str, why: str) -> Dict[str, Any]:
    return _rec(gate, None, f"{gate} 未运行（G5 短路）：{why}")


def verify_g0_to_g3(task: Any, source: str) -> Dict[str, Any]:
    """按 G0→G1→G2→G3 顺序验证，前序失败即短路（与 run_all_gates 同语义）。"""
    out: Dict[str, Any] = {}

    g0 = _g.run_g0(source)
    out["G0"] = g0.to_json()
    if g0.passed is False:
        return _pack(out, "G0")

    g1 = _g.run_g1(source)
    out["G1"] = g1.to_json()
    if g1.passed is False:
        return _pack(out, "G1")

    art = _g.evaluate_candidate(task, g1.details["_run_fn"])
    g2 = _g.run_g2(task, art)
    out["G2"] = g2.to_json()
    if g2.passed is not True:
        out["G3"] = _skip("G3", "G2 未通过")
        return _pack(out, "G2")

    g3 = _g.run_g3(task, art)
    out["G3"] = g3.to_json()
    if g3.passed is not True:
        return _pack(out, "G3")

    return {"passed": True, "blocked_by": None, "gates": out,
            "evidence": "G0–G3 全过（G4 计时由调用方在锚点窗口内单独执行）"}


def _pack(gates: Dict[str, Any], blocked_by: str) -> Dict[str, Any]:
    return {"passed": False, "blocked_by": blocked_by, "gates": gates,
            "evidence": gates[blocked_by].get("evidence", "")}


def _median(res: Any, task: Any) -> Optional[float]:
    if res.passed is not True:
        return None
    per = res.details.get("per_task", {}).get(getattr(task, "name", ""), {})
    v = per.get("median_ms")
    return None if v is None else float(v)


def time_candidate(task: Any, source: str, *, passes: int = 1) -> Optional[float]:
    """候选计时（独立子进程、独立时钟、与 M0 同协议）。失败返回 None。"""
    try:
        res = _g.run_g4([task], source=source, mode="candidate", passes=passes)
    except Exception:  # noqa: BLE001 —— 计时失败按「无成绩」处理，不编造数字
        return None
    return _median(res, task)


def time_reference(task: Any, *, passes: int = 1) -> float:
    """reference 计时；失败直接抛（锚点缺失时整块结果不可解释）。"""
    res = _g.run_g4([task], mode="reference", passes=passes)
    median = _median(res, task)
    if median is None:
        raise RuntimeError(f"reference 计时失败：{(res.evidence or '')[:200]}")
    return median