# -*- coding: utf-8 -*-
"""同窗锚点（评测口径，docs/05 §6）。

规则：每个任务块前后各测一次 reference，**取较快值**作分母。

原因（已实测）：同一协议、同一台机器，隔一段时间测 reference，0.347ms vs 1.72ms，
差 4–5 倍。跨时间相除会把加速比系统性放大或缩小，所以禁止拿 run 开头的常量当
所有候选的分母。

默认 runner 直接复用 `gates.run_g4(mode="reference")`——它已经在独立子进程里、
按 G4 的预热/重复协议计时，锚点和候选的计时口径天然一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = ["Anchor", "measure", "speedup", "default_reference_runner"]


@dataclass(frozen=True)
class Anchor:
    before_ms: float
    after_ms: float
    used_ms: float
    note: str = ""

    @property
    def drift_ratio(self) -> float:
        if self.used_ms <= 0:
            return float("nan")
        return max(self.before_ms, self.after_ms) / self.used_ms

    def to_json(self) -> dict:
        return {"anchor_before_ms": round(self.before_ms, 6),
                "anchor_after_ms": round(self.after_ms, 6),
                "anchor_ms_used": round(self.used_ms, 6),
                "drift_ratio": round(self.drift_ratio, 3),
                "note": self.note}


def measure(task: Any, runner: Optional[Callable[[Any], float]] = None) -> Anchor:
    """测 anchor_before / anchor_after，返回较快值。"""
    run = runner or default_reference_runner
    before = float(run(task))
    after = float(run(task))
    if before <= 0 or after <= 0:
        raise ValueError(f"anchor 计时非正：before={before} after={after}")
    return Anchor(before_ms=before, after_ms=after, used_ms=min(before, after),
                  note="同窗 A/B：取前后两次 reference 的较小值作分母")


def speedup(anchor: Anchor, candidate_ms: float) -> float:
    if candidate_ms <= 0:
        raise ValueError(f"候选计时非正：{candidate_ms}")
    return anchor.used_ms / float(candidate_ms)


def default_reference_runner(task: Any) -> float:
    """用 G4 的 reference 模式计时（独立子进程、独立时钟、同协议）。"""
    from .. import gates

    res = gates.run_g4([task], mode="reference")
    if res.passed is not True:
        raise RuntimeError(f"reference 计时失败：{res.evidence[:200]}")
    per = res.details.get("per_task", {}).get(getattr(task, "name", ""), {})
    median = per.get("median_ms")
    if median is None:
        raise RuntimeError(f"reference 计时缺少 {getattr(task, 'name', '?')} 的中位数")
    return float(median)