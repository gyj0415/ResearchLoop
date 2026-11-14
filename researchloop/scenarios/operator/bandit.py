# -*- coding: utf-8 -*-
"""C5 · 预算分配：族级早停 + UCB1。

依据（docs/04 事后分析）：M4 的 27 个实验里 67% 花在「已经证明上限很低」的族上；
7 个族中 5 个族的上限只有最优族的 44%，却占了 63% 的实验。这条模块就是把
「死族」冻结掉，把预算还给还没试过的族。

规则（写死，可核查）：
  1) 某族 tried≥2 且族内最优 < 0.5×全局最优 → frozen，配额 0（可显式复活一次）；
  2) 未尝试过的新族先给 2 个探索名额；
  3) 剩余配额按 UCB1 逐个分配（探索/利用平衡），总和严格等于 budget。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["FamilyState", "allocate", "FREEZE_RATIO", "MIN_TRIES",
           "NEW_FAMILY_SLOTS", "UCB_C"]

FREEZE_RATIO = 0.5
MIN_TRIES = 2
NEW_FAMILY_SLOTS = 2
UCB_C = 1.0


@dataclass
class FamilyState:
    family: str
    tried: int = 0
    best: float = 0.0            # 族内最优加速比
    cost: float = 0.0            # 云端记账成本（$）
    frozen: bool = False
    revives: int = 0

    def record(self, speedup: float, cost: float = 0.0) -> "FamilyState":
        self.tried += 1
        self.best = max(self.best, float(speedup))
        self.cost += float(cost)
        return self

    def revive(self) -> None:
        """冻结后允许复活一次（防止误杀慢热族）。"""
        if self.frozen and self.revives < 1:
            self.frozen = False
            self.revives += 1


def allocate(states: Dict[str, FamilyState], budget: int,
             global_best: float) -> List[Tuple[str, int]]:
    """返回 [(family, 配额)]；顺序按族名，冻结族配额为 0。"""
    budget = max(0, int(budget))
    names = sorted(states)

    for n in names:
        st = states[n]
        if (not st.frozen and st.tried >= MIN_TRIES
                and global_best > 0 and st.best < FREEZE_RATIO * global_best):
            st.frozen = True

    live = [n for n in names if not states[n].frozen]
    quotas: Dict[str, int] = {n: 0 for n in names}
    remaining = budget

    # 新族：先给固定探索名额
    for n in live:
        if remaining <= 0:
            break
        if states[n].tried == 0:
            q = min(NEW_FAMILY_SLOTS, remaining)
            quotas[n] += q
            remaining -= q

    # 其余按 UCB1 逐个分配
    total_pulls = sum(states[n].tried for n in live)
    while remaining > 0 and live:
        best_name: Optional[str] = None
        best_score = float("-inf")
        for n in live:
            st = states[n]
            pulls = st.tried + quotas[n]
            explore = math.sqrt(2.0 * math.log(total_pulls + 1.0) / max(1, pulls))
            score = st.best + UCB_C * explore
            if score > best_score + 1e-12 or (
                    abs(score - best_score) <= 1e-12
                    and (best_name is None or n < best_name)):
                best_score = score
                best_name = n
        if best_name is None:
            break
        quotas[best_name] += 1
        remaining -= 1

    return [(n, quotas[n]) for n in names]


def summary(states: Dict[str, FamilyState]) -> List[dict]:
    return [
        {"family": n, "tried": s.tried, "best": round(s.best, 4),
         "cost_usd": round(s.cost, 9), "frozen": s.frozen}
        for n, s in sorted(states.items())
    ]