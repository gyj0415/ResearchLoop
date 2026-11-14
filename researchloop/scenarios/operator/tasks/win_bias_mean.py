# -*- coding: utf-8 -*-
"""演示任务：mean(x + bias)。加性归约可以消元，预期能改写。"""

from __future__ import annotations

import torch

from src.tasks import OperatorTask


def _make():
    def make_inputs(shape, seed):
        b, t, c = int(shape[0]), int(shape[1]), int(shape[2])
        g = torch.Generator().manual_seed(seed)
        return {
            "x": torch.randn(b, t, c, generator=g),
            "bias": torch.randn(c, generator=g),
        }

    def reference(inputs):
        return (inputs["x"] + inputs["bias"]).mean(dim=1)

    return make_inputs, reference


_make_inputs, _reference = _make()

TASK = OperatorTask(
    name="win_bias_mean",
    description="偏置加 + 时间轴均值（演示：预期可改写）",
    category="elementwise + reduce",
    make_inputs=_make_inputs,
    reference=_reference,
    shapes=[[4, 64, 32], [1, 31, 16]],
    seeds=[0, 1, 2],
    rtol=1e-4,
    atol=1e-4,
)

ALL_TASKS = [TASK]
