"""算子研究层落进 ResearchLoop 之后的结构验证。不导入 torch，也不跑计时。"""

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

from researchloop.scenarios.operator.evidence import OpProfile, _decide
from researchloop.scenarios.operator.lifter import lift_source
from researchloop.scenarios.operator.rules import rewritable_reduce
from researchloop.scenarios.operator.search import search


WIN_SOURCE = '''
def reference(inputs):
    return (inputs["x"] + inputs["bias"]).mean(dim=1)
'''

LOSE_SOURCE = '''
def reference(inputs):
    return F.gelu(inputs["x"] + inputs["bias"]).mean(dim=1)
'''


def _spec():
    return SimpleNamespace(shape_rank=lambda key: {"x": 3, "bias": 1}[key])


class OperatorLogicTests(unittest.TestCase):
    def test_retrieval_import_does_not_load_operator_runtime(self):
        code = (
            "import sys\n"
            "import researchloop.runner\n"
            "assert 'torch' not in sys.modules, 'torch'\n"
            "assert 'src.gates' not in sys.modules, 'gates'\n"
            "assert 'researchloop.scenarios.operator.pipeline' not in sys.modules, 'pipeline'\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_win_task_is_rewritable(self):
        node = lift_source(WIN_SOURCE)
        self.assertEqual(node.to_source(), "reduce(mean, add(x, bias), axis=1)")
        spec = _spec()
        rule = rewritable_reduce(node, spec)
        self.assertEqual(rule, "R1_reduce_of_add_mean")
        found = search(node, spec, cap=4)
        self.assertTrue(found)
        self.assertIn("R1_reduce_of_add_mean", found[0].rule_chain)

    def test_lose_task_is_blocked_before_experiments(self):
        node = lift_source(LOSE_SOURCE)
        self.assertIn("act(gelu", node.to_source())
        spec = _spec()
        self.assertIsNone(rewritable_reduce(node, spec))
        profiles = [OpProfile("act:gelu", 10.0, 1.0, "transcendental", 10.0)]
        hint, factor, notes = _decide(spec, node, profiles, "transcendental")
        self.assertEqual(hint, "blocked_by_transcendental")
        self.assertEqual(factor, 0.0)
        self.assertTrue(notes)

    def test_rewritable_reduce_is_not_blocked_by_a_slow_reduction_profile(self):
        node = lift_source(WIN_SOURCE)
        spec = _spec()
        profiles = [
            OpProfile("add", 1.2, 1.0, "memory", 1.2),
            OpProfile("reduce:mean", 9.9, 1.0, "compute", 9.9),
        ]
        hint, factor, _notes = _decide(spec, node, profiles, "compute")
        self.assertEqual(hint, "winnable")
        self.assertEqual(factor, 1.0)


if __name__ == "__main__":
    unittest.main()
