"""研究方向对话：规则草拟、拒绝伪造计划、以及用户自己的 reference。"""

import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch

from researchloop.brief import draft, normalize
from researchloop.scenarios.operator.custom import save_task


class BriefTests(unittest.TestCase):
    def test_operator_direction_is_a_falsifiable_plan(self):
        plan = draft([{"role": "user", "content": "偏置加完再取均值，看能不能比 eager 快"}], model=None)
        self.assertEqual(plan["domain"], "operator")
        self.assertEqual(plan["source"], "rules")
        self.assertIn("1.10", plan["falsify_if"])
        self.assertNotIn("已经达到", plan["reply"])
        self.assertTrue(plan["steps"])

    def test_transcendental_direction_stops_before_claiming_speed(self):
        plan = draft([{"role": "user", "content": "GELU 之后再取均值，先判断值不值得做"}])
        self.assertEqual(plan["domain"], "operator")
        self.assertIn("实验前", "".join(plan["steps"]))
        self.assertNotIn("已经达到", plan["success"])

    def test_retrieval_direction_stays_on_existing_task(self):
        plan = draft([{"role": "user", "content": "我想改进 SciFact 上的证据检索"}])
        self.assertEqual(plan["domain"], "retrieval")
        self.assertEqual(plan["task_name"], "C")

    def test_plan_cannot_skip_gates_or_invent_a_measurement(self):
        with self.assertRaises(ValueError):
            normalize({"domain": "operator", "title": "x", "question": "q", "success": "过门控",
                       "steps": ["跳过门控直接报速度"], "out_of_scope": [], "falsify_if": "失败", "reply": "好"})
        with self.assertRaises(ValueError):
            normalize({"domain": "operator", "title": "x", "question": "q", "success": "实测为 3 倍",
                       "steps": ["做实验"], "out_of_scope": [], "falsify_if": "失败", "reply": "好"})

    def test_unreachable_model_falls_back_without_a_fake_plan(self):
        plan = draft([{"role": "user", "content": "给 mean(x + bias) 找一种更快的写法"}],
                     model={"ollama_url": "http://127.0.0.1:9", "model": "deepseek-r1:8b"})
        self.assertEqual(plan["source"], "rules")
        self.assertIn("model_error", plan)

    def test_unrelated_question_is_not_forced_into_an_operator_plan(self):
        plan = draft([{"role": "user", "content": "研究clip对比预训练在现代大规模图文对上对上下游任务的提升"}])
        self.assertEqual(plan["domain"], "unscoped")
        self.assertEqual(plan["log"]["reason_code"], "skipped")

    def test_timeout_is_recorded_on_the_plan(self):
        with patch("researchloop.brief._ask_model", side_effect=TimeoutError("timed out")):
            plan = draft([{"role": "user", "content": "研究 clip 预训练"}],
                         model={"ollama_url": "http://127.0.0.1:11434", "model": "deepseek-r1:14b"})
        self.assertEqual(plan["source"], "rules")
        self.assertEqual(plan["log"]["reason_code"], "timeout")
        self.assertEqual(plan["log"]["reason_label"], "超时")
        self.assertGreaterEqual(plan["log"]["elapsed_ms"], 0)

    def test_user_reference_is_added_only_when_the_structure_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = save_task("my_bias_mean", "def reference(inputs):\n    return (inputs['x'] + inputs['bias']).mean(dim=1)\n",
                              directory=directory)
            self.assertEqual(saved["ir"], "reduce(mean, add(x, bias), axis=1)")
            text = Path(saved["path"]).read_text(encoding="utf-8")
            self.assertIn("def reference", text)
            self.assertNotIn("perf_counter", text)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_task("reads_disk", "import os\ndef reference(inputs):\n    os.listdir('.')\n    return inputs['x']\n",
                          directory=directory)
            self.assertEqual(list(Path(directory).glob("*.py")), [])


if __name__ == "__main__":
    unittest.main()
