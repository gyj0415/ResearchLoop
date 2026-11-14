"""Research loop contracts: no operator task names, real subprocesses and rejection."""
import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from researchloop import storage
from researchloop.engine import core
from researchloop.engine.worker import validate


class ResearchEngineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name, value in (("RUNS", root / "runs"), ("MEMORY", root / "memory")):
            patcher = patch.object(storage, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def engine(self, question="研究随机采样的均值误差是否在合理范围内", **kwargs):
        return core.create({"question": question, "max_steps": 10, **kwargs})

    def actions(self, *, threshold=0.5, broken=False):
        rows = [{"tool": "inspect", "args": {}},
                {"tool": "hypothesis", "args": {"statement": "随机均值的绝对误差小于0.5", "prediction": "误差较小", "controls": "标准正态分布，相同样本量", "metric": "error", "relation": "lt", "threshold": threshold}}]
        if broken:
            rows.append({"tool": "experiment", "args": {"hypothesis_id": "h1", "code": "raise ValueError('错误方法，需要修正')"}})
        code = "values = rng.normal(size=200)\nemit_result({'metrics': {'error': float(abs(np.mean(values)))}})"
        identifier = "x2" if broken else "x1"
        rows.extend([{"tool": "experiment", "args": {"hypothesis_id": "h1", "code": code}},
                     {"tool": "verify", "args": {"experiment_id": identifier}},
                     {"tool": "finish", "args": {"conclusion": "仅适用于当前模拟机制", "citations": [identifier, "h1"], "lesson": "均值误差实验应保留独立随机重跑"}}])
        return iter(rows)

    def test_real_code_is_frozen_and_independently_reexecuted(self):
        engine = self.engine()
        actions = self.actions()
        engine.decide = lambda: next(actions)
        run = engine.execute()
        self.assertEqual(run["status"], "completed")
        item = run["experiments"][0]
        self.assertEqual(item["result"]["code_sha256"], item["verification"]["result"]["code_sha256"])
        self.assertNotEqual(item["result"]["seed"], item["verification"]["result"]["seed"])
        self.assertTrue((engine.folder / "report.md").exists())
        self.assertTrue((engine.folder / "source" / "researchloop" / "engine" / "core.py").exists())
        self.assertEqual(run["hypotheses"][0]["threshold"], 0.5)

    def test_failed_experiment_can_be_corrected_without_losing_failure(self):
        engine = self.engine()
        actions = self.actions(broken=True)
        engine.decide = lambda: next(actions)
        run = engine.execute()
        self.assertEqual(run["status"], "completed")
        self.assertEqual([e["status"] for e in run["experiments"]], ["failed", "completed"])
        self.assertIn("错误方法", run["steps"][2]["observation"]["error"])

    def test_refuted_hypothesis_is_an_observable_completed_research_result(self):
        engine = self.engine()
        actions = self.actions(threshold=-1)
        engine.decide = lambda: next(actions)
        run = engine.execute()
        self.assertEqual(run["status"], "completed")
        self.assertFalse(run["experiments"][0]["verification"]["check"]["passed"])

    def test_report_without_verified_experiment_is_incomplete(self):
        engine = self.engine()
        engine.decide = lambda: {"tool": "finish", "args": {"conclusion": "资料不足", "citations": []}}
        self.assertEqual(engine.execute()["status"], "incomplete")

    def test_unknown_evidence_and_unregistered_experiments_are_rejected(self):
        engine = self.engine()
        with self.assertRaises(ValueError):
            engine.apply("finish", {"citations": ["fabricated-paper"]})
        with self.assertRaises(ValueError):
            engine.apply("experiment", {"hypothesis_id": "missing", "code": "result = {}"})

    def test_material_is_readable_by_id_and_cannot_escape_workspace(self):
        engine = self.engine(assets=[{"name": "values.csv", "content": "value\n1\n2\n3\n"}])
        content = engine.apply("read", {"asset_id": "a1"})
        self.assertIn("value", content["text"])
        with self.assertRaises(ValueError):
            engine.apply("read", {"asset_id": "../../config.json"})

    def test_execution_has_a_restricted_interface(self):
        for code in ("import os", "open('config.json')", "np.load('x')", "x.__class__", "np.random.seed(1)"):
            with self.assertRaises(ValueError):
                validate(code)
        validate("values = rng.random(100)\nemit_result({'metrics': {'mean': float(np.mean(values))}})")

    def test_random_generator_rejection_names_an_actionable_fix(self):
        for code in ("rng = np.random.default_rng()\nvalues = rng.normal(size=10)",
                     "np.random.seed(42)\nvalues = rng.normal(size=10)",
                     "values = np.random.normal(0, 1, 10)"):
            with self.assertRaises(ValueError) as caught:
                validate(code)
            message = str(caught.exception)
            self.assertIn("np.random", message)
            self.assertIn("rng.normal", message)
            self.assertIn("不要自行创建生成器", message)
            self.assertIn("第1行", message)

    def test_protocol_calls_the_model_for_content_but_never_skips_real_verification(self):
        engine = self.engine()
        engine.decide = engine.ask
        h = self.actions().__next__()
        rows = list(self.actions())
        payloads = [{"feasible": True, "method": "比较独立模拟的误差", "limitations": "模拟情景", "literature_query": ""},
                    rows[1]["args"], rows[2]["args"], {"next_action": "finish", "reason": "已完成独立复核"}, rows[4]["args"]]
        responses = [io.BytesIO(json.dumps({"message": {"content": json.dumps(item)}, "prompt_eval_count": 50, "eval_count": 30}).encode()) for item in payloads]
        base_settings = storage.settings()
        offline = {**base_settings, "web": {**base_settings["web"], "enabled": False}}
        with patch("researchloop.storage.settings", return_value=offline), \
                patch("researchloop.engine.core.urllib.request.urlopen", side_effect=responses) as calls:
            run = engine.execute()
        self.assertEqual(run["status"], "completed")
        self.assertEqual([s["tool"] for s in run["steps"]], ["inspect", "plan", "hypothesis", "experiment", "verify", "reflect", "finish"])
        self.assertEqual(calls.call_count, 5)
        self.assertEqual(run["budget"]["total_tokens"], 400)
        self.assertEqual(run["steps"][4]["source"], "controller")

    def test_experience_versions_are_separate_and_rollback_preserves_history(self):
        engine = self.engine()
        actions = self.actions()
        engine.decide = lambda: next(actions)
        run = engine.execute()
        value = core.publish(run)
        self.assertEqual(value["version"], 1)
        self.assertEqual(core.activate(0)["active"], 0)
        self.assertEqual(len(core.memory_state()["versions"]), 1)

    def test_agent_chooses_the_next_tool_itself(self):
        engine = self.engine()
        payload = {"message": {"content": json.dumps({"tool": "inspect", "reason": "先看材料", "args": {}})},
                   "prompt_eval_count": 40, "eval_count": 20}
        with patch("researchloop.engine.core.urllib.request.urlopen",
                   return_value=io.BytesIO(json.dumps(payload).encode())):
            decision = engine.decide()
        self.assertEqual(decision["tool"], "inspect")
        self.assertEqual(decision["source"], "ollama")
        self.assertEqual(engine.run["calls"][0]["purpose"], "agent_step")

    def test_agent_cannot_choose_a_tool_outside_the_available_set(self):
        engine = self.engine()
        payload = {"message": {"content": json.dumps({"tool": "verify", "reason": "跳过前提", "args": {}})},
                   "prompt_eval_count": 40, "eval_count": 20}
        with patch("researchloop.engine.core.urllib.request.urlopen",
                   side_effect=lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaises(ValueError):
                engine.decide()

    def test_follow_up_continues_the_same_session(self):
        engine = self.engine()
        actions = self.actions()
        engine.decide = lambda: next(actions)
        run = engine.execute()
        first_steps = len(run["steps"])
        follow = core.resume(run, engine.folder, "继续追问：复核结果对样本量选择意味着什么？",
                             token_limit=20000, seconds_limit=120, max_steps=6)
        follow.decide = lambda: {"tool": "finish", "reason": "收束追问",
                                 "args": {"conclusion": "追问示例结论", "citations": ["x1"], "lesson": ""}}
        second = follow.execute()
        self.assertEqual(second["id"], run["id"])
        self.assertEqual(len(second["turns"]), 2)
        self.assertEqual(second["current_question"], "继续追问：复核结果对样本量选择意味着什么？")
        self.assertGreater(len(second["steps"]), first_steps)
        self.assertEqual(second["turns"][1]["status"], second["status"])
        self.assertTrue((engine.folder / "report-t2.md").exists())
        combined = (engine.folder / "report.md").read_text(encoding="utf-8")
        self.assertIn("继续追问", combined)

    def test_qualitative_finish_with_web_evidence_counts_as_completed(self):
        engine = self.engine()
        engine.run["research_kind"] = "qualitative"
        engine.run["sources"] = [{"id": "p1", "title": "媒体报道", "url": "https://news.example.com/a",
                                  "abstract": "奶龙相关内容", "evidence_scope":
                                  "网页搜索摘要（tavily），媒体/平台内容，不是同行评议"}]
        result = engine.apply("finish", {"conclusion": "依据本轮网页证据只能说明平台存在相关报道，不推断传播效果。",
                                         "citations": ["p1"], "lesson": "定性问题需要平台证据"})
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("evidence_gaps", engine.run["report"])

    def test_qualitative_finish_with_metadata_only_stays_inconclusive(self):
        engine = self.engine()
        engine.run["research_kind"] = "literature"
        engine.run["sources"] = [{"id": "p1", "title": "论文", "url": "https://doi.example/x",
                                  "abstract": "", "evidence_scope": "只有元数据，不是结论证据"}]
        result = engine.apply("finish", {"conclusion": "检索只有元数据，无法回答传播与影响。",
                                         "citations": ["p1"], "lesson": ""})
        self.assertEqual(result["status"], "inconclusive")
        self.assertTrue(engine.run["report"].get("evidence_gaps"))

    def test_budget_ceilings_are_configurable(self):
        engine = core.create({"question": "研究样本量对估计误差的影响", "token_limit": 300000,
                              "seconds_limit": 3600, "max_steps": 30})
        self.assertEqual(engine.run["token_limit"], 300000)
        base = storage.settings()
        stricter = {**base, "limits": {"max_tokens": 200000, "max_seconds": 3600, "max_steps": 20}}
        with patch("researchloop.storage.settings", return_value=stricter):
            with self.assertRaises(ValueError):
                core.create({"question": "研究样本量对估计误差的影响", "token_limit": 300000})
