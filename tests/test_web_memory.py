"""Web safety, fact memory, tool registry and workflow validation contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from researchloop import facts, storage, webaccess
from researchloop.engine import registry, roles, workflow


class WebSafetyTests(unittest.TestCase):
    def test_private_and_local_targets_are_rejected(self):
        for url in ("http://localhost/x", "http://127.0.0.1/x", "http://10.1.2.3/x",
                    "http://192.168.0.1/x", "http://169.254.1.1/x", "http://[::1]/x",
                    "ftp://example.com/x", "https://example.com:8443/x", "file:///etc/passwd"):
            with self.assertRaises(ValueError, msg=url):
                webaccess.validate_outbound(url)

    def test_public_urls_are_allowed(self):
        for url in ("https://example.com/page", "http://example.com"):
            self.assertEqual(webaccess.validate_outbound(url).hostname, "example.com")

    def test_redirect_to_local_target_is_rejected_before_following(self):
        import urllib.request
        handler = webaccess.CheckedRedirect(())
        request = urllib.request.Request("https://example.com/start")
        for target in ("http://127.0.0.1:11434/api/tags", "http://169.254.169.254/latest", "http://localhost/x"):
            with self.assertRaises(ValueError, msg=target):
                handler.redirect_request(request, None, 302, "Found", {}, target)
        followed = handler.redirect_request(request, None, 302, "Found", {}, "https://example.com/next")
        self.assertEqual(followed.full_url, "https://example.com/next")

    def test_search_input_is_validated_before_any_request(self):
        with self.assertRaises(ValueError):
            webaccess.search("", provider="searxng", endpoint="https://search.example.com")
        with self.assertRaises(ValueError):
            webaccess.search("query", provider="unknown", endpoint="https://search.example.com")


class FactMemoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name, value in (("RUNS", root / "runs"), ("MEMORY", root / "memory")):
            patcher = patch.object(storage, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_facts_are_recalled_by_keyword_score(self):
        facts.remember("s1", "current_plan", "模拟实验A", "先做标准正态模拟", tags=["模拟"])
        facts.remember("s1", "location", "实验室X", "数据保存在实验室X", tags=["数据"])
        rows = facts.match("模拟实验")
        self.assertTrue(rows)
        self.assertEqual(rows[0]["object"], "模拟实验A")
        self.assertIn("模拟", rows[0]["summary"])

    def test_exclusive_predicate_contradicts_older_fact(self):
        facts.remember("s1", "current_plan", "计划A", "先做模拟")
        facts.remember("s2", "current_plan", "计划B", "改做真实数据")
        active = facts.match("计划")
        objects = [row["object"] for row in active]
        self.assertIn("计划B", objects)
        self.assertNotIn("计划A", objects)

    def test_expired_fact_is_not_recalled(self):
        facts.remember("s1", "current_plan", "临时计划", "今天内有效", expires_at="2000-01-01T00:00:00+00:00")
        self.assertEqual(facts.match("临时计划"), [])
        stale = db_rows()
        self.assertEqual(stale[0]["status"], "stale")


def db_rows():
    from researchloop import db
    return db.facts(status=None)


class RegistryTests(unittest.TestCase):
    def test_web_and_memory_tools_are_registered(self):
        names = set(registry.names())
        self.assertTrue({"web_search", "web_fetch", "remember_fact", "get_memory_context"} <= names)
        self.assertTrue({"plan", "experiment", "verify", "finish"} <= names)

    def test_unknown_tool_is_rejected_by_dispatch(self):
        class FakeEngine:
            run = {"steps": []}
        with self.assertRaises(ValueError):
            registry.dispatch(FakeEngine(), "not_a_tool", {})


class WorkflowTests(unittest.TestCase):
    def test_default_workflow_can_handle_qualitative_or_experimental_research(self):
        stages = workflow.validate(workflow.default_stages())
        self.assertEqual([stage["role"] for stage in stages], ["planner", "generalist", "reporter"])

    def test_qualitative_workflow_without_experiment_is_allowed(self):
        stages = workflow.validate([{"role": "planner", "goal": "检索并评估证据"},
                                    {"role": "reporter", "goal": "按证据形成结论"}])
        self.assertEqual([stage["role"] for stage in stages], ["planner", "reporter"])

    def test_stage_reasons_are_preserved(self):
        stages = workflow.validate([{"role": "planner", "goal": "规划", "why": "先确认证据"},
                                    {"role": "reporter", "goal": "报告", "why": "汇总复核结果"}])
        self.assertEqual(stages[0]["why"], "先确认证据")

    def test_unknown_role_becomes_a_custom_generalist_stage(self):
        stages = workflow.validate([{"role": "planner", "goal": "规划"},
                                    {"role": "wizard", "goal": "用自定义方法验证"}])
        self.assertEqual(stages[1]["role"], "generalist")
        self.assertEqual(stages[1]["label"], "wizard")

    def test_workflow_can_have_eight_stages_and_longer_step_budgets(self):
        stages = [{"role": "analyst", "goal": f"阶段{i + 1}", "max_steps": 12} for i in range(8)]
        cleaned = workflow.validate(stages)
        self.assertEqual(len(cleaned), 8)
        self.assertEqual(cleaned[0]["max_steps"], 12)

    def test_stage_tool_subset_must_belong_to_the_role(self):
        with self.assertRaises(ValueError):
            workflow.validate([{"role": "planner", "goal": "规划", "tools": ["verify"]}])
        cleaned = workflow.validate([{"role": "analyst", "goal": "分析",
                                      "tools": ["literature", "web_search"]}])
        self.assertEqual(cleaned[0]["tools"], ["literature", "web_search"])

    def test_empty_tool_list_means_no_restriction(self):
        cleaned = workflow.validate([{"role": "reporter", "goal": "总结来源", "tools": []}])
        self.assertIsNone(cleaned[0]["tools"])
        with self.assertRaisesRegex(ValueError, "finish"):
            workflow.validate([{"role": "reporter", "goal": "总结来源", "tools": ["reflect"]}])


class SettingsMaskingTests(unittest.TestCase):
    def test_api_keys_never_appear_in_public_settings(self):
        temporary = tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "config.json").write_text(json.dumps({
            "model_source": "cloud",
            "cloud": {"enabled": True, "base_url": "https://api.example.com/v1",
                      "model": "model-x", "api_key": "secret-key-123"},
            "web": {"enabled": True, "search_provider": "tavily",
                    "search_endpoint": "https://api.tavily.com/search", "search_api_key": "secret-web-456"},
        }), encoding="utf-8")
        with patch.object(storage, "ROOT", root):
            public = storage.masked_settings()
        self.assertEqual(public["cloud"]["api_key"], "")
        self.assertTrue(public["cloud"]["api_key_set"])
        self.assertEqual(public["web"]["search_api_key"], "")
        self.assertTrue(public["web"]["search_api_key_set"])


class RoleTests(unittest.TestCase):
    def test_each_role_declares_a_tool_subset(self):
        for role in roles.ROLES:
            self.assertTrue(roles.tools_for(role), role)
        self.assertIn("experiment", roles.tools_for("experimenter"))
        self.assertNotIn("experiment", roles.tools_for("verifier"))
        self.assertNotIn("finish", roles.tools_for("planner"))
        self.assertNotIn("finish", roles.tools_for("experimenter"))
        self.assertIn("finish", roles.tools_for("reporter"))
        self.assertIn("finish", roles.tools_for("generalist"))
        self.assertIn("literature", roles.tools_for("analyst"))
        self.assertNotIn("experiment", roles.tools_for("analyst"))


if __name__ == "__main__":
    unittest.main()
