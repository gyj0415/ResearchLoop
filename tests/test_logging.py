"""Regression checks for real provenance, model usage and test isolation."""
import io
import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from researchloop import activity, storage
from researchloop.brief import draft
from researchloop.scenarios.operator import service
from researchloop.server import Handler


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root_patch = patch.object(storage, "ROOT", self.root)
        self.runs_patch = patch.object(storage, "RUNS", self.root / "runs")
        self.root_patch.start()
        self.runs_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.runs_patch.stop)

    def test_nested_measurements_survive_and_legacy_orphans_are_excluded(self):
        activity.record("measurement", "真实测量", source="operator", level="warning",
                        data={"gates": {"G2": {"passed": False}}, "ms": 1.5})
        legacy = self.root / ".cache" / "activity.jsonl"
        legacy.parent.mkdir()
        legacy.write_text(json.dumps({"at": "2026-11-01", "run_id": "deleted-fixture", "text": "mock"}) + "\n", encoding="utf-8")
        result = activity.query(source="operator", level="warning", text="真实")
        self.assertEqual(len(result["entries"]), 1)
        self.assertFalse(result["entries"][0]["data"]["gates"]["G2"]["passed"])
        self.assertEqual(result["ignored_legacy"], 1)
        self.assertTrue(legacy.exists())
        self.assertEqual(activity.query(level="error")["entries"], [])

    def test_explicit_temporary_operator_root_cannot_write_to_product_log(self):
        from test_operator_run import _task_run
        target = self.root / "injected-runs"
        service.execute("lose_gelu_pool", runs_root=target, runner=lambda *_: _task_run("blocked_by_transcendental"))
        self.assertTrue((target / "activity.jsonl").exists())
        self.assertFalse(activity.log_path().exists())
        rows = [json.loads(line) for line in (target / "activity.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[-1]["data"]["experiment_count"], 0)

    def test_usage_is_preserved_even_if_model_json_is_invalid(self):
        response = {"message": {"content": "not valid JSON"}, "prompt_eval_count": 29, "eval_count": 7}
        with patch("researchloop.brief.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            plan = draft([{"role": "user", "content": "改进证据检索"}],
                         model={"model": "local:8b", "ollama_url": "http://127.0.0.1:11434"})
        self.assertEqual(plan["source"], "rules")
        self.assertEqual(plan["log"]["usage"]["total_tokens"], 36)
        self.assertTrue(plan["log"]["model_called"])
        self.assertEqual(plan["log"]["reason_code"], "invalid_json")

    def test_timeout_usage_is_unknown_rather_than_zero(self):
        with patch("researchloop.brief._ask_model", side_effect=TimeoutError("timed out")):
            plan = draft([{"role": "user", "content": "改进证据检索"}],
                         model={"model": "local:8b", "ollama_url": "http://127.0.0.1:11434"})
        self.assertFalse(plan["log"]["usage"]["known"])
        self.assertNotIn("total_tokens", plan["log"]["usage"])

    def test_run_filter_searches_beyond_the_latest_page(self):
        path = activity.log_path()
        path.parent.mkdir(parents=True)
        rows = [{"id": str(i), "at": f"{i:08d}", "run_id": "old-run" if i == 0 else "new-run"}
                for i in range(550)]
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.assertEqual(activity.query(limit=10)["matched"], 550)
        result = activity.query(limit=10, run_id="old-run")
        self.assertEqual(result["entries"][0]["id"], "0")
        self.assertEqual(result["matched"], 1)

    def test_http_plan_events_are_linked_and_can_be_filtered(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        body = json.dumps({"messages": [{"role": "user", "content": "改进证据检索"}], "use_model": False}).encode()
        request = urllib.request.Request(base + "/api/brief", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            plan = json.load(response)
        with urllib.request.urlopen(base + "/api/logs?source=planner") as response:
            logs = json.load(response)["entries"]
        self.assertEqual(len(logs), 2)
        self.assertEqual({row["request_id"] for row in logs}, {plan["log"]["request_id"]})
        self.assertEqual(logs[0]["actor"], "rules")
        self.assertEqual(logs[1]["status"], "running")
        with urllib.request.urlopen(base + "/api/logs?level=error") as response:
            self.assertEqual(json.load(response)["entries"], [])
