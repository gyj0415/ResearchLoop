"""Regression checks for real provenance, model usage and test isolation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from researchloop import activity, storage
from researchloop.scenarios.operator import service


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

    def test_nested_measurements_and_source_remain_in_written_events(self):
        event = activity.record("measurement", "真实测量", source="operator", level="warning",
                                data={"gates": {"G2": {"passed": False}}, "ms": 1.5})
        saved = json.loads(activity.log_path().read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(saved["id"], event["id"])
        self.assertEqual(saved["source"], "operator")
        self.assertFalse(saved["data"]["gates"]["G2"]["passed"])

    def test_explicit_temporary_operator_root_cannot_write_to_product_log(self):
        from test_operator_run import _task_run
        target = self.root / "injected-runs"
        service.execute("lose_gelu_pool", runs_root=target, runner=lambda *_: _task_run("blocked_by_transcendental"))
        self.assertTrue((target / "activity.jsonl").exists())
        self.assertFalse(activity.log_path().exists())
        rows = [json.loads(line) for line in (target / "activity.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[-1]["data"]["experiment_count"], 0)
