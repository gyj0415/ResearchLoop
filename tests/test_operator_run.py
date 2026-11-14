"""算子运行记录：完成、超预算、以及检索摘要仍能读这条记录。"""

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from researchloop.scenarios.operator import service as operator_service
from researchloop.scenarios.operator.pipeline import TaskRun
from researchloop.scenarios.operator.service import execute
from researchloop.scenarios.operator.verdict import Verdict
from researchloop.server import SERVICE, Handler, summary
from researchloop import storage


def _verdict(reason, decision="keep_baseline"):
    return Verdict(task="win_bias_mean", decision=decision, reason_code=reason,
                   evidence_refs=["evidence:act:gelu"], best_speedup=0.0,
                   next_action="记下这条结构", detail="测试裁决")


def _task_run(reason, *, experiments=(), wallclock_s=0.2, decision="keep_baseline",
              hint="blocked_by_transcendental", skip="证据层实验前拦截"):
    return TaskRun(
        task="lose_gelu_pool", arm="full", verdict_hint=hint, dominant="transcendental",
        budget_alloc={}, experiments=list(experiments), verdict=_verdict(reason, decision),
        wallclock_s=wallclock_s, skip_reason=skip, evidence_json={"dominant": "transcendental"},
    )


class OperatorRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache")
        self.addCleanup(self.temporary.cleanup)
        patcher = patch.object(storage, "RUNS", Path(self.temporary.name) / "runs")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_blocked_run_is_recorded_and_publishable(self):
        with tempfile.TemporaryDirectory() as directory:
            run = execute("lose_gelu_pool", runs_root=Path(directory), seconds_limit=30,
                          runner=lambda name, limit: _task_run("blocked_by_transcendental"))
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["memory_candidate"]["reason_code"], "blocked_by_transcendental")
            self.assertEqual(run["memory_candidate"]["domain"], "operator")
            run_dir = Path(directory) / run["id"]
            for name in ("run.json", "events.jsonl", "evidence.json", "hypothesis.json",
                         "experiment.json", "verdict.json"):
                self.assertTrue((run_dir / name).exists(), name)
            card = summary(run)
            self.assertEqual(card["domain"], "operator")
            self.assertEqual(card["experiment_count"], 0)
            self.assertIsNone(card["baseline_metrics"])
            self.assertTrue(card["has_memory_candidate"])

    def test_over_time_run_is_not_publishable(self):
        with tempfile.TemporaryDirectory() as directory:
            run = execute("win_bias_mean", runs_root=Path(directory), seconds_limit=10,
                          runner=lambda name, limit: _task_run(
                              "target_met", decision="adopt", wallclock_s=50,
                              hint="winnable", skip=""))
            self.assertEqual(run["status"], "budget_stopped")
            self.assertNotIn("memory_candidate", run)
            card = summary(run)
            self.assertFalse(card["has_memory_candidate"])

    def test_unknown_operator_task_is_rejected(self):
        with self.assertRaises(ValueError):
            execute("heldout_gelu_pool_bias")

    def test_http_starts_operator_run_without_loading_retrieval(self):
        started, release = threading.Event(), threading.Event()
        original = operator_service.complete

        def fake(run, run_dir, runner=None):
            started.set()
            self.assertTrue(release.wait(3))
            return original(run, run_dir, runner=lambda name, limit: _task_run(
                "blocked_by_transcendental"))

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            with patch("researchloop.scenarios.operator.service.complete", fake):
                first = _post(host, port, {"task": "lose_gelu_pool", "seconds_limit": 30})
                self.assertEqual(first["domain"], "operator")
                self.assertEqual(first["status"], "running")
                self.assertTrue(started.wait(3))
                self.assertEqual(SERVICE.active, first["id"])
                with self.assertRaises(urllib.error.HTTPError) as busy:
                    _post(host, port, {"task": "win_bias_mean", "seconds_limit": 30})
                self.assertEqual(busy.exception.code, 400)
                release.set()
                for _ in range(50):
                    if SERVICE.active is None:
                        break
                    threading.Event().wait(0.05)
                self.assertIsNone(SERVICE.active)
        finally:
            server.shutdown()
            server.server_close()
            folder = Path("runs") / first["id"] if "first" in locals() else None
            if folder and folder.exists():
                for child in folder.glob("*"):
                    child.unlink()
                folder.rmdir()

    def test_unknown_domain_summary_does_not_compute_retired_retrieval_metrics(self):
        card = summary({"id": "r", "status": "completed", "task": "old"})
        self.assertEqual(card["domain"], "unknown")
        self.assertEqual(card["id"], "r")
        self.assertNotIn("baseline_metrics", card)


def _post(host, port, payload):
    request = urllib.request.Request(
        f"http://{host}:{port}/api/runs",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
