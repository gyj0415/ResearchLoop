import json
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from researchloop.dataset import evidence_groups, make_splits
from researchloop.ollama import Budget, BudgetError, Ollama
from researchloop.retrieval import BASELINE, Retriever, evaluate, guard_accepts, normalize_config, score_claim
from researchloop.runner import ResearchService, frontier
from researchloop import storage, runner


class EvidenceTests(unittest.TestCase):
    def test_alternative_rationale_is_sufficient(self):
        claim = {"id": 1, "evidence": {"10": [{"sentences": [1, 2]}, {"sentences": [4]}]}}
        score = score_claim(claim, {"hits": [{"id": "10:4"}]})
        self.assertEqual(score["complete_at_5"], 1)
        self.assertEqual(score["coverage_at_5"], 1)

    def test_partial_multi_sentence_evidence_does_not_count_complete(self):
        claim = {"evidence": {"10": [{"sentences": [1, 2]}]}}
        score = score_claim(claim, {"hits": [{"id": "10:1"}, {"id": "11:2"}]})
        self.assertEqual(score["complete_at_5"], 0)
        self.assertEqual(score["coverage_at_5"], .5)

    def test_cited_documents_are_not_gold(self):
        self.assertEqual(evidence_groups({"evidence": {}, "cited_doc_ids": [12]}), [])

    def test_guard_rejects_regression_in_either_required_metric(self):
        baseline = {"complete_at_5": .5, "coverage_at_5": .7}
        self.assertFalse(guard_accepts(baseline, {"complete_at_5": .6, "coverage_at_5": .6}))
        self.assertFalse(guard_accepts(baseline, {"complete_at_5": .4, "coverage_at_5": .8}))
        self.assertTrue(guard_accepts(baseline, {"complete_at_5": .5, "coverage_at_5": .7}))

    def test_split_components_cannot_leak(self):
        def row(i):
            return {"id": i, "claim": f"claim {i}", "evidence": {str(i // 2): [{"sentences": [0]}]}}
        train, dev = [row(i) for i in range(120)], [row(i) for i in range(120, 160)]
        # Deliberately share one evidence document across official train/dev.
        dev[0]["evidence"] = train[0]["evidence"]
        splits = make_splits(train, dev, target=18, test_size=10)
        rows = {str(r["id"]): r for r in train + dev}
        parts = [splits["A"]["explore"], splits["A"]["guard"], splits["B"]["explore"], splits["B"]["guard"], splits["test"]]
        documents = [{d for cid in part for d in rows[cid]["evidence"]} for part in parts]
        for i, left in enumerate(documents):
            for right in documents[i+1:]:
                self.assertFalse(left & right)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever([
            {"doc_id": 1, "title": "Inflammation", "abstract": ["Tumor necrosis factor (TNF) regulates inflammation.", "Protein signaling controls cell activity."]},
            {"doc_id": 2, "title": "Cells", "abstract": ["Unrelated proteins regulate cell division."]}])

    def test_expansions_are_grounded_and_sentence_ids_are_stable(self):
        baseline = self.retriever.search("TNF inflammation", BASELINE)
        expanded = self.retriever.search("TNF inflammation", {"strategy": "acronym"})
        self.assertTrue(expanded["expansion"])
        self.assertTrue(all(word in self.retriever.postings for word in expanded["expansion"]))
        self.assertEqual(len({h["id"] for h in expanded["hits"]}), len(expanded["hits"]))
        self.assertEqual(baseline["hits"][0]["id"], "1:0")

    def test_single_factor_and_duplicate_checks(self):
        experiments = [{"id": "e0", "config": dict(BASELINE)}]
        decision = {"action": "experiment", "parent_id": "e0", "config": {"strategy": "prf"},
                    **{k: "test" for k in ("hypothesis", "mechanism", "scope", "prediction", "falsify_if", "reason")}}
        ResearchService._validate(decision, experiments)
        decision["config"] = {"strategy": "prf", "gate": "short_query"}
        with self.assertRaises(ValueError):
            ResearchService._validate(decision, experiments)
        decision["config"] = dict(BASELINE)
        with self.assertRaises(ValueError):
            ResearchService._validate(decision, experiments)

    def test_frontier_is_unique_and_one_factor(self):
        history = [{"id": "e0", "config": dict(BASELINE)}, {"id": "e1", "config": normalize_config({"strategy": "prf"})}]
        proposals = frontier(history)
        self.assertEqual(len(proposals), len({json.dumps(p["config"], sort_keys=True) for p in proposals}))
        for proposal in proposals:
            parent = next(h for h in history if h["id"] == proposal["parent_id"])
            self.assertEqual(sum(proposal["config"][k] != parent["config"][k] for k in BASELINE), 1)


class BudgetTests(unittest.TestCase):
    def test_preflight_prevents_call(self):
        budget = Budget(10, 30)
        with self.assertRaises(BudgetError):
            budget.reserve(11)

    def test_all_purposes_and_unknown_usage_consume_budget(self):
        budget = Budget(100, 30)
        budget.record({"purpose": "planner", "input_tokens": 10, "output_tokens": 5})
        budget.record({"purpose": "retry", "input_tokens": 10, "output_tokens": 5})
        budget.record({"purpose": "reflection", "unknown_reserved": 20})
        self.assertEqual(budget.used, 50)
        self.assertEqual(budget.snapshot()["model_calls"], 3)

    def test_native_usage_and_invalid_json_are_still_charged(self):
        class Fake(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert body["think"] is False
                result = {"message": {"content": "invalid-json"}, "prompt_eval_count": 10, "eval_count": 5}
                data = json.dumps(result).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            budget = Budget(10000, 30)
            calls = []
            client = Ollama(f"http://127.0.0.1:{server.server_port}", "test-local", budget, calls.append)
            with self.assertRaises(ValueError):
                client.call("test", {}, "planner", 42)
            self.assertEqual(budget.used, 15)
            self.assertEqual(len(calls), 1)
        finally:
            server.shutdown()
            server.server_close()

    def test_cloud_model_is_rejected(self):
        with self.assertRaises(ValueError):
            Ollama("http://127.0.0.1:11434", "model-cloud", Budget(10000, 30), lambda _: None)


class MemoryTests(unittest.TestCase):
    def test_windows_transient_replace_lock_is_retried(self):
        with tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache") as directory:
            target = Path(directory) / "state.json"
            replace = storage.os.replace
            attempts = []
            def locked_once(source, destination):
                attempts.append(1)
                if len(attempts) < 3:
                    raise PermissionError("simulated Windows sharing lock")
                return replace(source, destination)
            with patch.object(storage.os, "replace", side_effect=locked_once):
                storage.write_json(target, {"saved": True})
            self.assertEqual(storage.read_json(target), {"saved": True})
            self.assertEqual(len(attempts), 3)

    def test_publish_gate_and_rollback_preserve_history(self):
        with tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache") as directory:
            base = Path(directory)
            with patch.object(storage, "RUNS", base / "runs"), patch.object(storage, "MEMORY", base / "memory"):
                run = {"id": "sample-1", "status": "completed", "mode": "agent", "memory_version": 0,
                       "guard": {"accepted": False, "delta": {"complete_at_5": 0, "coverage_at_5": 0}, "candidate": {"count": 5}},
                       "memory_candidate": {"lesson": "test"}, "best_config": BASELINE,
                       "selected_experiment_id": "e0", "experiments": [{"id": "e0", "config": BASELINE,
                       "result": {"count": 10, "metrics": {"complete_at_5": .5, "coverage_at_5": .5}}}]}
                storage.write_json(storage.RUNS / "sample-1" / "run.json", run)
                with self.assertRaises(ValueError):
                    storage.promote("sample-1")
                run["guard"]["accepted"] = True
                storage.write_json(storage.RUNS / "sample-1" / "run.json", run)
                version = storage.promote("sample-1")
                self.assertEqual(version["version"], 1)
                storage.rollback(0)
                self.assertEqual(storage.memory_state()["active"], 0)
                self.assertTrue((storage.MEMORY / "v1.json").exists())
                self.assertEqual(storage.promote("sample-1")["version"], 1)
                run["mode"] = "search"
                storage.write_json(storage.RUNS / "sample-1" / "run.json", run)
                with self.assertRaises(ValueError):
                    storage.promote("sample-1")


class ServiceTests(unittest.TestCase):
    def fixture(self):
        corpus = [{"doc_id": 1, "title": "Cell response", "abstract": ["Tumor necrosis factor (TNF) regulates inflammation."]}]
        claims = {str(i): {"id": i, "claim": "TNF regulates inflammation", "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]}} for i in range(10)}
        manifest = {"archive_sha256": "unit-test-fixture", "splits": {"A": {"explore": list(claims)[:5], "guard": list(claims)[5:8]}, "test": list(claims)[8:]}}
        dataset = SimpleNamespace(corpus=corpus, claims=claims, manifest=manifest)
        dataset.task = lambda task, part: [claims[cid] for cid in manifest["splits"][task][part]]
        service = ResearchService()
        service.dataset, service.retriever = dataset, Retriever(corpus)
        return service

    def test_complete_search_run_has_guard_and_no_agent_memory(self):
        with tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache") as directory:
            target = Path(directory) / "runs"
            with patch.object(storage, "RUNS", target), patch.object(runner, "RUNS", target):
                service = self.fixture()
                run = service.start({"mode": "search", "task": "A", "max_experiments": 2})
                deadline = time.monotonic() + 5
                while service.active and time.monotonic() < deadline:
                    time.sleep(.01)
                result = storage.read_json(target / run["id"] / "run.json")
                self.assertEqual(result["status"], "completed")
                self.assertEqual(len(result["experiments"]), 3)
                self.assertIn("guard", result)
                self.assertEqual(result["budget"]["model_calls"], 0)
                self.assertNotIn("memory_candidate", result)
                self.assertTrue((target / run["id"] / "source" / "researchloop" / "runner.py").exists())

    def test_final_test_is_one_time_and_freezes_only_completed_runs(self):
        with tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache") as directory:
            with patch.object(runner, "ROOT", Path(directory)), patch.object(runner, "list_runs", return_value=[{"id": "frozen", "status": "completed", "best_config": BASELINE, "source_hash": storage.source_hash()}]):
                service = self.fixture()
                first = service.finalize()
                self.assertEqual(first["frozen_run_ids"], ["frozen"])
                self.assertEqual(first["test_count"], 2)
                service.retriever = None
                second = service.finalize()
                self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
