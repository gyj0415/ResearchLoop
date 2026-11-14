import copy
import json
import unittest

from researchloop.dataset import Dataset, evidence_groups, load_lines
from researchloop.diagnostics import diagnose, query_profile, screen_accepts, select_memory
from researchloop.policy import BASELINE, FEATURES, config_changes, gate_matches, normalize_config, normalize_gate, signature
from researchloop.retrieval import Retriever, evaluate, subset_result, tokens
from researchloop.runner import ResearchService, behavior_key, frontier


class PolicyTests(unittest.TestCase):
    def test_negation_and_numbers_are_an_explicit_opt_in(self):
        text = "Treatment does not improve survival without 0.5 mg over 1 day."
        legacy, preserved = tokens(text), tokens(text, preserve=True)
        self.assertNotIn("not", legacy)
        self.assertNotIn("without", legacy)
        self.assertNotIn("1", legacy)
        self.assertTrue({"not", "without", "0.5", "1"} <= set(preserved))

    def test_compound_gates_are_observable_and_canonical(self):
        gate = normalize_gate({"all": ["has_number", "has_negation"]})
        self.assertTrue(gate_matches(gate, {"has_number": True, "has_negation": True}))
        self.assertFalse(gate_matches(gate, {"has_number": True, "has_negation": False}))
        self.assertEqual(normalize_gate({"all": ["has_acronym", "unique_acronym"]}), "unique_acronym")
        self.assertEqual(normalize_gate({"any": ["has_acronym", "unique_acronym"]}), "has_acronym")
        with self.assertRaises(ValueError):
            normalize_gate({"all": ["gold_found", "has_number"]})
        with self.assertRaises(ValueError):
            normalize_gate({"python": "open('data')"})

    def test_acronym_all_and_has_acronym_are_semantically_identical(self):
        self.assertEqual(signature({"strategy": "acronym", "gate": "all"}), signature({"strategy": "acronym", "gate": "has_acronym"}))

    def test_compound_policy_still_requires_one_factor(self):
        parent = {"id": "e1", "config": normalize_config({"normalization": "preserve_constraints"})}
        decision = {"action": "experiment", "parent_id": "e1", "config": {**parent["config"], "gate": {"any": ["has_number", "has_negation"]}},
                    **{k: "experiment rationale" for k in ("hypothesis", "mechanism", "scope", "prediction", "falsify_if", "reason")}}
        ResearchService._validate(decision, [parent])
        self.assertEqual(list(config_changes(parent["config"], decision["config"])), ["gate"])
        decision["config"]["strategy"] = "prf"
        with self.assertRaises(ValueError):
            ResearchService._validate(decision, [parent])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [{"doc_id": 1, "title": "Effects", "abstract": ["Treatment does not improve survival without 0.5 mg.", "Control improves survival."]},
                       {"doc_id": 2, "title": "Inflammation", "abstract": ["Tumor necrosis factor (TNF) regulates inflammation."]}]
        self.r = Retriever(self.corpus)
        self.claims = [{"id": 1, "claim": "Treatment does not improve survival without 0.5 mg.", "evidence": {"1": [{"sentences": [0]}]}},
                       {"id": 2, "claim": "TNF regulates inflammation.", "evidence": {"2": [{"sentences": [0]}]}}]

    def test_normalization_only_triggers_on_observable_constraints(self):
        cfg = {"normalization": "preserve_constraints"}
        self.assertEqual(self.r.search(self.claims[0]["claim"], cfg)["normalization_applied"], "preserve_constraints")
        self.assertEqual(self.r.search("TNF regulates inflammation", cfg)["normalization_applied"], "legacy")
        self.assertTrue(all(h["id"] in self.r.by_id for h in self.r.search(self.claims[0]["claim"], cfg)["hits"]))

    def test_inactive_and_feature_equivalent_candidates_are_removed(self):
        plain = [{"claim": "Control improves survival"}]
        f = [self.r.features(c["claim"]) for c in plain]
        self.assertEqual(behavior_key(BASELINE, f), behavior_key({"normalization": "preserve_constraints"}, f))
        self.assertEqual(behavior_key(BASELINE, f), behavior_key({"strategy": "acronym"}, f))
        stats = {}
        proposals = frontier([{"id": "e0", "config": BASELINE}], f, stats)
        self.assertGreater(stats["equivalent_or_inactive_removed"], 0)
        self.assertTrue(all(p["config"]["strategy"] == "prf" for p in proposals))

    def test_same_run_screen_rows_are_reused_with_identical_rankings(self):
        screen = evaluate(self.r, self.claims[:1], BASELINE)
        reused = evaluate(self.r, self.claims, BASELINE, previous=screen)
        plain = evaluate(self.r, self.claims, BASELINE)
        self.assertEqual(reused["reused_rows"], 1)
        self.assertEqual(reused["computed_rows"], 1)
        self.assertEqual(reused["metrics"], plain["metrics"])
        self.assertEqual(reused["rows"], plain["rows"])

    def test_cache_is_invalidated_by_config_corpus_or_gold_change(self):
        old = evaluate(self.r, self.claims[:1], BASELINE)
        changed = copy.deepcopy(self.claims[:1]); changed[0]["evidence"] = {"1": [{"sentences": [1]}]}
        self.assertEqual(evaluate(self.r, changed, BASELINE, previous=old)["reused_rows"], 0)
        self.assertEqual(evaluate(self.r, self.claims[:1], {"normalization": "preserve_constraints"}, previous=old)["reused_rows"], 0)
        other = Retriever(self.corpus + [{"doc_id": 9, "title": "Other", "abstract": ["Unrelated data."]}])
        self.assertEqual(evaluate(other, self.claims[:1], BASELINE, previous=old)["reused_rows"], 0)

    def test_screen_metrics_are_explicit_and_regression_is_rejected(self):
        result = evaluate(self.r, self.claims, BASELINE)
        subset = subset_result(result, ["1"])
        self.assertEqual(subset["count"], 1)
        worse = copy.deepcopy(subset)
        worse["metrics"]["complete_at_5"] = -1
        self.assertFalse(screen_accepts(subset, worse))
        self.assertTrue(screen_accepts(subset, subset))

    def test_diagnostic_failure_groups_have_operational_definitions(self):
        result = evaluate(self.r, self.claims, BASELINE)
        result["rows"][0]["failure_type"] = "ranking"
        result["rows"][0]["complete_at_5"] = 0
        d = diagnose(result, self.r)
        self.assertEqual(d["failure_counts"]["ranking"], 1)
        self.assertEqual(d["feature_groups"]["has_negation"]["count"], 1)
        self.assertEqual(d["examples"][0]["failure_type"], "ranking")

    def test_memory_rejects_legacy_and_does_not_need_target_gold(self):
        queries = [{"claim": "Treatment does not improve without 1 mg"}]
        profile = query_profile(self.r, queries)
        loaded, reason = select_memory({"version": 2}, profile, self.r, queries)
        self.assertIsNone(loaded)
        self.assertFalse(reason["applied"])
        memory = {"version": 3, "schema_version": 3, "applicability": {"feature_rates": profile["feature_rates"], "max_mean_distance": .2},
                  "best_config": normalize_config({"normalization": "preserve_constraints"}),
                  "priority": {"strategy": "baseline", "gate": "all", "normalization": "preserve_constraints"},
                  "guard_delta": {"complete_at_5": 0}, "source_run": "source-1", "source_profile": profile,
                  "lesson": "x" * 20000, "measured_effects": []}
        loaded, reason = select_memory(memory, profile, self.r, queries)
        self.assertTrue(reason["applied"])
        self.assertNotIn("lesson", loaded)
        self.assertLess(len(json.dumps(loaded)), 1500)
        dissimilar = copy.deepcopy(profile)
        dissimilar["feature_rates"] = {k: 1 - v for k, v in profile["feature_rates"].items()}
        self.assertIsNone(select_memory(memory, dissimilar, self.r, queries)[0])

    def test_memory_rejects_strategy_that_never_triggers(self):
        queries = [{"claim": "Control improves survival"}]
        profile = query_profile(self.r, queries)
        m = {"version": 3, "schema_version": 3, "applicability": {"feature_rates": profile["feature_rates"]}, "best_config": normalize_config({"strategy": "acronym"})}
        self.assertIsNone(select_memory(m, profile, self.r, queries)[0])


class FrozenTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset()

    def test_new_tasks_are_train_only_and_screens_are_subsets(self):
        train_ids = {str(c["id"]) for c in load_lines("claims_train.jsonl")}
        for task in self.dataset.registry["tasks"].values():
            self.assertEqual((len(task["explore"]), len(task["guard"]), len(task["screen"])), (24, 12, 8))
            self.assertTrue(set(task["screen"]) <= set(task["explore"]))
            self.assertTrue((set(task["explore"]) | set(task["guard"])) <= train_ids)

    def test_all_partitions_exclude_old_and_final_evidence_documents(self):
        d = self.dataset
        parts = [d.manifest["splits"]["test"]]
        for name in ("A", "B", "C", "D", "E", "F"):
            parts.extend([d.tasks[name]["explore"], d.tasks[name]["guard"]])
        doc_sets = [{doc for cid in ids for doc in d.claims[cid]["evidence"]} for ids in parts]
        for i, left in enumerate(doc_sets):
            for right in doc_sets[i + 1:]:
                self.assertFalse(left & right)


if __name__ == "__main__":
    unittest.main()
