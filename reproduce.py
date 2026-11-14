"""Recompute logged numeric experiments, without invoking a model."""
import argparse
import json
from pathlib import Path

from researchloop.dataset import Dataset
from researchloop.retrieval import BASELINE, Retriever, evaluate
from researchloop.storage import source_hash


def reproduce(path):
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset = Dataset()
    if dataset.manifest["archive_sha256"] != run["dataset_sha256"]:
        raise ValueError("数据归档哈希与运行记录不一致")
    retriever = Retriever(dataset.corpus)
    explore = [dataset.claims[cid] for cid in run["split_ids"]["explore"]]
    checks = []

    def check(label, claims, config, saved):
        actual = evaluate(retriever, claims, config)
        metric_match = all(abs(actual["metrics"][key] - expected) < 1e-12 for key, expected in saved["metrics"].items())
        hit_match = [[h["id"] for h in r["hits"]] for r in actual["rows"]] == [[h["id"] for h in r["hits"]] for r in saved["rows"]]
        checks.append({"experiment": label, "metrics_match": metric_match, "rankings_match": hit_match,
                       "metrics": actual["metrics"]})

    for experiment in run["experiments"]:
        claims = [c for c in explore if str(c["id"]) in run.get("screen_ids", [])] if experiment.get("result_scope") == "screen" else explore
        check(experiment["id"], claims, experiment["config"], experiment["result"])
        if experiment.get("screen_result"):
            screen = [c for c in explore if str(c["id"]) in run["screen_ids"]]
            check(experiment["id"] + "_screen", screen, experiment["config"], experiment["screen_result"])
    if run.get("guard"):
        guard = [dataset.claims[cid] for cid in run["split_ids"]["guard"]]
        check("guard_baseline", guard, BASELINE, run["guard"]["baseline"])
        check("guard_candidate", guard, run["frozen_candidate"]["config"], run["guard"]["candidate"])
    return {"run_id": run["id"], "source_hash_matches_current": source_hash() == run["source_hash"],
            "all_numeric_checks_passed": all(c["metrics_match"] and c["rankings_match"] for c in checks),
            "model_calls": 0, "checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_json", help="Path to a saved run.json")
    args = parser.parse_args()
    report = reproduce(args.run_json)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["all_numeric_checks_passed"] else 1)
