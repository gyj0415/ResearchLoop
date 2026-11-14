"""Gold-aware development diagnostics; memory applicability is query-only."""
from collections import Counter

from .policy import FEATURES, gate_matches, normalize_config


def query_profile(retriever, claims):
    values = [retriever.features(c["claim"]) for c in claims]
    return {"count": len(values), "feature_rates": {
        feature: round(sum(bool(f[feature]) for f in values) / max(1, len(values)), 4) for feature in FEATURES}}


def diagnose(result, retriever, max_examples=3):
    rows = result["rows"]
    counts = Counter(r.get("failure_type", "unknown") for r in rows)
    groups = {}
    for feature in FEATURES:
        subset = [r for r in rows if r["features"].get(feature)]
        groups[feature] = {"count": len(subset), "failed": sum(not r["complete_at_5"] for r in subset),
                           "complete_at_5": sum(r["complete_at_5"] for r in subset) / len(subset) if subset else None}
    examples, used = [], set()
    for kind in ("recall", "ranking", "partial_recall"):
        candidates = [r for r in rows if r.get("failure_type") == kind]
        candidates.sort(key=lambda r: (-int(r["features"].get("has_negation", False)) - int(r["features"].get("has_number", False)), r["claim_id"]))
        if not candidates:
            continue
        row = candidates[0]
        used.add(row["claim_id"])
        gold = [retriever.by_id[s]["text"][:180] for group in row["gold_sets"] for s in group if s in retriever.by_id]
        examples.append({"claim_id": row["claim_id"], "query": row["claim"], "failure_type": kind,
                         "features": row["features"], "retrieved": [h["text"][:140] for h in row["hits"][:2]],
                         "gold_explore_only": gold[:2]})
        if len(examples) >= max_examples:
            break
    return {"count": len(rows), "failure_counts": dict(counts), "feature_groups": groups,
            "examples": examples,
            "definitions": {"recall": "没有标注证据句进入初始查询 BM25 前50", "partial_recall": "仅部分标注证据进入初始查询前50", "ranking": "完整证据集合已在初始查询前50，但最终前5不完整"},
            "scope": "Development diagnostics only. Initial top50 is not the final RRF candidate pool. Gold is never used for runtime gating."}


def screen_accepts(parent, candidate):
    return all(candidate["metrics"][k] >= parent["metrics"][k] - 1e-12 for k in ("complete_at_5", "coverage_at_5"))


def select_memory(memory, profile, retriever, claims):
    if memory is None:
        return None, {"applied": False, "requested_version": 0, "reason": "未请求历史经验"}
    decision = {"applied": False, "requested_version": memory["version"], "profile": profile}
    source = memory.get("applicability", {}).get("feature_rates")
    if memory.get("schema_version", 1) < 3 or not source:
        return None, {**decision, "reason": "旧版经验缺少可观测特征依据；不注入 Agent，仅可作配置沿用对照"}
    distance = sum(abs(profile["feature_rates"].get(k, 0) - source.get(k, 0)) for k in FEATURES) / len(FEATURES)
    decision["profile_distance"] = round(distance, 4)
    if distance > memory["applicability"].get("max_mean_distance", .2):
        return None, {**decision, "reason": "问题特征分布差异超过经验的适用门槛"}
    config = normalize_config(memory["best_config"])
    triggered = sum(retriever.policy_triggers(c["claim"], config) for c in claims)
    if not triggered:
        return None, {**decision, "reason": "来源策略在目标问题上没有可触发的干预，不加载优先级"}
    effects = memory.get("measured_effects", [])
    compact = {"version": memory["version"], "status": "provisional", "priority": memory["priority"],
               "guard_delta": memory["guard_delta"], "source_count": memory.get("source_profile", {}).get("count"),
               "caution": "仅来源任务验收；新任务必须验证，不要盲从。", "source_run": memory["source_run"],
               "observed_effects": [{"strategy": e["strategy"], "gate": e["gate"],
                                      "delta_complete": round(e["explore_delta_from_baseline"]["complete_at_5"], 4)} for e in effects[:2]]}
    return compact, {**decision, "applied": True, "triggered_queries": triggered,
                     "reason": "通过查询特征与策略触发检查；仅加载短规则，迁移收益仍待验证"}
