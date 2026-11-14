"""Per-request routing; roles remain independent of model capacity tiers."""
from researchloop import storage


def clean(value):
    value = value if isinstance(value, dict) else {}
    result = {"enabled": bool(value.get("enabled")), "profiles": {}}
    for tier in ("light", "strong"):
        conf = (value.get("profiles") or {}).get(tier) or {}
        source, model = conf.get("model_source", "local"), str(conf.get("model") or "").strip()
        if source not in ("local", "cloud") or len(model) > 160:
            raise ValueError("路由模型来源或名称无效")
        result["profiles"][tier] = {"model_source": source, "model": model}
    return result


def choose(engine, purpose, prompt):
    model, source, role, agent = engine.active_identity()
    policy = engine.run.get("routing") or {}
    tier, reason = "default", "沿用当前角色或会话模型"
    if policy.get("enabled"):
        failed = bool(engine.run.get("decision_feedback")) or bool(
            engine.run.get("steps") and engine.run["steps"][-1].get("status") == "failed")
        complex_work = purpose.endswith("retry") or failed or role in ("experimenter", "verifier", "reporter")
        complex_work |= bool(engine.run.get("plan") and not engine.run.get("hypotheses"))
        tier = "strong" if complex_work else "light"
        reason = "修复失败、设计实验或核对结论" if complex_work else "材料检查、检索与常规调度"
        if engine.budget.token_limit - engine.budget.used < 16000 and not failed:
            tier, reason = "light", "剩余 token 较少，优先轻量档；仍受调用预留约束"
        conf = (policy.get("profiles") or {}).get(tier) or {}
        if conf.get("model"):
            candidate_source = conf.get("model_source", "local")
            if candidate_source != "cloud" or storage.settings()["cloud"].get("enabled"):
                model, source = conf["model"], candidate_source
            else:
                reason += "；云端未启用，沿用当前模型"
                tier = "default"
        else:
            tier, reason = "default", reason + "；该档未配置，沿用当前模型"
    return {"model": model, "model_source": source, "tier": tier, "reason": reason,
            "policy": "rules-v1", "role": role, "agent": agent}
