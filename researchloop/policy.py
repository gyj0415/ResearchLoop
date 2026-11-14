"""Validated retrieval policy DSL. No generated Python is executed."""
import hashlib
import json

FEATURES = ("has_acronym", "unique_acronym", "ambiguous_acronym", "short_query", "has_negation", "has_number", "rare_terms")
BASELINE = {"strategy": "baseline", "gate": "all", "prf_terms": 3, "normalization": "legacy"}
GATE_SCHEMA = {"anyOf": [
    {"type": "string", "enum": ["all", *FEATURES]},
    {"type": "object", "additionalProperties": False, "properties": {"all": {"type": "array", "items": {"type": "string", "enum": list(FEATURES)}, "minItems": 2, "maxItems": 3}}, "required": ["all"]},
    {"type": "object", "additionalProperties": False, "properties": {"any": {"type": "array", "items": {"type": "string", "enum": list(FEATURES)}, "minItems": 2, "maxItems": 3}}, "required": ["any"]}]}
CONFIG_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "strategy": {"type": "string", "enum": ["baseline", "acronym", "prf"]},
    "gate": GATE_SCHEMA, "prf_terms": {"type": "integer", "enum": [1, 3, 5]},
    "normalization": {"type": "string", "enum": ["legacy", "preserve_constraints"]}}, "required": list(BASELINE)}


def normalize_gate(value):
    if isinstance(value, str) and value in ("all", *FEATURES):
        return value
    if not isinstance(value, dict) or len(value) != 1 or next(iter(value)) not in ("all", "any"):
        raise ValueError("门控只能是可观测特征，或至多三个特征的 all/any 组合")
    op, items = next(iter(value.items()))
    if not isinstance(items, list) or not 1 <= len(items) <= 3 or any(not isinstance(i, str) or i not in FEATURES for i in items):
        raise ValueError("组合条件包含无效特征")
    items = sorted(set(items))
    if "unique_acronym" in items and "has_acronym" in items:
        items.remove("has_acronym" if op == "all" else "unique_acronym")
    return items[0] if len(items) == 1 else {op: items}


def normalize_config(value):
    if not isinstance(value, dict) or set(value) - set(BASELINE):
        raise ValueError("配置必须为允许字段构成的 JSON 对象")
    cfg = {**BASELINE, **value}
    if cfg["strategy"] not in ("baseline", "acronym", "prf"):
        raise ValueError("不支持的检索策略")
    if cfg["normalization"] not in ("legacy", "preserve_constraints"):
        raise ValueError("不支持的分词策略")
    cfg["gate"] = normalize_gate(cfg["gate"])
    if type(cfg["prf_terms"]) is not int or cfg["prf_terms"] not in (1, 3, 5):
        raise ValueError("扩展词数只能为 1、3、5")
    if cfg["strategy"] != "prf":
        cfg["prf_terms"] = 3
    if cfg["strategy"] == "baseline" and cfg["normalization"] == "legacy":
        cfg["gate"] = "all"
    return cfg


def gate_matches(gate, features):
    if isinstance(gate, str):
        return gate == "all" or bool(features.get(gate))
    op, items = next(iter(gate.items()))
    return (all if op == "all" else any)(bool(features.get(i)) for i in items)


def signature(config):
    cfg = normalize_config(config)
    if cfg["strategy"] == "acronym" and cfg["normalization"] == "legacy" and cfg["gate"] == "has_acronym":
        cfg["gate"] = "all"
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False)


def config_changes(left, right):
    left, right = normalize_config(left), normalize_config(right)
    return {k: {"from": left[k], "to": right[k]} for k in BASELINE if left[k] != right[k]}


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
