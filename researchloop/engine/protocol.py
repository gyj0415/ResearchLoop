"""Shared JSON schema validation for admitted tools (no model-generated defaults)."""
import math


def validate(value, schema, path="args"):
    kind = schema.get("type")
    checks = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
              "string": lambda x: isinstance(x, str), "boolean": lambda x: isinstance(x, bool),
              "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
              "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)}
    if kind in checks and not checks[kind](value):
        raise ValueError(f"{path} 必须是 {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} 必须在 {schema['enum']} 中")
    if kind == "object":
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise ValueError(f"{path} 缺少必需字段 {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(f"{path} 含未知字段 {sorted(set(value) - set(properties))}")
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], f"{path}.{key}")
    if kind == "array":
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError(f"{path} 至少需要 {schema['minItems']} 项")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path} 最多允许 {schema['maxItems']} 项；没有证据时请使用空数组")
        for index, item in enumerate(value):
            validate(item, schema.get("items", {}), f"{path}[{index}]")


def action_schema(catalog):
    # Bind the selected tool to its actual argument schema, including nested stages.
    return {"oneOf": [{"type": "object", "properties": {
        "tool": {"type": "string", "enum": [entry["tool"]]}, "reason": {"type": "string"},
        "args": entry["args_schema"]}, "required": ["tool", "reason", "args"],
        "additionalProperties": False} for entry in catalog]}
