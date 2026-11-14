"""One user-facing memory store: facts/preferences and versioned method lessons."""
import json
from . import db, facts, storage


def state():
    from .engine import core
    db.facts_expired()
    rows = db.facts(scope="research", status=None)
    for row in rows:
        row["tags"] = json.loads(row.get("tags_json") or "[]")
    return {"facts": rows, "methods": core.memory_state(),
            "recall": "新研究自动按问题召回启用的相关事实、偏好和方法；记忆不作为研究结论的证据。"}


def recall(query):
    from .engine import core
    rows = [row for row in facts.match(query, scope="research", limit=8) if row["predicate"] != "method_lesson"][:4]
    methods = core.memory_state()
    active = next((v for v in methods["versions"] if v["version"] == methods["active"]), None)
    keys = facts.keywords(query)
    method = None
    if active:
        text = " ".join(str(active.get(key) or "") for key in ("question", "lesson", "limitations"))
        if any(key.casefold() in text.casefold() for key in keys):
            method = active
    # User preferences apply across research questions; task facts remain scoped by relevance.
    known = {row["id"] for row in rows}
    for row in db.facts(scope="research"):
        if row["predicate"] == "user_preference" and row["id"] not in known:
            row["tags"] = json.loads(row.get("tags_json") or "[]")
            rows.append(row)
    rows = rows[:6]
    return {"query": query, "recalled_at": storage.now(), "facts": rows, "method": method,
            "text": facts.format_context(rows), "note": "方法和事实是待核验上下文，不能替代本轮证据或复核。"}


def add(body):
    kind = body.get("kind", "fact")
    if kind not in ("fact", "preference"):
        raise ValueError("记忆类型只能是事实或偏好；研究方法需从有证据的会话发布")
    text = str(body.get("text") or "").strip()
    if not 2 <= len(text) <= 2000:
        raise ValueError("记忆内容需要 2 至 2000 字符")
    tags = [value.strip() for value in str(body.get("tags") or "").split(",") if value.strip()][:8]
    return facts.remember(None, "user_preference" if kind == "preference" else "user_note", text, text,
                          tags=tags, confidence=1, source_message="用户在记忆页面明确添加")


def toggle(identifier, enabled):
    row = next((item for item in db.facts(scope="research", status=None) if item["id"] == int(identifier)), None)
    if not row:
        raise ValueError("记忆不存在")
    if row["status"] not in ("active", "disabled"):
        raise ValueError("冲突或过期的记忆不能直接重新启用，请核实后添加新记录")
    db._write("UPDATE facts SET status=? WHERE id=?", ("active" if enabled else "disabled", row["id"]))
    return {"id": row["id"], "enabled": bool(enabled)}
