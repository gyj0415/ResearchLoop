"""Keyword-scored fact-memory recall, mirroring the reference project.

Scoring weights follow the ReEscape Protocol design: summary 5, object 4,
predicate 3, tag 2, with a base of 1 so that an unscored fact is still usable
when the query has no extractable keyword.
"""
import json
import re

from . import db

PUNCTUATION = "，。！？,.:：；、\n\t"


def keywords(query):
    text = str(query or "").strip()
    if not text:
        return []
    normalized = text
    for character in PUNCTUATION:
        normalized = normalized.replace(character, " ")
    result, seen = [], set()

    def add(value):
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)

    add(text)
    for part in normalized.split():
        if len(part) >= 2:
            add(part)
    ignored = {"研究", "分析", "实验", "数据", "是否", "如何", "什么", "这个", "以及", "进行", "当前", "我们"}
    for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        for index in range(min(len(phrase) - 1, 48)):
            word = phrase[index:index + 2]
            if word not in ignored:
                add(word)
    return result


def score(fact, keys):
    total = 1
    for key in keys:
        if not key:
            continue
        if key in (fact.get("summary") or ""):
            total += 5
        if key in (fact.get("object") or ""):
            total += 4
        if key in (fact.get("predicate") or ""):
            total += 3
        for tag in fact.get("tags") or []:
            if key in tag or tag in key:
                total += 2
    return total


def match(query, scope=None, limit=4):
    db.facts_expired()
    rows = db.facts(scope=scope)
    keys = keywords(query)
    scored = []
    for row in rows:
        row["tags"] = json.loads(row.get("tags_json") or "[]")
        value = score(row, keys)
        if keys and value <= 1:
            continue
        scored.append((value, row))
    scored.sort(key=lambda item: (item[0], item[1].get("last_confirmed_at") or ""), reverse=True)
    return [row for _, row in scored[:max(1, int(limit or 4))]]


def format_context(rows):
    if not rows:
        return ""
    lines = [f"- [{row['predicate']}] {row['summary']}（置信度 {float(row.get('confidence') or 0):.2f}）"
             for row in rows]
    return "【事实记忆】\n" + "\n".join(lines)


def remember(session_id, predicate, object, summary, tags=None, confidence=0.6,
             source_message="", scope="research", expires_at=None):
    return db.fact_upsert(scope=scope, session_id=session_id, predicate=predicate, object=object,
                          summary=summary, tags=tags, confidence=confidence,
                          source_message=source_message, expires_at=expires_at)
