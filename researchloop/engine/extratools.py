"""Web and fact-memory tools, ported from the ReEscape Protocol design.

Registered through the same registry as built-in tools; no dispatch chain edit
is needed to add a capability.
"""
from .. import facts, storage, webaccess
from . import registry


def _web_settings():
    return storage.settings()["web"]


def web_search(engine, args):
    config = _web_settings()
    if not config.get("enabled"):
        raise ValueError("联网工具未启用：请先在设置中启用并配置搜索服务")
    query = str(args.get("query") or "").strip()
    result = webaccess.search(query, provider=config.get("search_provider"),
                              endpoint=config.get("search_endpoint"), api_key=config.get("search_api_key"),
                              max_results=args.get("max_results") or config.get("max_results") or 5,
                              recency=args.get("recency") or "any")
    for row in result["results"]:
        row["id"] = f"p{len(engine.run['sources']) + 1}"
        row.setdefault("abstract", row.get("snippet", ""))
        row["evidence_scope"] = "网页搜索摘要，不是论文结论"
        engine.run["sources"].append(row)
    engine.run["evidence_collected"] = True
    return {"provider": result["provider"], "query": result["query"], "results": result["results"],
            "note": "搜索摘要只作线索；需要正文时用 web_fetch 读取具体 URL"}


def web_fetch(engine, args):
    config = _web_settings()
    if not config.get("enabled"):
        raise ValueError("联网工具未启用：请先在设置中启用并配置搜索服务")
    page = webaccess.fetch(args.get("url"), max_chars=args.get("max_chars") or config.get("fetch_max_chars") or 6000)
    entry = {"id": f"p{len(engine.run['sources']) + 1}", "title": page["title"], "url": page["url"],
             "abstract": page["content"][:1800], "evidence_scope": "网页正文",
             "fetched_at": page["fetched_at"]}
    engine.run["sources"].append(entry)
    engine.run["evidence_collected"] = True
    return {"id": entry["id"], "url": page["url"], "title": page["title"],
            "chars": len(page["content"]), "truncated": page["truncated"]}


def remember_fact(engine, args):
    fact = facts.remember(engine.run["id"], str(args.get("predicate") or ""), str(args.get("object") or ""),
                          str(args.get("summary") or ""), tags=args.get("tags"),
                          confidence=args.get("confidence") or 0.6,
                          source_message=str(args.get("source_message") or engine.run["question"]))
    return {"id": fact["id"], "predicate": fact["predicate"], "object": fact["object"],
            "summary": fact["summary"], "status": fact["status"]}


def get_memory_context(engine, args):
    query = str(args.get("query") or engine.run["question"]).strip()
    rows = facts.match(query, scope="research", limit=4)
    engine.run["memory_context"] = {
        "query": query,
        "facts": [{key: row.get(key) for key in
                   ("id", "predicate", "object", "summary", "confidence", "last_confirmed_at")} for row in rows],
        "text": facts.format_context(rows)}
    return engine.run["memory_context"]


registry.register_tool("web_search", web_search,
                       description="搜索外部网页信息，适合最新消息、陌生实体或需要来源核实时使用。",
                       arguments={"query": "搜索关键词", "reason": "为什么需要搜索",
                                  "max_results": 5, "recency": "any"}, required=["query", "reason"])
registry.register_tool("web_fetch", web_fetch,
                       description="读取指定 http/https URL 的网页正文，用于确认搜索结果的原文。",
                       arguments={"url": "要读取的 URL", "reason": "为什么需要读取", "max_chars": 6000},
                       required=["url", "reason"])
registry.register_tool("remember_fact", remember_fact,
                       description="写入或确认长期事实记忆；只写明确出现的事实，不写推测。",
                       arguments={"predicate": "事实谓词", "object": "事实对象", "summary": "简短摘要",
                                  "tags": ["标签"], "confidence": 0.6}, required=["predicate", "object", "summary"])
registry.register_tool("get_memory_context", get_memory_context,
                       description="按查询召回事实记忆，返回适用条件与证据来源。",
                       arguments={"query": "检索文本"}, required=[])
