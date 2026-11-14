"""Local SQLite ledger.

One research task is one implicit session. The ledger keeps the session row,
every model call (token, cache and context metrics), every decision step, the
memory store and the tool registry. JSON run folders stay the full raw record;
this database is the queryable index and the memory/tool source of truth.
"""
import json
import sqlite3
import threading

from . import storage

LOCK = threading.RLock()
_INITIALISED_PATH = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
    id TEXT PRIMARY KEY, created_at TEXT, finished_at TEXT, question TEXT,
    model TEXT, model_source TEXT, status TEXT,
    budget_json TEXT, metrics_json TEXT);
CREATE TABLE IF NOT EXISTS calls(
    session_id TEXT, idx INTEGER, at TEXT, purpose TEXT, model TEXT, source TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cached_tokens INTEGER,
    context_tokens INTEGER, wall_seconds REAL, tok_per_second REAL, cloud INTEGER,
    PRIMARY KEY(session_id, idx));
CREATE TABLE IF NOT EXISTS steps(
    session_id TEXT, step_id TEXT, at TEXT, tool TEXT, status TEXT, reason TEXT,
    args_json TEXT, observation_json TEXT, elapsed_seconds REAL, agent TEXT, role TEXT,
    PRIMARY KEY(session_id, step_id));
CREATE TABLE IF NOT EXISTS agents(
    session_id TEXT, agent_id TEXT, role TEXT, goal TEXT, model TEXT, model_source TEXT,
    status TEXT, steps_json TEXT, PRIMARY KEY(session_id, agent_id));
CREATE TABLE IF NOT EXISTS facts(
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, scope TEXT, predicate TEXT, object TEXT,
    summary TEXT, tags_json TEXT, confidence REAL, source_message TEXT, status TEXT,
    created_at TEXT, last_confirmed_at TEXT, expires_at TEXT);
CREATE INDEX IF NOT EXISTS facts_lookup ON facts(scope, status);
CREATE TABLE IF NOT EXISTS memory_items(
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, version INTEGER,
    scope TEXT, lesson TEXT, applicability_json TEXT, evidence_json TEXT,
    status TEXT, source_run TEXT, parent_version INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS tools(
    name TEXT PRIMARY KEY, description TEXT, schema_json TEXT, handler TEXT,
    source TEXT, enabled INTEGER, registered_at TEXT);
CREATE INDEX IF NOT EXISTS memory_scope ON memory_items(scope, status);
"""


def path():
    # Follows the runs root so tests and exports never write the product ledger.
    return storage.RUNS.parent / "data" / "researchloop.db"


def _connect():
    database = path()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def _ensure_column(connection, table, column, ddl):
    columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init():
    global _INITIALISED_PATH
    with LOCK:
        current = str(path())
        if _INITIALISED_PATH == current:
            return
        connection = _connect()
        try:
            connection.executescript(SCHEMA)
            _ensure_column(connection, "steps", "agent", "TEXT")
            _ensure_column(connection, "steps", "role", "TEXT")
            connection.commit()
        finally:
            connection.close()
        _INITIALISED_PATH = current


def _write(sql, params=()):
    init()
    with LOCK:
        connection = _connect()
        try:
            connection.execute(sql, params)
            connection.commit()
        finally:
            connection.close()


def _read(sql, params=()):
    init()
    with LOCK:
        connection = _connect()
        try:
            return [dict(row) for row in connection.execute(sql, params).fetchall()]
        finally:
            connection.close()


def session_metrics(run):
    calls = run.get("calls") or []
    output = sum(int(c.get("output_tokens") or 0) for c in calls)
    input_total = sum(int(c.get("input_tokens") or 0) for c in calls)
    cached = sum(int(c.get("cached_tokens") or 0) for c in calls)
    wall = sum(float(c.get("wall_seconds") or 0) for c in calls)
    context = max([int(c.get("context_tokens") or 0) for c in calls] or [0])
    return {"model_calls": len(calls), "input_tokens": input_total, "output_tokens": output,
            "cached_tokens": cached, "context_tokens": context,
            "wall_seconds": round(wall, 3),
            "tok_per_second": round(output / wall, 3) if wall else None,
            "cache_hit_rate": round(cached / input_total, 4) if input_total else None}


def sync_run(run):
    """Idempotent mirror of one run folder into the ledger."""
    session = run["id"]
    metrics = session_metrics(run)
    _write("""INSERT INTO sessions(id, created_at, finished_at, question, model, model_source, status, budget_json, metrics_json)
              VALUES(?,?,?,?,?,?,?,?,?)
              ON CONFLICT(id) DO UPDATE SET finished_at=excluded.finished_at, status=excluded.status,
              budget_json=excluded.budget_json, metrics_json=excluded.metrics_json""",
           (session, run.get("created_at"), run.get("finished_at"), run.get("question", ""),
            run.get("model"), run.get("model_source", "local"), run.get("status"),
            json.dumps(run.get("budget"), ensure_ascii=False), json.dumps(metrics, ensure_ascii=False)))
    for index, call in enumerate(run.get("calls") or []):
        _write("""INSERT OR REPLACE INTO calls(session_id, idx, at, purpose, model, source, input_tokens,
                  output_tokens, cached_tokens, context_tokens, wall_seconds, tok_per_second, cloud)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
               (session, index, run.get("updated_at") or run.get("created_at"), call.get("purpose"),
                call.get("model"), call.get("source", "ollama"), int(call.get("input_tokens") or 0),
                int(call.get("output_tokens") or 0), int(call.get("cached_tokens") or 0),
                int(call.get("context_tokens") or 0), float(call.get("wall_seconds") or 0),
                call.get("tok_per_second"), 1 if call.get("cloud") else 0))
    for step in run.get("steps") or []:
        _write("""INSERT OR REPLACE INTO steps(session_id, step_id, at, tool, status, reason, args_json,
                  observation_json, elapsed_seconds, agent, role) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
               (session, step.get("id"), step.get("at"), step.get("tool"), step.get("status"),
                step.get("reason"), json.dumps(step.get("args"), ensure_ascii=False),
                json.dumps(step.get("observation"), ensure_ascii=False),
                float(step.get("elapsed_seconds") or 0), step.get("agent"), step.get("role")))
    for agent in run.get("agents") or []:
        _write("""INSERT OR REPLACE INTO agents(session_id, agent_id, role, goal, model, model_source, status, steps_json)
                  VALUES(?,?,?,?,?,?,?,?)""",
               (session, agent.get("id"), agent.get("role"), agent.get("goal"), agent.get("model"),
                agent.get("model_source"), agent.get("status"),
                json.dumps(agent.get("steps") or [], ensure_ascii=False)))


def sessions(limit=50):
    return _read("SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (int(limit),))


def session(session_id):
    rows = _read("SELECT * FROM sessions WHERE id=?", (session_id,))
    return rows[0] if rows else None


def calls(session_id):
    return _read("SELECT * FROM calls WHERE session_id=? ORDER BY idx", (session_id,))


def steps(session_id):
    return _read("SELECT * FROM steps WHERE session_id=? ORDER BY step_id", (session_id,))


def memory_add(*, session_id=None, version=None, scope="research", lesson="",
               applicability=None, evidence=None, status="provisional",
               source_run=None, parent_version=None, created_at=None):
    with LOCK:
        init()
        connection = _connect()
        try:
            cursor = connection.execute("""INSERT INTO memory_items(session_id, version, scope, lesson,
                applicability_json, evidence_json, status, source_run, parent_version, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (session_id, version, scope, lesson, json.dumps(applicability, ensure_ascii=False),
                 json.dumps(evidence, ensure_ascii=False), status, source_run, parent_version,
                 created_at or storage.now()))
            connection.commit()
            return cursor.lastrowid
        finally:
            connection.close()


def memory_list(scope=None, status=None):
    sql, params = "SELECT * FROM memory_items", []
    clauses = []
    if scope:
        clauses.append("scope=?")
        params.append(scope)
    if status:
        clauses.append("status=?")
        params.append(status)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    return _read(sql + " ORDER BY id DESC", tuple(params))


def memory_status(memory_id, status):
    _write("UPDATE memory_items SET status=? WHERE id=?", (status, int(memory_id)))
    return _read("SELECT * FROM memory_items WHERE id=?", (int(memory_id),))


def tool_register(name, description, schema, handler, source="builtin"):
    _write("""INSERT INTO tools(name, description, schema_json, handler, source, enabled, registered_at)
              VALUES(?,?,?,?,?,1,?)
              ON CONFLICT(name) DO UPDATE SET description=excluded.description,
              schema_json=excluded.schema_json, handler=excluded.handler, source=excluded.source,
              enabled=1, registered_at=excluded.registered_at""",
           (name, description, json.dumps(schema, ensure_ascii=False), handler, source, storage.now()))


def tool_list():
    return _read("SELECT * FROM tools ORDER BY name")


def migrate_memory_files():
    """Mirror legacy JSON memory versions once, never overwriting history."""
    if _read("SELECT COUNT(*) AS n FROM memory_items")[0]["n"]:
        return 0
    imported = 0
    for folder, scope in ((storage.MEMORY, "retrieval"), (storage.MEMORY / "research", "research")):
        for file in sorted(folder.glob("v*.json")):
            value = storage.read_json(file) or {}
            memory_add(session_id=None, version=value.get("version"), scope=scope,
                       lesson=value.get("lesson", ""), applicability=value.get("applicability") or value.get("priority"),
                       evidence=value.get("evidence") or value.get("measured_effects"),
                       status=value.get("status", "provisional"), source_run=value.get("source_run"),
                       parent_version=value.get("parent_version"), created_at=value.get("created_at"))
            imported += 1
    return imported


# --- fact memory (ported from the ReEscape Protocol design) ---
def _fact_now():
    return storage.now()


EXCLUSIVE_PREDICATES = {"name", "identity", "location", "current_plan", "research_goal", "method_lesson"}


def fact_upsert(*, scope="research", session_id=None, predicate="", object="", summary="",
                tags=None, confidence=0.6, source_message="", expires_at=None):
    predicate = str(predicate or "").strip()
    object = str(object or "").strip()
    summary = str(summary or "").strip()
    if not predicate or not object or not summary:
        raise ValueError("事实记忆需要 predicate、object 和 summary")
    tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
    confidence = max(0.0, min(1.0, float(confidence or 0.6)))
    now = _fact_now()
    init()
    with LOCK:
        connection = _connect()
        try:
            if predicate in EXCLUSIVE_PREDICATES:
                connection.execute("""UPDATE facts SET status='contradicted'
                                      WHERE scope=? AND predicate=? AND object<>? AND status='active'""",
                                   (scope, predicate, object))
            existing = connection.execute("""SELECT * FROM facts WHERE scope=? AND predicate=? AND object=? AND status='active'""",
                                          (scope, predicate, object)).fetchone()
            if existing:
                merged = sorted({*(json.loads(existing["tags_json"] or "[]")), *tags})
                connection.execute("""UPDATE facts SET summary=?, tags_json=?, confidence=?, source_message=?,
                                      status='active', last_confirmed_at=?, expires_at=COALESCE(?, expires_at)
                                      WHERE id=?""",
                                   (summary, json.dumps(merged, ensure_ascii=False),
                                    max(float(existing["confidence"] or 0), confidence),
                                    source_message, now, expires_at, existing["id"]))
                fact_id = existing["id"]
            else:
                cursor = connection.execute("""INSERT INTO facts(session_id, scope, predicate, object, summary,
                                              tags_json, confidence, source_message, status, created_at,
                                              last_confirmed_at, expires_at)
                                              VALUES(?,?,?,?,?,?,?,?, 'active', ?, ?, ?)""",
                                            (session_id, scope, predicate, object, summary,
                                             json.dumps(tags, ensure_ascii=False), confidence,
                                             source_message, now, now, expires_at))
                fact_id = cursor.lastrowid
            connection.commit()
            row = connection.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
        finally:
            connection.close()
    return dict(row)


def facts(scope=None, status="active"):
    sql, params = "SELECT * FROM facts", []
    clauses = []
    if scope:
        clauses.append("scope=?")
        params.append(scope)
    if status:
        clauses.append("status=?")
        params.append(status)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    return _read(sql + " ORDER BY last_confirmed_at DESC", tuple(params))


def facts_expired(now=None):
    now = now or _fact_now()
    _write("UPDATE facts SET status='stale' WHERE status='active' AND expires_at IS NOT NULL AND expires_at < ?", (now,))

