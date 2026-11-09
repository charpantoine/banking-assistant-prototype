"""SQLite audit log: one row per model call, tool call, guardrail result or action."""
import json
import sqlite3
from datetime import datetime, timezone

import config

_SCHEMA = """CREATE TABLE IF NOT EXISTS events (
    ts TEXT, session_id TEXT, step TEXT, agent TEXT, name TEXT,
    args_masked TEXT, result_summary TEXT, latency_ms REAL)"""


def _connect():
    """Open the audit database, creating the table on first use."""
    conn = sqlite3.connect(config.AUDIT_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def log_event(session_id, step, agent, name, args, result_summary, latency_ms):
    """Append one event with a UTC ISO 8601 timestamp (mask_pii is added here in stage 3)."""
    ts = datetime.now(timezone.utc).isoformat()
    row = (ts, session_id, step, agent, name, json.dumps(args, ensure_ascii=False, default=str),
           str(result_summary), latency_ms)
    conn = _connect()
    with conn:
        conn.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", row)
    conn.close()


def last_events(n):
    """Return the n most recent events as dicts, newest first."""
    conn = _connect()
    rows = conn.execute("SELECT * FROM events ORDER BY rowid DESC LIMIT ?", (n,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
