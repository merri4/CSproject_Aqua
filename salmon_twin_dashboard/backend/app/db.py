from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import settings


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
        _engine = create_engine(settings.DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
    return _engine


def sensor_columns() -> list[str]:
    return [c.strip() for c in settings.SENSOR_COLUMNS.split(",") if c.strip()]


def get_recent_sensor_rows(hours: float = 1.0, limit: int = 7200) -> list[dict[str, Any]]:
    """Read recent sensor rows. Assumes timestamp is ISO datetime or DB timestamp-compatible."""
    table = settings.SENSOR_TABLE
    time_col = settings.TIME_COLUMN
    cols = sensor_columns()
    selected = ", ".join([time_col] + cols)

    # Backend-agnostic cutoff parameter. SQLite ISO strings work if stored as ISO8601.
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    sql = text(
        f"""
        SELECT {selected}
        FROM {table}
        WHERE {time_col} >= :cutoff
        ORDER BY {time_col} ASC
        LIMIT :limit
        """
    )

    with get_engine().connect() as conn:
        rows = conn.execute(sql, {"cutoff": cutoff.isoformat(), "limit": limit}).mappings().all()
        return [dict(r) for r in rows]


def get_latest_sensor_row() -> dict[str, Any] | None:
    table = settings.SENSOR_TABLE
    time_col = settings.TIME_COLUMN
    cols = sensor_columns()
    selected = ", ".join([time_col] + cols)
    sql = text(f"SELECT {selected} FROM {table} ORDER BY {time_col} DESC LIMIT 1")
    with get_engine().connect() as conn:
        row = conn.execute(sql).mappings().first()
        return dict(row) if row else None


def append_action_log(proposal: dict[str, Any], decision: str) -> None:
    """Optional local audit table. Safe to use even when your sensor DB is SQLite/Postgres."""
    sql_create = text(
        """
        CREATE TABLE IF NOT EXISTS ai_action_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            decision TEXT NOT NULL,
            proposal_json TEXT NOT NULL
        )
        """
    )
    sql_insert = text(
        """
        INSERT INTO ai_action_log (created_at, decision, proposal_json)
        VALUES (:created_at, :decision, :proposal_json)
        """
    )
    import json

    with get_engine().begin() as conn:
        conn.execute(sql_create)
        conn.execute(
            sql_insert,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "decision": decision,
                "proposal_json": json.dumps(proposal, ensure_ascii=False),
            },
        )
