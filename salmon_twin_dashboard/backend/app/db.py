from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from .config import dashboard_root, resolve_path, settings


_engine: Engine | None = None
_SQLITE_BOOTSTRAPPED = False
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_sqlite() -> bool:
    return settings.DATABASE_URL.startswith("sqlite")


def _sqlite_path() -> Path | None:
    if not _is_sqlite():
        return None
    raw = settings.DATABASE_URL.split("///", 1)[-1]
    if raw == ":memory:":
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (dashboard_root() / "backend" / path).resolve()
    return path


def _quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _default_csv_path() -> str:
    return str((dashboard_root().parent / "rag" / "sensor_history.csv").resolve())


def _csv_path() -> Path | None:
    return resolve_path(settings.SENSOR_CSV_PATH, "../../rag/sensor_history.csv", _default_csv_path(), "/rag/sensor_history.csv")


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        sqlite_path = _sqlite_path()
        if sqlite_path is not None:
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False} if _is_sqlite() else {}
        _engine = create_engine(settings.DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
    ensure_sensor_table()
    return _engine


def sensor_columns() -> list[str]:
    return [c.strip() for c in settings.SENSOR_COLUMNS.split(",") if c.strip()]


def ensure_sensor_table() -> None:
    global _SQLITE_BOOTSTRAPPED
    if _SQLITE_BOOTSTRAPPED or not _is_sqlite() or _engine is None:
        return
    _SQLITE_BOOTSTRAPPED = True

    csv_path = _csv_path()
    if csv_path is None:
        return

    inspector = inspect(_engine)
    table_exists = inspector.has_table(settings.SENSOR_TABLE)
    if table_exists:
        with _engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {_quote_identifier(settings.SENSOR_TABLE)}")).scalar_one()
        if count:
            return

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    required = [settings.TIME_COLUMN] + sensor_columns()
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise RuntimeError(f"Sensor CSV missing required columns {missing}: {csv_path}")

    column_defs = [f"{_quote_identifier(settings.TIME_COLUMN)} TEXT NOT NULL"]
    column_defs.extend(f"{_quote_identifier(column)} REAL" for column in sensor_columns())
    create_sql = text(
        f"CREATE TABLE IF NOT EXISTS {_quote_identifier(settings.SENSOR_TABLE)} "
        f"({', '.join(column_defs)})"
    )

    insert_columns = required
    quoted_columns = ", ".join(_quote_identifier(column) for column in insert_columns)
    bind_columns = ", ".join(f":{column}" for column in insert_columns)
    insert_sql = text(
        f"INSERT INTO {_quote_identifier(settings.SENSOR_TABLE)} "
        f"({quoted_columns}) VALUES ({bind_columns})"
    )

    payload = []
    for row in rows:
        item: dict[str, Any] = {settings.TIME_COLUMN: row[settings.TIME_COLUMN]}
        for column in sensor_columns():
            value = row.get(column)
            item[column] = float(value) if value not in (None, "") else None
        payload.append(item)

    with _engine.begin() as conn:
        conn.execute(create_sql)
        if payload:
            conn.execute(insert_sql, payload)


def _selected_columns() -> tuple[str, str, list[str], str]:
    table = settings.SENSOR_TABLE
    time_col = settings.TIME_COLUMN
    cols = sensor_columns()
    selected = ", ".join(_quote_identifier(c) for c in [time_col] + cols)
    return table, time_col, cols, selected


def get_recent_sensor_rows(hours: float = 1.0, limit: int = 7200) -> list[dict[str, Any]]:
    table, time_col, _cols, selected = _selected_columns()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    sql = text(
        f"""
        SELECT {selected}
        FROM {_quote_identifier(table)}
        WHERE {_quote_identifier(time_col)} >= :cutoff
        ORDER BY {_quote_identifier(time_col)} ASC
        LIMIT :limit
        """
    )

    with get_engine().connect() as conn:
        rows = conn.execute(sql, {"cutoff": cutoff.isoformat(), "limit": limit}).mappings().all()
        if rows or not settings.RECENT_FALLBACK_TO_LATEST:
            return [dict(r) for r in rows]

        fallback_sql = text(
            f"""
            SELECT {selected}
            FROM {_quote_identifier(table)}
            ORDER BY {_quote_identifier(time_col)} DESC
            LIMIT :limit
            """
        )
        fallback_rows = conn.execute(fallback_sql, {"limit": limit}).mappings().all()
        return [dict(r) for r in reversed(fallback_rows)]


def get_latest_sensor_row() -> dict[str, Any] | None:
    table, time_col, _cols, selected = _selected_columns()
    sql = text(f"SELECT {selected} FROM {_quote_identifier(table)} ORDER BY {_quote_identifier(time_col)} DESC LIMIT 1")
    with get_engine().connect() as conn:
        row = conn.execute(sql).mappings().first()
        return dict(row) if row else None


def append_action_log(proposal: dict[str, Any], decision: str) -> None:
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


def append_control_command(payload: dict[str, Any]) -> int:
    sql_create = text(
        """
        CREATE TABLE IF NOT EXISTS control_commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            payload_json TEXT NOT NULL
        )
        """
    )
    sql_insert = text(
        """
        INSERT INTO control_commands (created_at, status, payload_json)
        VALUES (:created_at, 'pending', :payload_json)
        """
    )

    with get_engine().begin() as conn:
        conn.execute(sql_create)
        result = conn.execute(
            sql_insert,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "payload_json": json.dumps(payload, ensure_ascii=False),
            },
        )
        return int(result.lastrowid or 0)


def database_status() -> dict[str, Any]:
    try:
        latest = get_latest_sensor_row()
        table = settings.SENSOR_TABLE
        with get_engine().connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {_quote_identifier(table)}")).scalar_one()
        return {
            "ok": True,
            "url": settings.DATABASE_URL,
            "table": table,
            "rows": count,
            "latest_time": latest.get(settings.TIME_COLUMN) if latest else None,
            "csv_path": str(_csv_path()) if _csv_path() else None,
        }
    except Exception as exc:
        return {"ok": False, "url": settings.DATABASE_URL, "table": settings.SENSOR_TABLE, "error": str(exc)}
