from __future__ import annotations

import os
from decimal import Decimal
from typing import Any, Dict, Optional, Iterable, List, Tuple
import re

import time
import psycopg
from psycopg import sql
from psycopg.errors import InvalidCatalogName


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val


def get_conn_params() -> Dict[str, Any]:
    return {
        "host": _env("DB_HOST", _env("POSTGRES_HOST", "postgres")),
        "port": int(_env("DB_PORT", _env("POSTGRES_PORT", "5432")) or 5432),
        "dbname": _env("DB_NAME", _env("POSTGRES_DB", "appdb")),
        "user": _env("DB_USER", _env("POSTGRES_USER", "app")),
        "password": _env("DB_PASSWORD", _env("POSTGRES_PASSWORD", "app")),
        "connect_timeout": int(_env("DB_CONNECT_TIMEOUT", "10") or 10),
    }


def _ensure_database(params: Dict[str, Any]) -> None:
    """Ensure target database exists; create it if missing using maintenance DB 'postgres'."""
    try:
        with psycopg.connect(**params) as _conn:
            return  # DB exists and is connectable
    except InvalidCatalogName:
        maint = dict(params)
        maint["dbname"] = "postgres"
        dbname = params["dbname"]
        owner = params.get("user")
        with psycopg.connect(**maint) as conn:
            conn.execute(sql.SQL("CREATE DATABASE {} OWNER {}" ).format(
                sql.Identifier(dbname), sql.Identifier(owner)
            ))
            conn.commit()


def ensure_schema(retries: int = 30, delay: float = 2.0) -> None:
    params = get_conn_params()
    last_err: Optional[Exception] = None
    for _ in range(max(1, retries)):
        try:
            # First ensure the database exists
            _ensure_database(params)
            with psycopg.connect(**params) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS transactions (
                            id bigserial PRIMARY KEY,
                            ts timestamptz NOT NULL DEFAULT now(),
                            chatid bigint NOT NULL,
                            amount numeric NOT NULL,
                            description text NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS idx_transactions_chatid_ts
                            ON transactions(chatid, ts DESC);
                        """
                    )
                conn.commit()
            return
        except Exception as e:
            last_err = e
            time.sleep(delay)
    if last_err:
        raise last_err


def insert_transaction(chatid: int, amount: Decimal, description: str) -> None:
    params = get_conn_params()
    with psycopg.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO transactions (chatid, amount, description)
                VALUES (%s, %s, %s)
                """,
                (chatid, amount, description),
            )
        conn.commit()


def fetch_recent(chatid: int, limit: int = 5) -> List[Tuple[int, str, Decimal, str]]:
    params = get_conn_params()
    with psycopg.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, to_char(ts at time zone 'UTC', 'YYYY-MM-DD HH24:MI:SS') as ts_str,
                       amount, description
                FROM transactions
                WHERE chatid = %s
                ORDER BY ts DESC
                LIMIT %s
                """,
                (chatid, limit),
            )
            rows = cur.fetchall()
    return rows


def sum_period(chatid: int, period: str = "month") -> Optional[Decimal]:
    period = (period or "month").strip().lower()
    where = "chatid = %s"
    if period == "today":
        where += " AND ts >= date_trunc('day', now())"
    elif period == "week":
        where += " AND ts >= date_trunc('week', now())"
    elif period == "month":
        where += " AND ts >= date_trunc('month', now())"
    elif period == "all":
        pass
    else:
        # default to month
        where += " AND ts >= date_trunc('month', now())"
    params = get_conn_params()
    with psycopg.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COALESCE(SUM(amount),0) FROM transactions WHERE {where}", (chatid,))
            val = cur.fetchone()[0]
    return val


def delete_last(chatid: int) -> Optional[Tuple[int, str, Decimal, str]]:
    params = get_conn_params()
    with psycopg.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM transactions
                WHERE id IN (
                    SELECT id FROM transactions WHERE chatid = %s ORDER BY ts DESC LIMIT 1
                )
                RETURNING id, to_char(ts at time zone 'UTC', 'YYYY-MM-DD HH24:MI:SS') as ts_str, amount, description
                """,
                (chatid,),
            )
            row = cur.fetchone()
        conn.commit()
    return row


def fetch_for_export(chatid: int, period: Optional[str] = None) -> Iterable[Tuple[str, int, Decimal, str]]:
    params = get_conn_params()
    # Special case: YYYY-MM specific month
    if period and re.fullmatch(r"\d{4}-\d{2}", period.strip()):
        year_s, month_s = period.strip().split("-")
        year, month = int(year_s), int(month_s)
        with psycopg.connect(**params) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH range AS (
                        SELECT make_timestamp(%s, %s, 1, 0, 0, 0) at time zone 'UTC' AS start_ts,
                               (make_timestamp(%s, %s, 1, 0, 0, 0) + interval '1 month') at time zone 'UTC' AS end_ts
                    )
                    SELECT to_char(ts at time zone 'UTC', 'YYYY-MM-DD HH24:MI:SS') as ts_str,
                           chatid, amount, description
                    FROM transactions t, range r
                    WHERE t.chatid = %s AND t.ts >= r.start_ts AND t.ts < r.end_ts
                    ORDER BY ts ASC
                    """,
                    (year, month, year, month, chatid),
                )
                for row in cur:
                    yield row
        return

    # Named periods
    where = "chatid = %s"
    if period:
        p = period.strip().lower()
        if p == "today":
            where += " AND ts >= date_trunc('day', now())"
        elif p == "week":
            where += " AND ts >= date_trunc('week', now())"
        elif p == "month":
            where += " AND ts >= date_trunc('month', now())"
        elif p == "all":
            pass
        else:
            where += " AND ts >= date_trunc('month', now())"
    with psycopg.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT to_char(ts at time zone 'UTC', 'YYYY-MM-DD HH24:MI:SS') as ts_str,
                       chatid, amount, description
                FROM transactions
                WHERE {where}
                ORDER BY ts ASC
                """,
                (chatid,),
            )
            for row in cur:
                yield row


def month_summary(chatid: int, year: int, month: int) -> Dict[str, Any]:
    """Return totals and per-day sums for the given month.

    Keys:
    - income: Decimal (>=0)
    - expenses: Decimal (<=0)
    - net: Decimal
    - count: int
    - days: List[Tuple[str, Decimal]] with date as YYYY-MM-DD and daily sum
    """
    params = get_conn_params()
    with psycopg.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH range AS (
                    SELECT make_timestamp(%s, %s, 1, 0, 0, 0) at time zone 'UTC' AS start_ts,
                           (make_timestamp(%s, %s, 1, 0, 0, 0) + interval '1 month') at time zone 'UTC' AS end_ts
                )
                SELECT
                    COALESCE(SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END), 0) AS income,
                    COALESCE(SUM(CASE WHEN amount < 0 THEN amount ELSE 0 END), 0) AS expenses,
                    COALESCE(SUM(amount), 0) AS net,
                    COUNT(*) AS cnt
                FROM transactions t, range r
                WHERE t.chatid = %s AND t.ts >= r.start_ts AND t.ts < r.end_ts
                """,
                (year, month, year, month, chatid),
            )
            income, expenses, net, cnt = cur.fetchone()

            cur.execute(
                """
                WITH range AS (
                    SELECT make_timestamp(%s, %s, 1, 0, 0, 0) at time zone 'UTC' AS start_ts,
                           (make_timestamp(%s, %s, 1, 0, 0, 0) + interval '1 month') at time zone 'UTC' AS end_ts
                )
                SELECT to_char(date_trunc('day', ts), 'YYYY-MM-DD') as d, COALESCE(SUM(amount),0) AS sum
                FROM transactions t, range r
                WHERE t.chatid = %s AND t.ts >= r.start_ts AND t.ts < r.end_ts
                GROUP BY 1
                ORDER BY 1
                """,
                (year, month, year, month, chatid),
            )
            days = cur.fetchall()
    return {"income": income, "expenses": expenses, "net": net, "count": cnt, "days": days}
