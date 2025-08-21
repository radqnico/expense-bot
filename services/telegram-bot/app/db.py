from __future__ import annotations

import os
from decimal import Decimal
from typing import Any, Dict, Optional

import time
import psycopg


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


def ensure_schema(retries: int = 30, delay: float = 2.0) -> None:
    params = get_conn_params()
    last_err: Optional[Exception] = None
    for _ in range(max(1, retries)):
        try:
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
