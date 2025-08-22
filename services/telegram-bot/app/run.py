import logging
import os
import sys
import asyncio
from dataclasses import dataclass
from typing import Final, List, Tuple
import requests
import hashlib
import psycopg

from telegram import BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

LOG_LEVEL: Final = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("bot-spese.telegram-bot")

# Import handlers from the existing module to avoid duplicating logic
from .handlers.basic import cmd_start, cmd_help, cmd_health, cmd_status
from .handlers.reports import cmd_report, cmd_smartreport
from .handlers.navigation import cmd_navigation, handle_navigation_input
from .handlers.recurrent import cmd_recur_add, cmd_recur_list, cmd_recur_delete
from .main import (
    cmd_last,
    cmd_sum,
    cmd_undo,
    cmd_export,
    cmd_import,
    cmd_reset,
    cmd_month,
)

from .parser import to_csv_or_nd
from .handlers.quick import cmd_expense, cmd_income
from .llm import OllamaClient
from .db import (
    ensure_schema,
    insert_transaction,
    fetch_description_candidates,
)

from decimal import Decimal, InvalidOperation

# Minimal constants (duplicated to keep this thin and independent)
INFERENCE_QUEUES_KEY = "inference_queues"
INFERENCE_PROCESSING_KEY = "inference_processing"
HOSTS_KEY = "ollama_hosts"
HOST_RR_INDEX_KEY = "host_rr_index"
OLLAMA_LOCK_KEY = "ollama_lock"
NAV_STATE_KEY = "nav_state"
MODEL_READY_KEY = "ollama_model_ready"
MODEL_PULLING_KEY = "ollama_model_pulling"


def _normalize_host(h: str) -> str:
    h = h.strip().rstrip("/")
    if not h:
        return h
    if "://" not in h:
        h = f"http://{h}"
    return h


def parse_ollama_hosts() -> list[str]:
    raw = os.getenv("OLLAMA_HOSTS") or os.getenv("OLLAMA_HOST") or "localhost:11434"
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    hosts = [_normalize_host(p) for p in parts]
    return hosts or ["http://localhost:11434"]


async def echo(update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not (update.message and update.message.text):
        return
    if context.chat_data.get(NAV_STATE_KEY):
        try:
            await update.message.reply_text(
                "Navigation mode is active. Use /navigation to exit before inserting transactions."
            )
        except Exception:
            pass
        return

    text = update.message.text
    chat = update.effective_chat
    if not chat:
        return

    app = context.application
    hosts: list[str] = app.bot_data.get(HOSTS_KEY) or parse_ollama_hosts()
    queues: dict = app.bot_data.get(INFERENCE_QUEUES_KEY) or {}
    processing: dict = app.bot_data.get(INFERENCE_PROCESSING_KEY) or {}
    app.bot_data[HOSTS_KEY] = hosts
    app.bot_data[INFERENCE_QUEUES_KEY] = queues
    app.bot_data[INFERENCE_PROCESSING_KEY] = processing

    # Probe all hosts concurrently: (alive, has_model)
    async def _probe(h: str) -> tuple[str, bool, bool]:
        url = f"{h.rstrip('/')}/api/version"
        alive = False
        try:
            r = await asyncio.to_thread(lambda: requests.get(url, timeout=2.0))
            alive = bool(getattr(r, "ok", False))
        except Exception:
            alive = False
        has_model = False
        if alive:
            client = OllamaClient(host=h)
            try:
                has_model = bool(await asyncio.to_thread(client.has_model, None, 5.0))
            except Exception:
                has_model = False
        return h, alive, has_model

    results = await asyncio.gather(*[_probe(h) for h in hosts])
    ready_hosts = [h for h, alive, has in results if alive and has]
    alive_not_ready = [h for h, alive, has in results if alive and not has]
    down_hosts = [h for h, alive, has in results if not alive]

    if not ready_hosts:
        pulling: dict = app.bot_data.get(MODEL_PULLING_KEY)
        if pulling is None:
            pulling = {}
            app.bot_data[MODEL_PULLING_KEY] = pulling

        # If some hosts are alive but model missing, start background pull once per host
        if alive_not_ready:
            started = []
            for h in alive_not_ready:
                if not pulling.get(h, False):
                    pulling[h] = True
                    started.append(h)
                    async def _bg_pull(host: str) -> None:
                        client = OllamaClient(host=host)
                        try:
                            await asyncio.to_thread(client.pull_model, None, 3, 5.0, 300.0)
                        except Exception:
                            pass
                        finally:
                            pulling[host] = False
                    context.application.create_task(_bg_pull(h))
            # Inform the user
            msg = "⚠️ No ready backend right now. "
            if started:
                msg += f"Downloading the model on {len(started)} node(s): " + ", ".join(started) + "."
            else:
                msg += "Model download is already in progress on available node(s)."
            msg += " You can retry shortly, or use /expense or /income."
            try:
                await update.message.reply_text(msg)
            except Exception:
                pass
            return
        # No hosts alive at all
        else:
            try:
                await update.message.reply_text(
                    "⚠️ No backend online. I can't process this now. Use /expense or /income."
                )
            except Exception:
                pass
            return
    rr = app.bot_data.get(HOST_RR_INDEX_KEY, 0)
    host = ready_hosts[rr % len(ready_hosts)]
    app.bot_data[HOST_RR_INDEX_KEY] = rr + 1

    q: asyncio.Queue = queues.get(host)
    if q is None:
        q = asyncio.Queue()
        queues[host] = q

    position = q.qsize() + (1 if processing.get(host, False) else 0) + 1
    # Send a queue notice only if the user is not first; avoid duplicate replies otherwise
    if position > 1:
        try:
            await update.message.reply_text(
                f"⏳ Occupato. Sei in coda (#{position}). Ti avviso appena pronto."
            )
        except Exception:
            pass

    await q.put((chat.id, update.message.message_id, text))


def _parse_amount_token(tok: str) -> Decimal:
    s = tok.strip().replace(",", ".")
    return Decimal(s)


# Quick insert handlers are imported from .quick


async def register_workers(app: Application) -> None:
    hosts: list[str] = app.bot_data.get(HOSTS_KEY) or parse_ollama_hosts()
    app.bot_data[HOSTS_KEY] = hosts
    app.bot_data[INFERENCE_QUEUES_KEY] = {h: asyncio.Queue() for h in hosts}
    app.bot_data[INFERENCE_PROCESSING_KEY] = {h: False for h in hosts}
    app.bot_data[MODEL_PULLING_KEY] = {h: False for h in hosts}

    async def worker(host: str) -> None:
        q: asyncio.Queue = app.bot_data[INFERENCE_QUEUES_KEY][host]
        client = OllamaClient(host=host)
        while True:
            chat_id, message_id, text = await q.get()
            app.bot_data[INFERENCE_PROCESSING_KEY][host] = True
            try:
                # LLM + parse
                candidates = await asyncio.to_thread(fetch_description_candidates, int(chat_id), 50)
                result = await asyncio.to_thread(to_csv_or_nd, text, client, candidates)

                reply_text = None
                try:
                    if "," in result and result.upper() != "ND":
                        amount_str, description = result.split(",", 1)
                        amount = Decimal(amount_str.strip())
                        await asyncio.to_thread(insert_transaction, int(chat_id), amount, description.strip())
                        kind = "Entrata" if amount >= 0 else "Spesa"
                        emoji = "💰" if amount >= 0 else "💸"
                        reply_text = (
                            f"✅ {emoji} {kind} registrata\n"
                            f"Importo: {amount:+.2f}\n"
                            f"Descrizione: {description.strip()}"
                        )
                    else:
                        reply_text = (
                            "⚠️ Non determinato. Assicurati di includere un importo e una descrizione."
                        )
                except (InvalidOperation, Exception):
                    reply_text = (
                        "⚠️ Errore nell'inserimento. Controlla il formato e riprova."
                    )
                try:
                    await app.bot.send_message(chat_id=chat_id, text=reply_text or result, reply_to_message_id=message_id)
                except Exception:
                    pass
            finally:
                q.task_done()
                app.bot_data[INFERENCE_PROCESSING_KEY][host] = False

    # Start one worker per host
    for h in hosts:
        app.create_task(worker(h))


def main() -> None:
    ensure_schema()

    # Acquire a singleton lock so only one instance processes updates
    def _acquire_singleton_or_exit(token: str) -> None:
        dig = hashlib.sha1(token.encode()).digest()
        k1 = int.from_bytes(dig[:4], "big", signed=False)
        k2 = int.from_bytes(dig[4:8], "big", signed=False)
        if k1 >= 2**31:
            k1 -= 2**32
        if k2 >= 2**31:
            k2 -= 2**32
        dsn = (
            f"host={os.getenv('DB_HOST','postgres')} "
            f"port={os.getenv('DB_PORT','5432')} "
            f"dbname={os.getenv('DB_NAME','appdb')} "
            f"user={os.getenv('DB_USER','app')} "
            f"password={os.getenv('DB_PASSWORD','app')}"
        )
        try:
            conn = psycopg.connect(dsn)
            cur = conn.cursor()
            cur.execute("SELECT pg_try_advisory_lock(%s, %s)", (k1, k2))
            ok = bool(cur.fetchone()[0])
            if not ok:
                logger.error("Another bot instance is active (lock not acquired). Exiting.")
                try:
                    conn.close()
                except Exception:
                    pass
                raise SystemExit(0)
            # Keep connection open for the lifetime of the process to hold the lock
            globals()["_BOT_SINGLETON_CONN"] = conn  # prevent GC
            logger.info("Singleton lock acquired")
        except Exception as e:
            logger.warning("Could not acquire singleton lock: %s", e)
            # If DB unavailable, proceed without lock (best-effort) but warn about possible duplicates

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set")
        raise SystemExit(2)
    _acquire_singleton_or_exit(token)

    app = Application.builder().token(token).concurrent_updates(True).build()

    # Register commands
    commands: List[Tuple[str, str]] = [
        ("start", "Start the bot"),
        ("help", "Show help"),
        ("health", "Health check"),
        ("status", "Show backend + queue status"),
        ("last", "List recent entries"),
        ("sum", "Sum by period (today/week/month/all)"),
        ("undo", "Delete last entry"),
        ("export", "Export CSV"),
        ("import", "Import JSON/Excel/CSV"),
        ("report", "Charts + PDF"),
        ("smartreport", "LLM filtered report"),
        ("expense", "Quick expense: /expense AMOUNT DESCRIPTION"),
        ("income", "Quick income: /income AMOUNT DESCRIPTION"),
        ("navigation", "Browse/edit transactions"),
        ("reset", "Reset entries in period"),
        ("month", "Monthly summary"),
        ("recur_add", "Create recurrent op"),
        ("recur_list", "List recurrent ops"),
        ("recur_delete", "Delete recurrent op"),
    ]
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("health", cmd_health))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("last", cmd_last))
    app.add_handler(CommandHandler("sum", cmd_sum))
    app.add_handler(CommandHandler("undo", cmd_undo))
    app.add_handler(CommandHandler("export", cmd_export))
    app.add_handler(CommandHandler("import", cmd_import))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("smartreport", cmd_smartreport))
    app.add_handler(CommandHandler("expense", cmd_expense))
    app.add_handler(CommandHandler("income", cmd_income))
    app.add_handler(CommandHandler("navigation", cmd_navigation))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("month", cmd_month))
    app.add_handler(CommandHandler("recur_add", cmd_recur_add))
    app.add_handler(CommandHandler("recur_list", cmd_recur_list))
    app.add_handler(CommandHandler("recur_delete", cmd_recur_delete))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_navigation_input, block=False), group=0)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo), group=1)

    async def post_init(application: Application) -> None:
        await application.bot.set_my_commands([BotCommand(c, d) for c, d in commands])
        await register_workers(application)

    app.post_init = post_init  # type: ignore
    logger.info("Starting polling (run.py)…")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
