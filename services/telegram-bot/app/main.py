import logging
import os
import os
import sys
import asyncio
from dataclasses import dataclass
import requests
from typing import Final, List, Tuple

from telegram import BotCommand, Update
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


LOG_LEVEL: Final = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("bot-spese.telegram-bot")

from .llm import OllamaClient
from .db import ensure_schema, insert_transaction
from decimal import Decimal, InvalidOperation
from .parser import to_csv_or_nd

INFERENCE_QUEUES_KEY = "inference_queues"  # dict[host]->Queue
INFERENCE_PROCESSING_KEY = "inference_processing"  # dict[host]->bool
HOSTS_KEY = "ollama_hosts"  # list[str]
HOST_RR_INDEX_KEY = "host_rr_index"


@dataclass
class InferenceJob:
    chat_id: int
    text: str


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


def ping_host(base_url: str, timeout: float = 2.0) -> bool:
    url = f"{base_url}/api/version"
    try:
        r = requests.get(url, timeout=timeout)
        return r.ok
    except Exception:
        return False


BOT_NAME: Final = os.getenv("BOT_NAME", "RADQ Expenses Tracker").strip()
BOT_SHORT_DESCRIPTION: Final = (
    os.getenv("BOT_SHORT_DESCRIPTION", "Track expenses easily with RADQ.").strip()
)
BOT_DESCRIPTION: Final = (
    os.getenv(
        "BOT_DESCRIPTION",
        "A simple expenses tracker bot. Use /help to see commands.",
    ).strip()
)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info("/start by %s", user.username if user else "unknown")
    await update.message.reply_text(
        f"👋 Welcome to {BOT_NAME}!\nUse /help to see commands."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/start - greet\n/help - show this message\n/health - check health\n/status - backend + queue status"
    )


async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("OK")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    hosts: list[str] = app.bot_data.get(HOSTS_KEY) or parse_ollama_hosts()
    queues: dict = app.bot_data.get(INFERENCE_QUEUES_KEY) or {}
    processing: dict = app.bot_data.get(INFERENCE_PROCESSING_KEY) or {}

    alive_list = await asyncio.to_thread(lambda: [ping_host(h) for h in hosts])
    lines = []
    chat_id = update.effective_chat.id if update.effective_chat else None
    for h, alive in zip(hosts, alive_list):
        q = queues.get(h)
        qsize = q.qsize() if q else 0
        is_proc = bool(processing.get(h, False))
        # Try to estimate user's next position in this host queue
        pos = None
        if chat_id is not None and q is not None:
            try:
                pending = [it.chat_id for it in list(q._queue)]  # type: ignore[attr-defined]
                if chat_id in pending:
                    idx = pending.index(chat_id)
                    pos = idx + (1 if is_proc else 0) + 1
            except Exception:
                pass
        status = "up" if alive else "down"
        extra = f", you at #{pos}" if pos is not None else ""
        lines.append(f"{h} — {status}, queue {qsize}{extra}")

    text = "\n".join(lines) if lines else "No hosts configured"
    await update.message.reply_text(text)


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not (update.message and update.message.text):
        return

    text = update.message.text
    chat = update.effective_chat
    if not chat:
        return

    app = context.application
    # Resolve hosts and queues
    hosts: list[str] = app.bot_data.get(HOSTS_KEY) or []
    queues: dict = app.bot_data.get(INFERENCE_QUEUES_KEY) or {}
    processing: dict = app.bot_data.get(INFERENCE_PROCESSING_KEY) or {}

    # Ping all instances in a background thread
    alive_list = await asyncio.to_thread(lambda: [ping_host(h) for h in hosts])
    alive_hosts = [h for h, alive in zip(hosts, alive_list) if alive]

    # Choose target host: round-robin among alive, preserving order; fallback to first defined
    if alive_hosts:
        rr = app.bot_data.get(HOST_RR_INDEX_KEY, 0)
        host = alive_hosts[rr % len(alive_hosts)]
        app.bot_data[HOST_RR_INDEX_KEY] = rr + 1
    else:
        host = hosts[0] if hosts else "http://localhost:11434"

    q: asyncio.Queue = queues.get(host)
    if q is None:
        # Safety: initialize if missing
        q = asyncio.Queue()
        queues[host] = q
        app.bot_data[INFERENCE_QUEUES_KEY] = queues

    # Position in selected host queue
    is_processing = bool(processing.get(host, False))
    position = q.qsize() + (1 if is_processing else 0) + 1

    # Notify user
    try:
        if position > 1:
            postfix = " (nessuna istanza disponibile, attendo ripristino)" if not alive_hosts else ""
            await update.message.reply_text(
                f"⏳ Occupato. Sei in coda (#{position}). Ti avviso appena pronto.{postfix}"
            )
        else:
            await update.message.reply_text("🚀 Elaboro il tuo messaggio…")
    except Exception:
        pass

    await q.put(InferenceJob(chat_id=chat.id, text=text))


def get_token() -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set")
        raise SystemExit(2)
    return token


def main() -> None:
    token = get_token()

    # Optionally pull model on start on all configured hosts
    pull_on_start = os.getenv("OLLAMA_PULL_ON_START", "true").lower() not in {"0", "false", "no"}
    if pull_on_start:
        hosts = parse_ollama_hosts()
        for h in hosts:
            client = OllamaClient(host=h)
            try:
                logger.info("Ensuring model is available on %s: %s", h, client.model)
                client.pull_model()
                logger.info("Model ready on %s: %s", h, client.model)
            except Exception as e:
                logger.warning("Could not pull model on %s: %s", h, e)

    async def post_init(application: Application) -> None:
        commands: List[Tuple[str, str]] = [
            ("start", "Start the bot"),
            ("help", "Show help"),
            ("health", "Health check"),
            ("status", "Show backend + queue status"),
        ]

        try:
            await application.bot.set_my_name(name=BOT_NAME)
            logger.info("Set bot name: %s", BOT_NAME)
        except TelegramError as e:
            logger.warning("Could not set bot name: %s", e)

        try:
            await application.bot.set_my_short_description(
                short_description=BOT_SHORT_DESCRIPTION
            )
            logger.info("Set short description")
        except TelegramError as e:
            logger.warning("Could not set short description: %s", e)

        try:
            await application.bot.set_my_description(description=BOT_DESCRIPTION)
            logger.info("Set description")
        except TelegramError as e:
            logger.warning("Could not set description: %s", e)

        try:
            await application.bot.set_my_commands(
                [BotCommand(c, d) for c, d in commands]
            )
            logger.info("Set commands: %s", ", ".join(c for c, _ in commands))
        except TelegramError as e:
            logger.warning("Could not set commands: %s", e)

        # Initialize hosts and per-host queues
        hosts = parse_ollama_hosts()
        application.bot_data[HOSTS_KEY] = hosts
        application.bot_data[INFERENCE_QUEUES_KEY] = {}
        application.bot_data[INFERENCE_PROCESSING_KEY] = {}
        for h in hosts:
            application.bot_data[INFERENCE_QUEUES_KEY][h] = asyncio.Queue()

        # Start one worker per host
        try:
            for h in hosts:
                application.create_task(worker(application, h))
            logger.info("Started %d inference workers", len(hosts))
        except Exception as e:
            logger.warning("Could not start workers: %s", e)

    # Ensure DB schema before starting
    try:
        ensure_schema()
        logger.info("Database schema ensured")
    except Exception as e:
        logger.warning("Could not ensure DB schema: %s", e)

    app = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .concurrent_updates(True)
        .build()
    )

    # Create a single worker to process messages sequentially through Ollama
    async def worker(application: Application, host: str) -> None:
        q: asyncio.Queue = application.bot_data[INFERENCE_QUEUES_KEY][host]
        client = OllamaClient(host=host)
        while True:
            job: InferenceJob = await q.get()
            processing: dict = application.bot_data[INFERENCE_PROCESSING_KEY]
            processing[host] = True
            try:
                # Generate and parse
                result = await asyncio.to_thread(to_csv_or_nd, job.text, client)
                # Persist if CSV
                try:
                    if "," in result and result.upper() != "ND":
                        amount_str, description = result.split(",", 1)
                        amount = Decimal(amount_str.strip())
                        await asyncio.to_thread(
                            insert_transaction, int(job.chat_id), amount, description.strip()
                        )
                except (InvalidOperation, Exception):
                    pass
                # Reply to user
                try:
                    await application.bot.send_message(chat_id=job.chat_id, text=result)
                except Exception:
                    pass
            finally:
                q.task_done()
                processing[host] = False

    # Worker is started in post_init where event loop is running

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("health", cmd_health))
    app.add_handler(CommandHandler("status", cmd_status))

    # Fallback echo
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    logger.info("Starting polling...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
