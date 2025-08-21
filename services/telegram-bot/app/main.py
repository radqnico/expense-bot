import logging
import os
import os
import sys
import asyncio
from dataclasses import dataclass
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

INFERENCE_QUEUE_KEY = "inference_queue"
INFERENCE_PROCESSING_KEY = "inference_processing"


@dataclass
class InferenceJob:
    chat_id: int
    text: str


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
        "/start - greet\n/help - show this message\n/health - check health"
    )


async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("OK")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not (update.message and update.message.text):
        return

    text = update.message.text
    chat = update.effective_chat
    if not chat:
        return

    # Enqueue the message for sequential processing
    q: asyncio.Queue = context.application.bot_data.setdefault(INFERENCE_QUEUE_KEY, asyncio.Queue())
    processing: bool = bool(context.application.bot_data.get(INFERENCE_PROCESSING_KEY, False))

    # Calculate position: items in queue + (1 if currently processing) + this job
    position = q.qsize() + (1 if processing else 0) + 1

    if position > 1:
        try:
            await update.message.reply_text(f"⏳ Occupato. Sei in coda (#{position}). Ti avviso appena pronto.")
        except Exception:
            pass
    else:
        try:
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

    # Optionally pull model on start
    pull_on_start = os.getenv("OLLAMA_PULL_ON_START", "true").lower() not in {"0", "false", "no"}
    if pull_on_start:
        client = OllamaClient()
        try:
            logger.info("Ensuring model is available: %s", client.model)
            client.pull_model()
            logger.info("Model ready: %s", client.model)
        except Exception as e:
            logger.warning("Could not pull model on start: %s", e)

    async def post_init(application: Application) -> None:
        commands: List[Tuple[str, str]] = [
            ("start", "Start the bot"),
            ("help", "Show help"),
            ("health", "Health check"),
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

        # Start background worker once the loop is running
        try:
            application.create_task(worker(application))
            logger.info("Started inference worker")
        except Exception as e:
            logger.warning("Could not start worker: %s", e)

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
    async def worker(application: Application) -> None:
        q: asyncio.Queue = application.bot_data.setdefault(INFERENCE_QUEUE_KEY, asyncio.Queue())
        client = OllamaClient()
        while True:
            job: InferenceJob = await q.get()
            application.bot_data[INFERENCE_PROCESSING_KEY] = True
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
                application.bot_data[INFERENCE_PROCESSING_KEY] = False

    # Worker is started in post_init where event loop is running

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("health", cmd_health))

    # Fallback echo
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    logger.info("Starting polling...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
