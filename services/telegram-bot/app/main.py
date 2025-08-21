import logging
import os
import os
import sys
import asyncio
from dataclasses import dataclass
import requests
import csv
import io
import datetime as dt
import json
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
from .db import (
    ensure_schema,
    insert_transaction,
    bulk_insert_transactions,
    fetch_recent,
    sum_period,
    delete_last,
    fetch_for_export,
    month_summary,
    fetch_description_candidates,
    delete_period,
)
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


def get_commands() -> List[Tuple[str, str]]:
    return [
        ("start", "Start the bot"),
        ("help", "Show help"),
        ("health", "Health check"),
        ("status", "Show backend + queue status"),
        ("last", "List recent entries"),
        ("sum", "Sum by period (today/week/month/all)"),
        ("undo", "Delete last entry"),
        ("export", "Export CSV: /export YYYY-MM|day|week|month|year"),
        ("import", "Import JSON: send file or reply"),
        ("report", "Charts + PDF: /report day|week|month|year"),
        ("reset", "Reset entries: /reset day|month|all"),
        ("month", "Monthly summary: /month YYYY-MM"),
    ]


async def sync_commands(application: Application) -> None:
    try:
        cmds = [BotCommand(c, d) for c, d in get_commands()]
        await application.bot.set_my_commands(cmds)
        logger.info("Commands synced: %s", ", ".join(c for c, _ in get_commands()))
    except TelegramError as e:
        logger.warning("Could not set commands: %s", e)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/start - greet\n/help - show this message\n/health - check health\n"
        "/status - backend + queue status\n"
        "/last [n] - list last n entries (default 5)\n"
        "/sum [today|week|month|all] - sum amounts (default month)\n"
        "/undo - delete last entry\n"
        "/export [YYYY-MM|day|week|month|year] - CSV export\n"
        "/import - send a JSON file (or reply to one)\n"
        "/report day|week|month|year - charts + PDF\n"
        "/reset day|month|all - delete entries in period\n"
        "/month YYYY-MM - monthly summary"
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


async def cmd_last(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return
    args = (update.message.text.split()[1:] if update.message and update.message.text else [])
    try:
        n = int(args[0]) if args else 5
        n = max(1, min(n, 50))
    except Exception:
        n = 5
    rows = await asyncio.to_thread(fetch_recent, int(chat.id), n)
    if not rows:
        await update.message.reply_text("No entries yet.")
        return
    lines = []
    for _id, ts_str, amount, desc in rows:
        lines.append(f"{ts_str} | {amount:+.2f} | {desc}")
    await update.message.reply_text("\n".join(lines))


async def cmd_sum(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return
    args = (update.message.text.split()[1:] if update.message and update.message.text else [])
    period = (args[0] if args else "month").lower()
    total = await asyncio.to_thread(sum_period, int(chat.id), period)
    await update.message.reply_text(f"Total ({period}): {total:+.2f}")


async def cmd_undo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return
    row = await asyncio.to_thread(delete_last, int(chat.id))
    if not row:
        await update.message.reply_text("Nothing to undo.")
        return
    _id, ts_str, amount, desc = row
    await update.message.reply_text(f"Removed: {ts_str} | {amount:+.2f} | {desc}")


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return
    args = (update.message.text.split()[1:] if update.message and update.message.text else [])
    period = args[0] if args else None
    # Build CSV in memory via thread
    def build_csv() -> bytes:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp","chatid","amount","description"])
        for ts_str, chatid, amount, desc in fetch_for_export(int(chat.id), period):
            writer.writerow([ts_str, chatid, f"{amount:+.2f}", desc])
        return output.getvalue().encode("utf-8")

    data = await asyncio.to_thread(build_csv)
    bio = io.BytesIO(data)
    fname = f"transactions_{chat.id}.csv"
    bio.name = fname
    await context.bot.send_document(chat_id=chat.id, document=bio, filename=fname, caption="Export CSV")


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Generate plots and a PDF report for day|week|month|year
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    chat = update.effective_chat
    if not chat or not (update.message and update.message.text):
        return
    parts = update.message.text.split()
    if len(parts) < 2 or parts[1].lower() not in {"day", "week", "month", "year"}:
        await update.message.reply_text("Usage: /report day|week|month|year")
        return
    period = parts[1].lower()

    # Fetch data for the period using export iterator
    rows = list(await asyncio.to_thread(lambda: list(fetch_for_export(int(chat.id), period))))
    if not rows:
        await update.message.reply_text("No data for the selected period.")
        return

    # Parse rows into structures
    # rows: (ts_str, chatid, amount, desc)
    parsed = []
    for ts_str, _cid, amount, desc in rows:
        ts = dt.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        parsed.append((ts, float(amount), desc))
    parsed.sort(key=lambda x: x[0])

    # Build daily aggregation
    from collections import defaultdict
    daily_net = defaultdict(float)
    daily_income = defaultdict(float)
    daily_exp = defaultdict(float)
    by_desc_exp = defaultdict(float)
    for ts, amt, desc in parsed:
        d = ts.date()
        daily_net[d] += amt
        if amt >= 0:
            daily_income[d] += amt
        else:
            daily_exp[d] += amt
            by_desc_exp[desc] += abs(amt)

    # Sort days
    days = sorted(daily_net.keys())
    x = days
    income_y = [daily_income[d] for d in x]
    exp_y = [daily_exp[d] for d in x]
    net_cum = []
    s = 0.0
    for d in x:
        s += daily_net[d]
        net_cum.append(s)

    # 1) Cumulative net line plot
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, net_cum, marker="o")
    ax1.set_title(f"Cumulative net ({period})")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Amount")
    ax1.grid(True, alpha=0.3)
    for label in ax1.get_xticklabels():
        label.set_rotation(90)
        label.set_horizontalalignment("center")
    fig1.tight_layout()

    # 2) Daily bars: income (green) and expenses (red)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(x, income_y, color="#2e7d32", label="Income")
    ax2.bar(x, exp_y, color="#c62828", label="Expenses")
    ax2.set_title(f"Daily income/expenses ({period})")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Amount")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    for label in ax2.get_xticklabels():
        label.set_rotation(90)
        label.set_horizontalalignment("center")
    fig2.tight_layout()

    # 3) Top expense categories pie
    top_items = sorted(by_desc_exp.items(), key=lambda kv: kv[1], reverse=True)
    top5 = top_items[:5]
    other_sum = sum(v for _, v in top_items[5:])
    labels = [k for k, _ in top5] + ((["Others"] if other_sum > 0 else []))
    values = [v for _, v in top5] + (([other_sum] if other_sum > 0 else []))
    if values:
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax3.set_title(f"Top expense categories ({period})")
        fig3.tight_layout()
    else:
        fig3 = None

    # Export PDF
    pdf_bytes = io.BytesIO()
    with PdfPages(pdf_bytes) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        if fig3 is not None:
            pdf.savefig(fig3)
        # Summary page
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        ax4.axis("off")
        total_income = sum(max(0.0, a) for _, a, _ in parsed)
        total_exp = -sum(min(0.0, a) for _, a, _ in parsed)
        net = total_income - total_exp
        text = (
            f"Report ({period})\n\n"
            f"Entries: {len(parsed)}\n"
            f"Income: +{total_income:.2f}\n"
            f"Expenses: -{total_exp:.2f}\n"
            f"Net: {net:+.2f}\n"
        )
        ax4.text(0.05, 0.95, text, va="top", ha="left", fontsize=12)
        pdf.savefig(fig4)
    pdf_bytes.seek(0)

    # Send images
    images = []
    for fig in [fig1, fig2, fig3]:
        if fig is None:
            continue
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        images.append(buf)

    # Send photos individually to keep captions simple
    for i, img in enumerate(images, 1):
        try:
            await context.bot.send_photo(chat_id=chat.id, photo=img, caption=f"Report {period} ({i}/{len(images)})")
        except Exception:
            pass

    # Send PDF
    pdf_bytes.name = f"report_{period}_{chat.id}.pdf"
    await context.bot.send_document(chat_id=chat.id, document=pdf_bytes, filename=pdf_bytes.name, caption=f"Report {period}")


async def cmd_import(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Import JSON transactions from a file, a replied file, or inline JSON."""
    chat = update.effective_chat
    if not chat:
        return
    # Try sources in order: document in this message, replied document, inline JSON text after command
    content_bytes: bytes | None = None
    # 1) This message has a document
    if update.message and update.message.document:
        try:
            file = await context.bot.get_file(update.message.document.file_id)
            content_bytes = await file.download_as_bytearray()
        except Exception:
            content_bytes = None
    # 2) Replied message has a document
    if content_bytes is None and update.message and update.message.reply_to_message and update.message.reply_to_message.document:
        try:
            file = await context.bot.get_file(update.message.reply_to_message.document.file_id)
            content_bytes = await file.download_as_bytearray()
        except Exception:
            content_bytes = None
    # 3) Inline JSON in message text after command
    if content_bytes is None and update.message and update.message.text:
        parts = update.message.text.split(" ", 1)
        if len(parts) > 1:
            content_bytes = parts[1].encode("utf-8", "ignore")

    if not content_bytes:
        await update.message.reply_text("Send a JSON file or reply to one with /import")
        return

    try:
        data = json.loads(content_bytes.decode("utf-8", "ignore"))
    except Exception:
        await update.message.reply_text("Invalid JSON.")
        return
    if not isinstance(data, list):
        await update.message.reply_text("JSON must be a list of objects.")
        return

    rows = []
    inserted = 0
    skipped = 0
    for obj in data:
        try:
            date_str = obj.get("data") or obj.get("date")
            amt_raw = obj.get("importo") if "importo" in obj else obj.get("amount")
            op = (obj.get("operazione") or obj.get("descrizione") or obj.get("description") or "").strip()
            cat = (obj.get("categoria") or obj.get("category") or "").strip()
            if not date_str or amt_raw is None:
                skipped += 1
                continue
            ts = dt.datetime.strptime(str(date_str), "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            amount = Decimal(str(amt_raw))
            # Keep only the category as description (fallback to operation if category missing)
            desc = cat if cat else (op or "")
            rows.append((ts, amount, desc))
        except Exception:
            skipped += 1
    # Limit batch size for safety
    MAX_BATCH = 5000
    rows = rows[:MAX_BATCH]
    if not rows:
        await update.message.reply_text("Nothing to import.")
        return
    try:
        inserted = await asyncio.to_thread(bulk_insert_transactions, int(chat.id), rows)
    except Exception as e:
        await update.message.reply_text(f"Import failed: {e}")
        return
    await update.message.reply_text(f"Imported {inserted} entrie(s). Skipped {skipped}.")


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat or not (update.message and update.message.text):
        return
    parts = update.message.text.split()
    if len(parts) < 2 or parts[1].lower() not in {"day", "month", "all"}:
        await update.message.reply_text("Usage: /reset day|month|all")
        return
    period = parts[1].lower()
    try:
        deleted = await asyncio.to_thread(delete_period, int(chat.id), period)
    except ValueError:
        await update.message.reply_text("Usage: /reset day|month|all")
        return
    await update.message.reply_text(f"Deleted {deleted} entrie(s) for {period}.")


async def cmd_month(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat or not (update.message and update.message.text):
        return
    parts = update.message.text.split()
    ym = parts[1] if len(parts) > 1 else None
    import datetime as _dt
    if not ym:
        now = _dt.datetime.utcnow()
        year, month = now.year, now.month
    else:
        try:
            year_s, month_s = ym.split("-")
            year, month = int(year_s), int(month_s)
            if not (1 <= month <= 12):
                raise ValueError
        except Exception:
            await update.message.reply_text("Usage: /month YYYY-MM")
            return

    summary = await asyncio.to_thread(month_summary, int(chat.id), year, month)
    lines = [
        f"Summary {year:04d}-{month:02d}",
        f"Entries: {summary['count']}",
        f"Income: +{summary['income']:.2f}",
        f"Expenses: {summary['expenses']:.2f}",
        f"Net: {summary['net']:+.2f}",
    ]
    if summary.get("days"):
        lines.append("By day:")
        for d, s in summary["days"]:
            lines.append(f"- {d}: {s:+.2f}")
    await update.message.reply_text("\n".join(lines))


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
            # Ping first to avoid long hangs if host is down
            alive = False
            try:
                alive = ping_host(h, timeout=2.0)
            except Exception:
                alive = False
            if not alive:
                logger.warning("Skip model pull: Ollama host not reachable: %s", h)
                continue

            client = OllamaClient(host=h)
            try:
                logger.info("Ensuring model is available on %s: %s", h, client.model)
                # Use a bounded timeout per pull request to avoid long hangs
                client.pull_model(timeout=180.0)
                logger.info("Model ready on %s: %s", h, client.model)
            except Exception as e:
                logger.warning("Could not pull model on %s: %s", h, e)

    async def post_init(application: Application) -> None:

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

        await sync_commands(application)

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

        # Periodically refresh commands to ensure they stay updated
        async def _commands_refresher(app: Application) -> None:
            while True:
                try:
                    await asyncio.sleep(6 * 60 * 60)
                    await sync_commands(app)
                except Exception:
                    # keep looping regardless of transient errors
                    await asyncio.sleep(60)

        try:
            application.create_task(_commands_refresher(application))
        except Exception as e:
            logger.warning("Could not start commands refresher: %s", e)

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
                # Generate and parse; provide description candidates for this chat
                candidates = await asyncio.to_thread(fetch_description_candidates, int(job.chat_id), 50)
                result = await asyncio.to_thread(to_csv_or_nd, job.text, client, candidates)
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
    app.add_handler(CommandHandler("last", cmd_last))
    app.add_handler(CommandHandler("sum", cmd_sum))
    app.add_handler(CommandHandler("undo", cmd_undo))
    app.add_handler(CommandHandler("export", cmd_export))
    app.add_handler(CommandHandler("import", cmd_import))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("month", cmd_month))

    # Fallback echo
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    logger.info("Starting polling...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
