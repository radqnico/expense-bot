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
import re
import threading
from typing import Final, List, Tuple, Sequence, Set
import hashlib
import psycopg

from telegram import BotCommand, Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


LOG_LEVEL: Final = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
# Reduce third-party noise (httpx request logs)
_HTTPX_LVL = os.getenv("HTTPX_LOG_LEVEL", "WARNING").upper()
try:
    logging.getLogger("httpx").setLevel(getattr(logging, _HTTPX_LVL, logging.WARNING))
except Exception:
    pass
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
    fetch_transactions_in_range,
    update_transaction,
    delete_transaction_by_id,
    create_recurrent,
    list_recurrent,
    update_recurrent,
    deactivate_recurrent,
    get_active_recurrent_pending,
    mark_recurrent_ran,
)
from decimal import Decimal, InvalidOperation
from .parser import to_csv_or_nd
from .importer import read_any_excel_or_csv
from .quick import cmd_expense, cmd_income

INFERENCE_QUEUES_KEY = "inference_queues"  # dict[host]->Queue
INFERENCE_PROCESSING_KEY = "inference_processing"  # dict[host]->bool
HOSTS_KEY = "ollama_hosts"  # list[str]
HOST_RR_INDEX_KEY = "host_rr_index"
NAV_STATE_KEY = "nav_state"  # per-chat navigation state
OLLAMA_LOCK_KEY = "ollama_lock"  # global lock to serialize Ollama calls
MODEL_READY_KEY = "ollama_model_ready"  # per-host model availability


@dataclass
class InferenceJob:
    chat_id: int
    message_id: int
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
        ("expense", "Quick expense: /expense AMOUNT DESCRIPTION"),
        ("income", "Quick income: /income AMOUNT DESCRIPTION"),
        ("last", "List recent entries"),
        ("sum", "Sum by period (today/week/month/all)"),
        ("undo", "Delete last entry"),
        ("export", "Export CSV: /export YYYY-MM|day|week|month|year"),
        ("import", "Import JSON/Excel/CSV: send file or reply"),
        ("report", "Charts + PDF: /report day|week|month|year|YYYY-MM"),
        ("smartreport", "LLM filter: /smartreport PERIOD QUERY"),
        ("reset", "Reset entries: /reset day|month|all"),
        ("month", "Monthly summary: /month YYYY-MM"),
        ("navigation", "Browse/edit: /navigation YYYY-MM-DD YYYY-MM-DD"),
        ("recur_add", "Create recurrent: KIND AMOUNT DESCRIPTION"),
        ("recur_list", "List recurrent operations"),
        ("recur_delete", "Delete recurrent by ID"),
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
        "/expense AMOUNT DESCRIPTION - quick expense (no LLM)\n"
        "/income AMOUNT DESCRIPTION - quick income (no LLM)\n"
        "/last [n] - list last n entries (default 5)\n"
        "/sum [today|week|month|all] - sum amounts (default month)\n"
        "/undo - delete last entry\n"
        "/export [YYYY-MM|day|week|month|year] - CSV export\n"
        "/import - send a JSON or Excel/CSV file (or reply to one)\n"
        "/report day|week|month|year|YYYY-MM - charts + PDF\n"
        "/reset day|month|all - delete entries in period\n"
        "/month YYYY-MM - monthly summary\n"
        "/smartreport PERIOD QUERY - LLM filtered report by topic\n"
        "/navigation YYYY-MM-DD YYYY-MM-DD - browse/edit; /navigation to exit"
    )


async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("OK")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    hosts: list[str] = app.bot_data.get(HOSTS_KEY) or parse_ollama_hosts()
    queues: dict = app.bot_data.get(INFERENCE_QUEUES_KEY) or {}
    processing: dict = app.bot_data.get(INFERENCE_PROCESSING_KEY) or {}
    # Compute readiness on the fly per host (ping + model)
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
        model = "not-ready"
        if alive:
            try:
                client = OllamaClient(host=h)
                model = "ready" if await asyncio.to_thread(client.has_model, None, 5.0) else "not-ready"
            except Exception:
                model = "not-ready"
        extra = f", you at #{pos}" if pos is not None else ""
        lines.append(f"{h} — {status}, model {model}, queue {qsize}{extra}")

    text = "\n".join(lines) if lines else "No hosts configured"
    await update.message.reply_text(text)

# Recurrent operations UX-friendly commands
_ALLOWED_REC_KIND = {"daily", "weekly", "monthly", "yearly"}


def _recur_usage() -> str:
    return (
        "Usage:\n"
        "/recur_add KIND AMOUNT DESCRIPTION\n"
        "- KIND: daily|weekly|monthly|yearly\n"
        "- AMOUNT: decimal with dot (e.g., -50.00)\n"
        "Example: /recur_add monthly -50.00 abbonamento palestra\n\n"
        "/recur_list [all]\n"
        "/recur_delete ID\n"
        ""
    )


async def cmd_recur_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat or not (update.message and update.message.text):
        return
    parts = update.message.text.split(maxsplit=3)
    if len(parts) < 4:
        await update.message.reply_text(_recur_usage())
        return
    kind = parts[1].strip().lower()
    if kind not in _ALLOWED_REC_KIND:
        await update.message.reply_text("Invalid KIND. Allowed: daily, weekly, monthly, yearly\n\n" + _recur_usage())
        return
    try:
        amount = Decimal(parts[2].strip())
    except Exception:
        await update.message.reply_text("Invalid AMOUNT. Use a decimal like -50.00\n\n" + _recur_usage())
        return
    desc = parts[3].strip()
    if not desc:
        await update.message.reply_text("Missing DESCRIPTION.\n\n" + _recur_usage())
        return

    rid = await asyncio.to_thread(create_recurrent, int(chat.id), kind, amount, desc)
    try:
        await asyncio.to_thread(insert_transaction, int(chat.id), amount, desc)
    except Exception:
        pass
    await update.message.reply_text(
        f"✅ Recurrent created (ID {rid}): {kind} {amount:+.2f} {desc}\n"
        f"A corresponding transaction has been inserted."
    )


async def cmd_recur_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return
    args = (update.message.text.split()[1:] if update.message and update.message.text else [])
    include_inactive = bool(args and args[0].lower() == "all")
    rows = await asyncio.to_thread(list_recurrent, int(chat.id), include_inactive)
    if not rows:
        await update.message.reply_text("No recurrent operations.")
        return
    lines = ["ID | kind | amount | active | description"]
    for rid, kind, amount, desc, active in rows:
        lines.append(f"{rid} | {kind} | {amount:+.2f} | {'yes' if active else 'no'} | {desc}")
    footer = "\nUse /recur_delete ID to deactivate."
    await update.message.reply_text("\n".join(lines) + footer)


async def cmd_recur_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat or not (update.message and update.message.text):
        return
    parts = update.message.text.split()
    if len(parts) < 2:
        await update.message.reply_text("Usage: /recur_delete ID")
        return
    try:
        rid = int(parts[1])
    except Exception:
        await update.message.reply_text("ID must be an integer. Usage: /recur_delete ID")
        return
    ok = await asyncio.to_thread(deactivate_recurrent, int(chat.id), rid)
    if ok:
        await update.message.reply_text(f"✅ Recurrent {rid} deactivated.")
    else:
        await update.message.reply_text("Not found or already inactive.")


# (Recurrent navigation UI removed)




# Recurrent operations UX-friendly commands
_ALLOWED_REC_KIND = {"daily", "weekly", "monthly", "yearly"}


def _recur_usage() -> str:
    return (
        "Usage:\n"
        "/recur_add KIND AMOUNT DESCRIPTION\n"
        "- KIND: daily|weekly|monthly|yearly\n"
        "- AMOUNT: decimal with dot (e.g., -50.00)\n"
        "Example: /recur_add monthly -50.00 abbonamento palestra\n\n"
        "/recur_list [all]\n"
        "/recur_delete ID\n"
        ""
    )


async def cmd_recur_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat or not (update.message and update.message.text):
        return
    parts = update.message.text.split(maxsplit=3)
    if len(parts) < 4:
        await update.message.reply_text(_recur_usage())
        return
    kind = parts[1].strip().lower()
    if kind not in _ALLOWED_REC_KIND:
        await update.message.reply_text("Invalid KIND. Allowed: daily, weekly, monthly, yearly\n\n" + _recur_usage())
        return
    try:
        amount = Decimal(parts[2].strip())
    except Exception:
        await update.message.reply_text("Invalid AMOUNT. Use a decimal like -50.00\n\n" + _recur_usage())
        return
    desc = parts[3].strip()
    if not desc:
        await update.message.reply_text("Missing DESCRIPTION.\n\n" + _recur_usage())
        return

    rid = await asyncio.to_thread(create_recurrent, int(chat.id), kind, amount, desc)
    try:
        await asyncio.to_thread(insert_transaction, int(chat.id), amount, desc)
    except Exception:
        pass
    await update.message.reply_text(
        f"✅ Recurrent created (ID {rid}): {kind} {amount:+.2f} {desc}\n"
        f"A corresponding transaction has been inserted."
    )


async def cmd_recur_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return
    args = (update.message.text.split()[1:] if update.message and update.message.text else [])
    include_inactive = bool(args and args[0].lower() == "all")
    rows = await asyncio.to_thread(list_recurrent, int(chat.id), include_inactive)
    if not rows:
        await update.message.reply_text("No recurrent operations.")
        return
    lines = ["ID | kind | amount | active | description"]
    for rid, kind, amount, desc, active in rows:
        lines.append(f"{rid} | {kind} | {amount:+.2f} | {'yes' if active else 'no'} | {desc}")
    footer = "\nUse /recur_delete ID to deactivate."
    await update.message.reply_text("\n".join(lines) + footer)


async def cmd_recur_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat or not (update.message and update.message.text):
        return
    parts = update.message.text.split()
    if len(parts) < 2:
        await update.message.reply_text("Usage: /recur_delete ID")
        return
    try:
        rid = int(parts[1])
    except Exception:
        await update.message.reply_text("ID must be an integer. Usage: /recur_delete ID")
        return
    ok = await asyncio.to_thread(deactivate_recurrent, int(chat.id), rid)
    if ok:
        await update.message.reply_text(f"✅ Recurrent {rid} deactivated.")
    else:
        await update.message.reply_text("Not found or already inactive.")


def _choose_ollama_client(app: Application) -> OllamaClient:
    hosts: list[str] = app.bot_data.get(HOSTS_KEY) or parse_ollama_hosts()
    app.bot_data[HOSTS_KEY] = hosts
    # Optimistically pick first; try to prefer alive host via round-robin if available
    try:
        alive_list = [ping_host(h) for h in hosts]
        alive_hosts = [h for h, alive in zip(hosts, alive_list) if alive]
        if alive_hosts:
            rr = app.bot_data.get(HOST_RR_INDEX_KEY, 0)
            host = alive_hosts[rr % len(alive_hosts)]
            app.bot_data[HOST_RR_INDEX_KEY] = rr + 1
            return OllamaClient(host=host)
    except Exception:
        pass
    return OllamaClient(host=hosts[0] if hosts else None)


class _LockedOllama:
    def __init__(self, inner: OllamaClient, lock: "threading.Lock"):
        self._inner = inner
        self._lock = lock
        self.model = inner.model
        self.host = inner.host

    def generate(self, prompt: str) -> str:
        with self._lock:
            return self._inner.generate(prompt)


def _llm_select_indices(client: OllamaClient, topic: str, items: Sequence[str]) -> Set[int]:
    """Ask the LLM to select item indices (1-based) strictly matching the topic.

    Conservative filter: prefer precision over recall. On failure returns empty set.
    """
    if not items:
        return set()
    lines = [f"{i+1}) {s}" for i, s in enumerate(items)]
    prompt = (
        "Dato un TEMA, seleziona SOLO gli elementi la cui descrizione è chiaramente "
        "e direttamente pertinente a quel tema. SII MOLTO SEVERO: meglio escludere "
        "che includere falsi positivi. Se hai dubbi, NON selezionare.\n\n"
        f"Tema: \"{topic.strip()}\"\n\n"
        "Criteri:\n"
        "- Usa il significato, non solo la somiglianza di parole.\n"
        "- Considera sinonimi stretti del tema.\n"
        "- Escludi concetti diversi con parole simili (no omonimie).\n"
        "- Esempio (tema 'cibo'): scegli pranzo, ristorante, pizza, alimentari, spesa, caffè;\n"
        "  NON scegliere carburante, benzina, gasolio, olio motore, farmaci, bollette, arredo.\n\n"
        "Elenco (numero) descrizione:\n"
        + "\n".join(lines)
        + "\n\n"
        "Rispondi SOLO con i numeri separati da virgola (es: 1,3,8).\n"
        "Se nessuno, rispondi SOLO: NONE\n"
        "Output:\n"
    )
    try:
        out = client.generate(prompt).strip()
    except Exception:
        return set()
    if not out:
        return set()
    out = out.strip().strip("` ").strip()
    if out.upper().startswith("NONE"):
        return set()
    nums: Set[int] = set()
    for tok in out.replace("\n", ",").split(","):
        tok = tok.strip()
        if tok.isdigit():
            n = int(tok)
            if 1 <= n <= len(items):
                nums.add(n)
    return nums
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
    if len(parts) < 2:
        await update.message.reply_text("Usage: /report day|week|month|year|YYYY-MM")
        return
    arg = parts[1].strip()
    if re.fullmatch(r"\d{4}-\d{2}", arg):
        period = arg
    else:
        arg_l = arg.lower()
        if arg_l not in {"day", "week", "month", "year"}:
            await update.message.reply_text("Usage: /report day|week|month|year|YYYY-MM")
            return
        period = arg_l

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
        plt.close(fig4)

        # Transactions list pages (monospace), paginated
        header = "Timestamp           Amount    Description"
        lines = [
            f"{ts.strftime('%Y-%m-%d %H:%M'):19}  {a:+9.2f}  {d}"
            for ts, a, d in parsed
        ]
        # Reasonable number of lines per page considering margins/font size
        LPP = 36
        if lines:
            for i in range(0, len(lines), LPP):
                chunk = lines[i : i + LPP]
                figp, axp = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait size in inches
                axp.axis("off")
                page_text = "\n".join([header, "-" * len(header)] + chunk)
                axp.text(
                    0.03,
                    0.97,
                    page_text,
                    va="top",
                    ha="left",
                    family="monospace",
                    fontsize=9,
                )
                pdf.savefig(figp)
                plt.close(figp)
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


async def cmd_smartreport(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """LLM-filtered report: /smartreport PERIOD QUERY (runs in background)."""
    chat = update.effective_chat
    if not chat or not (update.message and update.message.text):
        return
    parts = update.message.text.split(maxsplit=2)
    if len(parts) < 3:
        await update.message.reply_text("Usage: /smartreport YYYY-MM|day|week|month|year QUERY")
        return
    period_arg = parts[1].strip()
    query = parts[2].strip()

    # Immediate feedback
    try:
        await update.message.reply_text(
            f"🧠 Smart report richiesto per {period_arg} — \"{query}\"."
            "\nCi vorrà un po', ti invierò il risultato appena pronto."
        )
    except Exception:
        pass

    # Run the heavy job in background
    async def _job(app: Application, chat_id: int, period: str, q: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        rows = list(await asyncio.to_thread(lambda: list(fetch_for_export(int(chat_id), period))))
        if not rows:
            try:
                await context.bot.send_message(chat_id=chat_id, text="No data for the selected period.")
            except Exception:
                pass
            return

        lock = app.bot_data.get(OLLAMA_LOCK_KEY)
        if lock is None:
            lock = threading.Lock()
            app.bot_data[OLLAMA_LOCK_KEY] = lock
        client = _LockedOllama(_choose_ollama_client(app), lock)
        keep_flags: list[bool] = []
        CHUNK = 60
        for i in range(0, len(rows), CHUNK):
            chunk = rows[i : i + CHUNK]
            descs = [d for (_ts, _cid, _amt, d) in chunk]
            indices = await asyncio.to_thread(_llm_select_indices, client, q, descs)
            sel = [(j + 1) in indices for j in range(len(chunk))]
            keep_flags.extend(sel)

        filtered = [r for r, keep in zip(rows, keep_flags) if keep]
        if not filtered:
            try:
                await context.bot.send_message(chat_id=chat_id, text="Nessuna transazione pertinente trovata per il tema richiesto.")
            except Exception:
                pass
            return

        parsed = []
        for ts_str, _cid, amount, desc in filtered:
            ts = dt.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            parsed.append((ts, float(amount), desc))
        parsed.sort(key=lambda x: x[0])

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

        days = sorted(daily_net.keys())
        x = days
        income_y = [daily_income[d] for d in x]
        exp_y = [daily_exp[d] for d in x]
        net_cum = []
        s = 0.0
        for d in x:
            s += daily_net[d]
            net_cum.append(s)

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(x, net_cum, marker="o")
        ax1.set_title(f"Smart cumulative net ({period}) — {q}")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Amount")
        ax1.grid(True, alpha=0.3)
        for label in ax1.get_xticklabels():
            label.set_rotation(90)
            label.set_horizontalalignment("center")
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.bar(x, income_y, color="#2e7d32", label="Income")
        ax2.bar(x, exp_y, color="#c62828", label="Expenses")
        ax2.set_title(f"Smart daily income/expenses ({period}) — {q}")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Amount")
        ax2.legend()
        ax2.grid(True, axis="y", alpha=0.3)
        for label in ax2.get_xticklabels():
            label.set_rotation(90)
            label.set_horizontalalignment("center")
        fig2.tight_layout()

        top_items = sorted(by_desc_exp.items(), key=lambda kv: kv[1], reverse=True)
        top5 = top_items[:5]
        other_sum = sum(v for _, v in top_items[5:])
        labels = [k for k, _ in top5] + ((["Others"] if other_sum > 0 else []))
        values = [v for _, v in top5] + (([other_sum] if other_sum > 0 else []))
        fig3 = None
        if values:
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            ax3.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
            ax3.set_title(f"Top expense categories ({period}) — {q}")
            fig3.tight_layout()

        pdf_bytes = io.BytesIO()
        with PdfPages(pdf_bytes) as pdf:
            pdf.savefig(fig1)
            pdf.savefig(fig2)
            if fig3 is not None:
                pdf.savefig(fig3)
            fig4, ax4 = plt.subplots(figsize=(7, 4))
            ax4.axis("off")
            total_income = sum(max(0.0, a) for _, a, _ in parsed)
            total_exp = -sum(min(0.0, a) for _, a, _ in parsed)
            net = total_income - total_exp
            text = (
                f"Smart Report ({period}) — {q}\n\n"
                f"Entries: {len(parsed)}\n"
                f"Income: +{total_income:.2f}\n"
                f"Expenses: -{total_exp:.2f}\n"
                f"Net: {net:+.2f}\n"
            )
            ax4.text(0.05, 0.95, text, va="top", ha="left", fontsize=12)
            pdf.savefig(fig4)
            plt.close(fig4)
            header = "Timestamp           Amount    Description"
            lines = [
                f"{ts.strftime('%Y-%m-%d %H:%M'):19}  {a:+9.2f}  {d}"
                for ts, a, d in parsed
            ]
            LPP = 36
            if lines:
                for i in range(0, len(lines), LPP):
                    chunk = lines[i : i + LPP]
                    figp, axp = plt.subplots(figsize=(8.27, 11.69))
                    axp.axis("off")
                    page_text = "\n".join([header, "-" * len(header)] + chunk)
                    axp.text(0.03, 0.97, page_text, va="top", ha="left", family="monospace", fontsize=9)
                    pdf.savefig(figp)
                    plt.close(figp)
        pdf_bytes.seek(0)

        images = []
        for fig in [fig1, fig2, fig3]:
            if fig is None:
                continue
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            images.append(buf)
        for i, img in enumerate(images, 1):
            try:
                await context.bot.send_photo(chat_id=chat_id, photo=img, caption=f"Smart report {period} ({i}/{len(images)}) — {q}")
            except Exception:
                pass
        pdf_bytes.name = f"smartreport_{period}_{chat_id}.pdf"
        await context.bot.send_document(chat_id=chat_id, document=pdf_bytes, filename=pdf_bytes.name, caption=f"Smart report {period} — {q}")

    context.application.create_task(_job(context.application, int(chat.id), period_arg, query))


def _nav_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [["Prev", "Next"], ["Edit", "Delete"], ["Exit"]], resize_keyboard=True
    )


async def cmd_navigation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return
    parts = update.message.text.split()
    # Exit nav mode if no params
    if len(parts) == 1:
        context.chat_data.pop(NAV_STATE_KEY, None)
        await update.message.reply_text("Navigation ended.", reply_markup=ReplyKeyboardRemove())
        return
    if len(parts) != 3:
        await update.message.reply_text("Usage: /navigation YYYY-MM-DD YYYY-MM-DD")
        return
    try:
        start = dt.date.fromisoformat(parts[1])
        end = dt.date.fromisoformat(parts[2])
    except Exception:
        await update.message.reply_text("Invalid dates. Use YYYY-MM-DD YYYY-MM-DD")
        return
    if end < start:
        await update.message.reply_text("End date must be >= start date.")
        return

    rows = await asyncio.to_thread(fetch_transactions_in_range, int(chat.id), start, end)
    if not rows:
        await update.message.reply_text("No transactions in the given range.")
        return
    # Save state
    context.chat_data[NAV_STATE_KEY] = {"start": str(start), "end": str(end), "rows": rows, "idx": 0, "await_edit": False}
    await _nav_show_current(update, context)


async def _nav_show_current(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return
    state = context.chat_data.get(NAV_STATE_KEY)
    if not state:
        return
    rows = state["rows"]
    idx = state.get("idx", 0)
    idx = max(0, min(idx, len(rows) - 1))
    state["idx"] = idx
    tx_id, ts_str, amount, desc = rows[idx]
    text = (
        f"[{idx+1}/{len(rows)}]\n"
        f"ID: {tx_id}\n"
        f"Date: {ts_str}\n"
        f"Amount: {Decimal(amount):+.2f}\n"
        f"Description: {desc}"
    )
    await update.message.reply_text(text, reply_markup=_nav_keyboard())


async def handle_navigation_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat or not update.message or not update.message.text:
        return
    state = context.chat_data.get(NAV_STATE_KEY)
    if not state:
        return  # not in navigation mode
    text = update.message.text.strip()

    # If awaiting edit, expect "amount,description" or "cancel"
    if state.get("await_edit"):
        if text.lower() in {"cancel", "annulla"}:
            state["await_edit"] = False
            await update.message.reply_text("Edit cancelled.", reply_markup=_nav_keyboard())
            return
        if "," not in text:
            await update.message.reply_text("Send: amount,description or 'cancel'", reply_markup=_nav_keyboard())
            return
        try:
            amount_str, description = text.split(",", 1)
            amount = Decimal(amount_str.strip())
            description = description.strip()
        except Exception:
            await update.message.reply_text("Invalid format. Use amount,description", reply_markup=_nav_keyboard())
            return
        rows = state["rows"]
        idx = state.get("idx", 0)
        tx_id, ts_str, _old_amount, _old_desc = rows[idx]
        ok = await asyncio.to_thread(update_transaction, int(chat.id), int(tx_id), amount, description)
        if not ok:
            await update.message.reply_text("Update failed.", reply_markup=_nav_keyboard())
            return
        # Update local state
        rows[idx] = (tx_id, ts_str, amount, description)
        state["await_edit"] = False
        await update.message.reply_text("Updated.", reply_markup=_nav_keyboard())
        await _nav_show_current(update, context)
        return

    # Not awaiting edit: handle commands
    cmd = text.lower()
    rows = state["rows"]
    idx = state.get("idx", 0)
    if cmd == "prev":
        if idx > 0:
            state["idx"] = idx - 1
        await _nav_show_current(update, context)
        return
    if cmd == "next":
        if idx < len(rows) - 1:
            state["idx"] = idx + 1
        await _nav_show_current(update, context)
        return
    if cmd == "edit":
        state["await_edit"] = True
        await update.message.reply_text("Send new: amount,description (or 'cancel')", reply_markup=_nav_keyboard())
        return
    if cmd == "delete":
        tx_id, *_ = rows[idx]
        ok = await asyncio.to_thread(delete_transaction_by_id, int(chat.id), int(tx_id))
        if not ok:
            await update.message.reply_text("Delete failed.")
            return
        # Remove from state
        rows.pop(idx)
        if not rows:
            context.chat_data.pop(NAV_STATE_KEY, None)
            await update.message.reply_text("No more transactions. Navigation ended.", reply_markup=ReplyKeyboardRemove())
            return
        if idx >= len(rows):
            state["idx"] = len(rows) - 1
        await update.message.reply_text("Deleted.")
        await _nav_show_current(update, context)
        return
    if cmd == "exit":
        context.chat_data.pop(NAV_STATE_KEY, None)
        await update.message.reply_text("Navigation ended.", reply_markup=ReplyKeyboardRemove())
        return
    # Otherwise, do nothing so other handlers (echo) can process normal text
    return


async def cmd_import(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Import JSON transactions from a file, a replied file, or inline JSON."""
    chat = update.effective_chat
    if not chat:
        return
    # Try sources in order: document in this message, replied document, inline JSON text after command
    content_bytes: bytes | None = None
    filename: str | None = None
    # 1) This message has a document
    if update.message and update.message.document:
        try:
            file = await context.bot.get_file(update.message.document.file_id)
            content_bytes = await file.download_as_bytearray()
            filename = update.message.document.file_name or None
        except Exception:
            content_bytes = None
    # 2) Replied message has a document
    if content_bytes is None and update.message and update.message.reply_to_message and update.message.reply_to_message.document:
        try:
            file = await context.bot.get_file(update.message.reply_to_message.document.file_id)
            content_bytes = await file.download_as_bytearray()
            filename = update.message.reply_to_message.document.file_name or None
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

    rows = []
    inserted = 0
    skipped = 0
    if filename and any(filename.lower().endswith(ext) for ext in (".xlsx", ".xls", ".csv")):
        # Excel/CSV path
        try:
            records = await asyncio.to_thread(read_any_excel_or_csv, bytes(content_bytes), filename)
        except Exception as e:
            await update.message.reply_text(f"Failed to parse file: {e}")
            return
        for obj in records:
            try:
                date_str = obj.get("data")
                amt_raw = obj.get("importo")
                cat = (obj.get("categoria") or "").strip()
                op = (obj.get("operazione") or "").strip()
                if not date_str or amt_raw is None:
                    skipped += 1
                    continue
                ts = dt.datetime.strptime(str(date_str), "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
                amount = Decimal(str(amt_raw))
                desc = cat if cat else (op or "")
                rows.append((ts, amount, desc))
            except Exception:
                skipped += 1
    else:
        # JSON path
        try:
            data = json.loads(content_bytes.decode("utf-8", "ignore"))
        except Exception:
            await update.message.reply_text("Invalid JSON.")
            return
        if not isinstance(data, list):
            await update.message.reply_text("JSON must be a list of objects.")
            return
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

    # If in navigation mode, do not insert/parse transactions
    if context.chat_data.get(NAV_STATE_KEY):
        try:
            await update.message.reply_text(
                "Navigation mode is active. Use /navigation to exit before inserting transactions."
            )
        except Exception:
            pass
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

    await q.put(InferenceJob(chat_id=chat.id, message_id=update.message.message_id, text=text))


def get_token() -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set")
        raise SystemExit(2)
    return token


def main() -> None:
    token = get_token()

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
            globals()["_BOT_SINGLETON_CONN_MAIN"] = conn
            logger.info("Singleton lock acquired")
        except Exception as e:
            logger.warning("Could not acquire singleton lock: %s", e)

    _acquire_singleton_or_exit(token)

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

        # Initialize hosts and per-host queues without clobbering any early state
        hosts = application.bot_data.get(HOSTS_KEY) or parse_ollama_hosts()
        application.bot_data[HOSTS_KEY] = hosts
        queues = application.bot_data.get(INFERENCE_QUEUES_KEY) or {}
        application.bot_data[INFERENCE_QUEUES_KEY] = queues
        processing = application.bot_data.get(INFERENCE_PROCESSING_KEY) or {}
        application.bot_data[INFERENCE_PROCESSING_KEY] = processing
        for h in hosts:
            if h not in queues:
                queues[h] = asyncio.Queue()
            processing.setdefault(h, False)

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

    # Pre-initialize hosts and queues so early messages before post_init don't get lost
    try:
        hosts = parse_ollama_hosts()
        app.bot_data[HOSTS_KEY] = hosts
        app.bot_data[INFERENCE_QUEUES_KEY] = {h: asyncio.Queue() for h in hosts}
        app.bot_data[INFERENCE_PROCESSING_KEY] = {h: False for h in hosts}
    except Exception:
        # Best-effort: leave empty; post_init will populate
        pass

    # Create a single worker to process messages sequentially through Ollama
    async def worker(application: Application, host: str) -> None:
        q: asyncio.Queue = application.bot_data[INFERENCE_QUEUES_KEY][host]
        lock = application.bot_data.get("OLLAMA_LOCK_KEY")
        if lock is None:
            lock = __import__("threading").Lock()
            application.bot_data["OLLAMA_LOCK_KEY"] = lock
        client = _LockedOllama(OllamaClient(host=host), lock)
        while True:
            job: InferenceJob = await q.get()
            processing: dict = application.bot_data[INFERENCE_PROCESSING_KEY]
            processing[host] = True
            try:
                # Generate and parse; provide description candidates for this chat
                candidates = await asyncio.to_thread(fetch_description_candidates, int(job.chat_id), 50)
                result = await asyncio.to_thread(to_csv_or_nd, job.text, client, candidates)
                # Persist if CSV
                reply_text = None
                try:
                    if "," in result and result.upper() != "ND":
                        amount_str, description = result.split(",", 1)
                        amount = Decimal(amount_str.strip())
                        await asyncio.to_thread(
                            insert_transaction, int(job.chat_id), amount, description.strip()
                        )
                        kind = "Entrata" if amount >= 0 else "Spesa"
                        emoji = "💰" if amount >= 0 else "💸"
                        reply_text = (
                            f"✅ {emoji} {kind} registrata\n"
                            f"Importo: {amount:+.2f}\n"
                            f"Descrizione: {description.strip()}"
                        )
                    else:
                        reply_text = (
                            "⚠️ Non determinato. "
                            "Assicurati di includere un importo e una descrizione."
                        )
                except (InvalidOperation, Exception):
                    reply_text = (
                        "⚠️ Errore nell\'inserimento. "
                        "Controlla il formato e riprova."
                    )
                # Reply to user
                try:
                    await application.bot.send_message(
                        chat_id=job.chat_id,
                        text=reply_text or result,
                        reply_to_message_id=getattr(job, "message_id", None) or None,
                    )
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
    # Quick insert commands that bypass LLM
    app.add_handler(CommandHandler("expense", cmd_expense))
    app.add_handler(CommandHandler("income", cmd_income))
    app.add_handler(CommandHandler("import", cmd_import))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("recur_add", cmd_recur_add))
    app.add_handler(CommandHandler("recur_list", cmd_recur_list))
    app.add_handler(CommandHandler("recur_delete", cmd_recur_delete))
    app.add_handler(CommandHandler("smartreport", cmd_smartreport))
    app.add_handler(CommandHandler("navigation", cmd_navigation))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("month", cmd_month))

    # Fallback echo
    # Navigation input handler before echo, so it catches keyboard inputs.
    # Set block=False to allow normal echo processing when not in navigation mode.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_navigation_input, block=False), group=0)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo), group=1)

    logger.info("Starting polling...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
