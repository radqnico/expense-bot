from __future__ import annotations

import asyncio
from decimal import Decimal
from telegram.ext import ContextTypes

from .db import insert_transaction


def parse_amount_token(tok: str) -> Decimal:
    s = (tok or "").strip().replace(",", ".")
    return Decimal(s)


async def cmd_expense(update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat or not (update.message and update.message.text):
        return
    parts = update.message.text.split(maxsplit=2)
    if len(parts) < 3:
        await update.message.reply_text("Usage: /expense AMOUNT DESCRIPTION")
        return
    try:
        amt = parse_amount_token(parts[1])
    except Exception:
        await update.message.reply_text("Invalid amount. Example: /expense 12.50 caffè")
        return
    amount = -abs(amt)
    desc = parts[2].strip()
    try:
        await asyncio.to_thread(insert_transaction, int(chat.id), amount, desc)
    except Exception as e:
        await update.message.reply_text(f"Insert failed: {e}")
        return
    await update.message.reply_text(
        f"✅ 💸 Spesa registrata\nImporto: {amount:+.2f}\nDescrizione: {desc}",
        reply_to_message_id=update.message.message_id,
    )


async def cmd_income(update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat or not (update.message and update.message.text):
        return
    parts = update.message.text.split(maxsplit=2)
    if len(parts) < 3:
        await update.message.reply_text("Usage: /income AMOUNT DESCRIPTION")
        return
    try:
        amt = parse_amount_token(parts[1])
    except Exception:
        await update.message.reply_text("Invalid amount. Example: /income 120 stipendio")
        return
    amount = abs(amt)
    desc = parts[2].strip()
    try:
        await asyncio.to_thread(insert_transaction, int(chat.id), amount, desc)
    except Exception as e:
        await update.message.reply_text(f"Insert failed: {e}")
        return
    await update.message.reply_text(
        f"✅ 💰 Entrata registrata\nImporto: {amount:+.2f}\nDescrizione: {desc}",
        reply_to_message_id=update.message.message_id,
    )

