from __future__ import annotations

from .llm import OllamaClient


BASE_PROMPT_TEMPLATE = """
Task: Parse a single human message about money and output exactly one line: either "<amount>,<description>" (CSV) or "ND".

Rules:
- amount: use a decimal with dot; convert commas to dot; strip currency symbols (€, eur, euro) and signs; thousands separators allowed.
- polarity: expenses negative (e.g., spesa, pagato, costo, acquisto), incomes positive (e.g., entrata, incasso, stipendio, rimborso, pagamento ricevuto). If unclear, do not guess.
- description: short text describing the item; remove amounts, currency, dates, and counts.
- multiple numbers: if there isn’t one clear amount, output ND.
- unrelated/unclear: output ND.
- Output must be exactly one line with no extra text or formatting.

If a list of existing descriptions is provided, prefer using one of them if the new description is similar; otherwise, output a concise new description.

Examples:
I: spesa 1,2 pranzo
O: -1.2,pranzo
I: entrata 2000 stipendio
O: 2000,stipendio
I: spesa maschera di merda 2.30
O: -2.3,maschera di merda
I: ho speso 12,50 € per pranzo
O: -12.5,pranzo
I: rimborso 15 biglietto
O: 15,biglietto
I: pagato bolletta luce 87,90 eur
O: -87.9,bolletta luce
I: incasso +120 consulenza
O: 120,consulenza
I: boh non so
O: ND
I: spesa 3 caffè 1.20
O: ND
"""


def to_csv_or_nd(text: str, client: OllamaClient, candidates: list[str] | None = None) -> str:
    text = text.strip()
    prompt = BASE_PROMPT_TEMPLATE
    if candidates:
        # Keep list compact; join with semicolons
        joined = "; ".join(c.strip() for c in candidates[:50] if c and c.strip())
        prompt += f"\nExisting descriptions (prefer exact reuse if similar):\n{joined}\n"
    prompt += f"\nNow convert:\nInput: {text}\nOutput:\n"
    try:
        out = client.generate(prompt)
    except Exception:
        return "ND"

    if not out:
        return "ND"

    # Post-process robustly: models often wrap output in code fences or add extra lines.
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]

    # Prefer the first plausible CSV line; ignore code fences like ``` or ```csv
    for raw in lines:
        if raw.startswith("```"):
            continue
        line = raw
        # Strip simple surrounding quotes/backticks
        if (line.startswith("`") and line.endswith("`")) or (
            line.startswith("\"") and line.endswith("\"")
        ):
            line = line[1:-1].strip()
        # Accept minimal sane CSV: contains a comma and at least one digit
        if "," in line and any(ch.isdigit() for ch in line):
            return line

    # If no CSV found, accept explicit ND on any line
    for ln in lines:
        if ln.upper() == "ND":
            return "ND"

    return "ND"
