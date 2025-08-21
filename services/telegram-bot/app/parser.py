from __future__ import annotations

from .llm import OllamaClient


PROMPT_TEMPLATE = """
Return exactly one line: either "amount,description" (CSV) or "ND".
Rules:
- amount: decimal with dot (convert commas), ignore currency/symbols; income positive, expense negative.
- description: short text without amount/currency.
- If no single clear amount or unrelated/unclear → ND.
Input: {text}
Output:
"""


def to_csv_or_nd(text: str, client: OllamaClient) -> str:
    prompt = PROMPT_TEMPLATE.format(text=text.strip())
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
