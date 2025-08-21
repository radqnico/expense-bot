from __future__ import annotations

from .llm import OllamaClient


PROMPT_TEMPLATE = """
You convert a single user message about expenses or income into either:

1) A single CSV line: amount,description
2) Or the exact string: ND

Rules:
- amount: a positive decimal number (e.g., 12.50, 1,3, 2000). If the input uses a comma, convert it to a dot. Ignore currency symbols.
- description: a short phrase describing the item or income, without the amount or currency symbols.
- If there is no clear single amount, output ND.
- If the message meaning is unclear or unrelated to expenses/income, output ND.
- Expenses are negative, income positive.
- Output EXACTLY one line with either "amount,description" or "ND". No extra text.

Examples:
Input: spesa caffè 1.30
Output: -1.30,caffè

Input: entrata stipendio 2000
Output: 2000,stipendio

Input: ho speso per pranzo oggi 12,50 euro
Output: -12.50,pranzo

Input: boh non so
Output: ND

Now convert the following input:
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

    # Post-process: keep only first line and trim
    line = out.strip().splitlines()[0].strip()
    # Guard against models adding code fencing or quotes
    if line.startswith("`") and line.endswith("`"):
        line = line.strip("`")
    if line.startswith("\"") and line.endswith("\""):
        line = line.strip("\"")
    # Minimal sanity: if line contains a comma and a digit, accept, else ND
    if "," in line and any(ch.isdigit() for ch in line):
        return line
    if line.upper() == "ND":
        return "ND"
    return "ND"

