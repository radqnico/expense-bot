from __future__ import annotations

from typing import List, Dict, Optional, Tuple

import io
import re
import unicodedata

import pandas as pd


HEADER_ALIASES = {
    "data": {
        r"^data$",
        r"^date$",
        r"^data\s*operazione$",
        r"^valuta$",
        r"^data\s*$",
    },
    "importo": {
        r"^importo$",
        r"^amount$",
        r"^entrate?$",
        r"^uscite?$",
        r"^addebito$",
        r"^accredito$",
        r"^importo\s*€$",
        r"^importo\s*\(eur\)$",
        r"^importo\s*$",
    },
    "operazione": {
        r"^operazione$",
        r"^descrizione$",
        r"^causale$",
        r"^descrizione\s*operazione$",
        r"^note?$",
        r"^operazione\s*$",
    },
    "categoria": {
        r"^categoria$",
        r"^category$",
        r"^tag$",
        r"^classe$",
        r"^categoria\s*$",
        r"^categoria\s+$",
    },
}


def _norm_header(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s


def _match_headers(df: pd.DataFrame) -> Dict[str, str] | Dict[str, Tuple]:
    col_map: Dict[str, str] | Dict[str, Tuple] = {}
    normalized = {c: _norm_header(str(c)) for c in df.columns}

    for key, patterns in HEADER_ALIASES.items():
        for col, norm in normalized.items():
            if any(re.match(pat, norm) for pat in patterns):
                col_map[key] = col
                break

    if "importo" not in col_map:
        debit_col = next((c for c, n in normalized.items() if re.match(r"^(addebito|debit|uscite?)$", n)), None)
        credit_col = next((c for c, n in normalized.items() if re.match(r"^(accredito|credit|entrate?)$", n)), None)
        if debit_col or credit_col:
            col_map["importo"] = ("_split_", debit_col, credit_col)

    return col_map


def _parse_amount(val) -> Optional[float]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    s = s.replace("\u00A0", " ").replace(" ", "")
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")

    try:
        num = float(s)
        return -num if neg else num
    except ValueError:
        try:
            return float(str(val))
        except Exception:
            return None


def _coerce_date(val) -> Optional[str]:
    if pd.isna(val):
        return None
    dt = pd.to_datetime(val, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None
    return dt.date().isoformat()


def _extract_records(df: pd.DataFrame) -> List[Dict]:
    df = df.dropna(how="all")
    if df.empty:
        return []

    col_map = _match_headers(df)
    required_missing = [k for k in ("data", "operazione") if k not in col_map]
    if "importo" not in col_map:
        required_missing.append("importo")
    if required_missing:
        raise ValueError(f"Cannot find required columns: {', '.join(required_missing)}")

    records: List[Dict] = []
    split_amount = isinstance(col_map["importo"], tuple) and col_map["importo"][0] == "_split_"
    debit_col = credit_col = None
    if split_amount:
        _, debit_col, credit_col = col_map["importo"]

    for _, row in df.iterrows():
        data_val = _coerce_date(row[col_map["data"]]) if "data" in col_map else None

        if split_amount:
            debit = _parse_amount(row[debit_col]) if debit_col in df.columns else None
            credit = _parse_amount(row[credit_col]) if credit_col in df.columns else None
            if debit is not None and debit != 0:
                importo_val = -abs(debit)
            elif credit is not None and credit != 0:
                importo_val = abs(credit)
            else:
                importo_val = None
        else:
            importo_val = _parse_amount(row[col_map["importo"]])

        operazione_val = (
            str(row[col_map["operazione"]]).strip()
            if "operazione" in col_map and not pd.isna(row[col_map["operazione"]])
            else None
        )
        categoria_val = None
        if "categoria" in col_map and not pd.isna(row[col_map["categoria"]]):
            categoria_val = str(row[col_map["categoria"]]).strip()

        if not any([data_val, importo_val, operazione_val, categoria_val]):
            continue

        records.append(
            {
                "data": data_val,
                "importo": importo_val,
                "operazione": operazione_val,
                "categoria": categoria_val,
            }
        )

    return records


def _find_header_row(df_no_header: pd.DataFrame) -> Optional[int]:
    keywords = [
        "data",
        "valuta",
        "importo",
        "descrizione",
        "operazione",
        "causale",
        "categoria",
        "addebito",
        "accredito",
    ]

    def norm(x):
        s = "" if pd.isna(x) else str(x).strip().lower()
        return re.sub(r"\s+", " ", s)

    for i in range(len(df_no_header)):
        row = [norm(v) for v in df_no_header.iloc[i].tolist()]
        hits = sum(any(k in cell for k in keywords) for cell in row)
        if hits >= 2:
            return i
    return None


def read_any_excel_or_csv(content: bytes, filename: str) -> List[Dict]:
    name = (filename or "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content), dtype=object, sep=None, engine="python")
        return _extract_records(df)

    if not (name.endswith(".xlsx") or name.endswith(".xls")):
        raise ValueError("Unsupported file type; expected .xlsx, .xls or .csv")

    with io.BytesIO(content) as fh:
        xls = pd.ExcelFile(fh)
        for sheet in xls.sheet_names:
            df_no_header = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=object)
            if df_no_header.dropna(how="all").empty:
                continue
            hdr = _find_header_row(df_no_header)
            if hdr is not None:
                df = pd.read_excel(xls, sheet_name=sheet, header=hdr, dtype=object)
                if not df.dropna(how="all").empty:
                    return _extract_records(df)
            else:
                df = pd.read_excel(xls, sheet_name=sheet, dtype=object)
                if not df.dropna(how="all").empty:
                    return _extract_records(df)
    raise ValueError("No table-like data found in the Excel file.")

