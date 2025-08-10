
"""
pdf_fin_table_extractor.py

Pure-Python table extraction for financial PDFs using pdfplumber.
- Extracts tables page-by-page
- Cleans text, parses numbers (currency, %, parentheses for negatives)
- Attempts to infer page-level units like "Dollars in millions"
- Returns tidy DataFrames (long form) and wide tables
- Optional: serialize rows to text and add to an existing VectorStore (from pdf_rag_pipeline)

Install:
  pip install pdfplumber pandas numpy

Optional (to index into Chroma):
  from pdf_rag_pipeline import VectorStore, Chunk, md5sum

Example:
  from pdf_fin_table_extractor import FinancialTableExtractor
  fte = FinancialTableExtractor()
  result = fte.extract("/path/to/10K.pdf")
  result["long"].head()
  result["wide"][0].head()  # first wide table on first page with headers

  # To index tables into vector store (RAG-friendly text with citations):
  from pdf_rag_pipeline import VectorStore, Chunk, md5sum
  store = VectorStore(db_dir="vectorstore")
  chunks = fte.index_into_vectorstore(
      store, pdf_path="/path/to/10K.pdf",
      company="AAPL", report_date="2025-12-31"
  )
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber
import pandas as pd
import numpy as np


UNIT_PAT = re.compile(r"(in|amounts in)\s+(thousands|millions|billions)", re.I)
CURR_PAT = re.compile(r"(usd|u\.s\.?\s*dollars?|dollars?)", re.I)
FOOTNOTE_MARK = re.compile(r"\s*\*+\s*$")
PCT_PAT = re.compile(r"%\s*$")

def _strip(s: Any) -> str:
    if s is None:
        return ""
    return str(s).replace("\n", " ").strip()

def _clean_cell(s: str) -> str:
    s = _strip(s)
    s = FOOTNOTE_MARK.sub("", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s

def _parse_number(cell: str, default_unit: Optional[str]) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Parse numbers with finance-friendly rules:
    - Parentheses = negative
    - Commas as thousands
    - Percent detection
    - Currency symbols
    - Respect default_unit: thousands/millions/billions
    """
    meta = {"is_percent": False, "currency": None, "raw": cell}
    s = cell.strip()

    if s == "" or s.lower() in {"na", "n/a", "—", "-", "— —"}:
        return None, meta

    # percent?
    if PCT_PAT.search(s):
        meta["is_percent"] = True
        s = PCT_PAT.sub("", s).strip()

    # currency
    if "$" in s or "USD" in s.upper():
        meta["currency"] = "USD"
    s = s.replace("$", "").replace("USD", "").replace("usd", "")

    # negatives via parentheses or minus
    neg = False
    if re.search(r"^\(.*\)$", s):
        neg = True
        s = s.strip("()")
    s = s.replace(",", "").replace(" ", "")
    try:
        val = float(s)
        if neg:
            val = -val
    except Exception:
        return None, meta

    # scale by default unit
    scale = 1.0
    if default_unit:
        du = default_unit.lower()
        if "thousand" in du:
            scale = 1e3
        elif "million" in du:
            scale = 1e6
        elif "billion" in du:
            scale = 1e9
    val *= scale
    return val, meta

def _infer_units_and_currency(text_block: str) -> Dict[str, Optional[str]]:
    units = None
    currency = None
    m = UNIT_PAT.search(text_block or "")
    if m:
        units = m.group(2).lower()  # thousands/millions/billions
    if CURR_PAT.search(text_block or "") or "$" in (text_block or ""):
        currency = "USD"
    return {"default_unit": units, "default_currency": currency}


@dataclass
class TableResult:
    page: int
    table_id: int
    wide: pd.DataFrame
    long: pd.DataFrame
    meta: Dict[str, Any]


class FinancialTableExtractor:
    def __init__(self, detect_units_window_chars: int = 1200):
        self.detect_units_window_chars = detect_units_window_chars

    def _page_context(self, page: pdfplumber.page.Page) -> str:
        # Take some text from the page top area to detect "in millions" notes
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        return text[: self.detect_units_window_chars]

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """
        Returns:
          {
            "tables": List[TableResult],
            "long": pd.DataFrame (concatenated),
            "wide": List[pd.DataFrame]
          }
        """
        tables: List[TableResult] = []
        all_long = []

        with pdfplumber.open(pdf_path) as pdf:
            for pi, page in enumerate(pdf.pages, start=1):
                context_text = self._page_context(page)
                hints = _infer_units_and_currency(context_text)

                try:
                    page_tables = page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
                except Exception:
                    page_tables = []

                for ti, raw in enumerate(page_tables):
                    if not raw or len(raw) < 2:
                        continue

                    # Build DataFrame and forward-fill headers if needed
                    df = pd.DataFrame([[ _clean_cell(c) for c in row ] for row in raw])
                    # try to detect header row: choose first non-empty row with >1 non-empty cells
                    header_idx = 0
                    for ridx, row in df.iterrows():
                        nonempty = (row != "").sum()
                        if nonempty >= max(2, int(df.shape[1]*0.6)):
                            header_idx = ridx
                            break

                    df.columns = [c if c else f"col_{i}" for i, c in enumerate(df.iloc[header_idx].tolist())]
                    df = df.iloc[header_idx+1:].reset_index(drop=True)

                    # drop fully empty columns
                    df = df.loc[:, (df != "").any(axis=0)]

                    # make wide copy
                    wide_df = df.copy()

                    # Tidy (long) form: first column is label if likely textual
                    if wide_df.shape[1] >= 2:
                        label_col = wide_df.columns[0]
                        long_df = wide_df.melt(id_vars=[label_col], var_name="column", value_name="value_raw")
                        long_df.rename(columns={label_col: "row_label"}, inplace=True)
                    else:
                        # fallback
                        long_df = pd.DataFrame(columns=["row_label", "column", "value_raw"])

                    # parse numbers
                    parsed_vals = []
                    for v in long_df["value_raw"].astype(str):
                        val, meta = _parse_number(v, hints["default_unit"])
                        parsed_vals.append((val, meta))
                    long_df["value_num"] = [v[0] for v in parsed_vals]
                    long_df["is_percent"] = [v[1]["is_percent"] for v in parsed_vals]
                    long_df["currency"] = [v[1]["currency"] or hints["default_currency"] for v in parsed_vals]

                    long_df["page"] = pi
                    long_df["table_id"] = ti

                    meta = {
                        "default_unit": hints["default_unit"],
                        "default_currency": hints["default_currency"],
                    }

                    tables.append(TableResult(page=pi, table_id=ti, wide=wide_df, long=long_df, meta=meta))
                    all_long.append(long_df)

        long_cat = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame(
            columns=["row_label", "column", "value_raw", "value_num", "is_percent", "currency", "page", "table_id"]
        )
        wide_list = [t.wide for t in tables]
        return {"tables": tables, "long": long_cat, "wide": wide_list}

    # --------- Optional: index table rows into VectorStore as text chunks ---------
    def index_into_vectorstore(self, store, pdf_path: str, company: str, report_date: str,
                               doc_type: str = "table", extra_meta: Optional[Dict[str, Any]] = None):
        """
        Serializes each row of each table to a compact text string and indexes it.
        Example serialized row:
          "[Income Statement] Net sales | 2024: 12,345,000 | 2023: 11,000,000 (USD, in millions)"
        """
        from pdf_rag_pipeline import Chunk, md5sum  # lazy import to avoid hard dependency at import time

        res = self.extract(pdf_path)
        chunks = []
        base = {
            "company": company,
            "report_date": report_date,
            "doc_type": doc_type,
            "source_path": pdf_path,
            "md5": md5sum(pdf_path),
            "table_serialized": True,
        }
        if extra_meta:
            base.update(extra_meta)

        # group by page/table_id to reconstruct a readable chunk per row_label
        if res["long"].empty:
            return []

        for (page, table_id, row_label), g in res["long"].groupby(["page", "table_id", "row_label"]):
            # Identify table "title" from the wide header if possible
            try:
                wide = res["tables"][0].wide  # placeholder; titles aren't trivial; skipping
                title = ""
            except Exception:
                title = ""

            vals = []
            for _, r in g.iterrows():
                v = r["value_raw"]
                if pd.isna(v) or str(v).strip() == "":
                    continue
                vals.append(f"{r['column']}: {v}")
            if not vals:
                continue
            serial = (f"{row_label} | " + " | ".join(vals)).strip()

            meta = dict(base)
            meta.update({
                "start_page": page,
                "end_page": page,
                "table_id": int(table_id),
                "row_label": row_label,
                "default_unit": res["tables"][0].meta.get("default_unit") if res["tables"] else None,
                "default_currency": res["tables"][0].meta.get("default_currency") if res["tables"] else None,
            })

            cid = f"{base['md5']}_p{page}_t{table_id}_{abs(hash(serial))%10**9}"
            chunks.append(Chunk(id=cid, text=serial, meta=meta))

        if chunks:
            store.add_chunks(chunks)
        return chunks
