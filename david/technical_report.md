# AI-Powered Financial Reports Q\&A, Forecasting & Strategy — Technical One-Pager

## 1) Goal & Scope

Build a production-ready, local-first system that ingests annual/quarterly PDFs, extracts narrative + tables, augments with free market data, and delivers grounded Q\&A, forecasts, and buy/hold/sell guidance—through a simple Streamlit app.

## 2) System Architecture (High Level)

* **Ingestion:** PyMuPDF text + pdfplumber tables → chunking → embeddings (bge-small) → **ChromaDB** (persistent).
* **Retrieval:** Metadata-filtered kNN (by company/year/doc type).
* **Reasoning (Local):** **Ollama** (`/v1/chat/completions`) for RAG answers & strategy.
* **Market Data (Free):** Alpha Vantage TIME\_SERIES\_DAILY/WEEKLY/MONTHLY (+ docs-only mode).
* **UI:** Streamlit with tabs (Ingest, Q\&A, Market, Assisted Forecast). Company knowledge base accumulates multiple years.

```
[PDFs (10-K/PR/ER)] → [Extractor+Table Parser] → [Chunker] → [Embeddings] → [ChromaDB]
                                                             ↑            ↓
                                                        [Retrieval (k)] ← [Q&A/Assisted Forecast (Ollama)]
                                                             ↑
                                           [Filters: company, report_date, doc_type]
        [Alpha Vantage (free)] → [ES Forecast] → [Chart + intervals] → [LLM strategy fusion]
```

## 3) Key Components

* **`pdf_rag_pipeline.py`**

  * `PDFExtractor` (PyMuPDF), `Chunker` (semantic w/ overlap), `VectorStore` (Chroma), `RAGQuery` (Ollama/OpenAI-compatible).
  * **Metadata filtering**: `where={"report_date":{"$in":["2024-12-31","2023-12-31"]}}`.
* **`pdf_fin_table_extractor.py`**

  * `pdfplumber` tables → tidy DataFrame (`row_label, column, value_raw, value_num, is_percent, currency, page, table_id`).
  * Optional: serialize rows into Chroma (exact numeric grounding + citations).
* **`market_forecast.py`**

  * **Free** Alpha Vantage fetch (`TIME_SERIES_DAILY/WEEKLY/MONTHLY`) → normalized `close` series.
  * Exponential Smoothing (statsmodels) → forecast + naïve 95% band; horizon parsed from NL (“next 6 weeks”).
* **`app_streamlit.py`**

  * Tabs: **Upload & Ingest**, **Document Q\&A**, **Market Data & Forecast**, **Assisted Forecast (Docs+Market+LLM)**.
  * **Company KB mode**: one collection per company; catalog (`kb_catalog.json`) lists all years ingested.
  * **Docs-only toggle**: disables Alpha Vantage, strategy is based purely on filings.

## 4) How to Run (Dev)

```bash
# 1) Install
pip install -r requirements.txt

# 2) Launch local model server (example)
# ollama run llama3  (ensure model is pulled)
# or ensure an OpenAI-compatible endpoint at:
# http://localhost:11434/v1/chat/completions

# 3) Start UI
streamlit run app_streamlit.py
```

Optional: `export ALPHAVANTAGE_API_KEY="YOUR_KEY"` (free tier works).

## 5) Core User Flows

* **Build KB:** Upload 2024, then 2025 report → same company collection (e.g., `finance_docs_mmm`).
* **Ask Qs:** Filter years (2024+2025) → “Compare revenue growth and cite pages.”
* **Plot & Forecast:** Fetch daily data → “next 6 weeks” → chart + band.
* **Assisted Forecast:** Fuse forecast + retrieved snippets → brief **Summary / Recommendation / Why / Risks**. Follow-ups keep chat & retrieval context.

## 6) Evaluation (Hackathon-Ready)

**Document Q\&A**

* *Retrieval:* Recall\@k (k∈{5,8,10}), MRR.
* *Answering:* Exact-match / relaxed numeric tolerance (±0.5–1.0%), citation correctness (page overlap).
* *Tables:* Numeric grounding accuracy (parsed value vs. table cell).

**Forecasting**

* Walk-forward on last N segments: MAPE, sMAPE, RMSE; prediction interval coverage (target ≈95%).

**Strategy**

* Checklist scoring: grounding (no hallucinated numbers), consistency with cited text, clarity of rationale/risks (Likert 1–5).

**Latency**

* p50/p95 for: ingest (per 100 pages), retrieval, LLM response, forecast.

## 7) Reliability & Guardrails

* **Docs-first grounding:** Always include citations; if absent → “Not found in provided documents.”
* **Numbers-first extract:** Prefer table rows; show units (“in millions”) and page.
* **Rate limits:** Cache Alpha Vantage responses; graceful errors; docs-only fallback.
* **Privacy:** Local embeddings & DB; no external uploads beyond Alpha Vantage HTTP GET.

## 8) Limitations & Mitigations

* OCR for scanned tables may require system deps (tesseract/ocrmypdf). *Mitigation:* enable OCR toggle.
* ES forecast is simple; won’t capture regime shifts. *Mitigation:* add Prophet/XGBoost/TFT option.
* LLM numeric errors. *Mitigation:* regex post-check + value cross-lookup from table index.

## 9) Next Steps

* Add **backtesting dashboard** (rolling-origin CV + plots).
* **Candlestick + volume** charts; event markers from filings (e.g., guidance).
* **Model ensemble** (ES + XGB) with SHAP feature attributions from fundamentals.
* **Report export**: one-click PDF with Q\&A, charts, citations, and strategy.

---

**Deliverables included:** Streamlit app, ingestion pipeline, table extractor, Alpha Vantage free-tier fetcher, and requirements file.
