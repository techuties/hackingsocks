# Project Summary

This project delivers a local-first AI assistant that turns annual reports into actionable insights, forecasts, and buy/hold/sell guidance. Users upload PDFs (10-K/annual reports), the app extracts narrative and tables, indexes them with embeddings, and answers questions with page-level citations. A free-tier market data pipeline adds stock history and lightweight forecasts; a “docs-only” mode keeps analysis strictly grounded in filings.

**How it works:**

* **Ingestion & Grounding:** PyMuPDF parses text; pdfplumber extracts financial tables and normalizes numbers (currency, parentheses, %). Content is chunked and embedded (bge-small) into a persistent ChromaDB with rich metadata (company, report date, pages).
* **Retrieval & Q\&A:** Queries retrieve top-K chunks (filterable by year) and call a local LLM via Ollama (`/v1/chat/completions`). Answers are concise and citation-backed.
* **Market Data & Forecasts:** Free Alpha Vantage endpoints (daily/weekly/monthly) feed an Exponential Smoothing model with simple 95% intervals; charts are interactive.
* **Assisted Strategy:** A combined tab fuses forecast signals with document snippets and a user “conditioning” prompt to produce a short, structured view: **Summary, Recommendation, Why, Risks**. Follow-ups keep chat and retrieval context.
* **Knowledge Base:** Multiple years per company accumulate in one collection with a visible catalog and year filters.

**Why it matters:** The system compresses hours of reading and model-wiring into minutes, providing grounded answers, reproducible numbers, and explainable forecasts—without sending sensitive documents to third parties. Evaluation includes retrieval quality (Recall\@K), numeric accuracy from tables, forecast error (MAPE/RMSE), and strategy clarity.
