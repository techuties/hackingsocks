
import os
import sys
import json
import requests
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st

HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from pdf_rag_pipeline import PDFIngestor, RAGQuery, md5sum
try:
    from pdf_fin_table_extractor import FinancialTableExtractor
    HAS_TABLES = True
except Exception:
    HAS_TABLES = False

try:
    from market_forecast import (
        fetch_alpha_vantage_free,
        parse_horizon,
        exp_smoothing_forecast,
    )
    HAS_MARKET = True
except Exception:
    HAS_MARKET = False

st.set_page_config(page_title="Annual Report Q&A + Forecast + AI Strategy", layout="wide")
st.title("Annual Report Q&A + Forecast + AI Strategy")

with st.sidebar:
    st.header("Settings")
    db_dir = st.text_input("Vector store directory", value=str(HERE / "vectorstore_app"))
    coll_prefix = st.text_input("Collection prefix", value="finance_docs")
    ollama_url = st.text_input("Ollama endpoint", value="http://localhost:11434/v1/chat/completions")
    model_name = st.text_input("Model name", value="llama3.3")
    k = st.slider("Top-K chunks", 3, 12, 6)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    do_ocr = st.checkbox("Enable OCR fallback (ocrmypdf + tesseract)", value=False)
    index_tables = st.checkbox("Also index table rows", value=True, disabled=not HAS_TABLES)

    st.markdown("---")
    st.subheader("Alpha Vantage (FREE endpoints)")
    default_key = os.getenv("ALPHAVANTAGE_API_KEY", "3B10HMD7VDM29PS9")
    av_api_key = st.text_input("API Key", value=default_key, type="password")
    granularity = st.selectbox("Granularity", options=["daily", "weekly", "monthly"], index=0)
    outputsize = st.selectbox("Output size (daily only)", options=["compact", "full"], index=0, disabled=(granularity!="daily"))

tab_ingest, tab_query, tab_market, tab_strategy = st.tabs(["Upload & Ingest", "Document Q&A", "Market Data & Forecast", "Assisted Forecast (Docs + Market + LLM)"])

for key, val in {"collection": None, "company": "", "report_date": "", "last_file_md5": None, "table_rows_indexed": 0}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------- Upload & Ingest ----------------
with tab_ingest:
    st.subheader("1) Upload annual report PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        company = st.text_input("Company (ticker or name)", value=st.session_state.company or "")
    with col_meta2:
        report_date = st.text_input("Report date (YYYY-MM-DD)", value=st.session_state.report_date or "")

    def save_upload(file):
        target_dir = Path(db_dir) / "uploads"
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / file.name
        with open(out_path, "wb") as f:
            f.write(file.getbuffer())
        return out_path

    def unique_collection_name(base, pdf_path: Path) -> str:
        h = md5sum(str(pdf_path))
        return f"{base}_{h[:10]}"

    if st.button("Ingest PDF"):
        if not uploaded:
            st.warning("Please upload a PDF first."); st.stop()
        if not company.strip():
            st.warning("Please provide a company/ticker."); st.stop()
        saved = save_upload(uploaded)
        st.session_state.company = company.strip()
        st.session_state.report_date = report_date.strip() or "NA"
        st.session_state.last_file_md5 = md5sum(str(saved))
        st.session_state.collection = unique_collection_name(coll_prefix, saved)

        st.info(f"Ingesting into collection: **{st.session_state.collection}**")
        with st.status("Extracting, chunking, and embedding…", expanded=True) as status:
            try:
                ingestor = PDFIngestor(db_dir=db_dir, collection=st.session_state.collection, embedding_model="BAAI/bge-small-en-v1.5", do_ocr=do_ocr)
                chunks = ingestor.ingest_pdf(str(saved), company=st.session_state.company, report_date=st.session_state.report_date, doc_type="annual_report")
                st.write(f"• Text chunks indexed: **{len(chunks)}**")
            except Exception as e:
                st.error(f"Ingestion error: {e}"); st.stop()

            st.session_state.table_rows_indexed = 0
            if index_tables and HAS_TABLES:
                try:
                    from pdf_rag_pipeline import VectorStore
                    fte = FinancialTableExtractor()
                    store = VectorStore(db_dir=db_dir, collection=st.session_state.collection)
                    table_chunks = fte.index_into_vectorstore(store, pdf_path=str(saved), company=st.session_state.company, report_date=st.session_state.report_date, doc_type="table")
                    st.session_state.table_rows_indexed = len(table_chunks)
                    st.write(f"• Table rows indexed: **{len(table_chunks)}**")
                except Exception as e:
                    st.warning(f"Table indexing skipped: {e}")
            status.update(state="complete", label="Ingestion complete")

# ---------------- Document Q&A ----------------
with tab_query:
    st.subheader("2) Ask a question about the uploaded report")
    q = st.text_input("Your question", placeholder="e.g., What was total revenue in 2016? Key risks?")
    if st.button("Ask"):
        if not st.session_state.collection:
            st.warning("Please ingest a PDF first."); st.stop()
        try:
            rag = RAGQuery(db_dir=db_dir, collection=st.session_state.collection, embedding_model="BAAI/bge-small-en-v1.5", ollama_url=ollama_url, model_name=model_name)
            with st.spinner("Retrieving and querying model…"):
                res = rag.answer(q, k=int(k), temperature=float(temperature))
        except Exception as e:
            st.error(f"Query error: {e}"); st.stop()

        st.markdown("### Answer")
        st.write(res.get("answer", ""))
        metas = res.get("citations", []) or []
        if metas:
            st.markdown("### Citations")
            for i, m in enumerate(metas, start=1):
                tag = f"{m.get('company','?')} | {m.get('report_date','?')} | p.{m.get('start_page','?')}-{m.get('end_page','?')}"
                with st.expander(f"Source {i}: {tag}"):
                    st.json(m, expanded=False)

# ---------------- Market Data & Forecast ----------------
with tab_market:
    st.subheader("3) Market Data & Forecast")
    if not HAS_MARKET:
        st.warning("market_forecast.py not found or dependencies missing. Place it next to this app and install: requests pandas numpy statsmodels plotly.")
    else:
        colA, colB = st.columns([2,1])
        with colA:
            ticker = st.text_input("Ticker (e.g., MMM, AAPL)", value=(st.session_state.company or "AAPL"), key="mdf_ticker")
        with colB:
            nl_prompt = st.text_input("Forecast request (optional, e.g., 'next 6 weeks')", value="next 4 weeks", key="mdf_prompt")

        periods_default = parse_horizon(nl_prompt or "", default_days=20)
        periods = st.slider("Forecast horizon (business days)", 5, 120, periods_default, step=5, key="mdf_periods")

        run_md = st.button("Fetch & Plot", key="btn_fetch_plot")
        run_fc = st.button("Forecast", key="btn_forecast")

        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_av(symbol: str, key: str, gran: str, size: str):
            return fetch_alpha_vantage_free(symbol, key, granularity=gran, outputsize=size)

        if run_md:
            try:
                with st.spinner("Fetching Alpha Vantage… (free endpoint)"):
                    dfp = cached_av(st.session_state.get("mdf_ticker"), av_api_key, granularity, outputsize)
                st.markdown(f"**{st.session_state.get('mdf_ticker')}** price history ({len(dfp)} rows) • granularity: **{granularity}**")
                fig = px.line(dfp, x="date", y="close", title=f"{st.session_state.get('mdf_ticker')} Close ({granularity})")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(dfp.tail(10), use_container_width=True, height=220)
                st.session_state["last_market_df"] = dfp
            except Exception as e:
                st.error(f"Alpha Vantage error: {e}")

        if run_fc:
            try:
                dfp = st.session_state.get("last_market_df")
                if dfp is None:
                    dfp = cached_av(st.session_state.get("mdf_ticker"), av_api_key, granularity, outputsize)
                with st.spinner("Fitting Exponential Smoothing…"):
                    fc = exp_smoothing_forecast(dfp, value_col="close", periods=int(st.session_state.get("mdf_periods")))
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=dfp["date"], y=dfp["close"], mode="lines", name="History"))
                fig2.add_trace(go.Scatter(x=fc["date"], y=fc["yhat"], mode="lines", name="Forecast"))
                fig2.add_trace(go.Scatter(
                    x=pd.concat([fc["date"], fc["date"][::-1]]),
                    y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
                    fill="toself", mode="lines", line=dict(width=0),
                    name="95% interval", opacity=0.2
                ))
                fig2.update_layout(title=f"{st.session_state.get('mdf_ticker')} Forecast ({st.session_state.get('mdf_periods')} business days)")
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(fc, use_container_width=True, height=260)
                st.session_state["last_forecast_df"] = fc
            except Exception as e:
                st.error(f"Forecast error: {e}")

# ---------------- Assisted Forecast (Docs + Market + LLM) ----------------
with tab_strategy:
    st.subheader("4) Assisted Forecast (combines document insights + market data + LLM)")
    if not HAS_MARKET:
        st.warning("market_forecast.py not found or dependencies missing.")
    else:
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            strat_ticker = st.text_input("Ticker", value=(st.session_state.company or "AAPL"), key="strat_ticker")
            conditioning = st.text_area("Strategy conditioning prompt", value="Focus on management guidance, segment trends, and risk disclosures. Incorporate recent revenue/earnings direction from the report.", height=120)
        with col2:
            nl_h = st.text_input("Forecast horizon (e.g., 'next 6 weeks')", value="next 6 weeks", key="strat_nl")
            periods = parse_horizon(nl_h or "", default_days=30)
            st.metric("Parsed horizon (business days)", periods)
        with col3:
            topk_docs = st.slider("Top-K doc chunks", 3, 12, int(k), key="strat_k")
            show_pi = st.checkbox("Show 95% forecast band", value=True)

        # Persistent chat state
        if "strategy_chat" not in st.session_state:
            st.session_state.strategy_chat = []   # list of dicts with role/content
        if "strategy_sources" not in st.session_state:
            st.session_state.strategy_sources = []  # metadata list
        if "strategy_last_fc" not in st.session_state:
            st.session_state.strategy_last_fc = None
        if "strategy_last_md" not in st.session_state:
            st.session_state.strategy_last_md = None

        def fetch_prices(symbol):
            return fetch_alpha_vantage_free(symbol, av_api_key, granularity="daily", outputsize="compact")

        def base_forecast(df, periods):
            return exp_smoothing_forecast(df, value_col="close", periods=int(periods))

        def summarize_market(df_hist, fc):
            last_close = float(df_hist["close"].iloc[-1])
            end_fc = float(fc["yhat"].iloc[-1])
            pct = (end_fc / last_close - 1.0) * 100.0
            width = float((fc["yhat_upper"].iloc[-1] - fc["yhat_lower"].iloc[-1]))
            vol20 = float(df_hist["close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
            return last_close, end_fc, pct, width, vol20

        def retrieve_docs(query_text, kdocs):
            if not st.session_state.collection:
                return {"documents":[[""]], "metadatas":[[]]}
            try:
                rag_tmp = RAGQuery(db_dir=db_dir, collection=st.session_state.collection, embedding_model="BAAI/bge-small-en-v1.5", ollama_url=ollama_url, model_name=model_name)
                res = rag_tmp.store.query(query_text, kdocs)
                return res
            except Exception as e:
                st.warning(f"Doc retrieval failed: {e}")
                return {"documents":[[""]], "metadatas":[[]]}

        def call_ollama(messages, temperature=0.1):
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "stream": False
            }
            r = requests.post(ollama_url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"].get("content", "")
            return str(data)[:2000]

        run_btn = st.button("Run Assisted Forecast")
        followup = st.text_input("Follow-up question (optional)")
        ask_followup = st.button("Ask Follow-up")

        if run_btn:
            try:
                with st.spinner("Fetching prices & computing base forecast…"):
                    dfp = fetch_prices(st.session_state.get("strat_ticker"))
                    fc = base_forecast(dfp, periods)
                    st.session_state.strategy_last_md = dfp
                    st.session_state.strategy_last_fc = fc
                # Retrieve doc context
                doc_query = conditioning + " outlook guidance risk revenue earnings cost headwinds tailwinds capex margins segments"
                res = retrieve_docs(doc_query, int(topk_docs))
                docs = res["documents"][0] if res.get("documents") else []
                metas = res["metadatas"][0] if res.get("metadatas") else []
                st.session_state.strategy_sources = metas

                # Build context
                last_close, end_fc, pct, width, vol20 = summarize_market(dfp, fc)
                market_ctx = (
                    f"Base forecast: last_close={last_close:.2f}, "
                    f"end_forecast={end_fc:.2f}, change_to_end={pct:.2f}%, "
                    f"interval_width≈{width:.2f}, annualized_vol20≈{vol20:.2f}."
                )
                doc_blurbs = []
                for d,m in zip(docs, metas):
                    tag = f"{m.get('company','?')} | {m.get('report_date','?')} | p.{m.get('start_page','?')}-{m.get('end_page','?')}"
                    doc_blurbs.append(f"[{tag}] {d[:800].replace(chr(10),' ')}")

                system_msg = {
                    "role": "system",
                    "content": (
                        "You are a disciplined equity analyst. Combine the provided base statistical forecast with "
                        "insights grounded ONLY in the cited document snippets. Do not invent figures. "
                        "If the document context is weak, say so. Provide a clear Buy/Hold/Sell with rationale, "
                        "and note uncertainties. Keep the answer under 200 words.\n"
                        "Return sections titled: Summary, Recommendation, Why, Risks."
                    )
                }
                user_msg = {
                    "role": "user",
                    "content": (
                        f"Ticker: {st.session_state.get('strat_ticker')}\n"
                        f"Horizon: {periods} business days\n"
                        f"Conditioning prompt: {conditioning}\n\n"
                        f"{market_ctx}\n\n"
                        "Document snippets:\n" + "\n\n".join(doc_blurbs)
                    )
                }
                st.session_state.strategy_chat = [system_msg, user_msg]
                with st.spinner("Reasoning with documents + market…"):
                    answer = call_ollama(st.session_state.strategy_chat, temperature=float(temperature))
                st.markdown("### Assisted Forecast — Model View")
                st.write(answer)

                # Plot
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=dfp['date'], y=dfp['close'], mode='lines', name='History'))
                fig3.add_trace(go.Scatter(x=fc['date'], y=fc['yhat'], mode='lines', name='Forecast'))
                if show_pi:
                    fig3.add_trace(go.Scatter(
                        x=pd.concat([fc["date"], fc["date"][::-1]]),
                        y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
                        fill="toself", mode="lines", line=dict(width=0),
                        name="95% interval", opacity=0.2
                    ))
                fig3.update_layout(title=f"{st.session_state.get('strat_ticker')} — Assisted Forecast Horizon ({periods}B)")
                st.plotly_chart(fig3, use_container_width=True)

                # Citations
                if metas:
                    st.markdown("### Citations")
                    for i, m in enumerate(metas, start=1):
                        tag = f"{m.get('company','?')} | {m.get('report_date','?')} | p.{m.get('start_page','?')}-{m.get('end_page','?')}"
                        with st.expander(f"Source {i}: {tag}"):
                            st.json(m, expanded=False)

            except Exception as e:
                st.error(f"Assisted forecast error: {e}")

        if ask_followup:
            if not st.session_state.strategy_chat:
                st.warning("Run an Assisted Forecast first.")
            else:
                # Optional: retrieve additional context for follow-up
                extra_res = retrieve_docs(followup, int(topk_docs))
                extra_docs = extra_res["documents"][0] if extra_res.get("documents") else []
                extra_metas = extra_res["metadatas"][0] if extra_res.get("metadatas") else []
                # Append brief "context update" message
                ctx_update = ""
                if extra_docs:
                    ctx_snips = " | ".join([d[:200].replace("\n"," ") for d in extra_docs[:2]])
                    ctx_update = f"\n\nAdditional context: {ctx_snips}"
                    st.session_state.strategy_sources.extend(extra_metas)

                st.session_state.strategy_chat.append({"role":"user", "content": followup + ctx_update})
                try:
                    with st.spinner("Thinking…"):
                        answer2 = call_ollama(st.session_state.strategy_chat, temperature=float(temperature))
                    st.markdown("### Follow-up")
                    st.write(answer2)
                except Exception as e:
                    st.error(f"Follow-up error: {e}")
