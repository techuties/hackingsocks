
import os
import sys
import hashlib
from datetime import date
from pathlib import Path

import streamlit as st

# Make sure our helper modules are importable if the app is launched elsewhere
HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

# Try to import our helper modules (assumes they're in the same dir or PYTHONPATH)
try:
    from pdf_rag_pipeline import PDFIngestor, RAGQuery, md5sum
except Exception as e:
    st.error("Cannot import pdf_rag_pipeline. Place pdf_rag_pipeline.py next to this app or in PYTHONPATH.")
    st.stop()

try:
    from pdf_fin_table_extractor import FinancialTableExtractor
    HAS_TABLES = True
except Exception:
    HAS_TABLES = False

# --------- UI CONFIG ---------
st.set_page_config(page_title="Annual Report Q&A (RAG + Ollama)", layout="wide")

st.markdown("""
# Annual Report Q&A
Upload an annual report PDF, index it into a local vector store, and ask grounded questions.  
**Backend:** Chroma + SentenceTransformers, **LLM:** Ollama (local), **Citations with pages**.
""")

with st.sidebar:
    st.header("Settings")
    db_dir = st.text_input("Vector store directory", value=str(HERE / "vectorstore_app"))
    coll_prefix = st.text_input("Collection prefix", value="finance_docs")
    ollama_url = st.text_input("Ollama endpoint", value="http://localhost:11434/v1/chat/completions")
    model_name = st.text_input("Model name", value="llama3.3")
    k = st.slider("Top-K chunks", 3, 12, 6)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    do_ocr = st.checkbox("Enable OCR fallback (needs ocrmypdf + tesseract installed)", value=False)
    index_tables = st.checkbox("Also index table rows", value=True, disabled=not HAS_TABLES)
    st.caption("Turn off if you don't have pdfplumber installed or prefer faster ingestion.")

# --------- SESSION STATE ---------
if "collection" not in st.session_state:
    st.session_state.collection = None
if "company" not in st.session_state:
    st.session_state.company = ""
if "report_date" not in st.session_state:
    st.session_state.report_date = str(date.today())
if "last_file_md5" not in st.session_state:
    st.session_state.last_file_md5 = None
if "table_rows_indexed" not in st.session_state:
    st.session_state.table_rows_indexed = 0

# --------- FILE UPLOAD ---------
st.subheader("1) Upload annual report PDF")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

col_meta1, col_meta2 = st.columns(2)
with col_meta1:
    company = st.text_input("Company (ticker or name)", value=st.session_state.company or "")
with col_meta2:
    report_date = st.text_input("Report date (YYYY-MM-DD)", value=st.session_state.report_date)

ingest_btn = st.button("Ingest PDF")

def save_upload(file) -> Path:
    target_dir = Path(db_dir) / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / file.name
    with open(out_path, "wb") as f:
        f.write(file.getbuffer())
    return out_path

def unique_collection_name(base: str, pdf_path: Path) -> str:
    # Use md5 of file so each upload gets its own collection
    h = md5sum(str(pdf_path))
    return f"{base}_{h[:10]}"

if ingest_btn:
    if not uploaded:
        st.warning("Please upload a PDF first.")
        st.stop()
    if not company.strip():
        st.warning("Please provide a company name/ticker.")
        st.stop()
    # Save and compute identifiers
    saved = save_upload(uploaded)
    st.session_state.company = company.strip()
    st.session_state.report_date = report_date.strip()
    st.session_state.last_file_md5 = md5sum(str(saved))
    collection = unique_collection_name(coll_prefix, saved)
    st.session_state.collection = collection

    st.info(f"Ingesting into collection: **{collection}**")
    with st.status("Extracting, chunking, and embedding…", expanded=True) as status:
        try:
            ingestor = PDFIngestor(
                db_dir=db_dir,
                collection=collection,
                embedding_model="BAAI/bge-small-en-v1.5",
                do_ocr=do_ocr,
            )
            chunks = ingestor.ingest_pdf(
                str(saved),
                company=st.session_state.company,
                report_date=st.session_state.report_date,
                doc_type="annual_report",
            )
            st.write(f"• Text chunks indexed: **{len(chunks)}**")
        except Exception as e:
            st.error(f"Ingestion error: {e}")
            st.stop()

        # Optional: index table rows
        st.session_state.table_rows_indexed = 0
        if index_tables and HAS_TABLES:
            try:
                fte = FinancialTableExtractor()
                from pdf_rag_pipeline import VectorStore  # import here to reuse same DB
                store = VectorStore(db_dir=db_dir, collection=collection)
                table_chunks = fte.index_into_vectorstore(
                    store,
                    pdf_path=str(saved),
                    company=st.session_state.company,
                    report_date=st.session_state.report_date,
                    doc_type="table",
                )
                st.session_state.table_rows_indexed = len(table_chunks)
                st.write(f"• Table rows indexed: **{len(table_chunks)}**")
            except Exception as e:
                st.warning(f"Table indexing skipped due to error: {e}")
        status.update(state="complete", label="Ingestion complete")

# --------- QUERY ---------
st.subheader("2) Ask questions about the report")
q = st.text_input("Your question", placeholder="e.g., What was total revenue for 2016? What key risks did management highlight?")

go = st.button("Ask")

if go:
    if not st.session_state.collection:
        st.warning("Please ingest a PDF first.")
        st.stop()
    try:
        rag = RAGQuery(
            db_dir=db_dir,
            collection=st.session_state.collection,
            embedding_model="BAAI/bge-small-en-v1.5",
            ollama_url=ollama_url,
            model_name=model_name,
        )
        with st.spinner("Retrieving and querying model…"):
            res = rag.answer(q, k=int(k), temperature=float(temperature))
    except Exception as e:
        st.error(f"Query error: {e}")
        st.stop()

    st.markdown("### Answer")
    st.write(res.get("answer", ""))

    # Citations
    metas = res.get("citations", []) or []
    if metas:
        st.markdown("### Citations")
        for i, m in enumerate(metas, start=1):
            tag = f"{m.get('company','?')} | {m.get('report_date','?')} | p.{m.get('start_page','?')}-{m.get('end_page','?')}"
            with st.expander(f"Source {i}: {tag}"):
                # show a small snippet if available via the store (requires another query by id).
                # For simplicity, we show the stored metadata only; advanced: fetch doc text by ID.
                st.json(m, expanded=False)

# --------- FOOTER ---------
st.divider()
st.caption("Tip: If numbers are missing, enable OCR fallback or ensure the tables were indexed. "
           "You can also adjust Top-K to 8-10 for broader context.")
