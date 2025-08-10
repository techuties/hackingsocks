
"""
pdf_rag_pipeline.py

A compact pipeline for:
- PDF extraction (PyMuPDF) with quality checks and optional OCR fallback
- Text cleaning + chunking with metadata
- Embeddings (SentenceTransformers) + ChromaDB vector store
- Retrieval + Ollama chat for grounded Q&A

Install (in your own Jupyter env):
  pip install pymupdf sentence-transformers chromadb requests

Optional (OCR fallback):
  # Requires system deps: ocrmypdf and tesseract
  pip install ocrmypdf pytesseract

Usage quick-start (see bottom for full example):
  from pdf_rag_pipeline import PDFIngestor, RAGQuery

  ingestor = PDFIngestor(db_dir="vectorstore")
  ingestor.ingest_pdf("path/to/file.pdf", company="AAPL", report_date="2025-05-01")

  rag = RAGQuery(db_dir="vectorstore", model_name="llama3")
  ans = rag.answer("What was Q2 revenue?", k=6)
  print(ans["answer"])
  print(ans["citations"])
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
import numpy as np

# Optional imports guarded
try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None

try:
    import subprocess
except Exception:
    subprocess = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None  # type: ignore
    Settings = None  # type: ignore

import requests

# -----------------------
# Utilities
# -----------------------

def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_ws(text: str) -> str:
    # Normalize weird whitespace/linebreaks; preserve bullet structure lightly
    text = text.replace("\r", "\n")
    # Keep hyphenated line breaks: join words split by hyphen at EOL
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse multiple newlines, but keep paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def text_density(txt: str) -> float:
    if not txt:
        return 0.0
    # crude: characters per newline (higher ~ more dense)
    lines = txt.splitlines() or [""]
    return len(txt) / max(1, len(lines))


# -----------------------
# Data classes
# -----------------------

@dataclass
class Chunk:
    id: str
    text: str
    meta: Dict[str, Any]


# -----------------------
# PDF Extraction
# -----------------------

class PDFExtractor:
    def __init__(self, ocr_threshold: float = 10.0, do_ocr: bool = False):
        """
        ocr_threshold: if average text density per page is below this, we try OCR (if enabled)
        do_ocr: try OCR fallback using ocrmypdf if available
        """
        self.ocr_threshold = ocr_threshold
        self.do_ocr = do_ocr

    def _maybe_ocr(self, pdf_path: str) -> str:
        if not self.do_ocr or subprocess is None:
            return pdf_path
        try:
            out_path = os.path.splitext(pdf_path)[0] + ".ocr.pdf"
            # If already exists, reuse
            if os.path.exists(out_path):
                return out_path
            subprocess.run(
                ["ocrmypdf", "--skip-text", "--quiet", pdf_path, out_path],
                check=True,
            )
            return out_path
        except Exception:
            # Fallback: return original
            return pdf_path

    def extract(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Returns a list of pages, each: {"page_num": int, "text": str}
        Attempts OCR if density is too low and do_ocr=True.
        """
        # First pass extraction
        pages = self._extract_plain(pdf_path)
        avg_density = np.mean([text_density(p["text"]) for p in pages]) if pages else 0.0

        if avg_density < self.ocr_threshold:
            # Try OCR fallback
            ocr_pdf = self._maybe_ocr(pdf_path)
            if ocr_pdf != pdf_path:
                pages = self._extract_plain(ocr_pdf)

        # Clean text
        for p in pages:
            p["text"] = normalize_ws(p["text"])

        return pages

    @staticmethod
    def _extract_plain(pdf_path: str) -> List[Dict[str, Any]]:
        out = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                txt = page.get_text("text")  # layout-preserving text
                out.append({"page_num": i + 1, "text": txt or ""})
        return out


# -----------------------
# Chunking
# -----------------------

class Chunker:
    def __init__(self, target_tokens: int = 350, overlap: int = 60):
        self.target_tokens = target_tokens
        self.overlap = overlap

    @staticmethod
    def _approx_tokens(text: str) -> int:
        # rough approximation: ~1.3 words/token, 5 chars/word -> ~4 chars/token
        return max(1, int(len(text) / 4))

    def chunk_pages(self, pages: List[Dict[str, Any]], base_meta: Dict[str, Any]) -> List[Chunk]:
        chunks: List[Chunk] = []
        buf = []
        tokens = 0
        start_page = None

        def flush(end_page):
            nonlocal buf, tokens, start_page
            if not buf:
                return
            text = "\n".join(buf).strip()
            cid = hashlib.md5(text.encode("utf-8")).hexdigest()
            meta = dict(base_meta)
            meta.update({"start_page": start_page, "end_page": end_page})
            chunks.append(Chunk(id=cid, text=text, meta=meta))
            buf = []
            tokens = 0
            start_page = None

        for p in pages:
            page_text = p["text"].strip()
            if not page_text:
                continue
            para_list = re.split(r"\n{2,}", page_text)

            for para in para_list:
                t = self._approx_tokens(para)
                if tokens == 0:
                    start_page = p["page_num"]
                if tokens + t > self.target_tokens and tokens > 0:
                    # flush current buffer
                    flush(p["page_num"])
                    # start new buffer, include overlap from last chunk
                    if self.overlap > 0 and chunks:
                        tail = chunks[-1].text.split()
                        keep = " ".join(tail[-self.overlap:])
                        buf = [keep, para]
                        tokens = self._approx_tokens(keep) + t
                        start_page = p["page_num"]
                    else:
                        buf = [para]
                        tokens = t
                        start_page = p["page_num"]
                else:
                    buf.append(para)
                    tokens += t

        # final flush
        if start_page is not None:
            flush(pages[-1]["page_num"])

        return chunks


# -----------------------
# Vector store
# -----------------------

class VectorStore:
    def __init__(self, db_dir: str = "vectorstore", collection: str = "finance_docs",
                 embedding_model: str = "BAAI/bge-small-en-v1.5"):
        if chromadb is None:
            raise ImportError("chromadb is not installed. pip install chromadb")
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. pip install sentence-transformers")

        self.client = chromadb.PersistentClient(path=db_dir, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.embedder = SentenceTransformer(embedding_model)

    def add_chunks(self, chunks: List[Chunk]):
        texts = [c.text for c in chunks]
        metas = [c.meta for c in chunks]
        ids = [c.id for c in chunks]
        embs = self.embedder.encode(texts, normalize_embeddings=True).tolist()
        self.collection.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)

    def query(self, query: str, k: int = 6) -> Dict[str, Any]:
        q = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        return self.collection.query(query_embeddings=[q], n_results=k)


# -----------------------
# High-level ingestion
# -----------------------

class PDFIngestor:
    def __init__(self, db_dir: str = "vectorstore", collection: str = "finance_docs",
                 embedding_model: str = "BAAI/bge-small-en-v1.5", do_ocr: bool = False):
        self.extractor = PDFExtractor(do_ocr=do_ocr)
        self.chunker = Chunker()
        self.store = VectorStore(db_dir=db_dir, collection=collection, embedding_model=embedding_model)

    def ingest_pdf(self, pdf_path: str, company: str, report_date: str, doc_type: str = "earnings_report",
                   extra_meta: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        pdf_path = os.path.abspath(pdf_path)
        pages = self.extractor.extract(pdf_path)
        base_meta = {
            "company": company,
            "report_date": report_date,
            "doc_type": doc_type,
            "source_path": pdf_path,
            "md5": md5sum(pdf_path),
        }
        if extra_meta:
            base_meta.update(extra_meta)
        chunks = self.chunker.chunk_pages(pages, base_meta)
        if chunks:
            self.store.add_chunks(chunks)
        return chunks


# -----------------------
# RAG + Ollama
# -----------------------

class RAGQuery:
    def __init__(self, db_dir: str = "vectorstore", collection: str = "finance_docs",
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 ollama_url: str = "http://localhost:11434/api/chat",
                 model_name: str = "llama3"):
        self.store = VectorStore(db_dir=db_dir, collection=collection, embedding_model=embedding_model)
        self.ollama_url = ollama_url
        self.model_name = model_name

    @staticmethod
    def _build_prompt(query: str, retrieved_docs: List[str], metadatas: List[Dict[str, Any]]) -> str:
        sources = []
        for doc, meta in zip(retrieved_docs, metadatas):
            tag = f"{meta.get('company','?')} | {meta.get('report_date','?')} | p.{meta.get('start_page','?')}-{meta.get('end_page','?')}"
            snippet = doc[:1200].replace("\n", " ")
            sources.append(f"[{tag}] {snippet}")
        sources_text = "\n\n".join(sources)

        instruction = (
            "You are a careful financial analyst. Answer ONLY from the provided sources. "
            "When giving numeric answers, extract the exact figure and currency from the sources; do not guess. "
            "If the answer is not in sources, say 'Not found in provided documents.' "
            "Return a concise answer followed by a bullet list of citations [Company | Date | p.start-end]."
        )
        prompt = (
            f"{instruction}\n\n"
            f"User question: {query}\n\n"
            f"Sources:\n{sources_text}\n\n"
            f"Answer:"
        )
        return prompt

    def answer(self, query: str, k: int = 6, temperature: float = 0.1) -> Dict[str, Any]:
        res = self.store.query(query, k=k)
        docs = res["documents"][0] if res and res.get("documents") else []
        metas = res["metadatas"][0] if res and res.get("metadatas") else []
        ids = res["ids"][0] if res and res.get("ids") else []

        prompt = self._build_prompt(query, docs, metas)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "options": {"temperature": temperature}
        }

        try:
            r = requests.post(self.ollama_url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            # Ollama streaming returns chunks; in /api/chat it's usually aggregated as "message"
            if "message" in data and "content" in data["message"]:
                answer_text = data["message"]["content"]
            else:
                # Fallback: try to aggregate 'content' fields if streaming response was proxied
                answer_text = json.dumps(data)[:2000]
        except Exception as e:
            answer_text = f"[Error calling Ollama: {e}]"
        return {
            "answer": answer_text,
            "citations": metas,
            "ids": ids
        }


# -----------------------
# Example helper: query to structured number extraction
# -----------------------

NUMERIC_PAT = re.compile(r"(?<![\d.,-])(?:\$|USD|US\$)?\s?([0-9]{1,3}(?:[,.\s][0-9]{3})*(?:\.[0-9]+)?)\s?(million|billion|thousand|m|bn|k)?", re.I)

def extract_numeric_candidates(text: str) -> List[str]:
    vals = []
    for m in NUMERIC_PAT.finditer(text):
        n, unit = m.group(1), m.group(2) or ""
        vals.append((m.group(0).strip(), n, unit))
    return [v[0] for v in vals]


# -----------------------
# End of module
# -----------------------
