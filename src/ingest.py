"""
src/ingest.py -- PDF -> page-level chunks with metadata, ready to embed.

Replaces the tutorial's load_transcripts(). Two things matter more here than
with transcripts:

  1. PAGE NUMBERS. Every chunk keeps the 0-based page it came from (PyMuPDF's
     index -- verified to line up exactly with FinanceBench's evidence_page_num).
     That's what lets eval_retriever score retrieval WITHOUT a judge: did a
     retrieved chunk come from the expected (doc_name, page)?

  2. METADATA. doc_name / company / doc_type / doc_period travel with every
     chunk, so later experiments (E1) can filter on them.

Chunks are split WITHIN a page, never across pages, so each chunk has exactly
one page. Parsed pages are cached to cache/pages/ (git-ignored), so changing
the chunking (E5) re-chunks in seconds instead of re-parsing hundreds of PDFs.

    uv run python -m src.ingest     -> parse + chunk the active phase, print stats
"""

import hashlib
import json
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.fetch_data import CACHE, company_key, fetch, pdfs_for_phase

PAGE_CACHE = Path("cache/pages")
DOC_INFO = CACHE / "data/financebench_document_information.jsonl"


# 1. METADATA ---- one dict per doc_name, from FinanceBench's document catalog
def load_doc_info():
    info = {}
    with open(DOC_INFO, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            info[row["doc_name"]] = row
    return info


def doc_metadata(doc_name, info):
    # a few PDFs have no catalog row: fall back to parsing the filename,
    # e.g. "AMCOR_2023Q2_10Q" -> period "2023Q2", type "10q"
    row = info.get(doc_name)
    parts = doc_name.split("_")
    return {
        "doc_name": doc_name,
        "company": company_key(doc_name),
        "doc_type": str(row["doc_type"]) if row else parts[2].lower(),
        "doc_period": str(row["doc_period"]) if row else parts[1],
    }


# 2. PARSE ---- PDF -> list of page texts (cached as JSON per document)
def parse_pdf(pdf_path):
    cached = PAGE_CACHE / f"{pdf_path.stem}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    with pymupdf.open(pdf_path) as pdf:
        pages = [page.get_text() for page in pdf]
    PAGE_CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(pages), encoding="utf-8")
    return pages


# 3. LOAD ---- every page of every PDF in the phase, as a Document
def load_pages(phase=None):
    phase = phase or config.PHASE
    info = load_doc_info()
    docs, seen = [], {}
    for name in pdfs_for_phase(phase):
        path, _ = fetch(f"pdfs/{name}")         # no-op if already cached

        # the source repo has a few byte-identical PDFs under different names;
        # indexing both would put duplicate chunks in the top-k
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            print(f"  skip duplicate {name} (same file as {seen[digest]})")
            continue
        seen[digest] = name

        meta = doc_metadata(path.stem, info)
        for page_num, text in enumerate(parse_pdf(path)):
            if text.strip():
                docs.append(Document(page_content=text, metadata={**meta, "page": page_num}))
    return docs


# 4. CHUNK ---- split within pages; stable ids so re-adding never duplicates
def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    counters = {}
    for c in chunks:
        key = (c.metadata["doc_name"], c.metadata["page"])
        i = counters[key] = counters.get(key, -1) + 1
        c.id = f"{key[0]}:p{key[1]}:c{i}"
    return chunks


def load_chunks(phase=None):
    return chunk_pages(load_pages(phase))


# quick stats: uv run python -m src.ingest
if __name__ == "__main__":
    pages = load_pages()
    chunks = chunk_pages(pages)
    n_docs = len({p.metadata["doc_name"] for p in pages})
    print(f"{config.PHASE}: {n_docs} docs, {len(pages)} non-empty pages, {len(chunks)} chunks")
    c = chunks[len(chunks) // 2]
    print(f"\nsample chunk {c.id}  {c.metadata}\n{c.page_content[:300]}")
