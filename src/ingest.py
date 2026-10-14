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
one page. Parsed pages are cached per parser (git-ignored), so changing the
chunking (E5) re-chunks in seconds instead of re-parsing hundreds of PDFs.

PARSERS (config.PARSER):
  "text"  PyMuPDF's plain get_text(). Tables come out flattened: each number on
          its own line, cut off from its row label.
  "rows"  (E4) rebuild each visual line from word positions; wide horizontal
          gaps become " | ". A table row stays one line:
              Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373)
          Prose is unaffected. ~3 ms/page, no extra dependency. Limitation:
          multi-column page layouts get their columns merged into one row.

    uv run python -m src.ingest     -> parse + chunk the active phase, print stats
"""

import hashlib
import json
import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.fetch_data import CACHE, company_key, fetch, pdfs_for_phase

PAGE_CACHE = {"text": Path("cache/pages"), "rows": Path("cache/pages_rows")}

ROW_Y_TOL = 3.0   # pt: words whose vertical centres differ less than this share a line
COL_GAP = 8.0     # pt: a horizontal gap wider than this between words is a column break
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
    # a few PDFs have no catalog row: fall back to parsing the filename.
    # Find the period token rather than assuming its position, since company
    # names can contain "_" ("JOHNSON_JOHNSON_2022_10K"); the token after it is
    # the type: "AMCOR_2023Q2_10Q" -> period "2023", type "10q"
    row = info.get(doc_name)
    if row:
        doc_type, period = str(row["doc_type"]), str(row["doc_period"])
    else:
        parts = doc_name.split("_")
        i = next(i for i, p in enumerate(parts) if re.match(r"(19|20)\d\d", p))
        doc_type, period = parts[i + 1].lower(), parts[i][:4]
    return {
        "doc_name": doc_name,
        "company": company_key(doc_name),
        "doc_type": doc_type,
        "doc_period": period,
    }


# 2. PARSE ---- PDF -> list of page texts (cached as JSON per document)
def page_rows(page):
    """E4: one line per visual row, " | " between columns."""
    words = page.get_text("words")              # (x0, y0, x1, y1, word, ...)
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows, row, row_y = [], [], None
    for w in words:
        y = (w[1] + w[3]) / 2
        if row and abs(y - row_y) > ROW_Y_TOL:
            rows.append(row)
            row = []
        if not row:
            row_y = y
        row.append(w)
    if row:
        rows.append(row)

    lines = []
    for row in rows:
        row.sort(key=lambda w: w[0])
        line = row[0][4]
        for prev, w in zip(row, row[1:], strict=False):
            line += (" | " if w[0] - prev[2] > COL_GAP else " ") + w[4]
        lines.append(line)
    return "\n".join(lines)


def parse_pdf(pdf_path, parser=None):
    parser = parser or config.PARSER
    cache_dir = PAGE_CACHE[parser]
    cached = cache_dir / f"{pdf_path.stem}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    with pymupdf.open(pdf_path) as pdf:
        pages = [page_rows(p) if parser == "rows" else p.get_text() for p in pdf]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(pages), encoding="utf-8")
    return pages


# 3. LOAD ---- every page of every PDF in the phase, as a Document
def load_pages(phase=None):
    phase = phase or config.PHASE
    info = load_doc_info()
    docs, seen = [], {}
    # catalog names first, so when two files are byte-identical we keep the
    # name the questions cite (AMCOR_2022_8K_dated-2022-07-01, not ..._2022-07-01)
    names = sorted(pdfs_for_phase(phase), key=lambda n: Path(n).stem not in info)
    for name in names:
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
# E5: CHUNKING = "page" keeps each page whole. A financial statement runs
# 3-5k chars, so 1000-char chunks cut most of its rows off from the year header
# ("(Millions) | 2018 | 2017 | 2016") and the model can't tell the columns apart.
# Pages are 3k chars at the median and 10k at most -- far inside the embedding
# model's limit -- and match FinanceBench's page-level labels one to one.
# A page over PAGE_MAX_CHARS (none in Phase 2) still gets split, as a safety net.
PAGE_MAX_CHARS = 12000


def chunk_pages(pages):
    if config.CHUNKING == "page":
        splitter = RecursiveCharacterTextSplitter(chunk_size=PAGE_MAX_CHARS, chunk_overlap=0)
    else:
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
