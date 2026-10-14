"""
src/config.py -- the single place that says WHICH pipeline we're running.

Every knob an experiment might turn lives here, so (a) an experiment is a one-
line change, and (b) run_suite can stamp the exact config into each snapshot.
A snapshot that doesn't record what it measured can't be compared honestly.

The phase can be overridden per run without editing code:

    FINRAG_PHASE=phase2 uv run python -m evals.run_suite
    FINRAG_SPLIT=test   ...   (the held-out set: run ONCE, at the very end)
"""

import os

from dotenv import load_dotenv

load_dotenv()

# which corpus (config/corpus.yaml) to index and which questions to score
PHASE = os.getenv("FINRAG_PHASE", "phase3")
SPLIT = os.getenv("FINRAG_SPLIT", "dev")      # "dev", "test" or "all"

# ingestion
PARSER = "rows"               # "text" (PyMuPDF plain text) | "rows" (E4: table rows kept on one line)
CHUNKING = "page"             # "chars" (CHUNK_SIZE / CHUNK_OVERLAP) | "page" (E5: one chunk per page)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "text-embedding-3-large"

# retrieval
RETRIEVER = "filtered"        # "vector" (naive) | "filtered" (E1: company + year filters)
                              # | "hybrid" (E2: E1 + BM25, fused with RRF; rejected)
                              # | "rerank" (E3: E1 + cross-encoder reranking)
TOP_K = 5
HYBRID_FETCH_K = 20           # E2: candidates from EACH of vector and BM25 before fusion
RRF_K = 60                    # E2: reciprocal rank fusion constant (the standard default)
RERANK_FETCH_K = 20           # E3: filtered vector candidates the cross-encoder re-scores
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# generation + judging (one judge model everywhere, unlike the tutorial)
GENERATOR_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"


def store_dir():
    """Vector store path, keyed by everything that changes its contents.

    Changing the phase, embedding model or chunking builds a NEW store instead of
    silently reusing a stale one -- and switching back reuses the old one for free.
    """
    emb = EMBEDDING_MODEL.replace("text-embedding-", "")
    parser = "" if PARSER == "text" else f"_{PARSER}"     # keeps pre-E4 store paths valid
    chunks = "cpage" if CHUNKING == "page" else f"c{CHUNK_SIZE}o{CHUNK_OVERLAP}"
    return f"chroma_store/{PHASE}_{emb}_{chunks}{parser}"


def as_dict():
    """What run_suite stamps into each snapshot's metadata."""
    return {
        "phase": PHASE,
        "split": SPLIT,
        "parser": PARSER,
        "chunking": CHUNKING,
        **({"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP} if CHUNKING == "chars" else {}),
        "embedding_model": EMBEDDING_MODEL,
        "retriever": RETRIEVER,
        "top_k": TOP_K,
        **({"hybrid_fetch_k": HYBRID_FETCH_K, "rrf_k": RRF_K} if RETRIEVER == "hybrid" else {}),
        **({"rerank_fetch_k": RERANK_FETCH_K, "reranker_model": RERANKER_MODEL}
           if RETRIEVER == "rerank" else {}),
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
    }
