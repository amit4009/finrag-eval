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
PHASE = os.getenv("FINRAG_PHASE", "phase2")
SPLIT = os.getenv("FINRAG_SPLIT", "dev")      # "dev", "test" or "all"

# ingestion
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "text-embedding-3-large"

# retrieval
RETRIEVER = "filtered"         # "vector" (naive baseline) | "filtered" (E1: company + year filters)
TOP_K = 5

# generation + judging (one judge model everywhere, unlike the tutorial)
GENERATOR_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"


def store_dir():
    """Vector store path, keyed by everything that changes its contents.

    Changing the phase, embedding model or chunking builds a NEW store instead of
    silently reusing a stale one -- and switching back reuses the old one for free.
    """
    emb = EMBEDDING_MODEL.replace("text-embedding-", "")
    return f"chroma_store/{PHASE}_{emb}_c{CHUNK_SIZE}o{CHUNK_OVERLAP}"


def as_dict():
    """What run_suite stamps into each snapshot's metadata."""
    return {
        "phase": PHASE,
        "split": SPLIT,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "retriever": RETRIEVER,
        "top_k": TOP_K,
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
    }
