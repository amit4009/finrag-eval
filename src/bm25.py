"""
src/bm25.py -- keyword (BM25) search over the same chunks as the vector store (E2).

Embeddings match MEANING, which is why "FY2021 capex" found capex sections of
every year. BM25 matches the exact WORDS, which filings repeat verbatim:
"Purchases of property, plant and equipment", "restructuring charges",
"EBITDAR". E2 runs both and fuses the two rankings (src/retriever.py).

The index is built from the Chroma store itself -- same chunk ids, same text,
same metadata -- so the two searches cover exactly the same corpus.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def get_all(store, include, page=5000):
    """store.get() in pages: one call for all 50k chunks exceeds SQLite's variable limit."""
    out = {"ids": [], **{k: [] for k in include}}
    offset = 0
    while True:
        part = store.get(include=include, limit=page, offset=offset)
        if not part["ids"]:
            return out
        for key in out:
            out[key].extend(part[key])
        offset += page


def matches(meta, where):
    """Evaluate the `where` clauses chroma_where() builds against one chunk's metadata."""
    if where is None:
        return True
    if "$and" in where:
        return all(matches(meta, w) for w in where["$and"])
    (field, cond), = where.items()
    return meta.get(field) in cond["$in"]


class BM25Index:
    def __init__(self, store):
        data = get_all(store, ["documents", "metadatas"])
        self.ids = data["ids"]
        self.texts = data["documents"]
        self.metas = data["metadatas"]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])

    def search(self, query, k, where=None):
        """Top-k (id, text, metadata) for the query, among chunks matching `where`."""
        scores = self.bm25.get_scores(tokenize(query))
        if where is not None:
            mask = np.array([matches(m, where) for m in self.metas])
            scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], self.texts[i], self.metas[i]) for i in top if scores[i] > 0]
