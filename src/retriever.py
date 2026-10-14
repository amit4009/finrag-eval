"""
src/retriever.py -- the RETRIEVER component (same role as the tutorial's).

    load_store()       -> the Chroma store for the active phase (built once, then on disk)
    build_retriever()  -> whatever config.RETRIEVER names; .invoke(query) -> [Document]

The store is built in batches with stable chunk ids, so an interrupted build
resumes where it stopped instead of re-embedding (and re-paying for) everything.

    uv run python -m src.retriever   -> build the store if needed, run a test query
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src import config
from src.bm25 import BM25Index, get_all
from src.ingest import load_chunks
from src.query_filters import chroma_where, extract_filters

BATCH = 200   # chunks per embedding request (~50k tokens, well under the API cap)


# 1. BUILD ---- chunk, embed once, keep it on disk so we never re-embed
def load_store():
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    path = config.store_dir()
    store = Chroma(persist_directory=path, embedding_function=embeddings)

    chunks = load_chunks()
    have = store._collection.count()
    if have >= len(chunks):
        return store

    existing = set(get_all(store, [])["ids"]) if have else set()
    todo = [c for c in chunks if c.id not in existing]
    print(f"building {path}: embedding {len(todo)} of {len(chunks)} chunks...")
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        store.add_documents(batch, ids=[c.id for c in batch])
        print(f"  {min(i + BATCH, len(todo))}/{len(todo)}", end="\r", flush=True)
    print()
    return store


# 2. RETRIEVE ---- Phase 1: plain vector similarity, top-k
class VectorRetriever:
    def __init__(self, store, top_k):
        self.store = store
        self.top_k = top_k

    def invoke(self, query):
        return self.store.similarity_search(query, k=self.top_k)


# E1: metadata filters. Narrow the search to the company + fiscal year named in
# the question, widening step by step if a filter matches nothing:
#   company + latest year -> company + any year named -> company -> no filter
class FilteredRetriever:
    def __init__(self, store, top_k):
        self.store = store
        self.top_k = top_k
        self.last_where = None     # the filter that produced the last results

    def attempts(self, query):
        f = extract_filters(query)
        companies, years = f["companies"], f["years"]
        wheres = [chroma_where(companies, years[:1]), chroma_where(companies, years),
                  chroma_where(companies, []), None]
        return [w for i, w in enumerate(wheres) if w not in wheres[:i]]   # dedupe, keep order

    def search(self, query, where):
        return self.store.similarity_search(query, k=self.top_k, filter=where)

    def invoke(self, query):
        for where in self.attempts(query):
            docs = self.search(query, where)
            if docs:
                self.last_where = where
                return docs
        self.last_where = None
        return []


# E2: hybrid search. Same filter cascade as E1, but each filtered search runs
# BM25 (exact words) alongside the vector search (meaning) and fuses the two
# rankings with reciprocal rank fusion: score(chunk) = sum over the lists it
# appears in of 1 / (RRF_K + rank). Rank-based, so BM25's unbounded scores and
# cosine similarities never have to be put on one scale.
class HybridRetriever(FilteredRetriever):
    def __init__(self, store, top_k):
        super().__init__(store, top_k)
        self.bm25 = BM25Index(store)

    def search(self, query, where):
        k = config.HYBRID_FETCH_K
        vector = [(d.id, d) for d in self.store.similarity_search(query, k=k, filter=where)]
        keyword = [(cid, Document(page_content=text, metadata=meta, id=cid))
                   for cid, text, meta in self.bm25.search(query, k, where)]

        scores, docs = {}, {}
        for ranking in (vector, keyword):
            for rank, (cid, doc) in enumerate(ranking, 1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (config.RRF_K + rank)
                docs.setdefault(cid, doc)
        best = sorted(scores, key=scores.get, reverse=True)[: self.top_k]
        return [docs[cid] for cid in best]


RETRIEVERS = {"vector": VectorRetriever, "filtered": FilteredRetriever,
              "hybrid": HybridRetriever}


def build_retriever():
    if config.RETRIEVER == "rerank":
        from src.reranker import (
            RerankingRetriever,  # here: reranker imports this module
        )
        return RerankingRetriever(load_store(), config.TOP_K)
    if config.RETRIEVER in RETRIEVERS:
        return RETRIEVERS[config.RETRIEVER](load_store(), config.TOP_K)
    raise ValueError(f"unknown retriever {config.RETRIEVER!r}")


# 3. TRY IT ---- uv run python -m src.retriever
if __name__ == "__main__":
    retriever = build_retriever()
    for d in retriever.invoke("What was PepsiCo's FY2021 capital expenditure?"):
        m = d.metadata
        print(f"[{m['doc_name']} p.{m['page']}] {d.page_content[:150]!r}\n")
