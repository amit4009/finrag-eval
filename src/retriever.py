"""
src/retriever.py -- the RETRIEVER component (same role as the tutorial's).

    load_store()       -> the Chroma store for the active phase (built once, then on disk)
    build_retriever()  -> whatever config.RETRIEVER names; .invoke(query) -> [Document]

The store is built in batches with stable chunk ids, so an interrupted build
resumes where it stopped instead of re-embedding (and re-paying for) everything.

    uv run python -m src.retriever   -> build the store if needed, run a test query
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src import config
from src.ingest import load_chunks

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

    existing = set(store.get(include=[])["ids"]) if have else set()
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


def build_retriever():
    if config.RETRIEVER == "vector":
        return VectorRetriever(load_store(), config.TOP_K)
    raise ValueError(f"unknown retriever {config.RETRIEVER!r}")


# 3. TRY IT ---- uv run python -m src.retriever
if __name__ == "__main__":
    retriever = build_retriever()
    for d in retriever.invoke("What was PepsiCo's FY2021 capital expenditure?"):
        m = d.metadata
        print(f"[{m['doc_name']} p.{m['page']}] {d.page_content[:150]!r}\n")
