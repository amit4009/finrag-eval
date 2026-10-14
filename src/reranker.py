"""
src/reranker.py -- cross-encoder reranking on top of E1's filters (E3).

Same idea as the tutorial's RerankingRetriever: over-retrieve with the fast
bi-encoder, then re-score each (question, chunk) pair TOGETHER with a cross-
encoder and keep the best. A bi-encoder embeds question and chunk separately,
so it can't tell "FY2018 capex" from a page that merely mentions capex; a
cross-encoder reads both at once.

Why it should help here: after E1 the right FILING is in the top 5 for 94% of
dev questions but the right PAGE for only 60%. Reranking 20 candidates from the
right filing targets exactly that gap.

The model is small and CPU-friendly; it downloads once (~90 MB) on first use.
"""

from langsmith import traceable
from sentence_transformers import CrossEncoder

from src import config
from src.retriever import FilteredRetriever


class RerankingRetriever(FilteredRetriever):
    def __init__(self, store, top_k):
        super().__init__(store, top_k)
        self.reranker = CrossEncoder(config.RERANKER_MODEL)

    @traceable(run_type="retriever", name="RerankingRetriever")
    def search(self, query, where):
        # 1. OVER-RETRIEVE: deliberately more than we need, inside E1's filter
        candidates = self.store.similarity_search(query, k=config.RERANK_FETCH_K, filter=where)
        if not candidates:
            return []

        # 2. RERANK: score each (query, chunk) pair together
        scores = self.reranker.predict([(query, d.page_content) for d in candidates])

        # 3. keep the best top_k
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]
