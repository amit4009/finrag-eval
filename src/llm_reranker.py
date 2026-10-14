"""
src/llm_reranker.py -- an LLM as the reranker, on top of E1's filters (E7).

E3's cross-encoder (ms-marco-MiniLM) failed because it preferred prose that
echoed the question over the financial-statement TABLES that hold the answers.
Here gpt-4o-mini does the reranking: it reads tables the way the generator does.

Per filtered search: take the 20 best pages by vector similarity, show the LLM
each page's label and its first RERANK_EXCERPT_CHARS characters (statement title,
year header and top rows), and ask for the 5 most useful, best first.

Trade-offs, measured in E7: one extra API call per question (cost, ~2-4 s of
latency), and retrieval stops being exactly repeatable -- an LLM can rank
differently run to run. Decisions are cached within a run, so every eval in one
run sees the same ranking for the same question.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

from src import config
from src.retriever import FilteredRetriever


class Ranking(BaseModel):
    ranking: list[int] = Field(description="candidate numbers, most useful first")


PROMPT = ChatPromptTemplate.from_template(
    """You are ranking excerpts from company filings by how useful they are for answering a question.

Question:
{question}

Candidates (each shows only the beginning of a page):
{candidates}

Return the numbers of the {top_k} most useful candidates, most useful first. Prefer the page that
contains the specific figures or statement the question needs (for example the income statement,
balance sheet or cash flow statement for the right company and fiscal year) over pages that only
discuss the topic."""
)


class LLMRerankRetriever(FilteredRetriever):
    def __init__(self, store, top_k):
        super().__init__(store, top_k)
        llm = ChatOpenAI(model=config.RERANK_MODEL, temperature=0)
        self.chain = PROMPT | llm.with_structured_output(Ranking)
        self.cache = {}            # (query, candidate ids) -> ranked ids, for this run

    @traceable(run_type="retriever", name="LLMRerankRetriever")
    def search(self, query, where):
        # 1. OVER-RETRIEVE inside E1's filter
        candidates = self.store.similarity_search(query, k=config.RERANK_FETCH_K, filter=where)
        if len(candidates) <= self.top_k:
            return candidates

        key = (query, tuple(d.id for d in candidates))
        if key not in self.cache:
            self.cache[key] = self._rank(query, candidates)
        order = self.cache[key]
        return [candidates[i] for i in order[: self.top_k]]

    def _rank(self, query, candidates):
        listing = "\n\n".join(
            f"[{i}] {d.metadata['doc_name']} p.{d.metadata['page']}\n"
            f"{d.page_content[: config.RERANK_EXCERPT_CHARS]}"
            for i, d in enumerate(candidates, 1)
        )
        try:
            picked = self.chain.invoke({"question": query, "candidates": listing,
                                        "top_k": self.top_k}).ranking
        except Exception as e:  # noqa: BLE001 -- a failed rerank must not fail retrieval
            print(f"  rerank failed ({type(e).__name__}); keeping vector order")
            picked = []
        # 1-based -> 0-based, drop invalid / repeated numbers, then fill up in vector order
        order = []
        for n in picked:
            if 1 <= n <= len(candidates) and n - 1 not in order:
                order.append(n - 1)
        order += [i for i in range(len(candidates)) if i not in order]
        return order
