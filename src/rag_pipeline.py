"""
src/rag_pipeline.py -- retrieve -> generate, in one call.

RagPipeline.invoke(query) returns the three legs of the RAG triad
{query, context, answer} exactly like the tutorial, plus `sources`: the
(doc_name, page) of each retrieved chunk, in rank order. That's what lets
eval_retriever compute hit@k / MRR against FinanceBench's evidence pages
without an LLM judge.

    uv run python -m src.rag_pipeline
"""

from langsmith import traceable

from src.generator import format_chunk, generate
from src.retriever import build_retriever


class RagPipeline:
    def __init__(self):
        # one retriever instance -- loads the store once
        self.retriever = build_retriever()

    @traceable(run_type="chain", name="RagPipeline")
    def invoke(self, query: str) -> dict:
        # 1. RETRIEVE
        docs = self.retriever.invoke(query)

        # 2. UNPACK: labelled chunks for the generator (and the triad metrics),
        #    plus the bare (doc, page) list for judge-free retrieval scoring
        context = [format_chunk(d.metadata["doc_name"], d.metadata["page"], d.page_content)
                   for d in docs]
        sources = [{"doc_name": d.metadata["doc_name"], "page": d.metadata["page"]}
                   for d in docs]

        # 3. GENERATE
        answer = generate(query, context)

        return {"query": query, "context": context, "answer": answer, "sources": sources}


# quick manual smoke test: uv run python -m src.rag_pipeline
if __name__ == "__main__":
    rag = RagPipeline()
    result = rag.invoke("What is the FY2021 capital expenditure amount (in USD billions) for PepsiCo?")
    print("ANSWER: ", result["answer"])
    print("\nSOURCES:", [f"{s['doc_name']} p.{s['page']}" for s in result["sources"]])
