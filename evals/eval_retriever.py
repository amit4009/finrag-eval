"""
evals/eval_retriever.py -- the RETRIEVER, scored in isolation.

Two kinds of metric, side by side:

  JUDGE-FREE (new vs the tutorial) -- FinanceBench labels the exact evidence
  page for every question, so we can check retrieval directly:
    hit_at_k      did any top-k chunk come from an expected (doc_name, page)?
    mrr           1 / rank of the first such chunk (0 if none) -- rewards ranking
    doc_hit_at_k  did any top-k chunk come from the right FILING at all? The gap
                  between this and hit_at_k separates "wrong filing" from "right
                  filing, wrong page"
  These are exact, free, and identical run to run.

  JUDGE -- DeepEval Contextual Recall: does the retrieved text support every
  statement in the ideal answer? Catches the case where the fact appears on a
  page other than the labelled one.

    uv run python -m evals.eval_retriever
"""

from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from evals.harness import (
    load_phase_goldens,
    print_summary,
    summarize_by_metric,
    summarize_scores,
)
from src import config
from src.generator import format_chunk
from src.retriever import build_retriever

THRESHOLD = 0.7


def rank_scores(retrieved, expected):
    """(hit, reciprocal_rank, doc_hit) for one question."""
    pages = {(e["doc_name"], e["page"]) for e in expected}
    docs = {e["doc_name"] for e in expected}
    rr = 0.0
    for rank, d in enumerate(retrieved, 1):
        if (d.metadata["doc_name"], d.metadata["page"]) in pages:
            rr = 1.0 / rank
            break
    doc_hit = any(d.metadata["doc_name"] in docs for d in retrieved)
    return float(rr > 0), rr, float(doc_hit)


def run(retriever):
    goldens = load_phase_goldens()
    k = config.TOP_K

    hits, rrs, doc_hits, test_cases = [], [], [], []
    for g in goldens:
        retrieved = retriever.invoke(g["question"])

        hit, rr, doc_hit = rank_scores(retrieved, g["expected_evidence"])
        hits.append(hit)
        rrs.append(rr)
        doc_hits.append(doc_hit)

        test_cases.append(LLMTestCase(
            input=g["question"],
            expected_output=g["ideal_answer"],
            retrieval_context=[format_chunk(d.metadata["doc_name"], d.metadata["page"], d.page_content)
                               for d in retrieved],
            actual_output="(generator not evaluated in this run)",
        ))

    result = evaluate(
        test_cases=test_cases,
        metrics=[ContextualRecallMetric(threshold=THRESHOLD, model=config.JUDGE_MODEL,
                                        include_reason=True)],
        hyperparameters={**config.as_dict(), "golden_set": f"financebench/{config.PHASE}"},
    )

    summary = summarize_by_metric(result)
    # threshold 1.0: a question passes only on a hit (or a rank-1 hit, for MRR)
    summary.update(summarize_scores(f"hit_at_{k}", hits, 1.0))
    summary.update(summarize_scores("mrr", rrs, 1.0))
    summary.update(summarize_scores(f"doc_hit_at_{k}", doc_hits, 1.0))
    return summary


def run_local():
    """Standalone convenience: build the retriever, then run."""
    return run(build_retriever())


if __name__ == "__main__":
    print_summary("retriever", run_local())
