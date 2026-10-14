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

  DUPLICATE EVIDENCE: a page that is a near-copy of a gold page, in another
  filing for the same company and period (AMD's annual report is its 10-K
  repackaged), counts as a hit. goldens/build_equivalents.py measures these.

  E1 DIAGNOSTIC (filtered retriever only):
    filter_accuracy  does the first filter tried (company + latest year read off
                     the question) target the gold filing's company and year?
                     Separates "extractor misread the question" from "right
                     filing, still ranked wrong"

  JUDGE -- DeepEval Contextual Recall: does the retrieved text support every
  statement in the ideal answer? Catches the case where the fact appears on a
  page other than the labelled one.

    uv run python -m evals.eval_retriever
"""

import json
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from evals.harness import (
    JUDGE_RUN,
    load_phase_goldens,
    print_summary,
    record,
    summarize_by_metric,
    summarize_scores,
)
from src import config
from src.generator import format_chunk
from src.ingest import doc_metadata, load_doc_info
from src.query_filters import extract_filters
from src.retriever import build_retriever

THRESHOLD = 0.7
EQUIVALENTS = Path("goldens/evidence_equivalents.json")


def load_equivalents():
    """question id -> [{doc_name, page}] of same-period copies of its gold pages."""
    if not EQUIVALENTS.exists():
        return {}
    data = json.loads(EQUIVALENTS.read_text(encoding="utf-8"))
    return {qid: [{"doc_name": e["doc_name"], "page": e["page"]} for e in items]
            for qid, items in data["equivalents"].items()}


def rank_scores(retrieved, expected, equivalents=()):
    """(hit, reciprocal_rank, doc_hit) for one question. A retrieved chunk counts
    if it comes from an expected page or a measured copy of one."""
    targets = list(expected) + list(equivalents)
    pages = {(e["doc_name"], e["page"]) for e in targets}
    docs = {e["doc_name"] for e in targets}
    rr = 0.0
    for rank, d in enumerate(retrieved, 1):
        if (d.metadata["doc_name"], d.metadata["page"]) in pages:
            rr = 1.0 / rank
            break
    doc_hit = any(d.metadata["doc_name"] in docs for d in retrieved)
    return float(rr > 0), rr, float(doc_hit)


def filter_targets_gold(question, gold_doc_name, info):
    gold = doc_metadata(gold_doc_name, info)
    f = extract_filters(question)
    return float(gold["company"] in f["companies"]
                 and f["years"][:1] == [gold["doc_period"]])


def run(retriever):
    goldens = load_phase_goldens()
    k = config.TOP_K

    use_filters = config.RETRIEVER in ("filtered", "hybrid", "rerank")
    info = load_doc_info()
    equivalents = load_equivalents()

    hits, rrs, doc_hits, filter_ok, test_cases = [], [], [], [], []
    for g in goldens:
        retrieved = retriever.invoke(g["question"])
        record("retriever", g["id"], sources=[{"doc_name": d.metadata["doc_name"],
                                               "page": d.metadata["page"]} for d in retrieved])

        hit, rr, doc_hit = rank_scores(retrieved, g["expected_evidence"],
                                       equivalents.get(g["id"], ()))
        hits.append(hit)
        rrs.append(rr)
        doc_hits.append(doc_hit)
        if use_filters:
            filter_ok.append(filter_targets_gold(g["question"], g["doc_name"], info))

        test_cases.append(LLMTestCase(
            input=g["question"],
            expected_output=g["ideal_answer"],
            retrieval_context=[format_chunk(d.metadata["doc_name"], d.metadata["page"], d.page_content)
                               for d in retrieved],
            actual_output="(generator not evaluated in this run)",
        ))

    result = evaluate(
        **JUDGE_RUN,
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
    if use_filters:
        summary.update(summarize_scores("filter_accuracy", filter_ok, 1.0))
    return summary


def run_local():
    """Standalone convenience: build the retriever, then run."""
    return run(build_retriever())


if __name__ == "__main__":
    print_summary("retriever", run_local())
