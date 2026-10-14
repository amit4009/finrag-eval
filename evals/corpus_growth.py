"""
evals/corpus_growth.py -- how retrieval degrades as the corpus grows, and
whether metadata filters stop it.

The QUESTIONS are held fixed while the CORPUS grows (1 -> 8 -> 40 companies),
so any change is caused by the added distractor filings alone:

    PepsiCo dev questions   on corpora of 1, 8 and 40 companies
    Phase 2 dev questions   on corpora of 8 and 40 companies

each with filters OFF (plain vector search) and ON (E1). Parsing and chunking
are the current config (rows parser, one chunk per page). Retrieval metrics only:
deterministic, no LLM calls beyond embedding the questions.

    uv run python -m evals.corpus_growth   -> baselines/corpus_growth.json
"""

import json
import sys
from pathlib import Path

from evals.eval_retriever import load_equivalents, rank_scores
from evals.harness import load_phase_goldens
from src import config
from src.retriever import FilteredRetriever, VectorRetriever, load_store

OUT = Path("baselines/corpus_growth.json")
CORPORA = {"phase1": 1, "phase2": 8, "phase3": 40}            # phase -> companies indexed
QUESTION_SETS = {"PepsiCo dev": "phase1", "Phase 2 dev": "phase2"}
RETRIEVERS = {"filters off": VectorRetriever, "filters on": FilteredRetriever}


def score(retriever, goldens, equivalents):
    rows = [rank_scores(retriever.invoke(g["question"]), g["expected_evidence"],
                        equivalents.get(g["id"], ())) for g in goldens]
    n = len(rows)
    return {"n": n, "hit_at_5": sum(r[0] for r in rows) / n,
            "mrr": sum(r[1] for r in rows) / n, "doc_hit_at_5": sum(r[2] for r in rows) / n}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    equivalents = load_equivalents()
    questions = {name: load_phase_goldens(phase, "dev") for name, phase in QUESTION_SETS.items()}
    results = []
    for phase, n_companies in CORPORA.items():
        config.PHASE = phase
        store = load_store()
        for qname, qphase in QUESTION_SETS.items():
            if list(CORPORA).index(qphase) > list(CORPORA).index(phase):
                continue                     # questions need their companies in the corpus
            for rname, cls in RETRIEVERS.items():
                s = score(cls(store, config.TOP_K), questions[qname], equivalents)
                results.append({"questions": qname, "companies": n_companies,
                                "pages": store._collection.count(), "retriever": rname, **s})
                print(f"{qname:<12} {n_companies:>2} companies ({results[-1]['pages']:>6} pages)  "
                      f"{rname:<11}  hit@5 {s['hit_at_5']:.3f}  doc_hit {s['doc_hit_at_5']:.3f}  "
                      f"mrr {s['mrr']:.3f}  (n={s['n']})")
    OUT.write_text(json.dumps({"parser": config.PARSER, "chunking": config.CHUNKING,
                               "top_k": config.TOP_K, "results": results}, indent=2),
                   encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
