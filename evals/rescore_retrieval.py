"""
evals/rescore_retrieval.py -- re-score past experiments' retrieval under a new rule.

Retrieval metrics are deterministic (no LLM), so when the scoring rule changes
-- here, counting duplicate evidence pages as hits -- every past configuration
can be re-scored exactly, without re-running its judges. Snapshots stay as they
were recorded; this writes a side table with both rules.

    uv run python -m evals.rescore_retrieval   -> baselines/retrieval_rescored.json
"""

import json
import sys
from pathlib import Path

from evals.eval_retriever import load_equivalents, rank_scores
from evals.harness import load_phase_goldens
from src import config
from src.retriever import build_retriever

OUT = Path("baselines/retrieval_rescored.json")

# (name, parser, retriever) for every experiment so far
EXPERIMENTS = [
    ("naive", "text", "vector"),
    ("E1 filters", "text", "filtered"),
    ("E2 hybrid", "text", "hybrid"),
    ("E3 rerank", "text", "rerank"),
    ("E4 rows parser", "rows", "filtered"),
]


def score(retriever, goldens, equivalents):
    rows = [rank_scores(retriever.invoke(g["question"]), g["expected_evidence"],
                        equivalents.get(g["id"], ())) for g in goldens]
    n = len(rows)
    return {"hit_at_5": sum(r[0] for r in rows) / n, "mrr": sum(r[1] for r in rows) / n,
            "doc_hit_at_5": sum(r[2] for r in rows) / n, "n": n}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    goldens, equivalents = load_phase_goldens(), load_equivalents()
    table = {}
    for name, parser, retriever in EXPERIMENTS:
        config.PARSER, config.RETRIEVER = parser, retriever
        r = build_retriever()
        table[name] = {"old_rule": score(r, goldens, {}), "duplicates_count": score(r, goldens, equivalents)}
        o, d = table[name]["old_rule"], table[name]["duplicates_count"]
        print(f"{name:<15} hit@5 {o['hit_at_5']:.3f} -> {d['hit_at_5']:.3f}   "
              f"mrr {o['mrr']:.3f} -> {d['mrr']:.3f}   doc_hit {o['doc_hit_at_5']:.3f} -> {d['doc_hit_at_5']:.3f}")
    OUT.write_text(json.dumps({"phase": config.PHASE, "split": config.SPLIT, "experiments": table},
                              indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
