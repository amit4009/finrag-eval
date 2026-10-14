"""
goldens/build_goldens.py -- FinanceBench jsonl -> our golden format.

Replaces the tutorial's hand-written golden files. One file holds all 150
expert-written questions; evals filter it to the active phase's companies at
load time (evals.harness.load_phase_goldens), so changing phase never means
rebuilding goldens.

Each golden carries everything the different evals need:

    question, ideal_answer      -> eval_application (GEval correctness, numeric match)
    expected_evidence           -> eval_retriever (hit@k / MRR on (doc_name, page))
    golden_context              -> eval_generator (generator tested in isolation)
    question_type, numeric      -> per-type reporting; which questions get numeric match
    split                       -> "dev" (tune on these) or "test" (scored once, at the end)

TEST-SET HYGIENE: this file is the exam, not the library. Nothing here is ever
indexed or put in a prompt outside eval_generator's isolated test.

    uv run python -m goldens.build_goldens
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from src.fetch_data import CACHE, company_key
from src.generator import format_chunk

SOURCE = CACHE / "data/financebench_open_source.jsonl"
OUT = Path("goldens/financebench.json")

# Frozen dev/test split. ~30% test, stratified by question_type, chosen by a
# seeded hash of the question id: deterministic, and decided once over all 150
# questions, so a question's split never depends on which phase is active.
# Changing SPLIT_SEED or TEST_FRACTION re-deals the test set -- don't.
SPLIT_SEED = "finrag-eval-v1"
TEST_FRACTION = 0.3


def assign_splits(goldens):
    by_type = defaultdict(list)
    for g in goldens:
        by_type[g["question_type"]].append(g)
    for group in by_type.values():
        group.sort(key=lambda g: hashlib.sha256(f"{SPLIT_SEED}:{g['id']}".encode()).hexdigest())
        n_test = round(TEST_FRACTION * len(group))
        for i, g in enumerate(group):
            g["split"] = "test" if i < n_test else "dev"


def build():
    goldens = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)

        evidence, context = [], []
        for e in q["evidence"]:
            key = {"doc_name": e["doc_name"], "page": e["evidence_page_num"]}
            if key in evidence:                       # same page cited twice
                continue
            evidence.append(key)
            context.append(format_chunk(e["doc_name"], e["evidence_page_num"],
                                        e["evidence_text_full_page"]))

        goldens.append({
            "id": q["financebench_id"],
            "company": company_key(q["doc_name"]),
            "doc_name": q["doc_name"],
            "question_type": q["question_type"],
            "question_reasoning": q["question_reasoning"],
            # every metrics-generated answer is a bare number ("$4.60", "16.5%")
            "numeric": q["question_type"] == "metrics-generated",
            "question": q["question"],
            "ideal_answer": q["answer"],
            "justification": q["justification"],
            "expected_evidence": evidence,
            "golden_context": context,
        })

    assign_splits(goldens)
    OUT.write_text(json.dumps(goldens, indent=2, ensure_ascii=False), encoding="utf-8")
    return goldens


if __name__ == "__main__":
    goldens = build()
    n_test = sum(g["split"] == "test" for g in goldens)
    print(f"wrote {len(goldens)} goldens ({len(goldens) - n_test} dev / {n_test} test) -> {OUT}")
