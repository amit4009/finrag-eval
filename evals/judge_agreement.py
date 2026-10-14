"""
evals/judge_agreement.py -- does the LLM judge agree with a human?

Every correctness number in this project comes from a GEval judge (gpt-4o-mini).
This measures how far to trust it: a human labels ~50 pipeline answers as
correct / incorrect, BLIND to the judge's scores, and we compute Cohen's kappa
between the two. The same labels also test the judge-free numeric matcher,
whose leniency (any matching number anywhere in the answer counts) was flagged
as a known risk in evals/numeric.py.

    uv run python -m evals.judge_agreement prepare   # answer + judge 50 dev questions,
                                                     # write the blank labeling sheet
    (label goldens/judge_labels.csv: column `correct` = Y or N)
    uv run python -m evals.judge_agreement score     # kappa, confusion matrix

Files:
  goldens/judge_labels.csv             the sheet a human fills in (no judge scores in it)
  goldens/judge_agreement_answers.json answers + judge scores, read only by `score`
"""

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from evals.eval_application import THRESHOLD, correctness_metric
from evals.harness import JUDGE_RUN, load_phase_goldens
from evals.numeric import numeric_match
from src import config
from src.rag_pipeline import RagPipeline

SHEET = Path("goldens/judge_labels.csv")
ANSWERS = Path("goldens/judge_agreement_answers.json")
SAMPLE_SIZE = 50
SEED = "judge-agreement-v1"


def sample(goldens):
    """SAMPLE_SIZE dev questions, stratified by question_type, by seeded hash."""
    by_type = defaultdict(list)
    for g in goldens:
        by_type[g["question_type"]].append(g)
    picked = []
    for _qtype, group in sorted(by_type.items()):
        group.sort(key=lambda g: hashlib.sha256(f"{SEED}:{g['id']}".encode()).hexdigest())
        picked += group[: round(SAMPLE_SIZE * len(group) / len(goldens))]
    return picked[:SAMPLE_SIZE]


def prepare():
    if SHEET.exists():
        raise SystemExit(f"{SHEET} exists -- refusing to overwrite labels. Delete it to start over.")
    picked = sample(load_phase_goldens())
    rag = RagPipeline()

    rows, cases = [], []
    for i, g in enumerate(picked, 1):
        answer = rag.invoke(g["question"])["answer"]
        rows.append({"n": i, "id": g["id"], "question_type": g["question_type"],
                     "question": g["question"], "expert_answer": g["ideal_answer"],
                     "model_answer": answer})
        cases.append(LLMTestCase(name=g["id"], input=g["question"], actual_output=answer,
                                 expected_output=g["ideal_answer"]))
        print(f"  answered {i}/{len(picked)}", end="\r", flush=True)
    print()

    result = evaluate(**JUDGE_RUN, test_cases=cases, metrics=[correctness_metric()])
    judged = {tr.name: (tr.metrics_data or [None])[0] for tr in result.test_results}

    record = []
    for r, g in zip(rows, picked, strict=True):
        m = judged.get(r["id"])
        record.append({**r, "judge_score": getattr(m, "score", None),
                       "judge_error": getattr(m, "error", None),
                       "numeric": g["numeric"],
                       "numeric_match": (numeric_match(r["model_answer"], g["ideal_answer"],
                                                       g["question"]) if g["numeric"] else None)})
    ANSWERS.write_text(json.dumps({"config": config.as_dict(), "threshold": THRESHOLD,
                                   "answers": record}, indent=2, ensure_ascii=False),
                       encoding="utf-8")

    # utf-8-sig so Excel shows apostrophes and currency symbols correctly
    with SHEET.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["n", "id", "question_type", "question",
                                          "expert_answer", "model_answer", "correct", "note"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "correct": "", "note": ""})
    print(f"wrote {SHEET} ({len(rows)} rows to label) and {ANSWERS}")


def kappa(a, b):
    """Cohen's kappa for two lists of booleans."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def report(name, human, machine):
    tp = sum(h and m for h, m in zip(human, machine, strict=True))
    tn = sum(not h and not m for h, m in zip(human, machine, strict=True))
    fp = sum(not h and m for h, m in zip(human, machine, strict=True))
    fn = sum(h and not m for h, m in zip(human, machine, strict=True))
    n = len(human)
    print(f"\n{name}  (n={n})")
    print(f"  agreement {(tp + tn) / n:.2f}   Cohen's kappa {kappa(human, machine):.2f}")
    print(f"  {'':<22}{'human: correct':>16}{'human: wrong':>14}")
    print(f"  {'machine: correct':<22}{tp:>16}{fp:>14}   <- false passes")
    print(f"  {'machine: wrong':<22}{fn:>16}{tn:>14}")
    return {"n": n, "agreement": (tp + tn) / n, "kappa": kappa(human, machine),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def score():
    with SHEET.open(encoding="utf-8-sig") as f:
        labels = {r["id"]: r["correct"].strip().upper()[:1] for r in csv.DictReader(f)}
    bad = {k: v for k, v in labels.items() if v not in ("Y", "N")}
    if bad:
        raise SystemExit(f"{len(bad)} rows not labelled Y or N yet: {sorted(bad)[:5]}...")
    data = json.loads(ANSWERS.read_text(encoding="utf-8"))
    answers = [a for a in data["answers"] if a["judge_score"] is not None]

    human = [labels[a["id"]] == "Y" for a in answers]
    judge = [a["judge_score"] >= data["threshold"] for a in answers]
    out = {"judge": report(f"GEval correctness (score >= {data['threshold']})", human, judge)}

    num = [a for a in answers if a["numeric"]]
    if num:
        out["numeric_match"] = report("numeric match (metrics-generated questions)",
                                      [labels[a["id"]] == "Y" for a in num],
                                      [a["numeric_match"] for a in num])

    print("\njudge score by human label (to see whether a different threshold would agree better):")
    for lab in (True, False):
        s = sorted(round(a["judge_score"], 2) for a, h in zip(answers, human, strict=True) if h == lab)
        print(f"  human {'correct' if lab else 'wrong  '}: {s}")
    Path("baselines/judge_agreement.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"prepare": prepare, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "prepare"]()
