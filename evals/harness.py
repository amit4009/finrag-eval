# evals/harness.py
"""
Shared harness for the eval suite.

The eval files each run their own DeepEval evaluation and return results; this
module holds the two pieces they all share:

  load_goldens(path)          -- read a golden JSON file
  summarize_by_metric(result) -- turn a DeepEval EvaluationResult into a
                                 PER-METRIC summary (pass rate + score stats)

Why per-metric and not pooled: a single eval file often runs several metrics at
once (correctness + completeness + style; the RAG triad; faithfulness +
relevancy). Pooling them into one number would let a regression in ONE metric
hide behind the others. summarize_by_metric keeps each metric separate, which is
exactly what the regression suite compares against the baseline.

The extractor is defensive across DeepEval versions: test results may live on
`.test_results` (or be a bare list), and per-test metrics on `.metrics_data`
(newer) or `.metrics` (older). It keys pass/fail off each metric's own `success`
flag, so it stays correct whether higher or lower score is "good".
"""

import json

from deepeval.evaluate import AsyncConfig, CacheConfig, ErrorConfig

FINANCEBENCH_GOLDENS = "goldens/financebench.json"

# Per-question outputs of the current run (answers, retrieved pages), written
# next to the snapshot by run_suite. Snapshots keep only aggregates; these let a
# later scoring-rule change re-score an old run exactly instead of re-running it.
RECORDS = []


def record(eval_name, golden_id, **fields):
    RECORDS.append({"eval": eval_name, "id": golden_id, **fields})


# Don't write DeepEval's test-run cache: it crashes on a second run in this
# version, and every snapshot should be a fresh measurement anyway.
NO_CACHE = CacheConfig(write_cache=False, use_cache=False)

# DeepEval's default runs 20 test cases at once, each firing several judge calls
# with whole 10-K excerpts. At Phase 2 size that queues requests at the API until
# they time out. 5 at a time is slower but finishes.
JUDGE_ASYNC = AsyncConfig(max_concurrent=5)

# A judge call can fail outright: gpt-4o-mini occasionally loops until it hits
# the 16k output-token limit on a structured judging step, and retrying
# reproduces it. Record that as an error on the one metric instead of failing
# the whole run. summarize_by_metric keeps errored scores out of the averages
# and counts them as n_errors, and run_suite stamps the total into the snapshot.
JUDGE_ERRORS = ErrorConfig(ignore_errors=True)

# everything every evaluate() call in the suite shares
JUDGE_RUN = {"cache_config": NO_CACHE, "async_config": JUDGE_ASYNC, "error_config": JUDGE_ERRORS}


def load_goldens(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_phase_goldens(phase=None, split=None):
    """The FinanceBench goldens for this phase's companies, in this split."""
    from src import config
    from src.fetch_data import company_keys_for_phase

    companies = company_keys_for_phase(phase or config.PHASE)
    split = split or config.SPLIT
    return [g for g in load_goldens(FINANCEBENCH_GOLDENS)
            if (companies is None or g["company"] in companies)
            and (split == "all" or g["split"] == split)]


def summarize_scores(name, scores, threshold):
    """Per-metric summary for metrics we compute ourselves (no DeepEval result),
    in the same shape summarize_by_metric returns, so run_suite flattens both alike."""
    n = len(scores)
    return {name: {
        "n": n,
        "pass_rate": (100 * sum(s >= threshold for s in scores) / n) if n else 0.0,
        "avg_score": (sum(scores) / n) if n else float("nan"),
        "min_score": min(scores) if n else float("nan"),
        "max_score": max(scores) if n else float("nan"),
    }}


def summarize_by_metric(result, keep=None):
    """DeepEval EvaluationResult -> {metric_name: {n, pass_rate, avg/min/max score}}.

    keep(test_result) -> bool restricts the summary to some test cases, e.g. only
    the questions the model actually answered.
    """
    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []
    if keep is not None:
        test_results = [tr for tr in test_results if keep(tr)]

    buckets = {}   # metric_name -> {"scores": [...], "passed": int, "total": int, "errors": int}
    for tr in test_results:
        metrics = getattr(tr, "metrics_data", None) or getattr(tr, "metrics", None) or []
        for m in metrics:
            name = getattr(m, "name", "unknown")
            b = buckets.setdefault(name, {"scores": [], "passed": 0, "total": 0, "errors": 0})
            if getattr(m, "error", None):      # judge call failed: not a score of 0
                b["errors"] += 1
                continue
            b["total"] += 1
            score = getattr(m, "score", None)
            if score is not None:
                b["scores"].append(score)
            if getattr(m, "success", False):
                b["passed"] += 1

    summary = {}
    for name, b in buckets.items():
        scores = b["scores"]
        summary[name] = {
            "n": b["total"],
            "pass_rate": (100 * b["passed"] / b["total"]) if b["total"] else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else float("nan"),
            "min_score": min(scores) if scores else float("nan"),
            "max_score": max(scores) if scores else float("nan"),
            "n_errors": b["errors"],
        }
    return summary


def print_summary(title, summary):
    """Readable per-metric recap, printed under DeepEval's own report when a file
    is run directly."""
    print("\n" + "=" * 60)
    print(f"{title}  (per-metric summary)")
    print("=" * 60)
    for name, s in summary.items():
        avg = f"{s['avg_score']:.2f}" if s["avg_score"] == s["avg_score"] else "nan"   # nan != nan
        print(f"  {name:<26} pass_rate={s['pass_rate']:5.0f}%  avg={avg}  n={s['n']}")
    print("=" * 60)