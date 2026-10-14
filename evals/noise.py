"""
evals/noise.py -- measure run-to-run noise and suggest guardrail tolerances.

Give it GROUPS of snapshots, where every snapshot in a group measured the
identical pipeline (same code, config and prompt). Within a group, any spread
is pure noise: the LLM judge and the generator aren't deterministic. It pools
the within-group variance per metric and suggests a tolerance:

    tol = 2.77 * sigma, rounded UP to 0.01, floor 0.03

2.77 = 1.96 * sqrt(2): the 95% band for the DIFFERENCE between two runs (the
baseline and the candidate are both noisy). The floor guards against a sigma
underestimated from few runs. Paste the results into metric_registry.JUDGE_TOL.

    uv run python -m evals.noise \
        baselines/phase2_naive.json,baselines/phase2_noise_rerun.json \
        baselines/e1_metadata_filters.json,baselines/baseline.json,baselines/e1_noise_rerun.json
"""

import json
import math
import sys

Z_DIFF = 1.96 * math.sqrt(2)
FLOOR = 0.03


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["metrics"]


def _ok(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x   # not NaN


def pooled_noise(groups):
    """{metric_id: (sigma, df, max_spread)} over every metric with df >= 1."""
    ids = {k for g in groups for snap in g for k in snap if k.endswith(".avg_score")}
    out = {}
    for mid in sorted(ids):
        ss = df = spread = 0
        for g in groups:
            vals = [s[mid] for s in g if _ok(s.get(mid))]
            if len(vals) > 1:
                mean = sum(vals) / len(vals)
                ss += sum((v - mean) ** 2 for v in vals)
                df += len(vals) - 1
                spread = max(spread, max(vals) - min(vals))
        if df:
            out[mid] = (math.sqrt(ss / df), df, spread)
    return out


def suggest_tolerance(sigma):
    return max(FLOOR, math.ceil(Z_DIFF * sigma * 100) / 100)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    groups = [[_load(p) for p in arg.split(",")] for arg in sys.argv[1:]]
    print(f"{'metric':<46}{'df':>4}{'sigma':>8}{'spread':>8}{'tol':>7}")
    for mid, (sigma, df, spread) in pooled_noise(groups).items():
        tol = "exact" if sigma == 0 else f"{suggest_tolerance(sigma):.2f}"
        print(f"{mid:<46}{df:>4}{sigma:>8.3f}{spread:>8.3f}{tol:>7}")


if __name__ == "__main__":
    main()
