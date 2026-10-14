# finrag-eval

An evaluation-driven RAG analyst over SEC filings, built on [FinanceBench](https://github.com/patronus-ai/financebench).
Every design choice is tested with a regression-gated eval suite (quality, safety, operational), and the gate runs in CI.

> 🚧 Work in progress. Results, architecture and design decisions will be added as each phase lands.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Project setup, data fetch | ✅ |
| 1 | One company (PepsiCo): naive baseline | ✅ |
| 2 | 8 companies: retrieval experiments | ⏳ E1 done |
| 3 | All 40 companies: full system, final test-set numbers | |
| 4 | Production: API, UI, tracing, online evals, CI | |
| 5 | Write-up | |

## Quickstart

```bash
uv sync                                     # install dependencies (Python 3.12)
cp .env.example .env                        # add your OPENAI_API_KEY
uv run python -m src.fetch_data phase1      # download the questions + PepsiCo PDFs
uv run python -m goldens.build_goldens      # FinanceBench jsonl -> goldens/financebench.json
uv run python -m src.retriever              # parse, chunk, embed once (~$0.40 for phase1)
uv run python -m src.rag_pipeline           # ask one question end to end

uv run python -m evals.run_suite --baseline --label "..."   # snapshot -> baselines/baseline.json
uv run python -m evals.run_suite --label "my change"        # snapshot -> baselines/candidate.json
uv run python -m evals.compare                              # PASS / REVIEW / FAIL
uv run pytest                                               # unit tests (numeric matcher)
```

Settings for every experiment (phase, chunking, embedding model, top-k, models) live in
[`src/config.py`](src/config.py) and are stamped into each snapshot. `FINRAG_PHASE=phase2` switches corpus.

## Data

FinanceBench (Patronus AI): 150 open questions with expert answers and evidence pages, plus 368 filings
(10-K, 10-Q, 8-K, earnings releases) from 40 public companies. PDFs are downloaded into a git-ignored
local cache and are **not redistributed** here. The source repo has no license file.

Phases are defined in [`config/corpus.yaml`](config/corpus.yaml). The questions are used **only for
evaluation**. They are never indexed or placed in the prompt.

## Results so far

### Phase 1: naive baseline (PepsiCo, 13 filings, 12,677 chunks, 11 questions)

Vector search top-5, PyMuPDF, 1000/150 chunks, `text-embedding-3-large`, `gpt-4o-mini`.
Snapshot: [`baselines/phase1_baseline.json`](baselines/phase1_baseline.json).

| Layer | Metric | Score | Judge? |
|---|---|---|---|
| Retriever | hit@5 (right filing **and** page in top 5) | 0.55 | no |
| | MRR | 0.43 | no |
| | doc hit@5 (right filing at all) | 0.64 | no |
| | contextual recall | 0.82 | yes |
| Generator, given the gold page | numeric match (3 numeric questions) | **1.00** | no |
| | faithfulness | 0.92 | yes |
| Full pipeline | numeric match | **0.00** | no |
| | correctness (GEval) | 0.30 | yes |
| | contextual relevancy | 0.19 | yes |

**Reading it:** the generator gets every number right when it's handed the correct page, and
gets none right through the pipeline, so for those questions the loss is entirely in retrieval.
5 of the 6 10-K questions missed the evidence page, all the same way: the right section from the
**wrong filing**. "FY2021 capex" retrieves the investing section of the 2015, 2016, 2017 and 2019
10-Ks. The FY2022 EBITDA questions retrieve the FY2022 *earnings release* instead of the 10-K.
With several years of near-identical filings, nearest-neighbour search can't tell them apart.
The 8-K and earnings-release questions, whose documents are one of a kind, hit at rank 1 or 2.
When retrieval misses, the generator abstains rather than inventing a number. Next experiment:
**E1, metadata filters on company and fiscal year.**

**Noise check:** the identical pipeline was run twice ([`phase1_noise_rerun.json`](baselines/phase1_noise_rerun.json)).
Judge-free metrics moved by exactly 0. Judge metrics moved by up to 0.052, so `compare` returned
REVIEW on an unchanged pipeline. The 0.05 tolerance inherited from the tutorial is too tight
for 11 questions, and is recalibrated on the larger Phase 2 set.

### Phase 2: 8 companies (80 filings, 50,314 chunks), 47 dev questions

The 150 questions are split once into **dev (105) / test (45)**, stratified by question type,
with a seeded hash of the question id. Phase 2 has 47 dev / 20 test. Experiments are tuned and
scored on dev only. The test set is run once, at the end of Phase 3.

| Metric | Naive (vector top-5) | **E1: metadata filters** | Judge? |
|---|---|---|---|
| doc hit@5 (right filing in top 5) | 0.55 | **0.94** | no |
| hit@5 (right filing and page) | 0.28 | **0.60** | no |
| MRR | 0.17 | **0.35** | no |
| contextual recall | 0.73 | 0.85 | yes |
| numeric match, full pipeline (7 numeric questions) | 2/7 | **3/7** | no |
| correctness (GEval) | 0.25 | **0.39** | yes |
| answer relevancy | 0.42 | **0.64** | yes |
| faithfulness | 0.91 | 0.85 ⚠️ | yes |
| questions the pipeline declined to answer | 26/47 | **14/47** | no |
| numeric match, generator given the gold page | 6/7 | 6/7 | no |

Snapshots: [`phase2_naive.json`](baselines/phase2_naive.json), [`e1_metadata_filters.json`](baselines/e1_metadata_filters.json). E1 is now [`baseline.json`](baselines/baseline.json), the reference for E2.

**Corpus growth:** going from 1 company to 8, naive hit@5 fell from 0.55 to 0.28. With 8
companies there are 6x more look-alike filings to confuse the right one with.

**E1** ([`src/query_filters.py`](src/query_filters.py)) reads the company and the fiscal years
off the question with plain rules, then filters the vector search to that company and the
**latest** year named. Filings report earlier years as comparatives, so "FY2015 to FY2017" lives
in the 2017 10-K. If a filter matches nothing it widens step by step. The extractor targets the
gold filing's company and year for 44/47 dev questions (`filter_accuracy` 0.94). Two of the three
misses name no year, and one never names the company.

**Verdict: REVIEW, accepted.** The one regression is faithfulness (−0.057). Refusals make no
claims, so they score as perfectly faithful, and E1 cut refusals from 26 to 14. Attempting 12
more answers is what raised correctness, numeric match and answer relevancy, and it also gives
the judge more claims to find unsupported. Faithfulness is still worth watching: a version scored
only on attempted answers would separate the two effects.

**Noise check, Phase 2** ([`phase2_noise_rerun.json`](baselines/phase2_noise_rerun.json)): retrieval
metrics moved by exactly 0, and judge metrics by up to 0.040. The **generator's numeric match
moved by 1/7 on an identical pipeline**: `gpt-4o-mini` at temperature 0 is not deterministic,
so judge-free does not mean noise-free. Only retrieval metrics are exactly repeatable. With 7
numeric questions one answer is 0.14, so numeric-match deltas below 2/7 are noise at this size.

**Judge failures:** `gpt-4o-mini` as judge sometimes loops to its 16k output-token limit on a
structured step. Those calls are recorded as `n_errors` and left out of the averages, not
scored 0 (1–2 per run, out of 329 judged metric evaluations: 47 questions × 7 judge metrics).
