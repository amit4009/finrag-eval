# finrag-eval

An evaluation-driven RAG analyst over SEC filings, built on [FinanceBench](https://github.com/patronus-ai/financebench).
Every design choice is tested with a regression-gated eval suite (quality, safety, operational), and the gate runs in CI.

> 🚧 Work in progress. Results, architecture and design decisions will be added as each phase lands.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Project setup, data fetch | ✅ |
| 1 | One company (PepsiCo): naive baseline | ✅ |
| 2 | 8 companies: retrieval experiments | |
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
