# finrag-eval

An evaluation-driven RAG analyst over SEC filings, built on [FinanceBench](https://github.com/patronus-ai/financebench).
Every design choice is tested with a regression-gated eval suite (quality, safety, operational), and the gate runs in CI.

> 🚧 Work in progress. Results, architecture and design decisions will be added as each phase lands.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Project setup, data fetch | ✅ |
| 1 | One company (PepsiCo): naive baseline | ⏳ |
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
