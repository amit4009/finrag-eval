# finrag-eval

An evaluation-driven RAG analyst over SEC filings, built on [FinanceBench](https://github.com/patronus-ai/financebench).
Every design choice is tested with a regression-gated eval suite (quality, safety, operational), and the gate runs in CI.

> 🚧 Work in progress. Results, architecture and design decisions will be added as each phase lands.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Project setup, data fetch | ✅ |
| 1 | One company (PepsiCo): naive baseline | ✅ |
| 2 | 8 companies: retrieval experiments | ⏳ E1 ✅, E2 ❌, E3 ❌, E4 ✅ |
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

**Noise and tolerances** (`python -m evals.noise`): five snapshots of identical pipelines
(2 naive: [`phase2_naive`](baselines/phase2_naive.json), [`phase2_noise_rerun`](baselines/phase2_noise_rerun.json);
3 E1: [`e1_metadata_filters`](baselines/e1_metadata_filters.json), [`baseline`](baselines/baseline.json),
[`e1_noise_rerun`](baselines/e1_noise_rerun.json)) give each metric's run-to-run noise. Three findings:

- **Retrieval metrics are exactly repeatable** (σ = 0), so any change is real and tolerance is 0.
- **The generator isn't deterministic**: `gpt-4o-mini` at temperature 0 changed one numeric answer
  in 7 between identical runs. Judge-free does not mean noise-free.
- **Judge noise differs about 3x by metric**: σ ≈ 0.007 for correctness, 0.024 for faithfulness.
  The single 0.05 tolerance inherited from the tutorial was too loose for some metrics and too
  tight for others, and gave false REVIEWs on unchanged pipelines. Each judge metric now has its
  own tolerance, 2.77σ (the 95% band for a difference of two noisy runs), with a floor of 0.03.

A first estimate from a single pair of runs said 0.05 was enough, and the next pair disproved it.
The current tolerances come from 3 degrees of freedom (1 for the two newest metrics), so they are
estimates. They were also fitted to the runs they now pass, and get re-measured as runs accumulate.

**Refusals and faithfulness:** a refusal makes no claims, so it scores 1.0 on faithfulness, and
plain faithfulness *falls* when better retrieval lets the model answer more. That is the REVIEW
that E1 triggered. The pipeline eval now also reports `answer_rate` and
`faithfulness_attempted` (answered questions only), and gates on those.

**Judge failures:** `gpt-4o-mini` as judge sometimes loops to its 16k output-token limit on a
structured step. Those calls are recorded as `n_errors` and left out of the averages, not
scored 0 (1–2 per run, out of 329 judged metric evaluations: 47 questions × 7 judge metrics).

### E2: hybrid BM25 + vector search. Rejected.

E1's filters, with BM25 keyword search run next to the vector search: top 20 from each, fused
by reciprocal rank fusion (k=60). Snapshot: [`e2_hybrid.json`](baselines/e2_hybrid.json).

| Metric | E1 | E2 | |
|---|---|---|---|
| hit@5 | 0.60 | **0.38** | regressed (exact metric: 2 hits gained, 12 lost) |
| MRR | 0.35 | 0.31 | regressed |
| doc hit@5 | 0.94 | 0.91 | regressed |
| contextual recall | 0.86 | 0.79 | regressed |
| correctness | 0.385 | 0.355 | regressed (just past its 0.03 tolerance) |
| numeric match, answer rate, faithfulness (answered) | 3/7, 0.66, 0.74 | 2/7, 0.57, 0.78 | within noise |

**Why it hurt:** for every lost question checked, BM25 never had the gold page in its top 20.
It only added wrong candidates, which fusion ranked above correct vector hits. Two causes:

- FinanceBench questions carry instructions ("If the quick ratio is not something a financial
  analyst would ask about… state that and explain why"), and BM25 matches those words literally.
  The AMCOR quick-ratio question pulled governance pages ("Removal of Directors").
- Questions and filings use different words: "capital expenditure" vs "Purchases of property,
  plant and equipment". Embeddings bridge that gap and keywords can't, the opposite of the usual
  case for hybrid search.

Not pursued: stripping the instruction text from the BM25 query, or weighting the fusion toward
vectors. Both would be tuned against the same 47 dev questions until something wins. The code
stays in the repo (`RETRIEVER = "hybrid"`), and E1 remains the baseline.

### E3: cross-encoder reranking. Rejected.

E1's filters, then 20 vector candidates re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`
(the tutorial's reranker), keeping the top 5. Snapshot: [`e3_rerank_minilm.json`](baselines/e3_rerank_minilm.json).

| Metric | E1 | E3 | |
|---|---|---|---|
| hit@5 | 0.60 | **0.38** | regressed (3 hits gained, 13 lost) |
| MRR | 0.35 | **0.21** | regressed |
| contextual recall | 0.86 | 0.68 | regressed |
| answer rate | 0.66 | 0.51 | regressed |
| answer relevancy | 0.63 | 0.46 | regressed |
| correctness | 0.385 | 0.318 | regressed |

The gold page is among the 20 candidates for 39/47 questions, so a good reranker could reach about
0.83. This one reorders them badly: **it prefers prose that echoes the question over the
financial-statement tables that hold the answer.** For 3M's FY2018 capex it ranked *"In 2018, 3M
expended approximately $27 million for capital projects related to protecting the environment"*
first and pushed the cash-flow statement from rank 4 to rank 9. For Amcor's net AR it moved the
balance sheet from rank 1 to rank 6. The model was trained on web search passages, and tables
are out of its domain.

**E2 and E3 share a root cause:** both judge text by its words, and both fail on tables.
PyMuPDF's plain text extraction flattens a table into a column of loose labels and numbers, so a
table chunk barely reads as text. That points to E4 (table-aware parsing) as the next experiment.

### E4: table-aware parsing (row reconstruction). Accepted.

PyMuPDF's plain text puts every table number on its own line, cut off from its row label.
E4 rebuilds each visual line from word positions and marks wide gaps as ` | `, so a row stays
whole: `Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373) | (1,420)`.
Prose is unaffected. It takes about 3 ms per page and needs no new dependency. PyMuPDF's own
`find_tables()` was tried first: the `lines` strategy shattered unruled statements into
fragments, and `text` split words across cells. Snapshot: [`e4_rows_parser.json`](baselines/e4_rows_parser.json).

| Metric | E1 | E4 | |
|---|---|---|---|
| **correctness** | 0.385 | **0.434** | improved, about 7σ of measured noise |
| MRR | 0.351 | 0.383 | improved |
| hit@5 | 0.596 | 0.596 | same (3 gained, 3 lost) |
| doc hit@5 | 0.936 | 0.894 | regressed: 2 filings, 1 a labelling artifact |
| contextual recall | 0.860 | 0.777 | regressed |
| numeric match | 3/7 | 3/7 | same |

**Verdict: REVIEW, accepted.** Correctness, the end-to-end metric, improved well beyond noise,
and page-level retrieval held steady. Of the two lost filings:

- **AMD** is not a real miss. `AMD_2022_annualreport` is the 10-K repackaged: the gold 10-K page
  matches annual-report page 8 at 95% similarity, and E4 retrieved that page. FinanceBench
  labels only the 10-K. Counting known duplicate documents as equivalent would fix the metric, but
  that change belongs in its own commit, applied to every snapshot, not to one pending verdict.
- **J&J** is real: the Q4 earnings release outranked the 10-K and lacks the inventory figures.

**Cost:** row parsing makes earnings releases (almost all tables) more searchable, and they now
sometimes outrank the 10-K for full-year questions. Hence the contextual-recall drop, and a
candidate follow-up: prefer 10-Ks for full-year questions.

E4 is now [`baseline.json`](baselines/baseline.json).
