# finrag-eval

An evaluation-driven RAG analyst over SEC filings, built on [FinanceBench](https://github.com/patronus-ai/financebench).
Every design choice is tested with a regression-gated eval suite (quality, safety, operational), and the gate runs in CI.

> 🚧 Work in progress. Results, architecture and design decisions will be added as each phase lands.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Project setup, data fetch | ✅ |
| 1 | One company (PepsiCo): naive baseline | ✅ |
| 2 | 8 companies: retrieval experiments | ✅ E1 ✅, E2 ❌, E3 ❌, E4 ✅, E5 ✅ |
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

~~**Corpus growth:** going from 1 company to 8, naive hit@5 fell from 0.55 to 0.28. With 8
companies there are 6x more look-alike filings to confuse the right one with.~~
**Correction (Phase 3):** this compared *different question sets* (PepsiCo's 11 vs 8 companies'
47) under different setups, so it was confounded. With the questions held fixed, adding companies
changes nothing; see [corpus growth](#corpus-growth-the-hard-distractor-is-the-same-company-in-other-years).

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

**Noise and tolerances** (`python -m evals.noise`): seven snapshots of identical pipelines
(2 naive: [`phase2_naive`](baselines/phase2_naive.json), [`phase2_noise_rerun`](baselines/phase2_noise_rerun.json);
3 E1: [`e1_metadata_filters`](baselines/e1_metadata_filters.json), [`e1_new_metrics_run`](baselines/e1_new_metrics_run.json),
[`e1_noise_rerun`](baselines/e1_noise_rerun.json); 2 E4: [`e4_rows_parser`](baselines/e4_rows_parser.json),
[`e4_rerun_dup_rule`](baselines/e4_rerun_dup_rule.json)) give each metric's run-to-run noise. Three findings:

- **Retrieval metrics are exactly repeatable** (σ = 0), so any change is real and tolerance is 0.
- **The generator isn't deterministic**: `gpt-4o-mini` at temperature 0 changed one numeric answer
  in 7 between identical runs. Judge-free does not mean noise-free.
- **Judge noise differs about 3x by metric**: σ ≈ 0.007 for correctness, 0.024 for faithfulness.
  The single 0.05 tolerance inherited from the tutorial was too loose for some metrics and too
  tight for others, and gave false REVIEWs on unchanged pipelines. Each judge metric now has its
  own tolerance, 2.77σ (the 95% band for a difference of two noisy runs), with a floor of 0.03.
  Recomputed as runs accumulate: the E4 pair narrowed four of them, and no verdict changed.

A first estimate from a single pair of runs said 0.05 was enough, and the next pair disproved it.
The current tolerances come from 4 degrees of freedom (2 for the two newest metrics), so they are
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

E4 became the baseline (re-run under the duplicate-evidence rule: [`e4_rerun_dup_rule.json`](baselines/e4_rerun_dup_rule.json)).

### Eval fix: duplicate evidence pages count as hits

FinanceBench labels one `(doc, page)` per piece of evidence, but some filings repeat others.
[`goldens/build_equivalents.py`](goldens/build_equivalents.py) compares each gold page with every
page of the same company's other filings **for the same fiscal period**, by Jaccard overlap of word
5-grams (≥ 0.8). Near-copies are recorded in
[`evidence_equivalents.json`](goldens/evidence_equivalents.json) and accepted by hit@k, MRR and
doc hit. In Phase 2 this covers 4 AMD questions (the annual report is the 10-K, Jaccard 0.85 to
0.995) and an AMCOR 8-K filed under two names.

Cross-year copies are **excluded on purpose**: companies repeat boilerplate pages verbatim
(AMCOR's 2022 and 2023 10-K page 4 match at Jaccard 1.0), but a question "as of FY2022" must be
answered from the FY2022 filing.

Retrieval is deterministic, so every past experiment was re-scored under both rules
([`evals/rescore_retrieval.py`](evals/rescore_retrieval.py) →
[`retrieval_rescored.json`](baselines/retrieval_rescored.json)). The old-rule column reproduces each
snapshot exactly:

| | hit@5 | MRR | doc hit@5 |
|---|---|---|---|
| naive | 0.277 → 0.277 | 0.173 → 0.173 | 0.553 → 0.553 |
| E1 filters | 0.596 → 0.596 | 0.351 → 0.351 | 0.936 → 0.936 |
| E2 hybrid ❌ | 0.383 → 0.383 | 0.312 → 0.312 | 0.915 → 0.936 |
| E3 rerank ❌ | 0.383 → 0.383 | 0.214 → 0.224 | 0.915 → 0.915 |
| **E4 rows ✅** | 0.596 → **0.617** | 0.383 → 0.390 | 0.894 → **0.915** |

**No verdict changes.** The fix landed after the E4 decision, as its own commit, applied to every
experiment. Under it E4's page hit rate beats E1 (0.617 vs 0.596), and its one remaining lost
filing (J&J) is a real miss. The E4 baseline was re-run under the new rule and compares PASS
against the original E4 run.

### E5: one chunk per page. Accepted.

With the rows parser a table row stays on one line, but a financial statement runs 3,000 to 5,000
characters, so 1000-character chunks cut most rows off from the year header
(`(Millions) | 2018 | 2017 | 2016`). E5 keeps each page whole: 3,000 characters at the median, 10,100
at most, well inside the embedding limit, and one-to-one with FinanceBench's page labels. Top-k stays
at 5. Snapshot: [`e5_page_chunks.json`](baselines/e5_page_chunks.json).

| Metric | E4 | E5 | |
|---|---|---|---|
| **numeric match (pipeline)** | 2/7 | **6/7** | improved: +4 questions, noise limit is 1 |
| **correctness** | 0.433 | **0.469** | improved, about 6σ |
| contextual recall | 0.780 | 0.809 | improved |
| contextual relevancy | 0.225 | 0.254 | improved |
| doc hit@5 | 0.915 | 0.936 | improved |
| hit@5 | 0.617 | 0.553 | regressed (3 gained, 6 lost) |
| MRR | 0.390 | 0.367 | regressed |

**Verdict: REVIEW, accepted.** The pipeline's numeric match now equals the generator's own score
on gold pages (6/7). Phase 1's gap (generator 3/3, pipeline 0/3) is largely closed for this
question type. The cost is retrieval precision: one embedding per page blurs several topics, so the
exact gold page drops out of the top 5 more often. But whole pages give the generator complete
tables, and every end-to-end metric rose.

Caveats: only 7 numeric questions (Phase 3's 105 dev questions test this at scale), and about 3x the
generator input per question, to be measured when the operational evals are ported. With answers
now close to the generator's ceiling, **retrieval is the weaker link again**, and a reranker on
whole, properly parsed pages is worth retrying in Phase 3. E3 failed on flattened tables.

E5 is now [`baseline.json`](baselines/baseline.json).

### Phase 2 summary (47 dev questions)

| | hit@5 | doc hit@5 | correctness | numeric match | verdict |
|---|---|---|---|---|---|
| naive | 0.28 | 0.55 | 0.25 | 2/7 | |
| E1 metadata filters | 0.60 | 0.94 | 0.39 | 3/7 | ✅ |
| E2 + BM25 hybrid | 0.38 | 0.94 | 0.35 | 2/7 | ❌ |
| E3 + MiniLM reranker | 0.38 | 0.91 | 0.32 | 2/7 | ❌ |
| E4 rows parser | 0.62 | 0.91 | 0.43 | 3/7 | ✅ |
| **E5 page chunks** | 0.55 | **0.94** | **0.47** | **6/7** | ✅ |

hit@5 and doc hit use the duplicate-evidence rule throughout. Correctness went from **0.25 to 0.47**
and numeric match from **2/7 to 6/7**. Two of the five experiments were rejected, each with a
diagnosed cause.

## Phase 3: all 40 companies (363 filings, 53,588 pages), 105 dev questions

The full corpus with the Phase 2 winning setup: rows parser, one chunk per page, company and year
filters, top 5. Company extraction was extended to all 40 companies. Names are matched ignoring
case, hyphens and both readings of an apostrophe ("Coca Cola" = "Coca-Cola", "Amcor's" = "Amcor",
"McDonald's" = "McDonalds"), plus stock tickers for all 40 companies, matched case-sensitively
because COST, BA, FL and KO are also ordinary words. Filter accuracy on dev is 94/105, with no
question tagged with a wrong company.

| Metric | Phase 2 (47 dev) | **Phase 3 (105 dev)** |
|---|---|---|
| hit@5 | 0.553 | **0.667** |
| doc hit@5 | 0.936 | 0.933 |
| correctness | 0.469 | **0.499** |
| numeric match, pipeline | 6/7 | **28/35** (0.80) |
| numeric match, generator on gold page | 6/7 | 30/35 (0.86) |
| answer rate | 0.68 | 0.73 |
| faithfulness (answered) | 0.77 | 0.83 |

Snapshots: [`phase3_e5_baseline.json`](baselines/phase3_e5_baseline.json), [`phase3_noise_rerun.json`](baselines/phase3_noise_rerun.json)
(the Phase 2 E5 baseline is kept as [`phase2_e5_baseline.json`](baselines/phase2_e5_baseline.json)).

- **Corpus growth costs nothing.** The 47 Phase 2 questions score identically against 53,588
  pages from 40 companies and against 12,762 pages from 8 (hit@5 0.553, MRR 0.367, doc hit 0.936,
  to three decimals). This first said the *company filter* was the reason; the corpus-growth
  measurement below shows growth costs nothing **even without filters**.
- **Numeric answers are within 2 of the generator's ceiling** (28/35 vs 30/35 on gold pages).
  Phase 1's gap between the pipeline and the generator is nearly closed at five times the sample
  size.
- **The tolerances passed an independent test.** An identical rerun on 105 questions, data the
  tolerances were *not* fitted to, compares PASS. Retrieval metrics again moved by exactly 0.

### Corpus growth: the hard distractor is the same company in other years

[`evals/corpus_growth.py`](evals/corpus_growth.py) → [`corpus_growth.json`](baselines/corpus_growth.json).
The questions are held fixed while the corpus grows, so only the added filings can change the
result. Current parsing and chunking; retrieval metrics only.

| Questions | Corpus | Filters off: hit@5 / doc hit | Filters on: hit@5 / doc hit |
|---|---|---|---|
| PepsiCo dev (8) | 1 company, 4,228 pages | 0.500 / 0.500 | 0.875 / 1.000 |
| PepsiCo dev (8) | 8 companies, 12,762 pages | 0.500 / 0.500 | 0.875 / 1.000 |
| PepsiCo dev (8) | 40 companies, 53,588 pages | 0.500 / 0.500 | 0.875 / 1.000 |
| Phase 2 dev (47) | 8 companies | 0.277 / 0.574 | 0.553 / 0.936 |
| Phase 2 dev (47) | 40 companies | 0.277 / 0.574 | 0.553 / 0.936 |

**A 12x larger corpus changes nothing, with or without filters.** This overturns the assumption
behind the plan's "corpus-growth chart", and two earlier README claims (corrected above). Where do
unfiltered top-5 results come from, across all 105 dev questions on 40 companies (525 pages)?

| Source of a retrieved page | Share |
|---|---|
| a different company | 8.4% |
| **the right company, a different fiscal year** | **61.9%** |
| the right company and year | 29.7% |

Questions name their company, and embeddings pick that up. The distractors that actually compete
are **the same company's other filings**: years of near-identical 10-Ks, plus its 10-Qs and
earnings releases. So E1's value comes from its **year** filter, not its company filter, and the
right way to stress this system is more years per company, not more companies.

### Can the LLM judge be trusted? Human labels vs the judge

[`evals/judge_agreement.py`](evals/judge_agreement.py). 50 dev questions, stratified by type
(17 numeric, 17 domain, 16 novel, 28 companies), answered by the current pipeline and labeled
correct/incorrect by a human **blind to the judge's scores**. The sheet held only the question, the
expert answer, FinanceBench's justification and the model's answer. Labels:
[`judge_labels.csv`](goldens/judge_labels.csv); results: [`judge_agreement.json`](baselines/judge_agreement.json).

| | agreement | Cohen's κ | false passes (judge ✓, human ✗) | false fails (judge ✗, human ✓) |
|---|---|---|---|---|
| **GEval correctness** (score ≥ 0.7) | 0.72 | **0.48** | **0** | 14 |
| **numeric match** (17 numeric questions) | 0.94 | **0.77** | 1 | 0 |

- **The judge is strict, not lenient.** It never accepted an answer the human rejected. Every
  human-rejected answer scored 0.0, except one at 0.50. All disagreements are correct answers
  scored below 0.7, and human-correct answers range from 0.40 to 1.0. **So correctness has been
  *understated*:** the human rates 30/50 answers correct (60%), and the judge passes 16/50 (32%).
  Regression verdicts gate on the *average* score, which still tracks quality, but the absolute
  numbers are conservative. The threshold was **not** retuned on these 50 labels; that would fit
  the judge to the very data used to check it.
- **The numeric matcher's leniency is real.** It accepts a correct number *anywhere* in the
  answer, the risk noted in `evals/numeric.py` since Phase 1. Its one false pass (Amazon DPO) has
  the right number in the working and a wrong conclusion.
- **Labeling found a generator bug: answers that contradict themselves.** The prompt said *"Lead
  with the direct answer in the first sentence"*, so the model opened with a number *before* doing
  the arithmetic, worked it out, and often concluded with a different one:

  | | expert | first sentence | conclusion | label |
  |---|---|---|---|---|
  | Kraft Heinz inventory turnover | 6.25 | 3.06 | "rounding gives **6.25**" | Y |
  | PepsiCo EBITDA less capex | $9,068M | 8,068 | **$9,068 million** | Y |
  | Corning 3-yr avg margin | 10.3% | 15.5% | "the correct average… is **10.3%**. I apologize for the confusion" | Y |
  | Best Buy 3-yr avg margin | 2.8% | 3.1% | "rounding… gives **3.1%**" (its own sum is 2.73%) | N |
  | Amazon DPO | 93.86 | 36.73 | **36.73** (its own working says 93.64) | N |

- **Labeling rule: judge the final conclusion.** The human labels follow it consistently. One
  label was changed from Y to N because the comparison looked at first sentences only, then
  restored once the conclusions were compared. The note in the sheet records this.
- **Strict numeric matcher** (`numeric_match_stated`): it reads only the `Final answer:` line,
  i.e. the conclusion, which the prompt requires since E6. A first-sentence version was tried and
  rejected: it picked the wrong number in 4 of the 17 labeled answers above. For pre-E6 answers
  with no such line, it falls back to the lenient check, because their conclusion has no fixed
  position.
- **Second labeling round:** the 17 numeric questions re-answered with the E6 prompt and labeled
  the same way ([`judge_labels_numeric.csv`](goldens/judge_labels_numeric.csv),
  [`judge_agreement_numeric.json`](baselines/judge_agreement_numeric.json)). Strict and lenient
  matchers agree on every E6 answer, because the stated conclusion and the working no longer
  disagree. Both disagreed with the human on one row: Amazon DPO (measured in days) answered
  `93.64 %`, right number with a stray percent sign. The matcher now treats matching digits with a
  mismatched `%` as a unit slip. It was re-checked on both rounds (34 labeled answers): **no new
  false passes**, κ 0.77 on round 1 (unchanged) and 1.00 on round 2. Round 2's 1.00 is not
  independent evidence, since that row prompted the fix.
- **First use of the per-question records:** the fix changes one baseline answer's score, so
  `baseline.json` was re-scored from its saved answers, not re-run (pipeline numeric match
  27/35 → 28/35, recorded in its metadata). Without the records this would have needed a full
  paid re-run.

### E6: work first, state the final answer last. Accepted (PASS).

Prompted by the judge-agreement finding: *"Lead with the direct answer in the first sentence"*
made the model commit to a number before doing the arithmetic. E6 asks it to work through the
figures first and end with `Final answer: …`, which must agree with the working. No other prompt
change. Snapshot: [`e6_answer_last.json`](baselines/e6_answer_last.json).

| Metric | Phase 3 baseline (E5) | **E6** | |
|---|---|---|---|
| **correctness** | 0.499 | **0.575** | +0.076, about 12σ |
| **answer rate** | 0.73 | **0.81** | 8 more questions answered instead of declined |
| answer relevancy (pipeline / generator) | 0.70 / 0.74 | **0.83 / 0.84** | improved |
| numeric match | 28/35 | 27/35 | within noise |
| faithfulness, retrieval | | | unchanged |

The contradiction cases from labeling now conclude correctly with a `Final answer:` line: Kraft
Heinz **6.24** (expert 6.25; 16,830 / 2,694 = 6.247), Best Buy **2.8%**, Amazon DPO **93.64**
(expert 93.86).

**Numeric match can't see this fix**, because its leniency masked the bug in the first place: it
credited the old answers for the right number buried in their working. The judge reads the stated
answer, and correctness rose sharply. A stricter matcher that reads the `Final answer:` line is
the next eval fix. Asking for the working first also made the model **decline less**: committing
to an answer up front made it give up more often.

E6 is now [`baseline.json`](baselines/baseline.json).
