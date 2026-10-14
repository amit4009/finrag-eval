"""
goldens/build_equivalents.py -- find pages elsewhere in the corpus that duplicate
a gold evidence page.

FinanceBench labels ONE (doc_name, page) per piece of evidence. But some filings
repeat others: AMD_2022_annualreport is the 10-K repackaged (its page 8 matches
the 10-K's page 3 at 95%). Retrieving the copy is retrieving the right evidence,
yet hit@k scored it as a miss (E4: AMD "lost" its filing).

So for each gold evidence page we look for near-identical pages in the same
company's other filings FOR THE SAME FISCAL PERIOD and record them as
equivalents. Same period matters: companies copy boilerplate pages verbatim
year to year (AMCOR's 2022 and 2023 10-K page 4 match at Jaccard 1.0), but a
question "as of FY2022" must be answered from the FY2022 filing -- last year's
identical page is not the right evidence. Similarity is the
Jaccard overlap of word 5-gram sets ("shingles"): robust to headers, spacing
and line breaks, and near 0 for pages that merely share vocabulary. Measured,
not hand-listed: no "annual report = 10-K" rule.

Output: goldens/evidence_equivalents.json, committed so the equivalences are
reviewable and fixed, not recomputed differently per run. Only companies whose
PDFs are downloaded are covered; `covered_companies` says which. Re-run when a
phase adds companies.

    uv run python -m goldens.build_equivalents
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from src.fetch_data import CACHE, company_key
from src.ingest import doc_metadata, load_doc_info, parse_pdf

GOLDENS = Path("goldens/financebench.json")
OUT = Path("goldens/evidence_equivalents.json")
THRESHOLD = 0.8      # Jaccard of 5-gram sets; real copies score ~0.9+, related pages < 0.3
SHINGLE = 5


def shingles(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def build():
    pdfs = sorted(p.stem for p in (CACHE / "pdfs").glob("*.pdf"))
    by_company = defaultdict(list)
    for doc in pdfs:
        by_company[company_key(doc)].append(doc)

    goldens = json.loads(GOLDENS.read_text(encoding="utf-8"))
    covered = sorted({g["company"] for g in goldens if g["company"] in by_company})

    info = load_doc_info()
    period = {doc: doc_metadata(doc, info)["doc_period"] for doc in pdfs}
    pages = {}       # doc -> [shingle set per page], parsed with the plain text parser

    def doc_pages(doc):
        if doc not in pages:
            pages[doc] = [shingles(t) for t in parse_pdf(CACHE / "pdfs" / f"{doc}.pdf", parser="text")]
        return pages[doc]

    equivalents = {}
    for g in goldens:
        if g["company"] not in by_company:
            continue
        found = []
        for ev in g["expected_evidence"]:
            gold = doc_pages(ev["doc_name"])[ev["page"]]
            for other in by_company[g["company"]]:
                if other == ev["doc_name"] or period[other] != period[ev["doc_name"]]:
                    continue
                for p, sh in enumerate(doc_pages(other)):
                    sim = jaccard(gold, sh)
                    if sim >= THRESHOLD:
                        found.append({"doc_name": other, "page": p, "of": ev, "similarity": round(sim, 3)})
        if found:
            equivalents[g["id"]] = found

    OUT.write_text(json.dumps({"threshold": THRESHOLD, "shingle": SHINGLE,
                               "covered_companies": covered, "equivalents": equivalents},
                              indent=2), encoding="utf-8")
    return covered, equivalents


if __name__ == "__main__":
    covered, eq = build()
    print(f"covered {len(covered)} companies; {len(eq)} questions have duplicate evidence pages")
    for qid, items in eq.items():
        for e in items:
            print(f"  {qid}: {e['of']['doc_name']} p.{e['of']['page']} == {e['doc_name']} p.{e['page']}"
                  f"  (jaccard {e['similarity']})")
