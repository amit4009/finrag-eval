"""
src/query_filters.py -- pull company + fiscal year out of a question (E1).

Phase 1 showed naive retrieval finding the right SECTION in the wrong FILING:
"FY2021 capex" matched the investing section of the 2015-2019 10-Ks equally
well. Embeddings barely separate "2021" from "2019". So we read the company and
year off the question with plain rules and filter the vector search by them.

Rules, not an LLM call: deterministic, free, unit-testable, and their accuracy
is scored directly against the goldens (eval_retriever's filter_accuracy).

    extract_filters("What is the FY2021 capex for PepsiCo?")
      -> {"companies": ["PEPSICO"], "years": ["2021"]}

YEAR CHOICE: questions often span years ("FY2015 - FY2017 average", "between
FY2023 and FY2022"), and filings report prior years as comparatives, so the
evidence is in the LATEST year's filing. FilteredRetriever (src/retriever.py)
tries that year first and widens only if it finds nothing.
"""

import json
import re
from functools import cache

from src.fetch_data import CACHE, company_key

DOC_INFO = CACHE / "data/financebench_document_information.jsonl"

# Names the catalog doesn't contain but questions use. Phase 3 (40 companies)
# will need more of these; filter_accuracy shows which.
EXTRA_ALIASES = {
    "jnj": "JOHNSON", "j&j": "JOHNSON",
    "amex": "AMERICANEXPRESS",
    "mgm": "MGMRESORTS",
}

FOUR_DIGIT_YEAR = re.compile(r"(?<!\d)((?:19|20)\d\d)(?!\d)")        # 2022, FY2022, FY2023Q1
TWO_DIGIT_FY = re.compile(r"\bFY\s?'?(\d\d)(?!\d)", re.IGNORECASE)   # FY22, FY '22


@cache
def company_aliases():
    """alias (lowercase) -> company key, from the FinanceBench catalog + extras."""
    aliases = dict(EXTRA_ALIASES)
    with open(DOC_INFO, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = company_key(row["doc_name"])
            aliases[row["company"].lower()] = key        # "johnson & johnson", "best buy"
            aliases[key.lower()] = key                   # "bestbuy", "americanexpress"
    return aliases


def find_companies(question):
    q = question.lower()
    found = []
    for alias, key in company_aliases().items():
        # whole-word match, so "3m" doesn't fire on "within 3 months"
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", q) and key not in found:
            found.append(key)
    return found


def find_years(question):
    years = {int(y) for y in FOUR_DIGIT_YEAR.findall(question)}
    years |= {2000 + int(y) for y in TWO_DIGIT_FY.findall(question)}
    return [str(y) for y in sorted(years, reverse=True) if 1990 <= y <= 2030]   # latest first


def extract_filters(question):
    return {"companies": find_companies(question), "years": find_years(question)}


def chroma_where(companies, years):
    """A Chroma `where` clause, or None for no filter."""
    clauses = []
    if companies:
        clauses.append({"company": {"$in": companies}})
    if years:
        clauses.append({"doc_period": {"$in": years}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
