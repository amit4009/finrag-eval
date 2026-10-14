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

# Names the catalog doesn't contain but questions use.
EXTRA_ALIASES = {
    "jnj": "JOHNSON", "j&j": "JOHNSON",
    "amex": "AMERICANEXPRESS",
    "mgm": "MGMRESORTS",
}

# Stock tickers for all 40 FinanceBench companies -- general knowledge, listed
# in full rather than only for the tickers that dev questions happened to use.
# Matched CASE-SENSITIVELY as written ("JPM's"), because several are ordinary
# words in lower case: COST, BA, FL, KO, GIS.
TICKERS = {
    "MMM": "3M", "ATVI": "ACTIVISIONBLIZZARD", "ADBE": "ADOBE", "AES": "AES",
    "AMZN": "AMAZON", "AMCR": "AMCOR", "AMD": "AMD", "AXP": "AMERICANEXPRESS",
    "AWK": "AMERICANWATERWORKS", "AAPL": "APPLE", "BBY": "BESTBUY", "SQ": "BLOCK",
    "BA": "BOEING", "KO": "COCACOLA", "GLW": "CORNING", "COST": "COSTCO",
    "CVS": "CVSHEALTH", "EBAY": "EBAY", "FDX": "FEDEX", "FL": "FOOTLOCKER",
    "GIS": "GENERALMILLS", "INTC": "INTEL", "JNJ": "JOHNSON", "JPM": "JPMORGAN",
    "KHC": "KRAFTHEINZ", "LMT": "LOCKHEEDMARTIN", "MCD": "MCDONALDS", "MGM": "MGMRESORTS",
    "MSFT": "MICROSOFT", "NFLX": "NETFLIX", "NKE": "NIKE", "ORCL": "ORACLE",
    "PYPL": "PAYPAL", "PEP": "PEPSICO", "PFE": "PFIZER", "PCG": "PG",
    "CRM": "SALESFORCE", "ULTA": "ULTABEAUTY", "VZ": "VERIZON", "WMT": "WALMART",
}

FOUR_DIGIT_YEAR = re.compile(r"(?<!\d)((?:19|20)\d\d)(?!\d)")        # 2022, FY2022, FY2023Q1
TWO_DIGIT_FY = re.compile(r"\bFY\s?'?(\d\d)(?!\d)", re.IGNORECASE)   # FY22, FY '22


APOSTROPHES = "'’"


def _norm(text, apostrophe=""):
    """Case and hyphen insensitive ("Coca Cola" == "Coca-Cola"). Apostrophes are
    dropped by default ("McDonald's" -> "mcdonalds") or replaced, e.g. by a
    space, so a possessive stays separate ("Amcor's" -> "amcor s")."""
    for a in APOSTROPHES:
        text = text.replace(a, apostrophe)
    return text.lower().replace("-", " ")


@cache
def company_aliases():
    """normalised alias -> company key, from the FinanceBench catalog + extras."""
    aliases = {_norm(a): k for a, k in EXTRA_ALIASES.items()}
    with open(DOC_INFO, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = company_key(row["doc_name"])
            aliases[_norm(row["company"])] = key         # "johnson & johnson", "coca cola"
            aliases[_norm(key)] = key                    # "bestbuy", "americanexpress"
    return aliases


def find_companies(question):
    # both readings of an apostrophe: "McDonald's" must match "mcdonalds",
    # "Amcor's" must match "amcor"
    variants = (_norm(question), _norm(question, apostrophe=" "))
    found = []
    for alias, key in company_aliases().items():
        # whole-word match, so "3m" doesn't fire on "within 3 months"
        pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        if any(re.search(pattern, q) for q in variants) and key not in found:
            found.append(key)
    for ticker, key in TICKERS.items():
        if re.search(rf"(?<![A-Za-z0-9]){ticker}(?![A-Za-z0-9])", question) and key not in found:
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
