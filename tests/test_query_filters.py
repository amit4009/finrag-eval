# Cases are phrased like DEV-split questions only; the test split never shaped
# these rules.
import pytest

from src.query_filters import chroma_where, extract_filters, find_companies, find_years


@pytest.mark.parametrize("question, companies", [
    ("What is the FY2021 capital expenditure amount for PepsiCo?", ["PEPSICO"]),
    ("Has Pepsico reported any legal battles?", ["PEPSICO"]),
    ("Are JnJ's FY2022 financials that of a high growth company?", ["JOHNSON"]),
    ("Does AMEX have an improving operating margin profile?", ["AMERICANEXPRESS"]),
    ("What drove gross margin change for American Express?", ["AMERICANEXPRESS"]),
    ("Which region had the Highest EBITDAR Contribution for MGM?", ["MGMRESORTS"]),
    ("What is the FY2019 total amount of inventories for Best Buy?", ["BESTBUY"]),
    ("Is 3M a capital-intensive business?", ["3M"]),
    ("What are the major products that AMD sells as of FY22?", ["AMD"]),
    ("What was AMCOR's Adjusted EBITDA?", ["AMCOR"]),
    ("What changed within 3 months?", []),            # "3m" must not fire inside text
])
def test_companies(question, companies):
    assert find_companies(question) == companies


@pytest.mark.parametrize("question, years", [
    ("What is the FY2021 capex?", ["2021"]),
    ("As of FY22, what does it sell?", ["2022"]),
    ("From FY21 to FY22, which segment grew most?", ["2022", "2021"]),
    ("What was the Adjusted EBITDA for FY 2023", ["2023"]),
    ("As of FY2023Q1, why did it raise guidance?", ["2023"]),
    ("Was there a drop between FY 2023 and Q2 of FY2024?", ["2024", "2023"]),
    ("the FY2015 - FY2017 3 year average net profit margin", ["2017", "2015"]),
    ("As of May 26, 2023, what may it borrow?", ["2023"]),
    ("the 8k filing dated 1st July 2022", ["2022"]),
    ("Does it maintain a stable dividend trend?", []),
    ("What is the capex (in USD millions)?", []),       # no stray numbers as years
])
def test_years_latest_first(question, years):
    assert find_years(question) == years


def test_extract_filters():
    f = extract_filters("Among operations, investing, and financing, which brought in the "
                        "most cash flow for Best Buy in FY2023?")
    assert f == {"companies": ["BESTBUY"], "years": ["2023"]}


def test_chroma_where():
    assert chroma_where([], []) is None
    assert chroma_where(["3M"], []) == {"company": {"$in": ["3M"]}}
    assert chroma_where(["3M"], ["2022"]) == {"$and": [
        {"company": {"$in": ["3M"]}}, {"doc_period": {"$in": ["2022"]}}]}


def test_bm25_where_matching_mirrors_chroma_filters():
    from src.bm25 import matches
    meta = {"company": "3M", "doc_period": "2022"}
    assert matches(meta, None)
    assert matches(meta, chroma_where(["3M"], ["2022"]))
    assert not matches(meta, chroma_where(["3M"], ["2021"]))
    assert not matches(meta, chroma_where(["AMD"], []))
