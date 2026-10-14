# Integration tests against real FinanceBench pages; skipped when the PDF cache
# (git-ignored, fetched by src/fetch_data.py) isn't present.
from pathlib import Path

import pymupdf
import pytest

from src.ingest import page_rows

PDFS = Path("financebench/pdfs")


def rows_of(doc, page):
    path = PDFS / f"{doc}.pdf"
    if not path.exists():
        pytest.skip(f"{path} not downloaded")
    with pymupdf.open(path) as pdf:
        return page_rows(pdf[page]).splitlines()


def test_cash_flow_row_keeps_label_and_numbers_together():
    rows = rows_of("3M_2018_10K", 59)
    assert "Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373) | (1,420)" in rows
    assert "(Millions) | 2018 | 2017 | 2016" in rows


def test_balance_sheet_row():
    assert "Trade receivables, net | 1,615.9 | 1,864.3" in rows_of("AMCOR_2020_10K", 49)


def test_prose_is_not_split_into_columns():
    rows = rows_of("PEPSICO_2022_10K", 25)
    assert not any(" | " in r for r in rows if r.startswith("Our principal executive office"))
