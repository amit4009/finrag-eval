import pytest

from evals.numeric import numeric_match, parse_numbers, question_scale

Q_BN = "What is the FY2021 capital expenditure amount (in USD billions) for PepsiCo?"
Q_MN = "What is the FY2022 unadjusted EBITDA less capex for PepsiCo? Answer in USD millions."
Q_PCT = "What is the FY2022 unadjusted EBITDA % margin for PepsiCo?"
Q_RATIO = "What is the FY2019 quick ratio for 3M?"


@pytest.mark.parametrize("answer, ideal, question", [
    ("PepsiCo's FY2021 capex was $4.6 billion [PEPSICO_2021_10K p.62].", "$4.60", Q_BN),
    ("Capital spending was $4,625 million.", "$4.60", Q_BN),
    ("The answer is 4.63.", "$4.60", Q_BN),                         # within 1%
    ("EBITDA less capex is $9,068 million.", "$9068.00", Q_MN),
    ("About $9.07 billion.", "$9068.00", Q_MN),
    ("The margin is 16.5%.", "16.5%", Q_PCT),
    ("The margin is 16.47%.", "16.5%", Q_PCT),                       # ideal's rounding
    ("The margin is 0.165.", "16.5%", Q_PCT),                        # fraction vs percent
    ("The quick ratio is 0.66.", "0.66", Q_RATIO),
    ("Change in margin: -0.02.", "-0.02", "What is the change?"),
    ("Capex was (1,577) in the cash flow statement.", "$1577.00", Q_MN),
])
def test_matches(answer, ideal, question):
    assert numeric_match(answer, ideal, question)


@pytest.mark.parametrize("answer, ideal, question", [
    ("PepsiCo's FY2021 capex was $4.2 billion.", "$4.60", Q_BN),     # wrong year's figure
    ("The margin is 15.8%.", "16.5%", Q_PCT),
    ("The quick ratio is 0.86.", "0.66", Q_RATIO),
    ("I don't have enough information in the provided filings to answer that.", "$4.60", Q_BN),
    # citation years / pages must not match by accident
    ("See [PEPSICO_2021_10K p.62] for details.", "62", "How many?"),
])
def test_mismatches(answer, ideal, question):
    assert not numeric_match(answer, ideal, question)


def test_parse_numbers_scales():
    assert parse_numbers("$4.6 billion and 12%") == [(4.6, 1e9, False), (12.0, 1.0, True)]


def test_question_scale():
    assert question_scale(Q_BN) == 1e9
    assert question_scale(Q_MN) == 1e6
    assert question_scale(Q_RATIO) == 1.0
