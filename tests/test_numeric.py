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


# ---- stated answer: the "Final answer:" line ----
from evals.numeric import numeric_match_stated, stated_answer  # noqa: E402

DPO_Q = "What is Amazon's FY2017 days payable outstanding (DPO)?"


def test_stated_answer_reads_the_final_line():
    a = "Working: 365 * 0.256 = 93.64.\n\nFinal answer: 93.64 days"
    assert stated_answer(a) == "93.64 days"
    assert stated_answer("**Final answer:** $4.6 billion") == "$4.6 billion"


def test_strict_match_ignores_right_numbers_in_the_working():
    # the lenient matcher's false pass from the labels: right in the working, wrong conclusion
    a = "DPO = 365 * 0.256 = 93.64\n\nFinal answer: 36.73 days"
    assert numeric_match(a, "93.86", DPO_Q)               # lenient: passes
    assert not numeric_match_stated(a, "93.86", DPO_Q)    # strict: fails


def test_without_final_line_strict_falls_back_to_lenient():
    a = "DPO is 36.73 days. Working: 365 * 0.256 = 93.64."
    assert stated_answer(a) is None
    assert numeric_match_stated(a, "93.86", DPO_Q) == numeric_match(a, "93.86", DPO_Q)


def test_stray_percent_sign_is_a_unit_slip_not_a_wrong_answer():
    # human-labelled correct: DPO is in days, the model wrote "93.64 %"
    assert numeric_match_stated("Final answer: 93.64 %", "93.86", DPO_Q)
    # but a genuinely different number with a % is still wrong
    assert not numeric_match_stated("Final answer: 36.73 %", "93.86", DPO_Q)
