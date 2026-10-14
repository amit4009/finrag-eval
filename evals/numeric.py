"""
evals/numeric.py -- judge-free correctness for number answers.

FinanceBench's 50 metrics-generated questions all have a bare-number answer
("$4.60", "16.5%", "0.66"), with the unit in the QUESTION ("in USD billions").
For those, an LLM judge is the wrong tool: it's slow, costs money, and wobbles
run to run. A number either matches or it doesn't.

numeric_match(answer, ideal, question) is True when ANY number in the model's
answer equals the ideal one, after:

  - scale normalisation:  "$4.6 billion" == "$4,625 million" == ideal "$4.60"
                          when the question asks for USD billions
  - tolerance:            max(1% relative, half a unit in the ideal's last
                          decimal place), so the ideal's own rounding never
                          fails a correct answer
  - sign:                 absolute values ("(1,577)" in a cash-flow statement
                          and "1,577 outflow" are the same capex)
  - percent vs fraction:  "16.5%" == "0.165" when only one side is a percent

Citations like [PEPSICO_2021_10K p.62] are stripped first so their years and
page numbers can't match by accident.

TWO VARIANTS:
  numeric_match         any number anywhere in the answer may match. Lenient: the
                        judge-agreement labels caught it crediting answers whose
                        WORKING held the right number while the STATED answer was
                        wrong (Amazon DPO: working 93.64, stated 36.73).
  numeric_match_stated  only the "Final answer:" line may match: the answer's
                        conclusion, which the prompt asks for since E6. This is the
                        guardrail; the lenient variant is kept as info.

  Answers without a "Final answer:" line (pre-E6 prompt) fall back to the lenient
  check: their conclusion has no fixed position. The first sentence was tried and
  was wrong four times in 17 labelled answers -- the model opened with one number
  and concluded with the corrected one (Kraft Heinz: "3.06" ... "gives 6.25").
"""

import re

SCALE_WORDS = {
    "thousand": 1e3, "thousands": 1e3, "k": 1e3,
    "million": 1e6, "millions": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6,
    "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9,
}

CITATION = re.compile(r"\[[^\]]*\]")
NUMBER = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?|\.\d+)"                         # 1,577.00  .5
    r"\s*(%|percent\b|thousands?\b|millions?\b|billions?\b|bn\b|mn\b|mm\b|[kmb]\b)?",
    re.IGNORECASE,
)
QUESTION_SCALE = re.compile(r"\b(thousands|millions|billions)\b", re.IGNORECASE)


def parse_numbers(text):
    """All numbers in text as (value_as_written, scale, is_percent)."""
    out = []
    for m in NUMBER.finditer(CITATION.sub(" ", text)):
        raw = m.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        unit = (m.group(2) or "").lower()
        is_pct = unit in ("%", "percent")
        out.append((value, SCALE_WORDS.get(unit, 1.0), is_pct))
    return out


def question_scale(question):
    """The scale the question asks the answer in: 'in USD millions' -> 1e6."""
    m = QUESTION_SCALE.search(question)
    return SCALE_WORDS[m.group(1).lower()] if m else 1.0


def _decimals(raw):
    raw = raw.strip().rstrip(".%")
    return len(raw.split(".")[1]) if "." in raw else 0


def _close(a, b, tol):
    return abs(abs(a) - abs(b)) <= tol


def numeric_match(answer, ideal, question=""):
    ideal_nums = parse_numbers(ideal)
    if not ideal_nums:
        return False
    gold, _, gold_pct = ideal_nums[0]
    gold_digits = re.search(r"[\d.,]+", ideal).group(0).replace(",", "")
    rel_tol = 0.01 * abs(gold)
    tol = max(rel_tol, 0.5 * 10 ** -_decimals(gold_digits))

    q_scale = question_scale(question)
    gold_abs = gold * q_scale

    for value, scale, is_pct in parse_numbers(answer):
        # 1. as written, in the question's unit ("4.6" vs "$4.60")
        if scale == 1.0 and is_pct == gold_pct and _close(value, gold, tol):
            return True
        # 2. scaled to absolute dollars ("$4,625 million" vs 4.60 billions)
        if scale != 1.0 and _close(value * scale, gold_abs, tol * q_scale):
            return True
        # 3. percent on exactly one side ("16.5%" vs "0.165")
        if is_pct != gold_pct:
            other = value / 100 if is_pct else value * 100
            if _close(other, gold, tol):
                return True
    return False


# "Final answer: X", tolerating Markdown bold on either side of the colon
FINAL_LINE = re.compile(r"^\s*\**\s*final answer\s*\**\s*:\s*\**\s*(.+?)\s*\**\s*$",
                        re.IGNORECASE | re.MULTILINE)


def stated_answer(answer):
    """The answer's conclusion: its last "Final answer:" line, or None."""
    finals = FINAL_LINE.findall(answer)
    return finals[-1].strip() if finals else None


def numeric_match_stated(answer, ideal, question=""):
    stated = stated_answer(answer)
    return numeric_match(answer if stated is None else stated, ideal, question)
