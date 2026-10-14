"""
evals/eval_application.py -- end-to-end answer quality vs the expert answers.

  correctness    GEval judge, all questions: does the answer agree with the
                 FinanceBench expert answer?
  numeric_match  no judge, metrics-generated questions only: is the number right?
                 (see evals/numeric.py)

The tutorial's Completeness and Style metrics are dropped: FinanceBench answers
are short facts, and a teaching voice is no longer the goal.

    uv run python -m evals.eval_application
"""

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from evals.harness import (
    JUDGE_RUN,
    load_phase_goldens,
    print_summary,
    summarize_by_metric,
    summarize_scores,
)
from evals.numeric import numeric_match
from src import config
from src.rag_pipeline import RagPipeline

THRESHOLD = 0.7


def correctness_metric():
    # reference-based, judges TRUTH against the expert answer -- not length or style
    return GEval(
        name="Correctness",
        evaluation_steps=[
            "Identify the key fact or figure in the expected output, including its company, fiscal period and unit.",
            "Check whether the actual output states the same fact or figure. Numbers match if they agree after rounding or a change of unit (e.g. $4.6 billion = $4,600 million).",
            "A figure from the wrong fiscal period, wrong company or wrong line item is incorrect, even if the number itself appears in the filing.",
            "If the actual output declines to answer or says it lacks information, it is incorrect.",
            "Do NOT deduct for brevity, citations, or extra correct detail such as the calculation steps.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="States the expected fact or figure correctly; no contradictions."),
            Rubric(score_range=(5, 8),  expected_outcome="Substantially correct but with a minor inaccuracy, e.g. small rounding error or one secondary point wrong."),
            Rubric(score_range=(0, 4),  expected_outcome="Wrong figure, wrong period, contradicts the expected output, or declines to answer."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                           LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=config.JUDGE_MODEL,
        strict_mode=False,
    )


def run(rag):
    goldens = load_phase_goldens()

    test_cases, numeric = [], []
    for g in goldens:
        result = rag.invoke(g["question"])
        test_cases.append(LLMTestCase(
            input=g["question"],
            actual_output=result["answer"],
            expected_output=g["ideal_answer"],
        ))
        if g["numeric"]:
            numeric.append(float(numeric_match(result["answer"], g["ideal_answer"], g["question"])))

    result = evaluate(**JUDGE_RUN, test_cases=test_cases, metrics=[correctness_metric()])

    summary = summarize_by_metric(result)
    summary.update(summarize_scores("numeric_match", numeric, 1.0))
    return summary


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("application", run_local())
