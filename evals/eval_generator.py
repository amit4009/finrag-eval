"""
evals/eval_generator.py -- the GENERATOR, in isolation.

ISOLATION: we feed the generator FinanceBench's labelled evidence pages
(golden_context), NOT the retriever's output. So any failure here is the
generator's fault -- the right page was in front of it.

  faithfulness      claims in the answer supported by the context? (judge)
  answer relevancy  does the answer address the question? (judge)
  numeric_match     right number, for the metrics-generated questions (no judge)

numeric_match here vs in eval_application is the key diagnostic: if the
generator gets the number from the gold page but the full pipeline doesn't,
the loss is in retrieval.

    uv run python -m evals.eval_generator
"""

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from evals.harness import (
    NO_CACHE,
    load_phase_goldens,
    print_summary,
    summarize_by_metric,
    summarize_scores,
)
from evals.numeric import numeric_match
from src import config
from src.generator import generate

THRESHOLD = 0.7


def run():
    goldens = load_phase_goldens()

    test_cases, numeric = [], []
    for g in goldens:
        answer = generate(g["question"], g["golden_context"])

        test_cases.append(LLMTestCase(
            input=g["question"],
            actual_output=answer,
            retrieval_context=g["golden_context"],
        ))
        if g["numeric"]:
            numeric.append(float(numeric_match(answer, g["ideal_answer"], g["question"])))

    result = evaluate(cache_config=NO_CACHE, test_cases=test_cases, metrics=[
        FaithfulnessMetric(threshold=THRESHOLD, model=config.JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=config.JUDGE_MODEL, include_reason=True),
    ])

    summary = summarize_by_metric(result)
    summary.update(summarize_scores("numeric_match", numeric, 1.0))
    return summary


if __name__ == "__main__":
    print_summary("generator", run())
