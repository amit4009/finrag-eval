"""
evals/eval_rag_pipeline.py -- the RAG triad on the live pipeline.

    contextual relevancy  is the retrieved context about the question?
    faithfulness          is the answer supported by the retrieved context?
    answer relevancy      does the answer address the question?

Same three metrics as the tutorial, now on FinanceBench questions.

    uv run python -m evals.eval_rag_pipeline
"""

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from evals.harness import (
    NO_CACHE,
    load_phase_goldens,
    print_summary,
    summarize_by_metric,
)
from src import config
from src.rag_pipeline import RagPipeline

THRESHOLD = 0.7


def run(rag):
    goldens = load_phase_goldens()

    test_cases = []
    for g in goldens:
        result = rag.invoke(g["question"])
        test_cases.append(LLMTestCase(
            input=g["question"],
            actual_output=result["answer"],
            retrieval_context=result["context"],
        ))

    judge = config.JUDGE_MODEL
    result = evaluate(cache_config=NO_CACHE, test_cases=test_cases, metrics=[
        ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ])
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())
