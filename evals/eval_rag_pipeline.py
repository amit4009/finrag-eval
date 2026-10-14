"""
evals/eval_rag_pipeline.py -- the RAG triad on the live pipeline.

    contextual relevancy  is the retrieved context about the question?
    faithfulness          is the answer supported by the retrieved context?
    answer relevancy      does the answer address the question?

Same three metrics as the tutorial, now on FinanceBench questions, plus two that
separate "answered less" from "made things up":

    answer_rate             share of questions answered rather than declined (no judge)
    faithfulness_attempted  faithfulness over the ANSWERED questions only

Why: a refusal makes no claims, so it scores 1.0 on faithfulness. Plain
faithfulness therefore rises when the pipeline gives up more and falls when
better retrieval lets it attempt more -- E1 cut refusals 26 -> 14 and plain
faithfulness "regressed". The registry gates on faithfulness_attempted and
keeps plain faithfulness as info.

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
    JUDGE_RUN,
    load_phase_goldens,
    print_summary,
    summarize_by_metric,
    summarize_scores,
)
from src import config
from src.generator import is_abstention
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
    result = evaluate(**JUDGE_RUN, test_cases=test_cases, metrics=[
        ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ])
    summary = summarize_by_metric(result)

    answered = summarize_by_metric(result, keep=lambda tr: not is_abstention(tr.actual_output))
    if "Faithfulness" in answered:
        summary["faithfulness_attempted"] = answered["Faithfulness"]
    summary.update(summarize_scores(
        "answer_rate", [float(not is_abstention(tc.actual_output)) for tc in test_cases], 1.0))
    return summary


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())
