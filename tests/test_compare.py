from evals.compare import compare
from evals.metric_registry import rule_for
from src.generator import ABSTAIN, is_abstention


def snap(**metrics):
    return {"metrics": {k.replace("__", "."): v for k, v in metrics.items()}}


def verdict(base, cand):
    return compare(snap(**base), snap(**cand))[0]


# ---- registry: which rule each metric gets ----

def test_retrieval_metrics_have_zero_tolerance():
    for mid in ["retriever.hit_at_5.avg_score", "retriever.mrr.avg_score",
                "retriever.doc_hit_at_5.avg_score", "retriever.filter_accuracy.avg_score"]:
        assert rule_for(mid)["tol"] == 0 and not rule_for(mid).get("questions"), mid


def test_generator_flip_metrics_allow_one_question():
    for mid in ["generator.numeric_match.avg_score", "application.numeric_match.avg_score"]:
        assert rule_for(mid)["questions"] == 1, mid


def test_judge_tolerances_are_per_metric():
    # correctness is ~3x less noisy than faithfulness: one shared 0.05 was wrong both ways
    assert rule_for("application.correctness_[geval].avg_score")["tol"] == 0.03
    assert rule_for("generator.faithfulness.avg_score")["tol"] == 0.06
    assert rule_for("some.unmeasured_metric.avg_score")["tol"] == 0.05   # default


def test_plain_faithfulness_is_info_attempted_is_guarded():
    assert rule_for("pipeline.faithfulness.avg_score")["kind"] == "info"
    assert rule_for("pipeline.faithfulness_attempted.avg_score")["kind"] == "guardrail"
    assert rule_for("generator.faithfulness.avg_score")["kind"] == "guardrail"


# ---- compare: the noise cases that produced false REVIEWs ----

def test_one_numeric_flip_out_of_7_is_flat():
    # Phase 2 noise rerun: 6/7 -> 5/7 on an identical pipeline
    assert verdict({"generator__numeric_match__avg_score": 6 / 7, "generator__numeric_match__n": 7},
                   {"generator__numeric_match__avg_score": 5 / 7, "generator__numeric_match__n": 7}
                   ) == "PASS"


def test_two_numeric_flips_out_of_7_regress():
    assert verdict({"generator__numeric_match__avg_score": 6 / 7, "generator__numeric_match__n": 7},
                   {"generator__numeric_match__avg_score": 4 / 7, "generator__numeric_match__n": 7}
                   ) == "REVIEW"


def test_one_lost_retrieval_hit_regresses():
    # deterministic: losing 1/47 hits is real, even though it's under 0.05
    assert verdict({"retriever__hit_at_5__avg_score": 28 / 47, "retriever__hit_at_5__n": 47},
                   {"retriever__hit_at_5__avg_score": 27 / 47, "retriever__hit_at_5__n": 47}
                   ) == "REVIEW"


def test_judge_noise_within_measured_band_is_flat():
    # E1 noise rerun: generator faithfulness 0.905 -> 0.849 on an identical pipeline
    assert verdict({"generator__faithfulness__avg_score": 0.9047},
                   {"generator__faithfulness__avg_score": 0.8488}) == "PASS"


def test_correctness_drop_beyond_its_noise_regresses():
    # 0.04 would have passed the old shared 0.05; correctness noise is ~0.007
    assert verdict({"application__correctness_[geval]__avg_score": 0.39},
                   {"application__correctness_[geval]__avg_score": 0.35}) == "REVIEW"


def test_plain_faithfulness_drop_no_longer_triggers_review():
    # E1: refusals 26 -> 14 dragged plain faithfulness 0.905 -> 0.848
    assert verdict({"pipeline__faithfulness__avg_score": 0.905},
                   {"pipeline__faithfulness__avg_score": 0.848}) == "PASS"


def test_without_n_per_question_rule_falls_back_to_zero_tolerance():
    assert verdict({"generator__numeric_match__avg_score": 6 / 7},
                   {"generator__numeric_match__avg_score": 5 / 7}) == "REVIEW"


# ---- abstention detection ----

def test_is_abstention():
    assert is_abstention(ABSTAIN)
    assert is_abstention("I don’t have enough information in the provided filings to answer that")
    assert not is_abstention("PepsiCo's FY2021 capex was $4.6 billion [PEPSICO_2021_10K p.62].")


# ---- duplicate evidence in hit@k ----

def _doc(name, page):
    from types import SimpleNamespace
    return SimpleNamespace(metadata={"doc_name": name, "page": page})


def test_duplicate_evidence_page_counts_as_hit():
    from evals.eval_retriever import rank_scores
    expected = [{"doc_name": "AMD_2022_10K", "page": 3}]
    copies = [{"doc_name": "AMD_2022_annualreport", "page": 8}]
    retrieved = [_doc("AMD_2022_annualreport", 10), _doc("AMD_2022_annualreport", 8)]
    assert rank_scores(retrieved, expected) == (0.0, 0.0, 0.0)          # old rule: miss
    assert rank_scores(retrieved, expected, copies) == (1.0, 0.5, 1.0)  # copy at rank 2
