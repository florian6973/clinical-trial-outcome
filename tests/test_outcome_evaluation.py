import pytest

from clinical_trial_outcome.evaluation.outcome import (
    PAPER_OUTCOME_RESULTS,
    OutcomePrediction,
    evaluate_outcome_predictions,
)


def test_outcome_paper_results_are_preserved() -> None:
    assert PAPER_OUTCOME_RESULTS.source == "paper/expected_results.json"
    assert PAPER_OUTCOME_RESULTS.status == "validation_expectation_only"
    assert PAPER_OUTCOME_RESULTS.distinct_input_expressions == 480_273
    assert PAPER_OUTCOME_RESULTS.distinct_normalized_concepts == 237_820
    assert PAPER_OUTCOME_RESULTS.held_out_examples == 50
    assert PAPER_OUTCOME_RESULTS.exact_match_accuracy == 0.88


def test_outcome_evaluation_exact_match_and_macro_f1() -> None:
    metrics = evaluate_outcome_predictions(
        [
            OutcomePrediction("GROUP: Clinical Response", "clinical response"),
            OutcomePrediction("Adverse Events (AEs)", "Adverse Events (AEs)"),
            OutcomePrediction("Pain", "Symptoms"),
        ]
    )
    assert metrics.number_of_examples == 3
    assert metrics.exact_matches == 2
    assert metrics.exact_match_accuracy == pytest.approx(2 / 3)
    assert 0 <= metrics.macro_f1 <= 1


def test_outcome_evaluation_requires_examples() -> None:
    with pytest.raises(ValueError, match="At least one"):
        evaluate_outcome_predictions([])
