"""
Тесты для recall_at_50
"""

import pytest

from avito_candidate_gen.metrics import recall_at_50


def test_example_from_task_description():
    """
    А: 1 релевантный, 1 найден  -> 1.0
    Б: 2 релевантных, 1 найден  -> 0.5
    В: 1 релевантный, 0 найдено -> 0.0
    Recall@50 = (1.0 + 0.5 + 0.0) / 3 = 0.5
    """
    predictions = {
        "A": ["item1", "item_noise"],
        "B": ["item2"],
        "V": ["item_wrong"],
    }
    relevant = {
        "A": {"item1"},
        "B": {"item2", "item3"},
        "V": {"item4"},
    }

    assert recall_at_50(predictions, relevant) == pytest.approx(0.5)


def test_perfect_predictions_give_recall_one():
    predictions = {"q1": ["a", "b", "c"]}
    relevant = {"q1": {"a", "b", "c"}}

    assert recall_at_50(predictions, relevant) == pytest.approx(1.0)


def test_missing_query_in_predictions_counts_as_zero():
    predictions = {"q1": ["a"]}
    relevant = {"q1": {"a"}, "q2": {"x"}}

    assert recall_at_50(predictions, relevant) == pytest.approx(0.5)
