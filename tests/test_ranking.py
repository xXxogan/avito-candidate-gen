"""
Тесты линейного реранка (boost_score / rerank_top_k).
"""

import pandas as pd

from avito_candidate_gen.features import FEATURE_COLUMNS
from avito_candidate_gen.ranking import DEFAULT_WEIGHTS, rerank_top_k


def make_df():
    """Два запроса по три кандидата; строки уже идут в BM25-порядке."""
    zeros = {c: [0.0] * 6 for c in FEATURE_COLUMNS if c not in ("bm25_norm", "loc_match")}
    return pd.DataFrame(
        {
            "query": ["q1"] * 3 + ["q2"] * 3,
            "item_id": ["a", "b", "c", "d", "e", "f"],
            "bm25_norm": [1.0, 0.8, 0.6, 1.0, 0.5, 0.25],
            "loc_match": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            **zeros,
        }
    )


def test_default_weights_cover_all_features():
    """Веса и признаки не должны разъезжаться после правок FEATURE_COLUMNS."""
    assert set(DEFAULT_WEIGHTS) == set(FEATURE_COLUMNS)


def test_pure_bm25_weights_preserve_order():
    pred = rerank_top_k(make_df(), {"bm25_norm": 1.0}, k=2)
    assert pred["q1"] == ["a", "b"]
    assert pred["q2"] == ["d", "e"]


def test_location_boost_lifts_candidate():
    # сильный вес локации поднимает b (второго по BM25) на первое место
    pred = rerank_top_k(make_df(), {"bm25_norm": 1.0, "loc_match": 5.0}, k=2)
    assert pred["q1"] == ["b", "a"]


def test_ties_keep_bm25_order():
    # все скоры равны -> устойчивая сортировка сохраняет BM25-порядок
    pred = rerank_top_k(make_df(), {}, k=3)
    assert pred["q1"] == ["a", "b", "c"]
