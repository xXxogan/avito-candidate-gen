"""
Тесты FeatureBuilder
"""

import numpy as np
import pandas as pd
import pytest

from avito_candidate_gen.features import (
    FEATURE_COLUMNS,
    FeatureBuilder,
    make_query_record,
)


@pytest.fixture()
def corpus():
    return pd.DataFrame(
        {
            "item_id": ["a1", "a2", "a3"],
            "item_location_id": [100, 200, 100],
            "item_infm_params_text": [
                "Вид услуги Красота, здоровье",
                "Вид услуги Ремонт",
                "",
            ],
            "item_title_lemma": ["маникюр педикюр", "ремонт квартира", "маникюр"],
            "item_rating": [4.5, np.nan, 5.0],
            "item_rating_reviews_count": [10.0, 0.0, 100.0],
        }
    )


def make_row(location_id=100, filter_text=""):
    return pd.Series(
        {
            "search_location_id": location_id,
            "search_infm_params_text": filter_text,
        }
    )


def test_build_returns_all_columns(corpus):
    fb = FeatureBuilder(corpus)
    qrec = make_query_record(make_row(), "маникюр")
    df = fb.build(qrec, ["a1", "a3"], np.array([2.0, 1.0]))
    assert list(df.columns) == ["item_id"] + FEATURE_COLUMNS
    assert len(df) == 2


def test_bm25_normalized_by_query_max(corpus):
    fb = FeatureBuilder(corpus)
    qrec = make_query_record(make_row(), "маникюр")
    df = fb.build(qrec, ["a1", "a3"], np.array([2.0, 1.0]))
    assert df["bm25_norm"].tolist() == pytest.approx([1.0, 0.5])


def test_loc_match(corpus):
    fb = FeatureBuilder(corpus)
    qrec = make_query_record(make_row(location_id=100), "маникюр")
    df = fb.build(qrec, ["a1", "a2"], np.array([1.0, 1.0]))
    assert df["loc_match"].tolist() == [1.0, 0.0]


def test_params_overlap_and_has_filter(corpus):
    fb = FeatureBuilder(corpus)
    # фильтр «Вид услуги Красота, здоровье» -> 4 токена
    qrec = make_query_record(
        make_row(filter_text="Вид услуги Красота, здоровье"), "маникюр"
    )
    df = fb.build(qrec, ["a1", "a2"], np.array([1.0, 1.0]))
    assert df["has_filter"].tolist() == [1.0, 1.0]
    assert df["params_overlap"].iloc[0] == pytest.approx(1.0)  # все 4 токена в a1
    assert df["params_overlap"].iloc[1] == pytest.approx(
        0.5
    )  # у a2 только «вид услуги»


def test_no_filter_gives_zero_overlap(corpus):
    fb = FeatureBuilder(corpus)
    qrec = make_query_record(make_row(filter_text=""), "маникюр")
    df = fb.build(qrec, ["a1"], np.array([1.0]))
    assert df["has_filter"].iloc[0] == 0.0
    assert df["params_overlap"].iloc[0] == 0.0


def test_title_coverage(corpus):
    fb = FeatureBuilder(corpus)
    qrec = make_query_record(make_row(), "маникюр педикюр")
    df = fb.build(qrec, ["a1", "a3"], np.array([1.0, 1.0]))
    # a1 содержит оба токена, a3 — только «маникюр»
    assert df["title_coverage"].tolist() == pytest.approx([1.0, 0.5])


def test_rating_and_missing_flag(corpus):
    fb = FeatureBuilder(corpus)
    qrec = make_query_record(make_row(), "маникюр")
    df = fb.build(qrec, ["a1", "a2"], np.array([1.0, 1.0]))
    assert df["rating_missing"].tolist() == [0.0, 1.0]
    assert df["rating_norm"].iloc[0] == pytest.approx(0.9)  # 4.5 / 5
    assert df["rating_norm"].iloc[1] == 0.0  # пропуск -> 0


def test_popularity_from_train_clicks(corpus):
    pop = pd.Series({"a1": 3, "a3": 0})  # a2 в train не кликали вообще
    fb = FeatureBuilder(corpus, popularity=pop)
    qrec = make_query_record(make_row(), "маникюр")
    df = fb.build(qrec, ["a1", "a2", "a3"], np.ones(3))
    assert df["pop_norm"].iloc[0] > 0
    assert df["pop_norm"].iloc[1] == 0.0
    assert df["pop_norm"].iloc[2] == 0.0
