"""
Дымовые тесты пайплайна на игрушечном корпусе:
полный цикл «запросы -> топ-k» и добивка коротких ответов.
"""

import numpy as np
import pandas as pd

from avito_candidate_gen.pipeline import (
    _backfill,
    _popular_fallback,
    generate_candidates,
)


def make_corpus():
    return pd.DataFrame(
        {
            "item_id": ["a1", "a2", "b1", "b2", "c1"],
            "item_text_processed": [
                "маникюр педикюр салон",
                "маникюр наращивание ногти",
                "ремонт квартира сантехник",
                "ремонт ванная под ключ",
                "свадебный маникюр невеста",
            ],
            "item_title_lemma": [
                "маникюр педикюр",
                "маникюр ногти",
                "ремонт квартира",
                "ремонт ванная",
                "маникюр свадебный",
            ],
            "item_location_id": [100, 100, 200, 200, 100],
            "item_infm_params_text": [
                "Вид услуги Красота",
                "Вид услуги Красота",
                "Вид услуги Ремонт",
                "",
                "",
            ],
            "item_rating": [4.5, 5.0, np.nan, 4.0, 4.9],
            "item_rating_reviews_count": [10.0, 20.0, 0.0, 5.0, 100.0],
        }
    )


def make_queries():
    return pd.DataFrame(
        {
            "query_id": ["q1", "q2"],
            "search_query": ["маникюр", "ремонт ванной"],
            "search_location_id": [100, 200],
            "search_infm_params_text": ["Вид услуги Красота", ""],
        }
    )


def test_generate_candidates_smoke():
    corpus = make_corpus()
    pop = pd.Series({"a1": 5, "b1": 2})
    preds = generate_candidates(
        make_queries(), corpus, popularity=pop, top_k=4, final_k=2, batch_size=1
    )
    assert set(preds) == {"q1", "q2"}
    for ids in preds.values():
        assert len(ids) == 2
        assert len(set(ids)) == 2
        assert set(ids) <= set(corpus["item_id"])
    # запрос про маникюр в локации 100 с фильтром «Красота»:
    # a1 совпадает и по тексту, и по локации, и по фильтру
    assert "a1" in preds["q1"]


def test_backfill_fills_short_predictions():
    corpus = make_corpus()
    pop = pd.Series({"c1": 10, "a2": 5})
    preds = {"q1": ["a1"], "q2": ["b1", "b2"]}
    out = _backfill(preds, ["q1", "q2"], corpus, pop, final_k=2)
    assert len(out["q1"]) == 2
    assert out["q1"][0] == "a1"  # существующий кандидат остаётся первым
    assert out["q1"][1] in {"c1", "a2"}  # добито популярным
    assert out["q2"] == ["b1", "b2"]  # полный ответ не тронут


def test_popular_fallback_order():
    corpus = make_corpus()
    pop = pd.Series({"a1": 1, "c1": 9})
    top2 = _popular_fallback(corpus, pop, 2)
    assert top2[0] == "c1"  # больше кликов
    assert top2[1] == "a1"  # вторые по кликам
