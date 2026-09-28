"""
Тесты BM25Retriever — в частности нового retrieve_batch,
который возвращает позиции и скоры для реранкера.
"""

import pandas as pd

from avito_candidate_gen.retrieval.bm25 import BM25Retriever

IDS = ["i1", "i2", "i3"]
TEXTS = ["маникюр педикюр", "ремонт квартир сантехник", "маникюр наращивание"]


def test_retrieve_batch_returns_positions_and_scores():
    r = BM25Retriever(IDS, TEXTS)
    pos, scores = r.retrieve_batch(["маникюр"], top_k=2)
    assert pos.shape == (1, 2)
    assert scores.shape == (1, 2)
    # первый результат — одно из объявлений про маникюр (0 или 2), не ремонт (1)
    assert pos[0][0] in (0, 2)
    # скоры отсортированы по убыванию
    assert scores[0][0] >= scores[0][1]


def test_search_batch_maps_positions_to_ids():
    r = BM25Retriever(IDS, TEXTS)
    assert r.search_batch(["ремонт квартир"], top_k=1) == [["i2"]]


def test_from_dataframe_custom_text_col():
    df = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "item_text_processed": ["общий текст про всё", "другой текст"],
            "item_title_lemma": ["маникюр", "ремонт"],
        }
    )
    r = BM25Retriever.from_dataframe(df, text_col="item_title_lemma")
    assert r.search("маникюр", top_k=1) == ["a"]
