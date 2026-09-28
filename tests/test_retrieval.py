"""
Тесты ретриверов: BM25Retriever (в частности retrieve_batch со скорами)
и CharNgramRetriever (поиск по символьным n-граммам).
"""

import pandas as pd

from avito_candidate_gen.retrieval.bm25 import BM25Retriever
from avito_candidate_gen.retrieval.char_ngram import CharNgramRetriever

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


# --- Char n-gram ---

CHAR_IDS = ["i1", "i2", "i3"]
CHAR_TEXTS = ["электрик проводка", "маникюр педикюр", "ремонт квартир"]


def test_char_ngram_finds_typo():
    # min_df=1: на игрушечном корпусе из 3 документов редкие граммы не выживают
    r = CharNgramRetriever(CHAR_IDS, CHAR_TEXTS, min_df=1)
    pos, sc = r.retrieve_batch(["электирик"], top_k=1)
    assert CHAR_IDS[pos[0][0]] == "i1"  # опечатка всё равно ведёт к «электрик»
    assert sc[0][0] > 0


def test_char_ngram_returns_k_for_unknown_query():
    r = CharNgramRetriever(CHAR_IDS, CHAR_TEXTS, min_df=1)
    pos, sc = r.retrieve_batch(["zzz qqq"], top_k=2)
    assert pos.shape == (1, 2)
    assert (sc == 0).all()  # ни одного совпадающего грамма
