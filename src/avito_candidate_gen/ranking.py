"""
Реранкинг кандидатов
"""

import numpy as np
import pandas as pd

from avito_candidate_gen.features import FEATURE_COLUMNS

# Веса подобраны в ноутбуках 03–04 (recall@50 0.3534 -> 0.4658 -> 0.4739).
# loc_match 0.5 — больше нельзя: работает как жёсткий фильтр и теряет 17%
# релевантных объявлений из других локаций.
# Популярность/рейтинг обнулены — в линейной формуле не работают (pop вредит),
# но признаки оставлены для обучаемого реранкера.
# bm25_title_norm = 0: заголовочный источник полезен как поставщик кандидатов
# в union-пул, но как линейный вес вредит (сетка в ноутбуке 04).
# char_sim 0.3 — единственный новый источник с ненулевым весом (опечатки, раскладка).
DEFAULT_WEIGHTS = {
    "bm25_norm": 1.0,
    "bm25_title_norm": 0.0,
    "char_sim": 0.3,
    "loc_match": 0.5,
    "title_coverage": 0.25,
    "params_overlap": 0.75,
    "pop_norm": 0.0,
    "rating_norm": 0.0,
    "reviews_norm": 0.0,
    "rating_missing": 0.0,
    "has_filter": 0.0,
}


def boost_score(df: pd.DataFrame, weights: dict) -> np.ndarray:
    """
    Линейная комбинация признаков: сумма (колонка * вес).
    Нулевые веса пропускаются
    """
    score = np.zeros(len(df), dtype=np.float64)
    for col, w in weights.items():
        if w:
            score += w * df[col].to_numpy(dtype=np.float64)
    return score


def rerank_top_k(
    df: pd.DataFrame,
    weights: dict,
    k: int = 50,
    query_col: str = "query",
) -> dict[str, list[str]]:
    """
    Топ-k item_id на каждый запрос по буст-скору.
    df единая таблица кандидатов всех запросов
    Возвращает dict в формате, который ожидает metrics.recall_at_50
    """
    scored = df.assign(_score=boost_score(df, weights))
    scored = scored.sort_values(
        [query_col, "_score"], ascending=[True, False], kind="mergesort"
    )
    top = scored.groupby(query_col, sort=False, observed=True).head(k)
    return {
        str(q): g["item_id"].tolist()
        for q, g in top.groupby(query_col, sort=False, observed=True)
    }
