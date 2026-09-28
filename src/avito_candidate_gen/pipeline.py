"""
Пайплайн кандидатогенерации: запросы -> топ-50 item_id.

Схема:
    BM25 top-K по обработанному тексту -> признаки (запрос, кандидат)
    -> линейный реранк (DEFAULT_WEIGHTS) -> top-50

Этот же код используется и для подготовки answer.csv
"""

import gc

import numpy as np
import pandas as pd

from avito_candidate_gen.features import (
    FEATURE_COLUMNS,
    FeatureBuilder,
    make_query_record,
)
from avito_candidate_gen.preprocessing import build_query_text
from avito_candidate_gen.ranking import DEFAULT_WEIGHTS, rerank_top_k
from avito_candidate_gen.retrieval.bm25 import BM25Retriever


def generate_candidates(
    df_queries: pd.DataFrame,
    corpus: pd.DataFrame,
    popularity: pd.Series | None = None,
    top_k: int = 200,
    final_k: int = 50,
    weights: dict = DEFAULT_WEIGHTS,
    batch_size: int = 2000,
    query_id_col: str = "query_id",
) -> dict[str, list[str]]:
    """
    Полный цикл генерации кандидатов

    - df_queries — запросы с search_*-признаками и колонкой идентификатора;
    - corpus — уникальные объявления с item_text_processed, item_title_lemma
    и атрибутами (локация, параметры, рейтинг, отзывы);
    - popularity — клики по item_id из train (популярность объявлений);
    - top_k — глубина пула кандидатов, final_k — сколько оставить в ответе.
    """
    queries = df_queries.copy()
    queries["query_text"] = queries["search_query"].apply(build_query_text)

    retriever = BM25Retriever.from_dataframe(corpus)
    fb = FeatureBuilder(corpus, popularity=popularity)
    print(
        f"Запросов: {len(queries):,} | BM25-индекс по корпусу {len(corpus):,} построен"
    )

    # Матрица признаков предаллоцируется целиком
    nq = len(queries)
    feat_all = np.empty((nq * top_k, len(FEATURE_COLUMNS)), dtype=np.float32)
    pos_all = np.empty(nq * top_k, dtype=np.int32)
    query_idx = np.empty(nq * top_k, dtype=np.int32)

    for start in range(0, nq, batch_size):
        batch = queries.iloc[start : start + batch_size]
        pos_batch, score_batch = retriever.retrieve_batch(
            batch["query_text"].tolist(), top_k=top_k
        )
        for j in range(len(batch)):
            assert len(pos_batch[j]) == top_k, "bm25s вернул меньше top_k кандидатов"
            row = batch.iloc[j]
            qrec = make_query_record(row, row["query_text"])
            gi = (start + j) * top_k
            feat_all[gi : gi + top_k] = fb.build_arrays(
                qrec, pos_batch[j], score_batch[j]
            )
            pos_all[gi : gi + top_k] = pos_batch[j]
            query_idx[gi : gi + top_k] = start + j
        print(f"  retrieve: {min(start + batch_size, nq):,}/{nq:,}", flush=True)

    # BM25-индекс больше не нужен
    del retriever
    gc.collect()

    features = pd.DataFrame(feat_all, columns=FEATURE_COLUMNS)
    # Строковые колонки как Categorical
    features["query"] = pd.Categorical.from_codes(
        query_idx, categories=queries[query_id_col].astype(str).to_numpy()
    )
    features["item_id"] = pd.Categorical.from_codes(
        pos_all, categories=corpus["item_id"].to_numpy()
    )

    predictions = rerank_top_k(features, weights, k=final_k)
    return _backfill(
        predictions,
        queries[query_id_col].astype(str).tolist(),
        corpus,
        popularity,
        final_k,
    )


def _popular_fallback(
    corpus: pd.DataFrame, popularity: pd.Series | None, n: int
) -> list[str]:
    """
    n самых «авторитетных» объявлений корпуса
    """
    ids = corpus["item_id"].to_numpy()
    clicks = (
        popularity.reindex(ids).fillna(0).to_numpy(dtype=float)
        if popularity is not None
        else np.zeros(len(ids))
    )
    reviews = np.nan_to_num(corpus["item_rating_reviews_count"].to_numpy(dtype=float))
    order = np.lexsort((-reviews, -clicks))
    return [str(ids[i]) for i in order[:n]]


def _backfill(
    predictions: dict[str, list[str]],
    query_keys: list[str],
    corpus: pd.DataFrame,
    popularity: pd.Series | None,
    final_k: int,
) -> dict[str, list[str]]:
    """
    Добивка ответов короче final_k популярными объявлениями

    При top_k=200 не должна срабатывать — это страховка от вырожденных
    случаев, чтобы в answer.csv гарантированно не оказалось коротких строк
    (недоиспользованные слоты — бесплатный потерянный recall)
    """
    short = [q for q in query_keys if len(predictions.get(q, [])) < final_k]
    if not short:
        return predictions

    fallback = _popular_fallback(corpus, popularity, final_k * 2)
    for q in short:
        have = list(predictions.get(q, []))
        seen = set(have)
        for iid in fallback:
            if len(have) >= final_k:
                break
            if iid not in seen:
                have.append(iid)
                seen.add(iid)
        predictions[q] = have[:final_k]
    return predictions
