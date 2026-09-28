"""
Пайплайн кандидатогенерации: запросы -> топ-50 item_id.

Схема (мультиисточник):
    BM25 (полный текст) + BM25 (заголовки) + char n-gram (заголовки)
    -> union кандидатов -> признаки -> линейный реранк -> top-50

Источники дают разных кандидатов: полнотекстовый BM25 — основной,
заголовочный ловит короткие точные совпадения, char n-gram — опечатки
и слова вне словаря лемматизатора. Запросы, набранные в английской
раскладке, предварительно декодируются (fix_keyboard_layout).

Общие шаги вынесены в функции prepare_queries / retrieve_source /
build_feature_matrix: их используют и экспериментальные ноутбуки, и
подготовка answer.csv — валидация и сабмит видят одинаковый текст
запросов и одинаковую сборку признаков.
"""

import gc

import numpy as np
import pandas as pd

from avito_candidate_gen.features import (
    FEATURE_COLUMNS,
    FeatureBuilder,
    make_query_record,
)
from avito_candidate_gen.preprocessing import (
    build_query_text,
    fix_keyboard_layout,
    normalize_text,
)
from avito_candidate_gen.ranking import DEFAULT_WEIGHTS, rerank_top_k
from avito_candidate_gen.retrieval.bm25 import BM25Retriever
from avito_candidate_gen.retrieval.char_ngram import CharNgramRetriever


def prepare_queries(df_queries: pd.DataFrame) -> pd.DataFrame:
    """
    Подготовка текстов запросов: +query_text (леммы — для BM25)
    и +query_norm (нормализованный сырой текст — для char n-gram).

    Раскладочные запросы («ktrnhbrf ghjdjlrf») декодируются первыми:
    оба источника должны видеть исправленный текст.
    """
    queries = df_queries.copy()
    fixed = queries["search_query"].apply(fix_keyboard_layout)
    queries["query_text"] = fixed.apply(build_query_text)
    queries["query_norm"] = fixed.apply(normalize_text)

    n_fixed = int((fixed != queries["search_query"]).sum())
    print(f"Запросов: {len(queries):,} | исправлена раскладка у {n_fixed}")
    return queries


def retrieve_source(
    retriever,
    query_texts: list[str],
    k: int,
    normalize: bool = True,
    batch_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Батчевый retrieve одного источника -> (позиции (nq, k), скоры (nq, k))

    normalize=True приводит скоры к 0..1 делением на максимум внутри запроса
    (для BM25); косинусные скоры char n-gram уже лежат в 0..1.
    """
    nq = len(query_texts)
    pos_all = np.empty((nq, k), dtype=np.int32)
    score_all = np.empty((nq, k), dtype=np.float32)

    for start in range(0, nq, batch_size):
        stop = min(start + batch_size, nq)
        pos, sc = retriever.retrieve_batch(query_texts[start:stop], top_k=k)
        if normalize:
            mx = sc.max(axis=1, keepdims=True)
            sc = np.divide(sc, mx, out=np.zeros_like(sc), where=mx > 0)
        pos_all[start:stop] = pos
        score_all[start:stop] = sc
        print(f"  retrieve: {stop:,}/{nq:,}", flush=True)

    return pos_all, score_all


def merge_sources(
    pos_arrays: list[np.ndarray], score_arrays: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Union кандидатов нескольких источников для одного запроса

    Возвращает (позиции, матрица скоров n_union × n_sources); 0 в колонке
    источника означает, что он кандидата не вернул.

    Порядок строк — по первому появлению: сначала кандидаты первого
    источника в его порядке, затем новые кандидаты следующих.
    Это делает детерминированными ничьи при реранке.
    """
    all_pos = np.concatenate(pos_arrays)
    union, first_idx = np.unique(all_pos, return_index=True)

    scores = np.zeros((len(union), len(pos_arrays)), dtype=np.float32)
    for s, (pos, sc) in enumerate(zip(pos_arrays, score_arrays)):
        idx = np.searchsorted(union, pos)
        scores[idx, s] = sc  # позиции внутри одного источника уникальны

    order = np.argsort(first_idx, kind="stable")
    return union[order], scores[order]


def build_feature_matrix(
    fb: FeatureBuilder,
    queries: pd.DataFrame,
    sources: list[tuple[str, np.ndarray, np.ndarray]],
    corpus_ids: np.ndarray,
    query_keys: np.ndarray | None = None,
    progress_every: int = 2000,
) -> pd.DataFrame:
    """
    Union источников + признаки -> единая таблица всех кандидатов.

    sources — список (колонка признака, позиции (nq, k), скоры (nq, k))
    в порядке retrieval: скор первого источника попадает в bm25_norm
    (основной сигнал), остальных — в свои колонки из extra_scores.
    Скоры должны быть уже в масштабе 0..1 (это делает retrieve_source).

    query_keys — значения колонки query (по умолчанию номера строк).
    Строковые колонки хранятся как Categorical — компактно при миллионах строк.
    """
    nq = len(queries)
    pos_mats = [pos for _, pos, _ in sources]
    score_mats = [sc for _, _, sc in sources]
    extra_names = [name for name, _, _ in sources[1:]]

    feat_blocks, pos_blocks, qidx_blocks = [], [], []
    for i in range(nq):
        row = queries.iloc[i]
        qrec = make_query_record(row, row["query_text"])
        union, src_scores = merge_sources(
            [pm[i] for pm in pos_mats], [sm[i] for sm in score_mats]
        )
        feat_blocks.append(
            fb.build_arrays(
                qrec,
                union,
                src_scores[:, 0],
                extra_scores={
                    name: src_scores[:, s + 1] for s, name in enumerate(extra_names)
                },
            )
        )
        pos_blocks.append(union)
        qidx_blocks.append(np.full(len(union), i, dtype=np.int32))
        if (i + 1) % progress_every == 0:
            print(f"  features: {i + 1:,}/{nq:,}", flush=True)

    features = pd.DataFrame(np.concatenate(feat_blocks), columns=FEATURE_COLUMNS)
    if query_keys is None:
        query_keys = np.arange(nq).astype(str)
    features["query"] = pd.Categorical.from_codes(
        np.concatenate(qidx_blocks), categories=np.asarray(query_keys)
    )
    features["item_id"] = pd.Categorical.from_codes(
        np.concatenate(pos_blocks), categories=np.asarray(corpus_ids)
    )
    return features


def generate_candidates(
    df_queries: pd.DataFrame,
    corpus: pd.DataFrame,
    popularity: pd.Series | None = None,
    top_k: int = 200,
    title_k: int = 200,
    char_k: int = 200,
    final_k: int = 50,
    weights: dict = DEFAULT_WEIGHTS,
    batch_size: int = 2000,
    query_id_col: str = "query_id",
) -> dict[str, list[str]]:
    """
    Полный цикл генерации кандидатов

    - df_queries — запросы с search_*-признаками и колонкой идентификатора;
    - corpus — уникальные объявления с item_text_processed, item_title_lemma,
      item_title_raw и атрибутами (локация, параметры, рейтинг, отзывы);
    - popularity — клики по item_id из train;
    - top_k/title_k/char_k — глубины источников, final_k — размер ответа.
    """
    queries = prepare_queries(df_queries)
    fb = FeatureBuilder(corpus, popularity=popularity)

    # Источники обрабатываются последовательно: индекс строится, отдаёт
    # результаты и освобождается — три индекса одновременно в память не лезут.
    # Порядок важен: первый источник задаёт порядок кандидатов при ничьих.
    specs = [
        # (колонка признака, фабрика ретривера, тексты запросов, глубина, нормировать)
        ("bm25_norm", lambda: BM25Retriever.from_dataframe(corpus), "query_text", top_k, True),
        (
            "bm25_title_norm",
            lambda: BM25Retriever.from_dataframe(corpus, text_col="item_title_lemma"),
            "query_text",
            title_k,
            True,
        ),
        ("char_sim", lambda: CharNgramRetriever.from_dataframe(corpus), "query_norm", char_k, False),
    ]

    sources = []
    for name, factory, text_col, k, normalize in specs:
        print(f"Источник {name}...", flush=True)
        retriever = factory()
        pos, sc = retrieve_source(
            retriever,
            queries[text_col].tolist(),
            k,
            normalize=normalize,
            batch_size=batch_size,
        )
        sources.append((name, pos, sc))
        del retriever
        gc.collect()

    features = build_feature_matrix(
        fb,
        queries,
        sources,
        corpus["item_id"].to_numpy(),
        query_keys=queries[query_id_col].astype(str).to_numpy(),
        progress_every=batch_size,
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
