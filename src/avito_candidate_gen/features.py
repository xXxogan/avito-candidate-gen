"""
Признаки пары (запрос, объявление-кандидат) для реранкера

Дополнительные сигналы позволяют поднять нужного
кандидата выше
- локация: 83% релевантных пар совпадают по location_id;
- фильтры поиска: токены search_infm_params_text почти полностью
  входят в параметры выбранного объявления (overlap 0.97);
- покрытие заголовка: запросы короткие и лексически близки к заголовкам;
- рейтинг/отзывы/популярность: пользователи чаще выбирают проверенных
"""

import sys

import numpy as np
import pandas as pd

# Фиксированный порядок признаков
FEATURE_COLUMNS = [
    "bm25_norm",  # BM25-скор, нормированный на максимум внутри запроса
    "loc_match",  # 1.0, если локация запроса == локации объявления
    "params_overlap",  # доля токенов фильтра, найденных в параметрах объявления
    "has_filter",  # 1.0, если у запроса вообще есть фильтр
    "title_coverage",  # доля токенов запроса, присутствующих в заголовке
    "rating_norm",  # рейтинг / 5
    "rating_missing",  # флаг пропуска рейтинга
    "reviews_norm",  # log1p(число отзывов), нормировано на максимум по корпусу
    "pop_norm",  # log1p(клики в train), нормировано по популярности объявления
]


def make_query_record(row: pd.Series, query_lemma: str) -> dict:
    """
    Упаковка признаков запроса для FeatureBuilder

    query_lemma — лемматизированный текст запроса (build_query_text)
    Фильтр сравнивается с параметрами объявления токенами без лемматизации
    """
    filter_text = row.get("search_infm_params_text", "")
    return {
        "tokens": set(str(query_lemma).split()),
        "location_id": int(row["search_location_id"]),
        "filter_tokens": (
            set(str(filter_text).lower().split()) if filter_text else set()
        ),
    }


class FeatureBuilder:
    """
    Сборка матрицы признаков для кандидатов одного запроса
    """

    def __init__(self, df_corpus: pd.DataFrame, popularity: pd.Series | None = None):
        """
        df_corpus — уникальные объявления с колонками:
            item_id, item_location_id, item_infm_params_text,
            item_title_lemma, item_rating, item_rating_reviews_count
        popularity — количество кликов по item_id в train-части.
            Только train: популярность, посчитанная по val-кликaм
        """
        df_corpus = df_corpus.reset_index(drop=True)
        self.item_ids = df_corpus["item_id"].to_numpy()

        self.loc = df_corpus["item_location_id"].to_numpy()

        rating = df_corpus["item_rating"].to_numpy(dtype=float)
        self.rating = (np.nan_to_num(rating, nan=0.0) / 5.0).astype(np.float32)
        self.rating_missing = np.isnan(rating).astype(np.float32)

        reviews = df_corpus["item_rating_reviews_count"].to_numpy(dtype=float)
        log_reviews = np.log1p(np.nan_to_num(reviews, nan=0.0))
        self.reviews = (log_reviews / max(log_reviews.max(), 1e-9)).astype(np.float32)

        if popularity is not None:
            pop = (
                popularity.reindex(df_corpus["item_id"]).fillna(0).to_numpy(dtype=float)
            )
        else:
            pop = np.zeros(len(df_corpus))
        log_pop = np.log1p(pop)
        self.pop = (log_pop / max(log_pop.max(), 1e-9)).astype(np.float32)

        # Токены фильтров и заголовков
        self.infm_tokens = [
            frozenset(map(sys.intern, str(s).lower().split()))
            for s in df_corpus["item_infm_params_text"]
        ]
        self.title_tokens = [
            frozenset(map(sys.intern, str(s).split()))
            for s in df_corpus["item_title_lemma"]
        ]

    def build_arrays(
        self, qrec: dict, pos: np.ndarray, bm25_scores: np.ndarray
    ) -> np.ndarray:
        """
        Матрица признаков (n_кандидатов * len(FEATURE_COLUMNS))

        pos — позиции кандидатов в корпусе (индексы из retrieve_batch),
        bm25_scores — их BM25-скоры в том же порядке
        """
        pos = np.asarray(pos)
        n = len(pos)
        out = np.empty((n, len(FEATURE_COLUMNS)), dtype=np.float32)

        # 0. bm25_norm — нормировка на максимум внутри запроса
        scores = np.asarray(bm25_scores, dtype=np.float32)
        mx = scores.max() if n else 0.0
        out[:, 0] = scores / mx if mx > 0 else 0.0

        # 1. loc_match — точное совпадение локации
        out[:, 1] = self.loc[pos] == qrec["location_id"]

        # 2-3. params_overlap + has_filter: без фильтра overlap всегда 0
        ftok = qrec["filter_tokens"]
        out[:, 3] = 1.0 if ftok else 0.0
        if ftok:
            out[:, 2] = [len(ftok & self.infm_tokens[j]) / len(ftok) for j in pos]
        else:
            out[:, 2] = 0.0

        # 4. title_coverage — доля токенов запроса в заголовке
        qtok = qrec["tokens"]
        if qtok:
            out[:, 4] = [len(qtok & self.title_tokens[j]) / len(qtok) for j in pos]
        else:
            out[:, 4] = 0.0

        # 5-8. атрибуты объявления — готовыми массивами по позициям
        out[:, 5] = self.rating[pos]
        out[:, 6] = self.rating_missing[pos]
        out[:, 7] = self.reviews[pos]
        out[:, 8] = self.pop[pos]
        return out

    def build(
        self, qrec: dict, candidate_ids: list[str], bm25_scores: np.ndarray
    ) -> pd.DataFrame:
        """
        То же, что build_arrays, но принимает item_id и возвращает DataFrame
        """
        pos_by_id = getattr(self, "_pos_by_id", None)
        if pos_by_id is None:
            pos_by_id = {iid: i for i, iid in enumerate(self.item_ids)}
            self._pos_by_id = pos_by_id

        pos = np.fromiter(
            (pos_by_id[i] for i in candidate_ids),
            dtype=np.int64,
            count=len(candidate_ids),
        )
        feats = self.build_arrays(qrec, pos, bm25_scores)

        df = pd.DataFrame(feats, columns=FEATURE_COLUMNS)
        df.insert(0, "item_id", list(candidate_ids))
        return df
