"""
Char n-gram retrieval — поиск по символьным n-граммам (TF-IDF + косинус).

Зачем: ловит опечатки («электирик» -> «электрик») и запросы со словами,
которых не знает лемматизатор. Работает без лемматизации — сравнивает
сырые символы, поэтому устойчив к ошибкам ввода.

Индекс строится только по заголовкам: они короткие, стилистически близки
к запросам, и матрица n-грамм комфортно помещается в память.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from avito_candidate_gen.preprocessing import normalize_text


class CharNgramRetriever:
    """
    TF-IDF по символьным n-граммам (char_wb 2–4) + косинусная близость.

    Интерфейс retrieve_batch совпадает с BM25Retriever —
    источники взаимозаменяемы в пайплайне.
    """

    def __init__(
        self,
        item_ids: list[str],
        texts: list[str],
        ngram_range: tuple[int, int] = (2, 4),
        min_df: int = 2,
    ):
        self.item_ids = item_ids
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=min_df,  # ультра-редкие граммы — шум
            lowercase=True,
            dtype=np.float32,
        )
        # Только нормализация (регистр/пунктуация), без лемматизации:
        # форма символов должна сохраниться, включая опечатки
        self.matrix = self.vectorizer.fit_transform([normalize_text(t) for t in texts])

    @classmethod
    def from_dataframe(
        cls, df_items: pd.DataFrame, text_col: str = "item_title_raw", **kwargs
    ) -> "CharNgramRetriever":
        return cls(
            item_ids=df_items["item_id"].tolist(),
            texts=df_items[text_col].tolist(),
            **kwargs,
        )

    def retrieve_batch(
        self, query_texts: list[str], top_k: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        (позиции в корпусе, косинусные близости 0..1).

        Считаем по одному запросу: плотная матрица «запросы × корпус»
        не влезла бы в память. Умножение sparse @ dense-вектор — это
        O(nnz) без транспонирования корпусной матрицы.
        """
        q = self.vectorizer.transform([normalize_text(t) for t in query_texts])
        nq = q.shape[0]
        top_k = min(top_k, self.matrix.shape[0])

        pos = np.empty((nq, top_k), dtype=np.int64)
        scores = np.empty((nq, top_k), dtype=np.float32)
        for i in range(nq):
            sims = self.matrix @ q[i].toarray().ravel()
            if top_k < sims.shape[0]:
                idx = np.argpartition(-sims, top_k - 1)[:top_k]
                idx = idx[np.argsort(-sims[idx], kind="stable")]
            else:
                idx = np.argsort(-sims, kind="stable")
            pos[i] = idx
            scores[i] = sims[idx]
        return pos, scores
