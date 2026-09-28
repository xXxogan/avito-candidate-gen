"""
BM25 retrieval — лексический поиск кандидатов по тексту.

Использует bm25s поверх подготовленного текста
Токенизация по пробелам, так как текст уже лемматизирован
и нормализован на этапе предобработки.
"""

import bm25s
import numpy as np
import pandas as pd


class BM25Retriever:
    """
    Обёртка над bm25s.BM25, которая хранит соответствие
    между позицией в индексе и item_id
    """

    def __init__(self, item_ids: list[str], texts: list[str]):
        if len(item_ids) != len(texts):
            raise ValueError("item_ids и texts должны быть одинаковой длины")

        self.item_ids = item_ids

        # Токенизация по пробелам
        corpus_tokens = bm25s.tokenize(texts, stopwords=[])

        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens)

    @classmethod
    def from_dataframe(
        cls, df_items: pd.DataFrame, text_col: str = "item_text_processed"
    ) -> "BM25Retriever":
        """
        Строит индекс из датафрейма с колонками item_id и текстовой колонкой.

        text_col позволяет по тому же корпусу строить разные индексы:
        по полному тексту (по умолчанию) или только по заголовкам.
        """
        return cls(
            item_ids=df_items["item_id"].tolist(),
            texts=df_items[text_col].tolist(),
        )

    def retrieve_batch(
        self, query_texts: list[str], top_k: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Сырой результат батчевого поиска: две матрицы (n_запросов × top_k) —
        позиции кандидатов в корпусе и их BM25-скоры.

        Позиции (а не item_id) нужны, чтобы строить признаки реранкера
        векторизованно, не гоняя лишние словари id→индекс.
        Результаты отсортированы по убыванию скора (стандартное поведение bm25s).
        """
        query_tokens = bm25s.tokenize(query_texts, stopwords=[])
        results, scores = self.bm25.retrieve(query_tokens, k=top_k)
        return np.asarray(results), np.asarray(scores, dtype=np.float32)

    def search(self, query_text: str, top_k: int = 50) -> list[str]:
        """
        Поиск для одного запроса — обёртка над search_batch
        """
        return self.search_batch([query_text], top_k=top_k)[0]

    def search_batch(self, query_texts: list[str], top_k: int = 50) -> list[list[str]]:
        """
        Основной метод поиска — bm25s, обрабатывающий пачку запросов.
        Возвращает списки item_id (скоры отбрасываются).
        """
        results, _scores = self.retrieve_batch(query_texts, top_k=top_k)
        return [[self.item_ids[i] for i in row] for row in results]
