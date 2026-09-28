"""
BM25 retrieval — лексический поиск кандидатов по тексту.

Использует bm25s поверх подготовленного текста
Токенизация по пробелам, так как текст уже лемматизирован
и нормализован на этапе предобработки.
"""

import bm25s
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
    def from_dataframe(cls, df_items: pd.DataFrame) -> BM25Retriever:
        """
        Строит индекс из датафрейма с колонками item_id и item_text_processed
        """
        return cls(
            item_ids=df_items["item_id"].tolist(),
            texts=df_items["item_text_processed"].tolist(),
        )

    def search(self, query_text: str, top_k: int = 50) -> list[str]:
        """
        Поиск для одного запроса — обёртка над search_batch
        """
        return self.search_batch([query_text], top_k=top_k)[0]

    def search_batch(self, query_texts: list[str], top_k: int = 50) -> list[list[str]]:
        """
        Основной метод поиска — bm25s, обрабатывающий пачку запросов
        """
        query_tokens = bm25s.tokenize(query_texts, stopwords=[])
        results, _scores = self.bm25.retrieve(query_tokens, k=top_k)
        return [[self.item_ids[i] for i in row] for row in results]
