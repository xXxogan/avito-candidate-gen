"""
Предобработка текста для дальнейшего использования в BM25 и эмбеддингах.

Шаги:
1. Нормализация текста: перевод в нижний регистр, схлопывание пробелов.
2. Лемматизация через pymorphy3: перевод в единую грамматическую форму
3. Сборка единого текстового поля объявления из title/description/params
    заголовок короче и ближе по духу к самому запросу,
    чем описание, поэтому его нужно "утяжелить"
    относительно более шумного описания.
"""

import re

import pandas as pd
import pymorphy3
from pathlib import Path

# Один экземпляр анализатора на модуль
_morph = pymorphy3.MorphAnalyzer()

# Кэш лемм на уровне процесса
_lemma_cache: dict[str, str] = {}


def _lemmatize_word(word: str) -> str:
    """
    Возвращение начальной формы слова с кэшем
    """
    if word not in _lemma_cache:
        _lemma_cache[word] = _morph.parse(word)[0].normal_form
    return _lemma_cache[word]


def normalize_text(text: str) -> str:
    """
    Нижний регистр + схлопывание пробелов/пунктуации в один пробел
    """
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_text(text: str) -> str:
    """
    Нормализация и лемматизация слов в тексте
    """
    normalized = normalize_text(text)
    if not normalized:
        return ""
    words = normalized.split()
    return " ".join(_lemmatize_word(w) for w in words)


def build_item_text(
    title: str,
    description: str,
    infm_params: str,
    title_repeats: int = 2,
) -> str:
    """
    Сборка единого текстового поля объявления для BM25/эмбеддингов

    Заголовок повторяется title_repeats раз, чтобы увеличить его вес
    относительно длинного описания без изменения самого алгоритма
    """
    title_lemma = lemmatize_text(title)
    description_lemma = lemmatize_text(description)
    infm_lemma = lemmatize_text(infm_params)

    parts = [title_lemma] * title_repeats + [description_lemma, infm_lemma]
    return " ".join(p for p in parts if p)


def build_query_text(search_query: str) -> str:
    """
    Подготовка текста запроса для поиска
    """
    return lemmatize_text(search_query)


def add_processed_text_columns(df_items: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление колонки item_text_processed к датафрейму объявлений,
    содержащую готовый текст для построения BM25-индекса
    """
    df_items = df_items.copy()
    df_items["item_text_processed"] = df_items.apply(
        lambda x: build_item_text(
            x["item_title_raw"], x["item_description_raw"], x["item_infm_params_text"]
        ),
        axis=1,
    )
    return df_items


def get_or_build_processed_items(
    df_items: pd.DataFrame,
    cache_path: Path,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Возврат датафрейма с колонкой item_text_processed, 
    используя кэш на диске, если он уже существует
    """
    if cache_path.exists() and not force_recompute:
        print(f"Читаю кэш: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Кэш не найден, считаю предобработку ({len(df_items):,} строк)...")
    df_processed = add_processed_text_columns(df_items)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(cache_path)
    print(f"Сохранено в {cache_path}")

    return df_processed
