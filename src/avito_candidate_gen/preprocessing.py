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


# Соответствие клавиш QWERTY -> ЙЦУКЕН для декодировки неверной раскладки
_LAYOUT_EN_TO_RU = str.maketrans(
    "qwertyuiop[]asdfghjkl;'zxcvbnm,./QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>?",
    "йцукенгшщзхъфывапролджэячсмитьбю.ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ:",
)

# Кэш проверки «знает ли pymorphy это слово»
_known_word_cache: dict[str, bool] = {}


def _is_russian_word(word: str) -> bool:
    """
    Слово известно словарю pymorphy.

    Проверять POS на «UNKN» нельзя: pymorphy для неизвестных слов
    предсказывает часть речи по суффиксу (FakeDictionary) и «узнаёт»
    любую кириллическую абракадабру. Надёжный признак — первый анализатор
    в стеке разбора: DictionaryAnalyzer (словарь) против предсказателей.
    """
    if word not in _known_word_cache:
        parse = _morph.parse(word)[0]
        analyzer = parse.methods_stack[0][0] if parse.methods_stack else None
        _known_word_cache[word] = type(analyzer).__name__ == "DictionaryAnalyzer"
    return _known_word_cache[word]


def fix_keyboard_layout(text: str) -> str:
    """
    Перекодировка запросов, набранных в английской раскладке:
    «ktrnhbrf ghjdjlrf» -> «лектрика проводка».

    Применяется только если в тексте нет кириллицы и ВСЕ декодированные
    слова нашлись в словаре: настоящие английские слова и артикулы должны
    остаться нетронутыми. Требование «все слова» строгое намеренно:
    словарь содержит редкие слова и одиночные буквы, поэтому правило
    «хотя бы половина» пропускало мусор («high heels» -> «ршпр рууды»).
    """
    s = str(text)
    if re.search(r"[а-яёА-ЯЁ]", s):
        return s

    decoded = s.translate(_LAYOUT_EN_TO_RU)
    if decoded == s:
        return s  # символы раскладки не встретились (цифры, артикулы)

    words = normalize_text(decoded).split()
    if not words:
        return s

    known = sum(_is_russian_word(w) for w in words)
    return decoded if known == len(words) else s


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


def add_title_lemma_column(
    df_items: pd.DataFrame,
    cache_path: Path,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Добавление колонки item_title_lemma – лемматизированного заголовка.

    Нужна для признака title_coverage реранкера и для BM25-индекса
    только по заголовкам

    В кэше хранится таблица «уникальный заголовок -> леммы»
    """
    if cache_path.exists() and not force_recompute:
        print(f"Читаю кэш заголовков: {cache_path}")
        title_map = pd.read_parquet(cache_path)
    else:
        unique_titles = df_items["item_title_raw"].astype(str).unique()
        print(f"Лемматизирую {len(unique_titles):,} уникальных заголовков...")
        title_map = pd.DataFrame({"item_title_raw": unique_titles})
        title_map["item_title_lemma"] = [lemmatize_text(t) for t in unique_titles]

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        title_map.to_parquet(cache_path)
        print(f"Сохранено в {cache_path}")

    lookup = dict(zip(title_map["item_title_raw"], title_map["item_title_lemma"]))
    df_items = df_items.copy()
    df_items["item_title_lemma"] = df_items["item_title_raw"].astype(str).map(lookup)
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
