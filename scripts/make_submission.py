"""
Подготовка answer.csv для бенчмарка — финального файла сабмита.

Запуск:
    uv run python scripts/make_submission.py

Первый запуск может быть долгий: лемматизация корпуса 189K объявлений.
Результат кэшируется в data/processed/, повторный запуск короче.

Пайплайн (валидируется в ноутбуках 03–04):
BM25 по полному тексту + BM25 по заголовкам + char n-gram по заголовкам
-> union кандидатов -> признаки -> линейный реранк -> top-50.
"""

from pathlib import Path

import pandas as pd

from avito_candidate_gen.data import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    load_benchmark_items,
    load_benchmark_queries,
)
from avito_candidate_gen.pipeline import generate_candidates
from avito_candidate_gen.preprocessing import (
    add_title_lemma_column,
    get_or_build_processed_items,
)

ROOT = Path(__file__).resolve().parents[1]
ANSWER_PATH = ROOT / "outputs" / "answer.csv"

ITEMS_CACHE = DATA_PROCESSED_DIR / "benchmark_items_processed.parquet"
TITLE_CACHE = DATA_PROCESSED_DIR / "benchmark_title_lemmas.parquet"


def load_popularity() -> pd.Series:
    """
    Популярность объявления = сколько раз его выбрали в train.

    Читаем только колонку item_id: train.parquet с текстами тяжёлый,
    остальное здесь не нужно
    """
    train_ids = pd.read_parquet(DATA_RAW_DIR / "train.parquet", columns=["item_id"])
    return train_ids.groupby("item_id").size()


def build_answer(
    df_queries: pd.DataFrame, predictions: dict[str, list[str]]
) -> pd.DataFrame:
    """
    Таблица для answer.csv: query_id + item_id через пробел.
    Порядок строк — как в benchmark_queries
    """
    query_ids = df_queries["query_id"].astype(str)

    missing = [qid for qid in query_ids if qid not in predictions]
    assert not missing, f"Нет предсказаний для {len(missing)} запросов, например {missing[:3]}"

    return pd.DataFrame(
        {
            "query_id": query_ids,
            "answer": [" ".join(predictions[qid]) for qid in query_ids],
        }
    )


def validate_answer(
    answer_path: Path, df_queries: pd.DataFrame, df_items: pd.DataFrame
) -> None:
    """
    Жёсткая проверка формата по ТЗ. Любое нарушение вызовет AssertionError

    Требования: 
        - ровно 2 колонки (query_id, answer); 
        - строка на каждый query_id бенчмарка без дублей; 
        - 1..50 item_id на строку, без повторов внутри; 
        - каждый id — 16 символов 0-9a-f и существует в benchmark_items
    """
    df = pd.read_csv(answer_path, dtype=str, keep_default_na=False)
    corpus_ids = set(df_items["item_id"].astype(str))
    expected_qids = set(df_queries["query_id"].astype(str))

    assert list(df.columns) == ["query_id", "answer"], (
        f"Нужны ровно колонки ['query_id', 'answer'], найдено {list(df.columns)}"
    )
    assert len(df) == len(expected_qids), f"Строк {len(df)}, ожидается {len(expected_qids)}"
    assert df["query_id"].is_unique, "Дубли query_id"
    assert set(df["query_id"]) == expected_qids, "Набор query_id не совпадает с benchmark_queries"
    assert df["query_id"].str.len().eq(16).all(), "query_id должен быть ровно 16 символов"

    ids_per_query = df["answer"].str.split()
    counts = ids_per_query.str.len() # type: ignore
    assert counts.between(1, 50).all(), "В каждой строке должно быть от 1 до 50 item_id"
    assert not ids_per_query.apply(lambda ids: len(set(ids)) != len(ids)).any(), (
        "Повторы item_id внутри строки"
    )

    all_ids = ids_per_query.explode()
    assert all_ids.str.fullmatch(r"[0-9a-f]{16}").all(), "item_id должен быть 16 символов 0-9a-f"
    assert all_ids.isin(corpus_ids).all(), "Встречены item_id, которых нет в benchmark_items"

    print(f"Валидация OK: {len(df):,} строк, {counts.min()}–{counts.max()} id на строку")


def main() -> None:
    df_queries = load_benchmark_queries()
    df_items = load_benchmark_items()
    print(f"Запросов: {len(df_queries):,} | Корпус: {len(df_items):,}")

    # Предобработка корпуса: лемматизированный текст для BM25
    # + леммы заголовков для признака title_coverage
    corpus = get_or_build_processed_items(df_items, ITEMS_CACHE)
    corpus = add_title_lemma_column(corpus, TITLE_CACHE)

    popularity = load_popularity()

    predictions = generate_candidates(df_queries, corpus, popularity=popularity)

    answer = build_answer(df_queries, predictions)
    ANSWER_PATH.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(ANSWER_PATH, index=False, encoding="utf-8")
    print(f"Сохранено: {ANSWER_PATH}")

    # Финальная проверка
    validate_answer(ANSWER_PATH, df_queries, df_items)


if __name__ == "__main__":
    main()
