from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DECIMAL_COLUMNS = ["item_price", "item_latitude", "item_longitude"]


def _resolve_path(path: Path | None, default: Path) -> Path:
    """
    Загрузка пути к файлу, если он передан, иначе default
    """

    path = path or default

    if not path.exists():
        raise FileNotFoundError(
            f"Путь {path} не найден. Проверьте, что файл лежит в data/raw/, или передайте путь явно"
        )

    return path


def _fix_decimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parquet-файлы хранят некоторые числовые поля как decimal128,
    из-за чего pandas читает их как object/Decimal вместо float.
    Приводим такие колонки к float
    """
    cols_present = [c for c in DECIMAL_COLUMNS if c in df.columns]
    df[cols_present] = df[cols_present].astype(float)
    return df


# Загрузка тренировочных данных
def load_train(path: Path | None = None) -> pd.DataFrame:
    path = _resolve_path(path=path, default=DATA_RAW_DIR / "train.parquet")
    df = pd.read_parquet(path)
    return _fix_decimal_columns(df)


# Данные для подготовки ответа
def load_benchmark_queries(path: Path | None = None) -> pd.DataFrame:
    path = _resolve_path(path=path, default=DATA_RAW_DIR / "benchmark_queries.parquet")
    return pd.read_parquet(path)


def load_benchmark_items(path: Path | None = None) -> pd.DataFrame:
    path = _resolve_path(path=path, default=DATA_RAW_DIR / "benchmark_items.parquet")
    df = pd.read_parquet(path)
    return _fix_decimal_columns(df)


# Загрузка объявлений с уже посчитанным item_text_processed
def load_processed_train(path: Path | None = None) -> pd.DataFrame:
    path = _resolve_path(
        path, DATA_RAW_DIR.parent / "processed" / "train_processed.parquet"
    )
    return pd.read_parquet(path)


def load_processed_items(path: Path | None = None) -> pd.DataFrame:
    path = _resolve_path(
        path, DATA_RAW_DIR.parent / "processed" / "benchmark_items_processed.parquet"
    )
    return pd.read_parquet(path)


def train_val_split(
    df_train: pd.DataFrame, val_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделение данных на обучающую и валидационную выборки
    по уникальным значениям search_query, дабы избежать утечек
    """

    unique_queries = np.asarray(df_train["search_query"].unique())
    _, val_q = train_test_split(
        unique_queries, test_size=val_size, random_state=random_state
    )

    val_q = set(val_q)
    is_val = df_train["search_query"].isin(val_q)

    return df_train[~is_val].reset_index(drop=True), df_train[is_val].reset_index(
        drop=True
    )
