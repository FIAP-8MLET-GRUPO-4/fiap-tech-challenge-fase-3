"""Carregamento eficiente do dataset de voos.

`flights.csv` tem ~5.8M linhas. Usamos dtypes explícitos para reduzir uso
de memória de ~3GB para ~700MB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config

FLIGHTS_DTYPES = {
    "YEAR": "int16",
    "MONTH": "int8",
    "DAY": "int8",
    "DAY_OF_WEEK": "int8",
    "AIRLINE": "category",
    "FLIGHT_NUMBER": "int32",
    "TAIL_NUMBER": "string",
    "ORIGIN_AIRPORT": "category",
    "DESTINATION_AIRPORT": "category",
    "SCHEDULED_DEPARTURE": "int16",
    "DEPARTURE_TIME": "float32",
    "DEPARTURE_DELAY": "float32",
    "TAXI_OUT": "float32",
    "WHEELS_OFF": "float32",
    "SCHEDULED_TIME": "float32",
    "ELAPSED_TIME": "float32",
    "AIR_TIME": "float32",
    "DISTANCE": "int32",
    "WHEELS_ON": "float32",
    "TAXI_IN": "float32",
    "SCHEDULED_ARRIVAL": "int16",
    "ARRIVAL_TIME": "float32",
    "ARRIVAL_DELAY": "float32",
    "DIVERTED": "int8",
    "CANCELLED": "int8",
    "CANCELLATION_REASON": "category",
    "AIR_SYSTEM_DELAY": "float32",
    "SECURITY_DELAY": "float32",
    "AIRLINE_DELAY": "float32",
    "LATE_AIRCRAFT_DELAY": "float32",
    "WEATHER_DELAY": "float32",
}


def load_flights(
    path: Optional[Path] = None,
    nrows: Optional[int] = None,
    sample_size: Optional[int] = None,
    random_state: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Carrega flights.csv com dtypes otimizados.

    Args:
        path: caminho do CSV (default `config.FLIGHTS_CSV`).
        nrows: lê apenas as N primeiras linhas (útil para debug).
        sample_size: após ler tudo, amostra aleatoriamente N linhas
            (passe 0 ou None para usar o dataset completo).
        random_state: seed para amostragem.
    """
    path = Path(path or config.FLIGHTS_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"flights.csv não encontrado em {path}. "
            "Rode `python scripts/download_data.py` ou ver `data/README.md`."
        )

    df = pd.read_csv(path, dtype=FLIGHTS_DTYPES, nrows=nrows, low_memory=False)

    if sample_size and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    return df


def load_airlines(path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(path or config.AIRLINES_CSV)
    return pd.read_csv(path)


def load_airports(path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(path or config.AIRPORTS_CSV)
    return pd.read_csv(path)


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Uso de memória do DataFrame em MB."""
    return df.memory_usage(deep=True).sum() / (1024**2)


def describe_dataset(df: pd.DataFrame) -> dict:
    """Resumo rápido do DataFrame (shape, missing, memory)."""
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "memory_mb": round(memory_usage_mb(df), 2),
        "missing_total": int(df.isna().sum().sum()),
        "missing_pct_per_col": (df.isna().mean() * 100).round(2).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
