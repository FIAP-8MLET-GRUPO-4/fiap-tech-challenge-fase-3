"""Engenharia de features.

Variáveis derivadas para enriquecer o modelo:

- Tempo: hora agendada de partida, período do dia, estação do ano.
- Calendário: feriados federais dos EUA em 2015, fim de semana.
- Rota: par (origem, destino).
- Histórico: estatísticas agregadas por aeroporto/companhia (média de
  atraso em treino — calculadas SEM vazamento da partição de teste).
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

US_FEDERAL_HOLIDAYS_2015 = {
    date(2015, 1, 1),   # New Year's Day
    date(2015, 1, 19),  # MLK Day
    date(2015, 2, 16),  # Presidents Day
    date(2015, 5, 25),  # Memorial Day
    date(2015, 7, 3),   # Independence Day (observed)
    date(2015, 7, 4),   # Independence Day
    date(2015, 9, 7),   # Labor Day
    date(2015, 10, 12), # Columbus Day
    date(2015, 11, 11), # Veterans Day
    date(2015, 11, 26), # Thanksgiving
    date(2015, 12, 24), # Christmas Eve (heavy travel)
    date(2015, 12, 25), # Christmas
    date(2015, 12, 31), # New Year's Eve (heavy travel)
}


def hhmm_to_hour(series: pd.Series) -> pd.Series:
    """Converte HHMM (int) → hora (0–23). Trata NaN."""
    hours = (series.fillna(0).astype(int) // 100) % 24
    return hours.astype("int8")


def hour_to_period(hour: pd.Series) -> pd.Series:
    """Bucketiza hora em período do dia."""
    bins = [-1, 5, 11, 17, 21, 24]
    labels = ["madrugada", "manha", "tarde", "noite", "madrugada"]
    out = pd.cut(hour, bins=bins, labels=labels[:-1], ordered=False)
    return out.astype("category")


def month_to_season(month: pd.Series) -> pd.Series:
    """Estação do ano no hemisfério norte (EUA)."""
    mapping = {
        12: "inverno", 1: "inverno", 2: "inverno",
        3: "primavera", 4: "primavera", 5: "primavera",
        6: "verao", 7: "verao", 8: "verao",
        9: "outono", 10: "outono", 11: "outono",
    }
    return month.map(mapping).astype("category")


def is_us_holiday(year: pd.Series, month: pd.Series, day: pd.Series) -> pd.Series:
    """1 se a data é feriado federal dos EUA em 2015, 0 caso contrário."""
    dates = pd.to_datetime(
        {"year": year, "month": month, "day": day},
        errors="coerce",
    ).dt.date
    holidays = US_FEDERAL_HOLIDAYS_2015
    return dates.map(lambda d: 1 if d in holidays else 0).astype("int8")


def is_weekend(day_of_week: pd.Series) -> pd.Series:
    """1 se sábado (6) ou domingo (7)."""
    return day_of_week.isin([6, 7]).astype("int8")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de tempo derivadas de SCHEDULED_DEPARTURE / DATE."""
    df = df.copy()
    df["DEP_HOUR"] = hhmm_to_hour(df["SCHEDULED_DEPARTURE"])
    df["DEP_PERIOD"] = hour_to_period(df["DEP_HOUR"])
    df["SEASON"] = month_to_season(df["MONTH"])
    df["IS_WEEKEND"] = is_weekend(df["DAY_OF_WEEK"])
    df["IS_HOLIDAY"] = is_us_holiday(df["YEAR"], df["MONTH"], df["DAY"])
    return df


def add_route_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a coluna ROUTE = 'ORIGIN-DESTINATION'."""
    df = df.copy()
    df["ROUTE"] = (
        df["ORIGIN_AIRPORT"].astype(str) + "-" + df["DESTINATION_AIRPORT"].astype(str)
    ).astype("category")
    return df


def build_target_encoding(
    train: pd.DataFrame,
    cols: Iterable[str],
    target: str = "IS_DELAYED",
    min_count: int = 30,
    smoothing: float = 10.0,
) -> dict:
    """Calcula target encoding suavizado a partir do conjunto de TREINO.

    A taxa global é usada como prior para grupos com poucas observações,
    evitando overfit em categorias raras.

    Retorna um dict {coluna: pd.Series indexada pela categoria}.
    """
    global_mean = train[target].mean()
    encodings = {}
    for col in cols:
        agg = train.groupby(col, observed=True)[target].agg(["mean", "count"])
        smoothed = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
            agg["count"] + smoothing
        )
        smoothed = smoothed.where(agg["count"] >= min_count, other=global_mean)
        encodings[col] = smoothed
    encodings["__global_mean__"] = global_mean
    return encodings


def apply_target_encoding(
    df: pd.DataFrame, encodings: dict, cols: Iterable[str]
) -> pd.DataFrame:
    """Aplica encoders calculados via `build_target_encoding`."""
    df = df.copy()
    global_mean = encodings["__global_mean__"]
    for col in cols:
        enc = encodings[col]
        df[f"{col}_DELAY_RATE"] = (
            df[col].map(enc).fillna(global_mean).astype("float32")
        )
    return df


def feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas as transformações de feature engineering."""
    df = add_time_features(df)
    df = add_route_feature(df)
    return df
