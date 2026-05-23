"""Limpeza e tratamento de valores ausentes.

Decisões justificadas no README e nos notebooks de EDA. Resumo:

- Voos cancelados/desviados não têm `ARRIVAL_DELAY` → removidos para tarefas
  de predição de atraso (o problema deles é cancelamento, não atraso).
- Colunas `*_DELAY` granulares (AIRLINE_DELAY, WEATHER_DELAY, etc.) só
  existem para voos com ARRIVAL_DELAY >= 15 min → `fillna(0)` é correto.
- `TAIL_NUMBER` faltando (~0.25%) → preenchido com "UNKNOWN".
"""

from __future__ import annotations

import pandas as pd

from . import config


def drop_cancelled_and_diverted(df: pd.DataFrame) -> pd.DataFrame:
    """Remove voos cancelados ou desviados (sem ARRIVAL_DELAY válido).

    São ~1.5% dos voos e a tarefa deles é outra (predição de cancelamento).
    """
    mask = (df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)
    return df.loc[mask].copy()


def fill_delay_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche colunas de motivo de atraso com 0 quando ausentes.

    Essas colunas só são preenchidas quando o atraso na chegada >= 15 min;
    portanto NaN aqui significa "sem contribuição de atraso", ou seja, 0.
    """
    df = df.copy()
    for col in config.DELAY_REASON_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def fill_tail_number(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TAIL_NUMBER" in df.columns:
        df["TAIL_NUMBER"] = df["TAIL_NUMBER"].fillna("UNKNOWN")
    return df


def drop_missing_arrival(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas restantes sem ARRIVAL_DELAY (caso ainda existam)."""
    return df.dropna(subset=["ARRIVAL_DELAY"]).copy()


def add_target_classification(
    df: pd.DataFrame,
    threshold: int = config.DELAY_THRESHOLD_MIN,
) -> pd.DataFrame:
    """Cria a variável alvo `IS_DELAYED` (1 se ARRIVAL_DELAY >= threshold)."""
    df = df.copy()
    df["IS_DELAYED"] = (df["ARRIVAL_DELAY"] >= threshold).astype("int8")
    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de limpeza para tarefas de predição de atraso."""
    df = drop_cancelled_and_diverted(df)
    df = drop_missing_arrival(df)
    df = fill_delay_reasons(df)
    df = fill_tail_number(df)
    df = add_target_classification(df)
    return df


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela com contagem e % de valores ausentes por coluna."""
    n = len(df)
    out = pd.DataFrame(
        {
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(3),
            "dtype": df.dtypes.astype(str),
        }
    )
    out = out.sort_values("missing", ascending=False)
    out.attrs["total_rows"] = n
    return out
