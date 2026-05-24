"""Visualizações reutilizáveis para EDA e relatórios."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PALETTE = "viridis"
sns.set_theme(style="whitegrid")


def plot_delay_distribution(
    df: pd.DataFrame,
    col: str = "ARRIVAL_DELAY",
    clip: tuple[int, int] = (-60, 180),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    data = df[col].clip(*clip)
    sns.histplot(data, bins=80, ax=ax, color="steelblue", edgecolor="white")
    ax.axvline(15, color="red", linestyle="--", label="Limiar 15 min")
    ax.set_title(f"Distribuição de {col} (clipped {clip[0]}..{clip[1]} min)")
    ax.set_xlabel("Minutos")
    ax.legend()
    return ax


def plot_delay_rate_by(
    df: pd.DataFrame,
    by: str,
    target: str = "IS_DELAYED",
    top_n: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    agg = (
        df.groupby(by, observed=True)[target]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    if top_n:
        agg = agg.head(top_n)
    sns.barplot(x=agg.index, y=agg["mean"] * 100, hue=agg.index, ax=ax, palette=PALETTE, legend=False)
    ax.set_title(f"Taxa de atraso (%) por {by}")
    ax.set_ylabel("% voos atrasados")
    ax.set_xlabel(by)
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_delay_heatmap(
    df: pd.DataFrame,
    row: str = "DAY_OF_WEEK",
    col: str = "DEP_HOUR",
    target: str = "IS_DELAYED",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    pivot = (
        df.groupby([row, col], observed=True)[target]
        .mean()
        .unstack(col)
        * 100
    )
    sns.heatmap(pivot, cmap="rocket_r", annot=False, ax=ax, cbar_kws={"label": "% atrasos"})
    ax.set_title(f"Taxa de atraso por {row} × {col}")
    return ax


def plot_top_routes(
    df: pd.DataFrame, top_n: int = 20, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    routes = (
        df.assign(
            ROUTE=df["ORIGIN_AIRPORT"].astype(str) + "→" + df["DESTINATION_AIRPORT"].astype(str)
        )
        .groupby("ROUTE", observed=True)
        .size()
        .sort_values(ascending=False)
        .head(top_n)
    )
    sns.barplot(x=routes.values, y=routes.index, hue=routes.index, ax=ax, palette=PALETTE, legend=False)
    ax.set_title(f"Top {top_n} rotas por volume de voos")
    ax.set_xlabel("Número de voos")
    return ax


def plot_correlation_matrix(
    df: pd.DataFrame, cols: list[str], ax: Optional[plt.Axes] = None
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de correlação")
    return ax
