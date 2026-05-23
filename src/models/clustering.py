"""Clusterização e redução de dimensionalidade.

Agrupamos **aeroportos** (não voos) por perfil operacional/atraso e
projetamos com PCA para visualização 2D.

Features por aeroporto (agregadas a partir do dataset de voos):

- Volume de voos (origem + destino).
- Taxa de atraso na partida e na chegada.
- Atraso médio e mediano.
- Distância média dos voos atendidos.
- Diversidade de companhias e rotas.
- Distribuição de motivos de atraso (clima, NAS, companhia, etc.).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def build_airport_features(flights: pd.DataFrame, min_flights: int = 5_000) -> pd.DataFrame:
    """Constrói uma linha por aeroporto, agregando o ponto de vista de ORIGIN.

    Filtra aeroportos com menos de `min_flights` voos para evitar ruído.
    """
    df = flights.copy()

    if "IS_DELAYED" not in df.columns:
        df["IS_DELAYED"] = (df["ARRIVAL_DELAY"] >= 15).astype("int8")
    df["DEPARTURE_DELAY_FILLED"] = df["DEPARTURE_DELAY"].fillna(0)

    grp = df.groupby("ORIGIN_AIRPORT", observed=True)

    feats = pd.DataFrame({
        "n_flights": grp.size(),
        "delay_rate": grp["IS_DELAYED"].mean(),
        "mean_arrival_delay": grp["ARRIVAL_DELAY"].mean(),
        "median_arrival_delay": grp["ARRIVAL_DELAY"].median(),
        "mean_departure_delay": grp["DEPARTURE_DELAY_FILLED"].mean(),
        "mean_distance": grp["DISTANCE"].mean(),
        "n_airlines": grp["AIRLINE"].nunique(),
        "n_destinations": grp["DESTINATION_AIRPORT"].nunique(),
        "share_weather_delay": grp["WEATHER_DELAY"].sum() / grp["ARRIVAL_DELAY"].sum().clip(lower=1),
        "share_nas_delay": grp["AIR_SYSTEM_DELAY"].sum() / grp["ARRIVAL_DELAY"].sum().clip(lower=1),
        "share_airline_delay": grp["AIRLINE_DELAY"].sum() / grp["ARRIVAL_DELAY"].sum().clip(lower=1),
    })

    feats = feats[feats["n_flights"] >= min_flights].copy()
    return feats


def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """Avalia K-Means para diferentes K usando inércia e silhouette."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else np.nan
        results.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": sil,
        })
    return pd.DataFrame(results)


def cluster_airports(
    features: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Roda K-Means e retorna o DataFrame com label de cluster."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)

    out = features.copy()
    out["cluster"] = labels
    return out, km, scaler


def reduce_pca(
    X: np.ndarray, n_components: int = 2, random_state: int = 42
) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=random_state)
    X_red = pca.fit_transform(X)
    return X_red, pca


def cluster_profiles(clustered: pd.DataFrame) -> pd.DataFrame:
    """Média de cada feature por cluster — facilita a interpretação."""
    return (
        clustered.groupby("cluster")
        .mean(numeric_only=True)
        .round(3)
        .sort_index()
    )
