"""Pipeline end-to-end: carrega → limpa → feature engineering → treina → avalia.

Roda os principais experimentos de forma headless (sem notebook), salva
métricas em CSV e modelos serializados em `models_artifacts/`.

Uso:
    python scripts/run_pipeline.py --task classification
    python scripts/run_pipeline.py --task regression
    python scripts/run_pipeline.py --task clustering
    python scripts/run_pipeline.py --task all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import load_flights
from src.evaluation import (
    classification_metrics,
    metrics_table,
    regression_metrics,
)
from src.features import feature_pipeline
from src.models.classification import (
    build_lightgbm_pipeline as build_clf_lgbm,
    build_logistic_pipeline,
)
from src.models.clustering import (
    build_airport_features,
    cluster_airports,
    cluster_profiles,
    find_optimal_k,
)
from src.models.regression import (
    build_lightgbm_regressor,
    build_ridge_pipeline,
    build_feature_matrix,
)
from src.preprocessing import clean_pipeline


def prepare_data() -> pd.DataFrame:
    print(f"→ Carregando {config.FLIGHTS_CSV} (sample_size={config.SAMPLE_SIZE})…")
    t0 = time.time()
    df = load_flights(sample_size=config.SAMPLE_SIZE)
    print(f"  shape={df.shape}, {time.time() - t0:.1f}s")

    print("→ Limpando…")
    df = clean_pipeline(df)

    print("→ Feature engineering…")
    df = feature_pipeline(df)
    return df


def run_classification(df: pd.DataFrame) -> dict:
    print("\n=== Classificação ===")
    X = build_feature_matrix(df)
    y = df["IS_DELAYED"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
    )

    pipelines = {
        "logistic_regression": build_logistic_pipeline(config.RANDOM_SEED),
        "lightgbm": build_clf_lgbm(config.RANDOM_SEED),
    }

    metrics = {}
    for name, pipe in pipelines.items():
        print(f"→ Treinando {name}…")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics[name] = {**classification_metrics(y_test, y_pred, y_proba), "train_s": round(elapsed, 1)}

        out_path = config.MODELS_DIR / f"clf_{name}.joblib"
        joblib.dump(pipe, out_path)
        print(f"  saved → {out_path}")

    table = metrics_table(metrics)
    print("\n", table)
    table.to_csv(config.REPORTS_DIR / "metrics_classification.csv")
    return metrics


def run_regression(df: pd.DataFrame) -> dict:
    print("\n=== Regressão ===")
    X = build_feature_matrix(df)
    y = df["ARRIVAL_DELAY"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED
    )

    pipelines = {
        "ridge": build_ridge_pipeline(config.RANDOM_SEED),
        "lightgbm": build_lightgbm_regressor(config.RANDOM_SEED),
    }

    metrics = {}
    for name, pipe in pipelines.items():
        print(f"→ Treinando {name}…")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = pipe.predict(X_test)
        metrics[name] = {**regression_metrics(y_test, y_pred), "train_s": round(elapsed, 1)}

        out_path = config.MODELS_DIR / f"reg_{name}.joblib"
        joblib.dump(pipe, out_path)
        print(f"  saved → {out_path}")

    table = metrics_table(metrics)
    print("\n", table)
    table.to_csv(config.REPORTS_DIR / "metrics_regression.csv")
    return metrics


def run_clustering(df: pd.DataFrame) -> dict:
    print("\n=== Clusterização de aeroportos ===")
    feats = build_airport_features(df, min_flights=5000)
    print(f"  aeroportos válidos: {len(feats)}")

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(feats.values)

    elbow = find_optimal_k(X, k_range=range(2, 9))
    elbow.to_csv(config.REPORTS_DIR / "clustering_elbow.csv", index=False)
    print(elbow)

    k = int(elbow.loc[elbow["silhouette"].idxmax(), "k"])
    print(f"→ Melhor K por silhouette: {k}")

    clustered, km, scaler = cluster_airports(feats, n_clusters=k, random_state=config.RANDOM_SEED)
    clustered.to_csv(config.REPORTS_DIR / "airport_clusters.csv")
    profiles = cluster_profiles(clustered)
    profiles.to_csv(config.REPORTS_DIR / "cluster_profiles.csv")
    print("\nPerfis médios por cluster:\n", profiles)

    joblib.dump({"kmeans": km, "scaler": scaler}, config.MODELS_DIR / "airport_clustering.joblib")
    return {"k": k, "n_airports": int(len(clustered))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "clustering", "all"],
        default="all",
    )
    args = parser.parse_args()

    df = prepare_data()

    summary = {}
    if args.task in {"classification", "all"}:
        summary["classification"] = run_classification(df)
    if args.task in {"regression", "all"}:
        summary["regression"] = run_regression(df)
    if args.task in {"clustering", "all"}:
        summary["clustering"] = run_clustering(df)

    with open(config.REPORTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✓ Pipeline concluído. Resumo em {config.REPORTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
