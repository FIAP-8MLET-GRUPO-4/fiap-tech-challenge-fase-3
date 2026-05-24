"""Pipelines de classificação binária para predição de atraso.

Dois algoritmos comparados:

1. **Regressão Logística** com `class_weight='balanced'` (baseline linear).
2. **Gradient Boosting** (LightGBM por velocidade; XGBoost via parâmetro).

Ambos são empacotados num `Pipeline` do scikit-learn:
ColumnTransformer → modelo, garantindo que pré-processamento e modelo
fiquem juntos para serialização e inferência consistente.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


NUMERIC_FEATURES = [
    "DEP_HOUR",
    "MONTH",
    "DAY_OF_WEEK",
    "DISTANCE",
    "SCHEDULED_TIME",
    "IS_WEEKEND",
    "IS_HOLIDAY",
]

CATEGORICAL_FEATURES = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "DEP_PERIOD",
    "SEASON",
]


def make_preprocessor(
    numeric_features: list[str] = NUMERIC_FEATURES,
    categorical_features: list[str] = CATEGORICAL_FEATURES,
    high_cardinality_strategy: Literal["onehot", "passthrough"] = "onehot",
) -> ColumnTransformer:
    """Cria o pré-processador de colunas.

    Modelos de árvore (LightGBM/XGBoost) podem consumir categóricas
    diretamente — nesse caso, prefira `high_cardinality_strategy='passthrough'`
    com `categorical_feature='auto'` no LGBM.
    """
    cat_transformer = (
        OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=50)
        if high_cardinality_strategy == "onehot"
        else "passthrough"
    )
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_logistic_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(high_cardinality_strategy="onehot")),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    solver="saga",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_lightgbm_pipeline(
    random_state: int = 42, n_estimators: int = 500, learning_rate: float = 0.05
) -> Pipeline:
    if not _HAS_LGBM:
        raise ImportError("lightgbm não instalado. `pip install lightgbm`.")
    return Pipeline(
        [
            ("preprocess", make_preprocessor(high_cardinality_strategy="onehot")),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=63,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(
    random_state: int = 42, n_estimators: int = 500, learning_rate: float = 0.05
) -> Pipeline:
    if not _HAS_XGB:
        raise ImportError("xgboost não instalado. `pip install xgboost`.")
    return Pipeline(
        [
            ("preprocess", make_preprocessor(high_cardinality_strategy="onehot")),
            (
                "clf",
                XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=6,
                    tree_method="hist",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def predict_with_threshold(
    pipeline: Pipeline, X: pd.DataFrame, threshold: float = 0.5
) -> np.ndarray:
    """Predição binária com threshold customizado."""
    proba = pipeline.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int)
