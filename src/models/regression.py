"""Pipelines de regressão para predizer a duração do atraso (em minutos).

Dois algoritmos comparados:

1. **Ridge Regression** (baseline linear regularizado).
2. **Gradient Boosting** (LightGBM).

A variável alvo (`ARRIVAL_DELAY`) tem cauda longa positiva. Suportamos
transformação log1p (com sinal preservado) via `log_target=True`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from .classification import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    make_preprocessor,
)

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _signed_expm1(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.expm1(np.abs(x))


def _wrap_log_target(regressor) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=regressor,
        func=_signed_log1p,
        inverse_func=_signed_expm1,
    )


def build_ridge_pipeline(
    random_state: int = 42, alpha: float = 1.0, log_target: bool = True
) -> Pipeline:
    ridge = Ridge(alpha=alpha, random_state=random_state)
    inner = _wrap_log_target(ridge) if log_target else ridge
    return Pipeline(
        [
            ("preprocess", make_preprocessor(high_cardinality_strategy="onehot")),
            ("reg", inner),
        ]
    )


def build_lightgbm_regressor(
    random_state: int = 42,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    log_target: bool = True,
) -> Pipeline:
    if not _HAS_LGBM:
        raise ImportError("lightgbm não instalado. `pip install lightgbm`.")
    base = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=63,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    inner = _wrap_log_target(base) if log_target else base
    return Pipeline(
        [
            ("preprocess", make_preprocessor(high_cardinality_strategy="onehot")),
            ("reg", inner),
        ]
    )


def build_feature_matrix(
    df: pd.DataFrame,
    numeric: list[str] = NUMERIC_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
) -> pd.DataFrame:
    return df[numeric + categorical].copy()
