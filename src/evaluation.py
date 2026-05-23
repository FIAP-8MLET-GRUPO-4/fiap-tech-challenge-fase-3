"""Métricas e visualizações de avaliação de modelos."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# --------------------------------------------------------------------------- #
# Classificação                                                                #
# --------------------------------------------------------------------------- #
def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
        out["pr_auc"] = average_precision_score(y_true, y_proba)
    return out


def plot_confusion_matrix(
    y_true, y_pred, ax: Optional[plt.Axes] = None, title: str = "Matriz de confusão"
):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No atraso", "Atrasou"],
        yticklabels=["No atraso", "Atrasou"],
        ax=ax,
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)
    return ax


def plot_roc_curves(results: dict, ax: Optional[plt.Axes] = None):
    """`results` = {nome_modelo: (y_true, y_proba)}."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curvas ROC")
    ax.legend(loc="lower right")
    return ax


def print_classification_report(y_true, y_pred, title: str = ""):
    if title:
        print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, target_names=["No atraso", "Atrasou"]))


# --------------------------------------------------------------------------- #
# Regressão                                                                    #
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def plot_residuals(y_true, y_pred, ax: Optional[plt.Axes] = None, title: str = "Resíduos"):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.2, s=8)
    ax.axhline(0, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("Predição (min)")
    ax.set_ylabel("Resíduo (real - predito, min)")
    ax.set_title(title)
    return ax


# --------------------------------------------------------------------------- #
# Comparação                                                                   #
# --------------------------------------------------------------------------- #
def metrics_table(results: dict) -> pd.DataFrame:
    """Recebe {modelo: dict_metricas} e retorna DataFrame ordenado."""
    return pd.DataFrame(results).T.round(4)
