"""Outils communs aux modèles : chargement des features, découpage train/test, indicateurs."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from credit_risk_scoring.config import ID, SEED, TARGET
from credit_risk_scoring.features import FEATURES_PATH

TEST_SIZE = 0.2


def load_features() -> pd.DataFrame:
    """Charge features.parquet avec des types simples (float / object) compris par toutes les bibliothèques."""
    df = pd.read_parquet(FEATURES_PATH)
    for col in df.columns:
        if col in (ID, TARGET):
            continue
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        else:
            df[col] = df[col].astype("float64")
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Découpage 80 / 20 stratifié : même taux de défaut dans le train et le test."""
    X = df.drop(columns=[ID, TARGET])
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov : écart maximal entre les distributions de score des bons et des mauvais."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_score) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_score)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks_statistic(y_true, y_score)}
