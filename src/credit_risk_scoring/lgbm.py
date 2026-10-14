"""Modèle LightGBM (gradient boosting d'arbres de décision).

Contrairement à la scorecard, LightGBM utilise directement les 170 variables brutes :
il gère seul les valeurs manquantes, les relations non linéaires et les interactions.
Le nombre d'arbres est choisi par validation croisée (5 plis) avec arrêt anticipé.
Si l'optimisation Optuna a été lancée (tune.py), ses meilleurs réglages sont utilisés.

Usage :
    uv run python -m credit_risk_scoring.lgbm
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from credit_risk_scoring.config import ROOT, SEED
from credit_risk_scoring.modeling import evaluate, load_features, split

MODEL_PATH = ROOT / "models" / "lgbm.joblib"
TUNED_PARAMS_PATH = ROOT / "reports" / "lgbm_best_params.json"

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 31,            # complexité de chaque arbre
    "min_child_samples": 100,    # au moins 100 demandes par feuille : évite d'apprendre des cas isolés
    "feature_fraction": 0.7,     # chaque arbre ne voit que 70 % des variables
    "bagging_fraction": 0.8,     # … et 80 % des demandes
    "bagging_freq": 1,
    "lambda_l2": 1.0,            # régularisation
    "seed": SEED,
    "verbose": -1,
}


def tuned_params() -> dict:
    """Meilleurs réglages trouvés par Optuna (vide si l'optimisation n'a pas été lancée)."""
    if not TUNED_PARAMS_PATH.exists():
        return {}
    return json.loads(TUNED_PARAMS_PATH.read_text(encoding="utf-8"))["params"]


class LGBMModel:
    def __init__(self, params: dict | None = None, max_rounds: int = 5000, nfold: int = 5):
        self.params = {**PARAMS, **(params or {})}
        self.max_rounds = max_rounds
        self.nfold = nfold

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Variables texte → type « category », avec les catégories vues à l'entraînement."""
        X = X.copy()
        for col, cats in self.categories.items():
            X[col] = pd.Categorical(X[col], categories=cats)
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMModel":
        self.categories = {
            c: sorted(X[c].dropna().unique()) for c in X.columns if not is_numeric_dtype(X[c])
        }
        self.features = list(X.columns)
        data = lgb.Dataset(self._prepare(X), label=y, free_raw_data=False)

        cv = lgb.cv(
            self.params, data, num_boost_round=self.max_rounds, nfold=self.nfold, stratified=True,
            seed=SEED, callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        self.n_rounds = len(cv["valid auc-mean"])
        self.cv_auc = cv["valid auc-mean"][-1]
        self.cv_auc_std = cv["valid auc-stdv"][-1]

        self.booster = lgb.train(self.params, data, num_boost_round=self.n_rounds)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilité de défaut (PD) de chaque demande."""
        return self.booster.predict(self._prepare(X[self.features]))

    def feature_importance(self) -> pd.Series:
        """Gain total apporté par chaque variable dans les arbres."""
        gain = self.booster.feature_importance(importance_type="gain")
        return pd.Series(gain / gain.sum(), index=self.features).sort_values(ascending=False)


def main() -> None:
    df = load_features()
    X_train, X_test, y_train, y_test = split(df)

    params = tuned_params()
    print("Réglages optimisés par Optuna" if params else "Réglages par défaut")
    model = LGBMModel(params=params).fit(X_train, y_train)
    print(f"Validation croisée : AUC {model.cv_auc:.3f} ± {model.cv_auc_std:.3f} ({model.n_rounds} arbres)")
    for name, X, y in [("Train", X_train, y_train), ("Test ", X_test, y_test)]:
        m = evaluate(y, model.predict_proba(X))
        print(f"{name} : AUC {m['AUC']:.3f} · Gini {m['Gini']:.3f} · KS {m['KS']:.3f}")
    print("Variables les plus importantes :")
    print(model.feature_importance().head(10).round(3).to_string())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modèle enregistré : {MODEL_PATH}")


if __name__ == "__main__":
    # Ré-import depuis le package pour que le modèle enregistré soit rechargeable ailleurs
    from credit_risk_scoring.lgbm import main as run

    run()
