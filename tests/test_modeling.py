import numpy as np
import pandas as pd
import pytest

from credit_risk_scoring.modeling import evaluate, ks_statistic
from credit_risk_scoring.scorecard import Scorecard


def test_perfect_model_metrics():
    y = [0, 0, 1, 1]
    m = evaluate(y, [0.1, 0.2, 0.8, 0.9])
    assert m["AUC"] == pytest.approx(1.0)
    assert m["Gini"] == pytest.approx(1.0)
    assert m["KS"] == pytest.approx(1.0)


def test_random_model_has_gini_near_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000)
    m = evaluate(y, rng.random(20_000))
    assert abs(m["Gini"]) < 0.05
    assert ks_statistic(y, rng.random(20_000)) < 0.05


@pytest.fixture
def synthetic():
    """Le risque baisse avec `score_externe`, monte avec `retards` ; `bruit` n'apporte rien."""
    rng = np.random.default_rng(42)
    n = 50_000  # avec 5 000 lignes, le bruit atteint parfois IV >= 0,02 par hasard (découpage trop fin)
    X = pd.DataFrame({
        "score_externe": rng.random(n),
        "retards": rng.poisson(1, n).astype(float),
        "bruit": rng.random(n),
        "categorie": rng.choice(["A", "B", "C"], n).astype(object),
    })
    logit = -2 - 3 * X["score_externe"] + 0.8 * X["retards"] + (X["categorie"] == "C") * 1.0
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_scorecard_keeps_predictive_variables_and_drops_noise(synthetic):
    X, y = synthetic
    sc = Scorecard().fit(X, y)
    assert {"score_externe", "retards"} <= set(sc.variables)
    assert "bruit" not in sc.variables
    assert (sc.coefficients()["coefficient"] < 0).all()


def test_scorecard_outputs_probabilities(synthetic):
    X, y = synthetic
    pd_ = Scorecard().fit(X, y).predict_proba(X)
    assert pd_.shape == (len(X),)
    assert ((pd_ > 0) & (pd_ < 1)).all()
    assert evaluate(y, pd_)["AUC"] > 0.7


def test_lgbm_learns_and_handles_unseen_categories(synthetic):
    from credit_risk_scoring.lgbm import LGBMModel

    X, y = synthetic
    model = LGBMModel(params={"learning_rate": 0.1}, max_rounds=200, nfold=3).fit(X, y)
    assert evaluate(y, model.predict_proba(X))["AUC"] > 0.7
    assert model.feature_importance().index[-1] == "bruit"

    nouveau = X.head(3).copy()
    nouveau["categorie"] = "Z"   # catégorie jamais vue à l'entraînement : traitée comme manquante
    assert len(model.predict_proba(nouveau)) == 3
