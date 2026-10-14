import numpy as np
import pandas as pd
import pytest

from credit_risk_scoring import decision as d
from credit_risk_scoring.modeling import evaluate
from credit_risk_scoring.scorecard import Scorecard


def test_breakeven_threshold():
    # gain 10 %, perte 60 % : rentable tant que PD < 0,10 / 0,70
    assert d.breakeven_threshold(0.10, 0.60) == pytest.approx(1 / 7)


def test_expected_profit_counts_only_accepted_loans():
    y = [0, 1, 0, 1]
    pd_ = [0.05, 0.05, 0.50, 0.50]
    amount = [100, 100, 100, 100]
    # seuil 0,2 : on accepte les deux premiers → +10 (remboursé) − 60 (défaut)
    assert d.expected_profit(y, pd_, amount, threshold=0.2) == pytest.approx(-50)
    assert d.expected_profit(y, pd_, amount, threshold=0.0) == 0


def test_score_scale_conventions():
    assert d.pd_to_score(1 / 51) == pytest.approx(d.BASE_SCORE)     # cote 50:1 → 600 points
    pd_half_odds = 1 / (1 + 2 * d.BASE_ODDS)                         # cote doublée (100:1)
    assert d.pd_to_score(pd_half_odds) == pytest.approx(d.BASE_SCORE + d.PDO)
    pds = np.array([0.01, 0.1, 0.5])
    assert np.allclose(d.score_to_pd(d.pd_to_score(pds)), pds)
    assert np.all(np.diff(d.pd_to_score(pds)) < 0)                   # plus risqué = moins de points


def test_risk_grades_follow_pd_bands():
    t = d.breakeven_threshold()
    assert list(d.risk_grade([0.01, 0.03, 0.07, t - 0.001, t + 0.001])) == ["A", "B", "C", "D", "E"]


@pytest.fixture
def fitted_scorecard():
    rng = np.random.default_rng(1)
    n = 20_000
    X = pd.DataFrame({"a": rng.random(n), "b": rng.poisson(1, n).astype(float)})
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(2 + 3 * X["a"] - 0.8 * X["b"]))).astype(int))
    return Scorecard().fit(X, y), X, y


def test_points_grid_reproduces_model_score(fitted_scorecard):
    sc, X, y = fitted_scorecard
    assert np.allclose(d.scorecard_score(sc, X), d.pd_to_score(sc.predict_proba(X)))
    points = d.scorecard_points(sc)
    assert set(points["variable"]) == set(sc.variables)
    # « a » protège : sa meilleure classe rapporte plus de points que sa pire
    pa = points[points["variable"] == "a"].sort_values("taux_defaut")
    assert pa["points"].iloc[0] > pa["points"].iloc[-1]
    assert evaluate(y, -d.scorecard_score(sc, X))["AUC"] > 0.65
