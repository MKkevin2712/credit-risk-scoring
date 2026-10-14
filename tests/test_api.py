"""Tests de l'API et de la cohérence entre l'entraînement (SQL) et la production (Python)."""
import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_risk_scoring.api import create_app
from credit_risk_scoring.features import read_sql
from credit_risk_scoring.lgbm import LGBMModel
from credit_risk_scoring.scoring import REQUIRED, add_derived_features

APPLICATIONS = [
    # salarié classique
    {"DAYS_BIRTH": -12000, "DAYS_EMPLOYED": -2000, "AMT_INCOME_TOTAL": 150000.0, "AMT_CREDIT": 450000.0,
     "AMT_ANNUITY": 22500.0, "AMT_GOODS_PRICE": 400000.0, "CNT_FAM_MEMBERS": 3.0},
    # retraité : code 365243
    {"DAYS_BIRTH": -23000, "DAYS_EMPLOYED": 365243, "AMT_INCOME_TOTAL": 90000.0, "AMT_CREDIT": 200000.0,
     "AMT_ANNUITY": 12000.0, "AMT_GOODS_PRICE": None, "CNT_FAM_MEMBERS": 1.0},
]
DERIVED = ["DAYS_EMPLOYED", "DAYS_EMPLOYED_ANOM", "AGE_YEARS", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO",
           "CREDIT_TERM", "CREDIT_GOODS_RATIO", "INCOME_PER_PERSON", "EMPLOYED_AGE_RATIO"]


def test_python_derived_features_match_sql():
    """Le même calcul en SQL (entraînement) et en Python (API) doit donner exactement les mêmes valeurs."""
    con = duckdb.connect()
    rows = pd.DataFrame([{"SK_ID_CURR": i, "TARGET": 0, "CODE_GENDER": "F", **a} for i, a in enumerate(APPLICATIONS)])
    con.register("application_train", rows)
    sql = con.execute(read_sql("01_application.sql")).df().set_index("SK_ID_CURR")
    for i, application in enumerate(APPLICATIONS):
        python = add_derived_features(application)
        for col in DERIVED:
            expected, got = sql.loc[i, col], python[col]
            if pd.isna(expected):
                assert got is None, col
            else:
                assert got == pytest.approx(float(expected)), col


def test_pensioner_has_no_employment_ratio():
    f = add_derived_features(APPLICATIONS[1])
    assert f["DAYS_EMPLOYED"] is None and f["DAYS_EMPLOYED_ANOM"] == 1
    assert f["EMPLOYED_AGE_RATIO"] is None


@pytest.fixture(scope="module")
def client():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({
        "EXT_SOURCE_2": rng.random(n),
        "AMT_INCOME_TOTAL": rng.uniform(5e4, 3e5, n),
        "AMT_CREDIT": rng.uniform(1e5, 1e6, n),
        "AMT_ANNUITY": rng.uniform(5e3, 5e4, n),
        "DAYS_BIRTH": -rng.uniform(20, 70, n) * 365,
        "CREDIT_TERM": rng.uniform(0.02, 0.12, n),
        "BUREAU_N_CREDITS": rng.poisson(2, n).astype(float),
        "NAME_EDUCATION_TYPE": rng.choice(["Higher education", "Secondary"], n).astype(object),
    })
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(1 + 4 * X["EXT_SOURCE_2"]))).astype(int))
    model = LGBMModel(params={"learning_rate": 0.1}, max_rounds=100, nfold=3).fit(X, y)
    with TestClient(create_app(model)) as c:
        yield c


VALID = {"EXT_SOURCE_2": 0.7, "AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 500000, "AMT_ANNUITY": 25000,
         "DAYS_BIRTH": -14000, "NAME_EDUCATION_TYPE": "Higher education"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["statut"] == "ok"


def test_features_lists_types(client):
    features = {f["nom"]: f for f in client.get("/features").json()}
    assert features["NAME_EDUCATION_TYPE"]["type"] == "categorielle"
    assert features["AMT_CREDIT"]["type"] == "numerique"


def test_predict_returns_a_full_assessment(client):
    r = client.post("/predict", json={"features": VALID})
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["pd"] < 1
    assert body["decision"] in {"ACCEPTÉ", "REFUSÉ"}
    assert body["classe"] in list("ABCDE")
    assert (body["pd"] < body["seuil_pd"]) == (body["decision"] == "ACCEPTÉ")


def test_better_external_score_lowers_risk(client):
    low = client.post("/predict", json={"features": {**VALID, "EXT_SOURCE_2": 0.05}}).json()
    high = client.post("/predict", json={"features": {**VALID, "EXT_SOURCE_2": 0.95}}).json()
    assert high["pd"] < low["pd"] and high["score"] > low["score"]


def test_unknown_variable_is_rejected(client):
    r = client.post("/predict", json={"features": {**VALID, "FOO": 1}})
    assert r.status_code == 422 and "FOO" in r.json()["detail"]


@pytest.mark.parametrize("missing", REQUIRED)
def test_required_variables(client, missing):
    r = client.post("/predict", json={"features": {k: v for k, v in VALID.items() if k != missing}})
    assert r.status_code == 422 and missing in r.json()["detail"]


def test_policy(client):
    policy = client.get("/policy").json()
    assert policy["seuil_pd"] == pytest.approx(policy["marge"] / (policy["marge"] + policy["lgd"]))
    assert [c["classe"] for c in policy["classes"]] == list("ABCDE")
