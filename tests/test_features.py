"""Tests des requêtes SQL sur de petites tables fictives dont on connaît le résultat attendu."""
import duckdb
import pandas as pd
import pytest

from credit_risk_scoring.features import build_features


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE application_train AS SELECT * FROM (VALUES
            (1, 0, 'F', -10000, -1000,  100000.0, 300000.0, 15000.0, 250000.0, 2.0),
            (2, 1, 'M', -20000, 365243, 50000.0,  100000.0, 10000.0, 100000.0, 1.0)
        ) t(SK_ID_CURR, TARGET, CODE_GENDER, DAYS_BIRTH, DAYS_EMPLOYED, AMT_INCOME_TOTAL,
            AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, CNT_FAM_MEMBERS)
    """)
    # Client 1 : 2 crédits au bureau (1 actif), dont un avec 1 mois de retard sur 2 mois connus
    con.execute("""
        CREATE TABLE bureau AS SELECT * FROM (VALUES
            (1, 10, 'Active', 'Consumer credit', -500, 0, 1000.0, 400.0, 0.0, 0.0, 0),
            (1, 11, 'Closed', 'Credit card',     -900, 0, 1000.0, 0.0,   0.0, 0.0, 0)
        ) t(SK_ID_CURR, SK_ID_BUREAU, CREDIT_ACTIVE, CREDIT_TYPE, DAYS_CREDIT, CREDIT_DAY_OVERDUE,
            AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT, AMT_CREDIT_SUM_OVERDUE, AMT_CREDIT_MAX_OVERDUE,
            CNT_CREDIT_PROLONG)
    """)
    con.execute("""
        CREATE TABLE bureau_balance AS SELECT * FROM (VALUES
            (10, -1, '0'), (10, -2, '2'), (10, -3, 'X'), (11, -1, 'C')
        ) t(SK_ID_BUREAU, MONTHS_BALANCE, STATUS)
    """)
    con.execute("""
        CREATE TABLE previous_application AS SELECT * FROM (VALUES
            (100, 1, 'Approved', 1000.0, 900.0, 12.0, -300),
            (101, 1, 'Refused',  2000.0, 0.0,   NULL, -100)
        ) t(SK_ID_PREV, SK_ID_CURR, NAME_CONTRACT_STATUS, AMT_APPLICATION, AMT_CREDIT, CNT_PAYMENT,
            DAYS_DECISION)
    """)
    # Échéance 1 payée en deux fois, la seconde 5 jours en retard ; échéance 2 payée à temps mais incomplète
    con.execute("""
        CREATE TABLE installments_payments AS SELECT * FROM (VALUES
            (100, 1, 1, 1, -100, -102, 500.0, 300.0),
            (100, 1, 1, 1, -100,  -95, 500.0, 200.0),
            (100, 1, 1, 2,  -70,  -70, 500.0, 400.0)
        ) t(SK_ID_PREV, SK_ID_CURR, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER, DAYS_INSTALMENT,
            DAYS_ENTRY_PAYMENT, AMT_INSTALMENT, AMT_PAYMENT)
    """)
    con.execute("""
        CREATE TABLE credit_card_balance AS SELECT * FROM (VALUES
            (200, 1, -1, 500.0, 1000.0, 0.0, 0),
            (200, 1, -2, 1000.0, 1000.0, 100.0, 3)
        ) t(SK_ID_PREV, SK_ID_CURR, MONTHS_BALANCE, AMT_BALANCE, AMT_CREDIT_LIMIT_ACTUAL,
            AMT_DRAWINGS_ATM_CURRENT, SK_DPD)
    """)
    con.execute("""
        CREATE TABLE POS_CASH_balance AS SELECT * FROM (VALUES
            (300, 1, -1, 12.0, 3.0, 0), (300, 1, -2, 12.0, 4.0, 10)
        ) t(SK_ID_PREV, SK_ID_CURR, MONTHS_BALANCE, CNT_INSTALMENT, CNT_INSTALMENT_FUTURE, SK_DPD)
    """)
    yield con
    con.close()


@pytest.fixture
def features(con):
    return build_features(con).set_index("SK_ID_CURR")


def test_one_row_per_application(features):
    assert len(features) == 2 and features.index.is_unique


def test_gender_is_excluded(features):
    assert "CODE_GENDER" not in features.columns


def test_days_employed_anomaly(features):
    assert features.loc[2, "DAYS_EMPLOYED_ANOM"] == 1
    assert pd.isna(features.loc[2, "DAYS_EMPLOYED"])
    assert features.loc[1, "DAYS_EMPLOYED"] == -1000


def test_application_ratios(features):
    assert features.loc[1, "CREDIT_INCOME_RATIO"] == pytest.approx(3.0)
    assert features.loc[1, "ANNUITY_INCOME_RATIO"] == pytest.approx(0.15)
    assert features.loc[1, "CREDIT_TERM"] == pytest.approx(0.05)


def test_bureau_aggregates(features):
    assert features.loc[1, "BUREAU_N_CREDITS"] == 2
    assert features.loc[1, "BUREAU_ACTIVE_RATIO"] == pytest.approx(0.5)
    assert features.loc[1, "BUREAU_DEBT_CREDIT_RATIO"] == pytest.approx(0.2)
    assert features.loc[1, "BUREAU_LATE_MONTHS_RATIO"] == pytest.approx(0.5)  # 'X' et 'C' ignorés
    assert features.loc[1, "BUREAU_WORST_STATUS"] == 2


def test_previous_applications(features):
    assert features.loc[1, "PREV_REFUSED_RATIO"] == pytest.approx(0.5)
    assert features.loc[1, "PREV_LAST_REFUSAL_DAYS"] == -100


def test_installments_group_partial_payments(features):
    assert features.loc[1, "INST_N_INSTALMENTS"] == 2           # 3 paiements, 2 échéances
    assert features.loc[1, "INST_MAX_DAYS_LATE"] == 5           # dernier paiement à -95 pour -100
    assert features.loc[1, "INST_LATE_RATIO"] == pytest.approx(0.5)
    assert features.loc[1, "INST_TOTAL_UNPAID"] == pytest.approx(100.0)


def test_credit_card_and_pos(features):
    assert features.loc[1, "CC_MEAN_UTILIZATION"] == pytest.approx(0.75)
    assert features.loc[1, "CC_MAX_DPD"] == 3
    assert features.loc[1, "POS_REMAINING_RATIO"] == pytest.approx(0.25)


def test_client_without_history_has_zero_counts(features):
    for col in ["BUREAU_N_CREDITS", "PREV_N_APPLICATIONS", "INST_N_INSTALMENTS", "CC_N_CARDS", "POS_N_LOANS"]:
        assert features.loc[2, col] == 0
    assert pd.isna(features.loc[2, "INST_LATE_RATIO"])  # inconnu, pas 0
