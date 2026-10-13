"""Chemins et constantes du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
DUCKDB_PATH = DATA_PROCESSED / "home_credit.duckdb"

KAGGLE_COMPETITION = "home-credit-default-risk"
TARGET = "TARGET"   # 1 = défaut de paiement, 0 = remboursé
ID = "SK_ID_CURR"   # identifiant client
SEED = 42

# Tables attendues dans data/raw après téléchargement
TABLES = [
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "previous_application",
    "installments_payments",
    "credit_card_balance",
    "POS_CASH_balance",
]
