"""Téléchargement des données Kaggle et chargement dans DuckDB.

Usage :
    uv run python -m credit_risk_scoring.data download   # télécharge et décompresse dans data/raw
    uv run python -m credit_risk_scoring.data load       # crée data/processed/home_credit.duckdb
    uv run python -m credit_risk_scoring.data check      # vérifie que tout est en place
"""
import sys
import zipfile

import duckdb

from credit_risk_scoring.config import DATA_RAW, DUCKDB_PATH, KAGGLE_COMPETITION, TABLES


def download() -> None:
    """Nécessite d'être connecté (`uv run kaggle auth login`) et d'avoir accepté le règlement de la compétition."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    try:
        api.competition_download_files(KAGGLE_COMPETITION, path=DATA_RAW, quiet=False)
    except Exception as e:
        if "403" in str(e):
            sys.exit(
                "Accès refusé (403) : acceptez le règlement sur "
                f"https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/rules "
                "(bouton « I Understand and Accept »), puis relancez."
            )
        raise
    archive = DATA_RAW / f"{KAGGLE_COMPETITION}.zip"
    with zipfile.ZipFile(archive) as z:
        z.extractall(DATA_RAW)
    archive.unlink()
    print(f"Données décompressées dans {DATA_RAW}")


def missing_tables() -> list[str]:
    return [t for t in TABLES if not (DATA_RAW / f"{t}.csv").exists()]


def load() -> None:
    """Charge chaque CSV comme une table DuckDB (beaucoup plus rapide à requêter ensuite)."""
    if missing := missing_tables():
        sys.exit(f"Fichiers manquants dans {DATA_RAW} : {missing}. Lancez d'abord 'download'.")
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        for t in TABLES:
            csv = (DATA_RAW / f"{t}.csv").as_posix()
            con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM read_csv_auto('{csv}')")
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"{t:<25} {n:>12,} lignes")
    print(f"Base créée : {DUCKDB_PATH}")


def check() -> None:
    missing = missing_tables()
    print("CSV : OK" if not missing else f"CSV manquants : {missing}")
    print("DuckDB : OK" if DUCKDB_PATH.exists() else "DuckDB : pas encore créée")


if __name__ == "__main__":
    commands = {"download": download, "load": load, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]]()
