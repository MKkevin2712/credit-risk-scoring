# Scoring de risque de crédit — Home Credit

> 🚧 Projet en cours (octobre–novembre 2026)

## Objectif

Prédire la **probabilité de défaut de paiement** d'un demandeur de crédit, et transformer cette
probabilité en **décision d'octroi** compréhensible par un métier bancaire.

Le projet compare deux approches :

1. **Scorecard** (découpage WoE / IV + régression logistique) : l'approche classique des banques,
   lisible et auditable.
2. **LightGBM** : un modèle de gradient boosting plus performant, rendu interprétable avec **SHAP**.

Il est ensuite industrialisé : code testé, API FastAPI, conteneur Docker, démo Streamlit et
intégration continue.

## Données

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (Kaggle) :
environ 307 000 demandes de crédit et plusieurs tables d'historique (crédits passés auprès
d'autres organismes, demandes précédentes, remboursements).

| Table | Contenu |
|---|---|
| `application_train` | Une ligne par demande, avec la cible `TARGET` (1 = défaut) |
| `bureau`, `bureau_balance` | Crédits du client auprès d'autres organismes |
| `previous_application` | Demandes précédentes chez Home Credit |
| `installments_payments` | Historique des remboursements |
| `credit_card_balance`, `POS_CASH_balance` | Soldes mensuels des cartes et crédits à la consommation |

Les données ne sont pas publiées dans ce dépôt (règlement Kaggle).

## Plan

- [x] Mise en place du projet
- [x] Analyse exploratoire ([notebook](notebooks/01_exploration.ipynb))
- [x] Features SQL (DuckDB) à partir des tables d'historique : 51 variables ([requêtes](src/credit_risk_scoring/sql/))
- [ ] Scorecard WoE / IV + régression logistique
- [ ] LightGBM, calibration, explicabilité SHAP
- [ ] Seuil de décision métier et grille de score
- [ ] API FastAPI, Docker, démo Streamlit, CI GitHub Actions

## Lancer le projet

Prérequis : [uv](https://docs.astral.sh/uv/) et un compte Kaggle.

```bash
uv sync                                              # environnement Python 3.12 + dépendances
uv run kaggle auth login                             # connexion Kaggle (une seule fois)
uv run python -m credit_risk_scoring.data download   # télécharge les données Kaggle
uv run python -m credit_risk_scoring.data load       # charge les CSV dans DuckDB
uv run python -m credit_risk_scoring.features        # construit data/processed/features.parquet
uv run pytest                                        # tests
uv run jupyter lab                                   # notebooks
```

## Structure

```
data/raw/          données Kaggle (non versionnées)
data/processed/    base DuckDB, features (non versionnées)
notebooks/         analyses exploratoires et résultats
src/credit_risk_scoring/
    config.py      chemins et constantes
    data.py        téléchargement et chargement dans DuckDB
    features.py    assemblage des features (une ligne par demande)
    sql/           une requête d'agrégation par table d'historique
tests/             tests pytest
reports/figures/   graphiques
```

## Auteur

Kevin Moukam Nkameni — [github.com/MKkevin2712](https://github.com/MKkevin2712)
