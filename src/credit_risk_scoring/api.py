"""API de scoring : renvoie la PD, le score, la classe de risque, la décision et ses raisons.

Lancer en local :
    uv run uvicorn credit_risk_scoring.api:app --reload
Documentation interactive : http://127.0.0.1:8000/docs
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_risk_scoring import decision
from credit_risk_scoring.lgbm import MODEL_PATH, LGBMModel
from credit_risk_scoring.scoring import LABELS, assess, load_model, to_frame


class CreditApplication(BaseModel):
    """Caractéristiques de la demande. Les variables absentes sont traitées comme manquantes."""

    features: dict[str, float | str | None] = Field(
        examples=[{
            "EXT_SOURCE_1": 0.45, "EXT_SOURCE_2": 0.62, "EXT_SOURCE_3": 0.51,
            "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 500000, "AMT_ANNUITY": 25000,
            "AMT_GOODS_PRICE": 450000, "DAYS_BIRTH": -14000, "DAYS_EMPLOYED": -2500,
            "NAME_EDUCATION_TYPE": "Higher education", "ORGANIZATION_TYPE": "Business Entity Type 3",
        }],
    )


def create_app(model: LGBMModel | None = None) -> FastAPI:
    """Crée l'application ; un modèle peut être fourni (tests), sinon il est chargé au démarrage."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model or load_model(os.environ.get("MODEL_PATH", MODEL_PATH))
        yield

    app = FastAPI(
        title="API de scoring crédit",
        description="Probabilité de défaut, score, classe de risque et décision d'octroi (modèle LightGBM).",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        m = app.state.model
        return {"statut": "ok", "modele": "LightGBM", "variables": len(m.features), "arbres": m.n_rounds}

    @app.get("/features")
    def features() -> list[dict]:
        """Liste des variables attendues, avec leur type et, pour les catégorielles, les valeurs connues."""
        m = app.state.model
        return [
            {
                "nom": f,
                "libelle": LABELS.get(f, f),
                "type": "categorielle" if f in m.categories else "numerique",
                **({"valeurs": list(m.categories[f])} if f in m.categories else {}),
            }
            for f in m.features
        ]

    @app.get("/policy")
    def policy() -> dict:
        """Règle d'octroi et classes de risque en vigueur."""
        grades = decision.risk_grades().replace([float("inf"), float("-inf")], None)
        return {
            "marge": decision.MARGIN,
            "lgd": decision.LGD,
            "seuil_pd": decision.breakeven_threshold(),
            "seuil_score": float(decision.pd_to_score(decision.breakeven_threshold())),
            "classes": grades.to_dict(orient="records"),
        }

    @app.post("/predict")
    def predict(application: CreditApplication) -> dict:
        try:
            X = to_frame(app.state.model, application.features)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return assess(app.state.model, X).to_dict()

    return app


app = create_app()
