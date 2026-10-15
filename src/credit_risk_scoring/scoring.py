"""Évaluation d'une demande de crédit : PD, score, classe, décision et principales raisons.

Utilisé à la fois par l'API (api.py) et par la démo Streamlit (app/demo.py).
"""
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd

from credit_risk_scoring import decision
from credit_risk_scoring.lgbm import MODEL_PATH, LGBMModel

# Libellés lisibles des variables les plus importantes (les autres gardent leur nom technique)
LABELS = {
    "EXT_SOURCE_1": "Score externe 1",
    "EXT_SOURCE_2": "Score externe 2",
    "EXT_SOURCE_3": "Score externe 3",
    "ORGANIZATION_TYPE": "Type d'employeur",
    "OCCUPATION_TYPE": "Métier",
    "NAME_EDUCATION_TYPE": "Niveau d'études",
    "CREDIT_TERM": "Mensualité / montant du crédit",
    "CREDIT_GOODS_RATIO": "Montant du crédit / prix du bien",
    "ANNUITY_INCOME_RATIO": "Taux d'effort (mensualité / revenu)",
    "CREDIT_INCOME_RATIO": "Montant du crédit / revenu",
    "DAYS_EMPLOYED": "Ancienneté dans l'emploi",
    "DAYS_BIRTH": "Âge",
    "AGE_YEARS": "Âge",
    "AMT_ANNUITY": "Mensualité",
    "AMT_CREDIT": "Montant du crédit",
    "AMT_GOODS_PRICE": "Prix du bien financé",
    "AMT_INCOME_TOTAL": "Revenu",
    "INST_LATE_RATIO": "Part des échéances payées en retard",
    "INST_LATE_RATIO_LAST_YEAR": "Retards de paiement sur la dernière année",
    "INST_N_INSTALMENTS": "Nombre d'échéances déjà remboursées",
    "PREV_REFUSED_RATIO": "Part des demandes précédentes refusées",
    "PREV_CREDIT_APPLICATION_RATIO": "Montant accordé / demandé par le passé",
    "PREV_MEAN_N_PAYMENTS": "Durée moyenne des crédits précédents",
    "BUREAU_DEBT_CREDIT_RATIO": "Endettement auprès d'autres organismes",
    "BUREAU_ACTIVE_RATIO": "Part de crédits en cours ailleurs",
    "CC_MEAN_UTILIZATION": "Utilisation du plafond de carte",
    "CC_MAX_UTILIZATION": "Utilisation maximale du plafond de carte",
    "CC_MEAN_ATM_DRAWINGS": "Retraits d'espèces par carte",
    "PREV_LAST_REFUSAL_DAYS": "Ancienneté du dernier refus de crédit",
    "PREV_APPROVED_RATIO": "Part des demandes précédentes acceptées",
    "INST_LATE_30_RATIO": "Part des échéances payées avec plus de 30 jours de retard",
    "INST_MAX_DAYS_LATE": "Retard de paiement maximal",
    "INST_TOTAL_UNPAID": "Montant total impayé",
    "BUREAU_LAST_CREDIT_DAYS": "Ancienneté du dernier crédit ouvert ailleurs",
    "BUREAU_OLDEST_CREDIT_DAYS": "Ancienneté du premier crédit ouvert ailleurs",
    "BUREAU_LATE_MONTHS_RATIO": "Mois en retard sur les crédits ailleurs",
    "BUREAU_N_CREDITS": "Nombre de crédits ailleurs",
    "POS_REMAINING_RATIO": "Part restant à payer sur les crédits conso",
    "POS_LATE_MONTHS_RATIO": "Mois en retard sur les crédits conso",
    "DAYS_REGISTRATION": "Ancienneté de l'adresse déclarée",
    "DAYS_ID_PUBLISH": "Ancienneté de la pièce d'identité",
    "DAYS_LAST_PHONE_CHANGE": "Dernier changement de téléphone",
    "OWN_CAR_AGE": "Âge du véhicule",
    "NAME_CONTRACT_TYPE": "Type de crédit",
    "NAME_INCOME_TYPE": "Type de revenu",
    "NAME_FAMILY_STATUS": "Situation familiale",
    "REGION_RATING_CLIENT": "Note de la région",
    "REGION_RATING_CLIENT_W_CITY": "Note de la région (avec la ville)",
    "DAYS_EMPLOYED_ANOM": "Sans emploi déclaré (retraité)",
    "EMPLOYED_AGE_RATIO": "Ancienneté dans l'emploi / âge",
    "INCOME_PER_PERSON": "Revenu par personne du foyer",
    "FLAG_OWN_CAR": "Possède une voiture",
    "FLAG_OWN_REALTY": "Possède un logement",
    "CNT_FAM_MEMBERS": "Taille du foyer",
    "REGION_POPULATION_RELATIVE": "Densité de population de la région",
    "HOUR_APPR_PROCESS_START": "Heure de la demande",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Consultations du bureau de crédit (1 an)",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Défauts dans l'entourage",
    "PREV_MEAN_CREDIT": "Montant moyen des crédits précédents",
    "PREV_CANCELED_RATIO": "Part des demandes précédentes annulées",
    "BUREAU_MAX_AMT_OVERDUE": "Plus gros impayé auprès d'autres organismes",
    "FLAG_DOCUMENT_3": "Justificatif n° 3 fourni",
}


# Variables minimales pour évaluer une demande : sans elles, les ratios métier ne peuvent pas être calculés
REQUIRED = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH"]

# Compteurs d'historique : non fournis = client sans historique = 0 (comme à l'entraînement)
COUNT_PREFIXES = ("BUREAU_N_", "PREV_N_", "INST_N_", "CC_N_", "POS_N_")


def _div(a, b):
    return None if a is None or b in (None, 0) else a / b


def add_derived_features(features: dict) -> dict:
    """Recalcule les variables dérivées exactement comme sql/01_application.sql.

    Indispensable en production : le modèle a appris sur ces variables ; si l'API ne les recalculait pas,
    elles arriveraient vides et les prédictions seraient faussées (« training-serving skew »).
    Un test vérifie que ce code et la requête SQL donnent les mêmes résultats.
    """
    f = dict(features)
    days_employed = f.get("DAYS_EMPLOYED")
    anomaly = days_employed == 365243
    if days_employed is not None:
        f["DAYS_EMPLOYED_ANOM"] = int(anomaly)
        f["DAYS_EMPLOYED"] = None if anomaly else days_employed
    days_birth = f.get("DAYS_BIRTH")
    income, credit = f.get("AMT_INCOME_TOTAL"), f.get("AMT_CREDIT")
    annuity, goods = f.get("AMT_ANNUITY"), f.get("AMT_GOODS_PRICE")
    f["AGE_YEARS"] = None if days_birth is None else -days_birth / 365.25
    f["CREDIT_INCOME_RATIO"] = _div(credit, income)
    f["ANNUITY_INCOME_RATIO"] = _div(annuity, income)
    f["CREDIT_TERM"] = _div(annuity, credit)
    f["CREDIT_GOODS_RATIO"] = _div(credit, goods)
    f["INCOME_PER_PERSON"] = _div(income, f.get("CNT_FAM_MEMBERS"))
    f["EMPLOYED_AGE_RATIO"] = _div(f.get("DAYS_EMPLOYED"), days_birth)
    return f


@dataclass
class Reason:
    variable: str
    libelle: str
    valeur: object
    effet: float          # contribution SHAP (log-odds) : > 0 augmente le risque


@dataclass
class Assessment:
    pd: float
    score: float
    classe: str
    decision: str
    seuil_pd: float
    raisons_risque: list[Reason]
    raisons_favorables: list[Reason]

    def to_dict(self) -> dict:
        return asdict(self)


def load_model(path=MODEL_PATH) -> LGBMModel:
    return joblib.load(path)


def to_frame(model: LGBMModel, features: dict) -> pd.DataFrame:
    """Dictionnaire {variable: valeur} → tableau d'une ligne avec toutes les colonnes du modèle.

    Les variables dérivées (ratios, âge…) sont recalculées ; les compteurs d'historique absents valent 0 ;
    les autres variables absentes sont considérées comme manquantes, ce que LightGBM sait gérer.
    """
    unknown = set(features) - set(model.features)
    if unknown:
        raise ValueError(f"Variables inconnues : {sorted(unknown)}")
    missing = [c for c in REQUIRED if features.get(c) is None]
    if missing:
        raise ValueError(f"Variables obligatoires manquantes : {missing}")
    features = add_derived_features(features)
    for col in model.features:
        if col.startswith(COUNT_PREFIXES) and features.get(col) is None:
            features[col] = 0
    row = {}
    for col in model.features:
        value = features.get(col)
        if col in model.categories:
            row[col] = value if value is not None else None
        else:
            row[col] = np.nan if value is None else float(value)
    df = pd.DataFrame([row], columns=model.features)
    for col in model.categories:
        df[col] = df[col].astype(object)
    return df


def _reason(model: LGBMModel, X: pd.DataFrame, name: str, effect: float) -> Reason:
    value = X[name].iloc[0]
    if isinstance(value, float) and np.isnan(value):
        value = None
    elif isinstance(value, (np.floating, np.integer)):
        value = value.item()
    return Reason(name, LABELS.get(name, name), value, round(float(effect), 4))


def assess(model: LGBMModel, X: pd.DataFrame, n_reasons: int = 5) -> Assessment:
    """Évalue une demande (tableau d'une ligne)."""
    pd_ = float(model.predict_proba(X)[0])
    threshold = decision.breakeven_threshold()
    contrib = model.booster.predict(model._prepare(X[model.features]), pred_contrib=True)[0][:-1]
    order = np.argsort(contrib)
    risk = [_reason(model, X, model.features[i], contrib[i]) for i in order[::-1][:n_reasons] if contrib[i] > 0]
    good = [_reason(model, X, model.features[i], contrib[i]) for i in order[:n_reasons] if contrib[i] < 0]
    return Assessment(
        pd=round(pd_, 6),
        score=round(float(decision.pd_to_score(pd_)), 1),
        classe=str(decision.risk_grade([pd_])[0]),
        decision="ACCEPTÉ" if pd_ < threshold else "REFUSÉ",
        seuil_pd=round(threshold, 6),
        raisons_risque=risk,
        raisons_favorables=good,
    )
