"""Démo interactive du modèle de scoring crédit.

Lancer en local :
    uv run streamlit run app/demo.py

En local, la démo propose des clients réels du jeu de test (si data/processed/features.parquet existe).
En ligne, les données Kaggle ne sont pas publiées (règlement de la compétition) : la démo propose
alors des profils types, modifiables avec les curseurs.
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Sur Streamlit Community Cloud, le package n'est pas installé : on l'importe depuis src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk_scoring import decision
from credit_risk_scoring.features import FEATURES_PATH
from credit_risk_scoring.scoring import assess, load_model, to_frame

st.set_page_config(page_title="Scoring crédit", page_icon=":material/account_balance:", layout="wide")

GRADE_COLORS = {"A": "green", "B": "green", "C": "yellow", "D": "orange", "E": "red"}
EDUCATION = ["Higher education", "Secondary / secondary special", "Incomplete higher",
             "Lower secondary", "Academic degree"]
YEAR = 365.25

# Profils types (fictifs) utilisés quand les données réelles ne sont pas disponibles
PROFILES = {
    "Profil solide": {
        "EXT_SOURCE_1": 0.75, "EXT_SOURCE_2": 0.72, "EXT_SOURCE_3": 0.70, "AMT_INCOME_TOTAL": 250_000,
        "AMT_CREDIT": 400_000, "AMT_ANNUITY": 20_000, "AGE_YEARS": 48, "DAYS_EMPLOYED": -12 * YEAR,
        "NAME_EDUCATION_TYPE": "Higher education",
    },
    "Profil moyen": {
        "EXT_SOURCE_1": 0.50, "EXT_SOURCE_2": 0.55, "EXT_SOURCE_3": 0.50, "AMT_INCOME_TOTAL": 150_000,
        "AMT_CREDIT": 600_000, "AMT_ANNUITY": 30_000, "AGE_YEARS": 38, "DAYS_EMPLOYED": -4 * YEAR,
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    },
    "Profil fragile": {
        "EXT_SOURCE_2": 0.25, "EXT_SOURCE_3": 0.20, "AMT_INCOME_TOTAL": 100_000,
        "AMT_CREDIT": 900_000, "AMT_ANNUITY": 45_000, "AGE_YEARS": 26, "DAYS_EMPLOYED": -0.5 * YEAR,
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    },
    "Retraité": {
        "EXT_SOURCE_2": 0.60, "EXT_SOURCE_3": 0.65, "AMT_INCOME_TOTAL": 90_000, "AMT_CREDIT": 200_000,
        "AMT_ANNUITY": 12_000, "AGE_YEARS": 63, "DAYS_EMPLOYED_ANOM": 1,
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    },
}


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data(max_entries=1)
def get_test_clients(n: int = 300) -> pd.DataFrame | None:
    """Échantillon de demandes du jeu de test (jamais vues par le modèle), avec leur issue réelle."""
    if os.environ.get("DEMO_PROFILS_SEULEMENT") or not FEATURES_PATH.exists():
        return None
    from credit_risk_scoring.modeling import load_features, split

    df = load_features()
    _, X_test, _, y_test = split(df)
    sample = X_test.sample(n, random_state=7).copy()
    sample["TARGET"] = y_test.loc[sample.index]
    sample["SK_ID_CURR"] = df.loc[sample.index, "SK_ID_CURR"]
    return sample.set_index("SK_ID_CURR")


def clean(value):
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else value


def show(value) -> str:
    if value is None:
        return "manquante"
    if isinstance(value, (int, float)):
        return f"{value:,.0f}".replace(",", " ") if abs(value) >= 1000 else f"{value:.3g}"
    return str(value)


model = get_model()
clients = get_test_clients()

st.title("Scoring de risque de crédit")
st.caption("Modèle LightGBM entraîné sur 246 000 demandes de crédit (Home Credit) · décision fondée sur la "
           f"rentabilité : marge {decision.MARGIN:.0%}, perte en cas de défaut {decision.LGD:.0%}")

# ---------- Choix de la demande ----------
with st.sidebar:
    st.header("Demande de crédit")
    sources = ["Clients réels", "Profils types"] if clients is not None else ["Profils types"]
    source = st.segmented_control("Source", sources, default=sources[0], required=True)

    if source == "Clients réels":
        client_id = st.selectbox("Client du jeu de test", clients.index,
                                 help="Demandes jamais vues par le modèle ; on connaît leur issue réelle.")
        row = clients.loc[client_id]
        base = {c: clean(row[c]) for c in model.features}
        st.caption("Issue réelle : " + ("défaut de paiement :material/close:" if row["TARGET"] == 1
                                        else "crédit remboursé :material/check:"))
        key = str(client_id)
    else:
        profile = st.selectbox("Profil", list(PROFILES), help="Profils fictifs, modifiables ci-dessous.")
        base = dict(PROFILES[profile])
        key = profile

    st.subheader("Simuler une modification")

    def num(label, name, default, lo, hi, step, fmt=None):
        value = base.get(name)
        value = default if value is None else float(np.clip(value, lo, hi))
        return st.slider(label, float(lo), float(hi), float(value), float(step), format=fmt, key=f"{name}_{key}")

    values = {}
    for name in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        known = st.toggle(f"Score externe {name[-1]} connu", value=base.get(name) is not None,
                          key=f"has_{name}_{key}")
        values[name] = num(f"Score externe {name[-1]}", name, 0.5, 0.0, 1.0, 0.01) if known else None
    values["AMT_INCOME_TOTAL"] = num("Revenu annuel", "AMT_INCOME_TOTAL", 150_000, 20_000, 1_000_000, 5_000, "%d")
    values["AMT_CREDIT"] = num("Montant du crédit", "AMT_CREDIT", 500_000, 40_000, 3_000_000, 10_000, "%d")
    values["AMT_ANNUITY"] = num("Mensualité", "AMT_ANNUITY", 25_000, 2_000, 200_000, 500, "%d")
    values["DAYS_BIRTH"] = -num("Âge (ans)", "AGE_YEARS", 40, 20, 70, 1, "%d") * YEAR
    retired = base.get("DAYS_EMPLOYED_ANOM") == 1
    if st.toggle("Sans emploi ou retraité", value=retired, key=f"retired_{key}"):
        values["DAYS_EMPLOYED"] = 365243
    else:
        seniority = -(base.get("DAYS_EMPLOYED") or -5 * YEAR) / YEAR
        base["_SENIORITY"] = seniority
        values["DAYS_EMPLOYED"] = -num("Ancienneté dans l'emploi (ans)", "_SENIORITY", 5, 0, 40, 0.5) * YEAR
    education = base.get("NAME_EDUCATION_TYPE")
    values["NAME_EDUCATION_TYPE"] = st.selectbox(
        "Niveau d'études", EDUCATION, index=EDUCATION.index(education) if education in EDUCATION else 1,
        key=f"education_{key}",
    )

features = {k: v for k, v in {**base, **values}.items() if k in model.features}
# Les variables dérivées sont recalculées à partir des valeurs saisies (comme dans l'API)
for derived in ["AGE_YEARS", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
                "CREDIT_GOODS_RATIO", "INCOME_PER_PERSON", "EMPLOYED_AGE_RATIO", "DAYS_EMPLOYED_ANOM"]:
    features.pop(derived, None)
if features.get("AMT_GOODS_PRICE") is None:
    features["AMT_GOODS_PRICE"] = 0.9 * features["AMT_CREDIT"]

result = assess(model, to_frame(model, features), n_reasons=6)

# ---------- Résultat ----------
with st.container(horizontal=True):
    st.metric("Probabilité de défaut", f"{result.pd:.1%}", border=True)
    st.metric("Score", f"{result.score:.0f} points", border=True)
    st.metric("Classe de risque", result.classe, border=True)
    st.metric("Décision", result.decision, border=True)

accepted = result.decision == "ACCEPTÉ"
st.badge(f"Classe {result.classe} · {decision.risk_grades().set_index('classe').loc[result.classe, 'libelle']}",
         color=GRADE_COLORS[result.classe])
st.progress(min(result.pd / 0.4, 1.0),
            text=f"PD {result.pd:.1%} · seuil de refus {result.seuil_pd:.1%} "
                 f"({decision.pd_to_score(result.seuil_pd):.0f} points)")
if accepted:
    st.success("La demande est rentable en moyenne pour la banque : crédit accordé.", icon=":material/check_circle:")
else:
    st.error("Le risque de défaut dépasse le seuil de rentabilité : crédit refusé.", icon=":material/block:")

st.subheader("Pourquoi cette décision ?")
st.caption("Contribution de chaque variable au risque de ce client (valeurs SHAP). "
           "Rouge : augmente le risque · vert : le diminue.")
reasons = result.raisons_risque + result.raisons_favorables
if reasons:
    df = pd.DataFrame([{"variable": r.libelle, "effet": r.effet, "valeur": show(r.valeur)} for r in reasons])
    df = df.sort_values("effet")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 0.6))
    ax.barh(df["variable"], df["effet"], color=["#C62828" if e > 0 else "#2E7D32" for e in df["effet"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Effet sur le risque (log-odds)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    st.pyplot(fig, alt="Contribution des principales variables au risque du client")
    st.dataframe(df[::-1].set_index("variable"), alt="Détail des contributions par variable")

with st.expander("Classes de risque et règle d'octroi"):
    grades = decision.risk_grades()
    st.dataframe(grades.style.format({"pd_min": "{:.1%}", "pd_max": "{:.1%}",
                                      "score_min": "{:.0f}", "score_max": "{:.0f}"}),
                 hide_index=True, alt="Classes de risque A à E")
    st.markdown(f"Un crédit est rentable si **(1 − PD) × {decision.MARGIN:.0%} > PD × {decision.LGD:.0%}**, "
                f"soit **PD < {decision.breakeven_threshold():.1%}**.")

st.caption("Projet de Kevin Moukam Nkameni · [code source sur GitHub](https://github.com/MKkevin2712/credit-risk-scoring)")
