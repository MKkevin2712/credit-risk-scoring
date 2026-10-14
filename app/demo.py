"""Démo interactive du modèle de scoring crédit.

Lancer :
    uv run streamlit run app/demo.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from credit_risk_scoring import decision
from credit_risk_scoring.features import FEATURES_PATH
from credit_risk_scoring.modeling import split
from credit_risk_scoring.scoring import assess, load_model, to_frame

st.set_page_config(page_title="Scoring crédit", page_icon="🏦", layout="wide")

GRADE_COLORS = {"A": "#2E7D32", "B": "#66BB6A", "C": "#F9A825", "D": "#EF6C00", "E": "#C62828"}
EDUCATION = ["Higher education", "Secondary / secondary special", "Incomplete higher",
             "Lower secondary", "Academic degree"]


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_test_clients(n: int = 300) -> pd.DataFrame | None:
    """Échantillon de demandes du jeu de test (jamais vues par le modèle), avec leur issue réelle."""
    if not FEATURES_PATH.exists():
        return None
    from credit_risk_scoring.modeling import load_features

    df = load_features()
    _, X_test, _, y_test = split(df)
    sample = X_test.sample(n, random_state=7).copy()
    sample["TARGET"] = y_test.loc[sample.index]
    sample["SK_ID_CURR"] = df.loc[sample.index, "SK_ID_CURR"]
    return sample.set_index("SK_ID_CURR")


def clean(value):
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else value


model = get_model()
clients = get_test_clients()

st.title("🏦 Scoring de risque de crédit")
st.caption("Modèle LightGBM entraîné sur les données Home Credit · décision fondée sur la rentabilité "
           f"(marge {decision.MARGIN:.0%}, perte en cas de défaut {decision.LGD:.0%})")

# ---------- Choix de la demande ----------
with st.sidebar:
    st.header("Demande de crédit")
    base: dict = {}
    if clients is not None:
        client_id = st.selectbox("Client du jeu de test", clients.index,
                                 help="Demandes jamais vues par le modèle ; on connaît leur issue réelle.")
        row = clients.loc[client_id]
        base = {c: clean(row[c]) for c in model.features}
        st.caption("Issue réelle : " + ("❌ défaut de paiement" if row["TARGET"] == 1 else "✅ crédit remboursé"))
        st.divider()
        st.subheader("Simuler une modification")
    else:
        st.info("Données non disponibles : saisie manuelle.")

    def num(label, key, default, lo, hi, step, fmt=None):
        value = base.get(key)
        value = default if value is None else float(np.clip(value, lo, hi))
        return st.slider(label, lo, hi, value, step, format=fmt)

    ext = {k: st.checkbox(f"Score externe {k[-1]} connu", value=base.get(k) is not None or not base, key=f"has_{k}")
           for k in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]}
    values = {}
    for k, known in ext.items():
        values[k] = num(f"Score externe {k[-1]}", k, 0.5, 0.0, 1.0, 0.01) if known else None
    values["AMT_INCOME_TOTAL"] = num("Revenu annuel", "AMT_INCOME_TOTAL", 150_000.0, 20_000.0, 1_000_000.0, 5_000.0, "%d")
    values["AMT_CREDIT"] = num("Montant du crédit", "AMT_CREDIT", 500_000.0, 40_000.0, 3_000_000.0, 10_000.0, "%d")
    values["AMT_ANNUITY"] = num("Mensualité", "AMT_ANNUITY", 25_000.0, 2_000.0, 200_000.0, 500.0, "%d")
    age = num("Âge (ans)", "AGE_YEARS", 40.0, 20.0, 70.0, 1.0, "%d")
    values["DAYS_BIRTH"] = -age * 365.25
    retired = base.get("DAYS_EMPLOYED_ANOM") == 1
    if not st.checkbox("Sans emploi / retraité", value=retired):
        seniority = num("Ancienneté dans l'emploi (ans)", "_seniority",
                        -(base.get("DAYS_EMPLOYED") or -1825) / 365.25, 0.0, 40.0, 0.5)
        values["DAYS_EMPLOYED"] = -seniority * 365.25
    else:
        values["DAYS_EMPLOYED"] = 365243
    education = base.get("NAME_EDUCATION_TYPE") or EDUCATION[1]
    values["NAME_EDUCATION_TYPE"] = st.selectbox("Niveau d'études", EDUCATION, index=EDUCATION.index(education)
                                                 if education in EDUCATION else 1)

features = {**base, **values}
# Les variables dérivées sont recalculées à partir des valeurs saisies
for derived in ["AGE_YEARS", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
                "CREDIT_GOODS_RATIO", "INCOME_PER_PERSON", "EMPLOYED_AGE_RATIO", "DAYS_EMPLOYED_ANOM"]:
    features.pop(derived, None)
if features.get("AMT_GOODS_PRICE") is None:
    features["AMT_GOODS_PRICE"] = features["AMT_CREDIT"]

result = assess(model, to_frame(model, features), n_reasons=6)

# ---------- Résultat ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Probabilité de défaut", f"{result.pd:.1%}")
c2.metric("Score", f"{result.score:.0f} points")
color = GRADE_COLORS[result.classe]
c3.markdown(f"**Classe de risque**<br><span style='font-size:2.2rem;color:{color};font-weight:700'>"
            f"{result.classe}</span>", unsafe_allow_html=True)
c4.markdown(f"**Décision**<br><span style='font-size:1.6rem;font-weight:700;color:"
            f"{'#2E7D32' if result.decision == 'ACCEPTÉ' else '#C62828'}'>{result.decision}</span>",
            unsafe_allow_html=True)

st.progress(min(result.pd / 0.4, 1.0),
            text=f"PD {result.pd:.1%} · seuil de refus {result.seuil_pd:.1%} ({decision.pd_to_score(result.seuil_pd):.0f} points)")

st.subheader("Pourquoi cette décision ?")
st.caption("Contribution de chaque variable au risque du client (valeurs SHAP). "
           "Rouge : augmente le risque · vert : le diminue.")
reasons = result.raisons_risque + result.raisons_favorables
if reasons:
    def show(value):
        if value is None:
            return "manquante"
        if isinstance(value, (int, float)):
            return f"{value:,.0f}".replace(",", " ") if abs(value) >= 1000 else f"{value:.3g}"
        return str(value)

    df = pd.DataFrame([{"variable": r.libelle, "effet": r.effet, "valeur": show(r.valeur)} for r in reasons])
    df = df.sort_values("effet")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 0.6))
    ax.barh(df["variable"], df["effet"], color=["#C62828" if e > 0 else "#2E7D32" for e in df["effet"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Effet sur le risque (log-odds)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    st.pyplot(fig)
    st.dataframe(df[::-1].set_index("variable"), width="stretch")

with st.expander("Classes de risque et règle d'octroi"):
    grades = decision.risk_grades()
    st.dataframe(grades.style.format({"pd_min": "{:.1%}", "pd_max": "{:.1%}",
                                      "score_min": "{:.0f}", "score_max": "{:.0f}"}),
                 hide_index=True, width="stretch")
    st.markdown(f"Un crédit est rentable si **(1 − PD) × {decision.MARGIN:.0%} > PD × {decision.LGD:.0%}**, "
                f"soit **PD < {decision.breakeven_threshold():.1%}**.")
