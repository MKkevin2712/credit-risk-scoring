"""Du risque à la décision : seuil d'octroi, rentabilité et grille de points.

Hypothèses économiques (illustratives, à ajuster avec le métier) :
- un crédit remboursé rapporte MARGIN × montant (intérêts nets du coût du financement) ;
- un crédit en défaut fait perdre LGD × montant (LGD = Loss Given Default, part non récupérée).

On accorde le crédit si le gain espéré est positif :
    (1 − PD) × MARGIN − PD × LGD > 0   ⟺   PD < MARGIN / (MARGIN + LGD)
"""
import numpy as np
import pandas as pd

MARGIN = 0.10   # gain net si le client rembourse : 10 % du montant prêté
LGD = 0.60      # perte si le client fait défaut : 60 % du montant prêté

# Grille de points : convention classique des scorecards bancaires
BASE_SCORE = 600   # score attribué à…
BASE_ODDS = 50     # … une cote de 50 bons payeurs pour 1 mauvais
PDO = 20           # Points to Double the Odds : +20 points = cote doublée
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def breakeven_threshold(margin: float = MARGIN, lgd: float = LGD) -> float:
    """PD au-delà de laquelle un crédit fait perdre de l'argent en moyenne."""
    return margin / (margin + lgd)


def expected_profit(y_true, pd_, amount, threshold: float, margin: float = MARGIN, lgd: float = LGD):
    """Gain réel du portefeuille si l'on accepte les demandes dont la PD est sous le seuil."""
    y_true, pd_, amount = map(np.asarray, (y_true, pd_, amount))
    accepted = pd_ < threshold
    gains = np.where(y_true == 1, -lgd * amount, margin * amount)
    return float(gains[accepted].sum())


def threshold_table(y_true, pd_, amount, thresholds, margin: float = MARGIN, lgd: float = LGD) -> pd.DataFrame:
    """Pour chaque seuil : taux d'acceptation, défauts parmi les acceptés, mauvais payeurs évités, gain."""
    y_true, pd_ = np.asarray(y_true), np.asarray(pd_)
    rows = []
    for t in thresholds:
        acc = pd_ < t
        rows.append({
            "seuil_PD": t,
            "taux_acceptation": acc.mean(),
            "taux_defaut_acceptes": y_true[acc].mean() if acc.any() else np.nan,
            "mauvais_payeurs_refuses": (~acc & (y_true == 1)).sum() / max((y_true == 1).sum(), 1),
            "bons_payeurs_refuses": (~acc & (y_true == 0)).sum() / max((y_true == 0).sum(), 1),
            "gain": expected_profit(y_true, pd_, amount, t, margin, lgd),
        })
    return pd.DataFrame(rows)


def pd_to_score(pd_) -> np.ndarray:
    """Probabilité de défaut → score en points (plus le score est haut, moins le client est risqué)."""
    pd_ = np.clip(np.asarray(pd_, dtype=float), 1e-9, 1 - 1e-9)
    return OFFSET + FACTOR * np.log((1 - pd_) / pd_)


def score_to_pd(score) -> np.ndarray:
    return 1 / (1 + np.exp((np.asarray(score, dtype=float) - OFFSET) / FACTOR))


def risk_grades(threshold: float | None = None) -> pd.DataFrame:
    """Classes de risque définies par des tranches de PD ; la classe E correspond aux refus."""
    threshold = breakeven_threshold() if threshold is None else threshold
    bands = [
        ("A", "Risque très faible", 0.0, 0.02),
        ("B", "Risque faible", 0.02, 0.05),
        ("C", "Risque modéré", 0.05, 0.10),
        ("D", "Risque élevé, accepté", 0.10, threshold),
        ("E", "Risque trop élevé, refusé", threshold, 1.0),
    ]
    grades = pd.DataFrame(bands, columns=["classe", "libelle", "pd_min", "pd_max"])
    grades["score_min"] = pd_to_score(grades["pd_max"])
    grades["score_max"] = pd_to_score(grades["pd_min"])
    grades.loc[grades.index[0], "score_max"] = np.inf    # PD = 0 : pas de plafond de score
    grades.loc[grades.index[-1], "score_min"] = -np.inf  # PD = 1 : pas de plancher
    return grades


def risk_grade(pd_, threshold: float | None = None) -> np.ndarray:
    grades = risk_grades(threshold)
    idx = np.searchsorted(grades["pd_max"].to_numpy()[:-1], np.asarray(pd_, dtype=float), side="right")
    return grades["classe"].to_numpy()[idx]


def scorecard_points(scorecard) -> pd.DataFrame:
    """Grille de points de la scorecard : nombre de points pour chaque classe de chaque variable.

    Le score d'un client = somme des points de ses classes. Comme la régression logistique donne
    ln(cote des mauvais) = α + Σ βᵢ·WoEᵢ, on a score = OFFSET − FACTOR × (α + Σ βᵢ·WoEᵢ),
    que l'on répartit entre les n variables.
    """
    coefs = dict(zip(scorecard.variables, scorecard.model.coef_[0]))
    intercept = scorecard.model.intercept_[0]
    n = len(scorecard.variables)
    base = (OFFSET - FACTOR * intercept) / n
    rows = []
    for var in scorecard.variables:
        table = scorecard.binning_table(var).iloc[:-1]   # sans la ligne « Totals »
        for _, r in table.iterrows():
            if r["Count"] == 0:
                continue
            rows.append({
                "variable": var,
                "classe": str(r["Bin"]),
                "part_clients": float(r["Count (%)"]),
                "taux_defaut": float(r["Event rate"]),
                "points": base - FACTOR * coefs[var] * float(r["WoE"]),
            })
    return pd.DataFrame(rows)


def scorecard_score(scorecard, X: pd.DataFrame) -> np.ndarray:
    """Score de chaque demande calculé comme une somme de points (identique à pd_to_score(PD))."""
    woe = scorecard.binning.transform(X, metric="woe")[scorecard.variables]
    logit_bad = scorecard.model.intercept_[0] + woe.to_numpy() @ scorecard.model.coef_[0]
    return OFFSET - FACTOR * logit_bad
