-- Soldes mensuels des cartes de crédit chez Home Credit. SK_DPD = jours de retard.
SELECT
    SK_ID_CURR,
    COUNT(DISTINCT SK_ID_PREV)                                     AS CC_N_CARDS,
    AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))          AS CC_MEAN_UTILIZATION,   -- part du plafond utilisée
    MAX(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))          AS CC_MAX_UTILIZATION,
    AVG(AMT_DRAWINGS_ATM_CURRENT)                                  AS CC_MEAN_ATM_DRAWINGS,  -- retraits d'espèces
    AVG((SK_DPD > 0)::INTEGER)                                     AS CC_LATE_MONTHS_RATIO,
    MAX(SK_DPD)                                                    AS CC_MAX_DPD
FROM credit_card_balance
GROUP BY SK_ID_CURR
