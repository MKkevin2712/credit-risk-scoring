-- Soldes mensuels des crédits à la consommation (POS) et prêts personnels chez Home Credit.
SELECT
    SK_ID_CURR,
    COUNT(DISTINCT SK_ID_PREV)                                     AS POS_N_LOANS,
    AVG((SK_DPD > 0)::INTEGER)                                     AS POS_LATE_MONTHS_RATIO,
    MAX(SK_DPD)                                                    AS POS_MAX_DPD,
    -- part du crédit restant à payer lors du dernier relevé connu
    AVG(CNT_INSTALMENT_FUTURE / NULLIF(CNT_INSTALMENT, 0)) FILTER (WHERE MONTHS_BALANCE = -1) AS POS_REMAINING_RATIO
FROM POS_CASH_balance
GROUP BY SK_ID_CURR
