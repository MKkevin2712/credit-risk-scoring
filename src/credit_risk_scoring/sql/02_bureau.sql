-- Crédits du client auprès d'autres organismes (bureau de crédit).
-- bureau_balance.STATUS : '0' = à jour, '1' à '5' = retard croissant (5 = > 120 jours ou passé en perte),
-- 'C' = clôturé, 'X' = inconnu.
WITH balance_by_credit AS (
    SELECT
        SK_ID_BUREAU,
        COUNT(*) FILTER (WHERE STATUS IN ('0', '1', '2', '3', '4', '5'))     AS n_known_months,
        COUNT(*) FILTER (WHERE STATUS IN ('1', '2', '3', '4', '5'))          AS n_late_months,
        MAX(CASE WHEN STATUS IN ('1', '2', '3', '4', '5') THEN STATUS::INTEGER ELSE 0 END) AS worst_status
    FROM bureau_balance
    GROUP BY SK_ID_BUREAU
)
SELECT
    b.SK_ID_CURR,
    COUNT(*)                                                       AS BUREAU_N_CREDITS,
    COUNT(*) FILTER (WHERE b.CREDIT_ACTIVE = 'Active')             AS BUREAU_N_ACTIVE,
    AVG((b.CREDIT_ACTIVE = 'Active')::INTEGER)                     AS BUREAU_ACTIVE_RATIO,
    COUNT(*) FILTER (WHERE b.CREDIT_ACTIVE = 'Bad debt')           AS BUREAU_N_BAD_DEBT,
    AVG((b.CREDIT_TYPE = 'Credit card')::INTEGER)                  AS BUREAU_CREDIT_CARD_RATIO,
    SUM(b.AMT_CREDIT_SUM)                                          AS BUREAU_TOTAL_CREDIT,
    SUM(b.AMT_CREDIT_SUM_DEBT)                                     AS BUREAU_TOTAL_DEBT,
    SUM(b.AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(b.AMT_CREDIT_SUM), 0)  AS BUREAU_DEBT_CREDIT_RATIO,
    SUM(b.AMT_CREDIT_SUM_OVERDUE)                                  AS BUREAU_TOTAL_OVERDUE,
    MAX(b.CREDIT_DAY_OVERDUE)                                      AS BUREAU_MAX_DAYS_OVERDUE,
    MAX(b.AMT_CREDIT_MAX_OVERDUE)                                  AS BUREAU_MAX_AMT_OVERDUE,
    SUM(b.CNT_CREDIT_PROLONG)                                      AS BUREAU_N_PROLONGATIONS,
    MIN(b.DAYS_CREDIT)                                             AS BUREAU_OLDEST_CREDIT_DAYS,
    MAX(b.DAYS_CREDIT)                                             AS BUREAU_LAST_CREDIT_DAYS,
    -- historique mensuel des remboursements
    SUM(bb.n_late_months) / NULLIF(SUM(bb.n_known_months), 0)      AS BUREAU_LATE_MONTHS_RATIO,
    MAX(bb.worst_status)                                           AS BUREAU_WORST_STATUS
FROM bureau AS b
LEFT JOIN balance_by_credit AS bb USING (SK_ID_BUREAU)
GROUP BY b.SK_ID_CURR
