-- Demande de crédit : nettoyage et ratios métier (une ligne par client).
-- Corrections issues de l'analyse exploratoire (notebooks/01_exploration.ipynb).
SELECT
    * EXCLUDE (CODE_GENDER, DAYS_EMPLOYED),   -- CODE_GENDER : donnée sensible, exclue du modèle

    -- 365243 = code « sans emploi déclaré » (surtout des retraités), pas une vraie durée
    CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE DAYS_EMPLOYED END AS DAYS_EMPLOYED,
    (DAYS_EMPLOYED = 365243)::INTEGER                          AS DAYS_EMPLOYED_ANOM,

    -DAYS_BIRTH / 365.25                                       AS AGE_YEARS,
    AMT_CREDIT  / NULLIF(AMT_INCOME_TOTAL, 0)                  AS CREDIT_INCOME_RATIO,
    AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0)                  AS ANNUITY_INCOME_RATIO,   -- taux d'effort
    AMT_ANNUITY / NULLIF(AMT_CREDIT, 0)                        AS CREDIT_TERM,            -- inverse de la durée
    AMT_CREDIT  / NULLIF(AMT_GOODS_PRICE, 0)                   AS CREDIT_GOODS_RATIO,     -- part financée
    AMT_INCOME_TOTAL / NULLIF(CNT_FAM_MEMBERS, 0)              AS INCOME_PER_PERSON,
    DAYS_EMPLOYED / NULLIF(DAYS_BIRTH, 0)                      AS EMPLOYED_AGE_RATIO
FROM application_train
