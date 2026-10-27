-- One row per patient: static info and label summary
CREATE OR REPLACE TABLE patients AS
SELECT
    patient_id,
    hospital,
    ANY_VALUE(Age)          AS age,
    ANY_VALUE(Gender)       AS gender,
    ANY_VALUE(Unit1)        AS unit1,
    ANY_VALUE(Unit2)        AS unit2,
    ANY_VALUE(HospAdmTime)  AS hosp_adm_time,
    MIN(ICULOS)             AS first_iculos,
    MAX(ICULOS)             AS last_iculos,
    COUNT(*)                AS n_rows,
    MAX(SepsisLabel)        AS septic,
    MIN(ICULOS) FILTER (WHERE SepsisLabel = 1) AS first_label_iculos
FROM observations
GROUP BY patient_id, hospital;