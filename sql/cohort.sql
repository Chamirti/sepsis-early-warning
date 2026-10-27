-- Cohort: inclusion rule + patient-level splits
CREATE OR REPLACE TABLE cohort AS
WITH flagged AS (
    SELECT
        *,
        CASE
            WHEN septic = 1 AND first_label_iculos = first_iculos
                THEN 'septic_from_first_recorded_hour'
        END AS exclusion_reason
    FROM patients
),
included AS (
    SELECT *, (exclusion_reason IS NULL) AS included
    FROM flagged
),
a_ranked AS (
    -- Hospital A: shuffle reproducibly within septic / non-septic groups
    SELECT
        patient_id,
        ROW_NUMBER() OVER (PARTITION BY septic ORDER BY md5(patient_id || '-sepsis-split-v1')) AS rn,
        COUNT(*)     OVER (PARTITION BY septic)                                                AS n
    FROM included
    WHERE hospital = 'A' AND included
)
SELECT
    i.*,
    i.first_label_iculos + 6 AS onset_iculos,      -- label turns on 6 h before onset
    CASE
        WHEN NOT i.included       THEN NULL
        WHEN i.hospital = 'B'     THEN 'external'
        WHEN r.rn <= 0.70 * r.n   THEN 'train'
        WHEN r.rn <= 0.85 * r.n   THEN 'val'
        ELSE 'test'
    END AS split
FROM included i
LEFT JOIN a_ranked r USING (patient_id);