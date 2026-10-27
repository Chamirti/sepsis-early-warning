-- Hourly rows for included patients, tagged with their split
CREATE OR REPLACE VIEW cohort_observations AS
SELECT o.*, c.split
FROM observations o
JOIN cohort c USING (patient_id)
WHERE c.included;