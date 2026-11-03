"""Feature engineering.

feature_sql(source) returns a SQL query that computes one row of features per
patient-hour from an hourly table, using only that hour and earlier hours.

Build the features table from the project root:  python -m src.features
"""
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "sepsis.duckdb"

# Variables used. Dropped: EtCO2 (never measured at Hospital A) and labs missing
# >= 98% of hours at both hospitals (AST, Alkalinephos, Bilirubin_direct,
# Bilirubin_total, Fibrinogen, TroponinI).
VITAL_FEATS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LAB_FEATS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "BUN", "Calcium",
             "Chloride", "Creatinine", "Glucose", "Lactate", "Magnesium", "Phosphate",
             "Potassium", "Hct", "Hgb", "PTT", "WBC", "Platelets"]
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

# Columns that are identifiers or the label, never model inputs
META = ["patient_id", "hospital", "split", "ICULOS", "SepsisLabel"]

# Feature groups for the ablation experiments
ORDERING_FEATURES = ([f"{l}_hrs_since" for l in LAB_FEATS]
                     + [f"{l}_n24" for l in LAB_FEATS]
                     + ["lab_draws24"])
TIME_FEATURES = ["ICULOS"]


def feature_sql(source: str, keep=("hospital", "split", "SepsisLabel")) -> str:
    """SQL computing features for every row of `source`.

    `source` needs patient_id, ICULOS, the vital/lab/static columns, and the
    columns listed in `keep`, which are passed through unchanged.
    """
    keep = list(keep)

    # Step 1: window aggregates over each patient's past
    agg = []
    for v in VITAL_FEATS:
        agg += [
            f"AVG({v}) OVER w6 AS {v}_mean6",
            f"MIN({v}) OVER w6 AS {v}_min6",
            f"MAX({v}) OVER w6 AS {v}_max6",
            f"AVG({v}) OVER w24 AS {v}_mean24",
            f"COUNT({v}) OVER w6 AS {v}_n6",
            f"LAST_VALUE({v} IGNORE NULLS) OVER wall AS {v}_last",
            f"ICULOS - LAST_VALUE(CASE WHEN {v} IS NOT NULL THEN ICULOS END IGNORE NULLS) OVER wall AS {v}_hrs_since_raw",
        ]
    for l in LAB_FEATS:
        agg += [
            f"LAST_VALUE({l} IGNORE NULLS) OVER wall AS {l}_last",
            f"ICULOS - LAST_VALUE(CASE WHEN {l} IS NOT NULL THEN ICULOS END IGNORE NULLS) OVER wall AS {l}_hrs_since_raw",
            f"FIRST_VALUE({l} IGNORE NULLS) OVER w24 AS {l}_first24",
            f"COUNT({l}) OVER w24 AS {l}_n24",
        ]
    agg += [
        "SUM(CASE WHEN MAP < 65 THEN 1 ELSE 0 END) OVER w12 AS hrs_map_lt65_12",
        "SUM(CASE WHEN HR > 90 THEN 1 ELSE 0 END) OVER w12 AS hrs_hr_gt90_12",
        "SUM(CASE WHEN Resp > 20 THEN 1 ELSE 0 END) OVER w12 AS hrs_resp_gt20_12",
        "COUNT(*) OVER w24 AS hours_available24",
    ]

    # Step 2: features derived from the aggregates
    out = ["patient_id", "ICULOS"] + keep + STATIC
    for v in VITAL_FEATS:
        out += [f"{v}_mean6", f"{v}_min6", f"{v}_max6", f"{v}_mean24", f"{v}_n6",
                f"{v}_last",
                f"CASE WHEN {v}_hrs_since_raw > 24 THEN 24 ELSE {v}_hrs_since_raw END AS {v}_hrs_since",
                f"{v}_mean6 - {v}_mean24 AS {v}_delta6_24"]
    for l in LAB_FEATS:
        out += [f"{l}_last",
                f"CASE WHEN {l}_hrs_since_raw > 24 THEN 24 ELSE {l}_hrs_since_raw END AS {l}_hrs_since",
                f"{l}_n24",
                f"{l}_last - {l}_first24 AS {l}_change24"]
    out += [
        "HR_last / NULLIF(SBP_last, 0) AS shock_index",
        "(COALESCE(HR_last > 90, false)::INT"
        " + COALESCE(Resp_last > 20, false)::INT"
        " + COALESCE(Temp_last > 38 OR Temp_last < 36, false)::INT"
        " + COALESCE(WBC_last > 12 OR WBC_last < 4, false)::INT) AS warning_criteria_count",
        "hrs_map_lt65_12", "hrs_hr_gt90_12", "hrs_resp_gt20_12", "hours_available24",
        " + ".join(f"{l}_n24" for l in LAB_FEATS) + " AS lab_draws24",
    ]

    base_cols = ["patient_id", "ICULOS"] + keep + STATIC
    nl = ",\n        "
    return f"""
WITH agg AS (
    SELECT
        {nl.join(base_cols + agg)}
    FROM {source}
    WINDOW
        w6   AS (PARTITION BY patient_id ORDER BY ICULOS RANGE BETWEEN 5 PRECEDING AND CURRENT ROW),
        w12  AS (PARTITION BY patient_id ORDER BY ICULOS RANGE BETWEEN 11 PRECEDING AND CURRENT ROW),
        w24  AS (PARTITION BY patient_id ORDER BY ICULOS RANGE BETWEEN 23 PRECEDING AND CURRENT ROW),
        wall AS (PARTITION BY patient_id ORDER BY ICULOS RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
SELECT
    {nl.join(out)}
FROM agg
"""


def main():
    sql = feature_sql("cohort_observations")
    out_dir = ROOT / "sql" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "features.sql").write_text(sql)       # saved so you can read the full query

    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE features AS " + sql + " ORDER BY patient_id, ICULOS")
    n_rows = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    cols = [r[0] for r in con.execute("DESCRIBE features").fetchall()]
    n_features = len([c for c in cols if c not in META])
    print(f"features table: {n_rows:,} rows, {n_features} feature columns (+ {len(META)} meta)")
    con.close()


if __name__ == "__main__":
    main()