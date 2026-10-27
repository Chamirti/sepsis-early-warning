"""Build the patients, cohort and cohort_observations tables.

Run from the project root:  python -m src.build_cohort
"""
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "sepsis.duckdb"
SQL_DIR = ROOT / "sql"

# Run in this order: each step depends on the previous one
STEPS = ["patients.sql", "cohort.sql", "cohort_observations.sql"]


def main():
    con = duckdb.connect(str(DB_PATH))
    for name in STEPS:
        con.execute((SQL_DIR / name).read_text())
        print(f"Ran {name}")

    print(con.sql("""
        SELECT hospital, split, COUNT(*) AS patients, SUM(septic) AS septic_patients
        FROM cohort
        WHERE included
        GROUP BY ALL
        ORDER BY ALL
    """))
    con.close()


if __name__ == "__main__":
    main()