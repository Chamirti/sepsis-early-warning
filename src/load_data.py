"""Load the PhysioNet 2019 .psv files into a DuckDB database.

Run from the project root:  python -m src.load_data
"""
from pathlib import Path
import duckdb
from src.columns import VITALS, LABS, DEMO, LABEL

ROOT = Path(__file__).resolve().parents[1]      # the project folder
RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "sepsis.duckdb"


def main():
    # Declare every column's type explicitly instead of letting DuckDB guess
    types = {c: "DOUBLE" for c in VITALS + LABS + DEMO}
    types["ICULOS"] = "INTEGER"
    types[LABEL] = "INTEGER"
    types_sql = "{" + ", ".join(f"'{c}': '{t}'" for c, t in types.items()) + "}"

    pattern = (RAW / "training_set*" / "*.psv").as_posix()

    con = duckdb.connect(str(DB_PATH))
    con.execute("DROP TABLE IF EXISTS observations")
    con.execute(rf"""
        CREATE TABLE observations AS
        SELECT
            regexp_extract(filename, '(p[0-9]+)\.psv', 1)                AS patient_id,
            CASE WHEN filename LIKE '%training_setA%' THEN 'A' ELSE 'B' END AS hospital,
            * EXCLUDE (filename)
        FROM read_csv('{pattern}',
                      delim = '|',
                      header = true,
                      nullstr = 'NaN',
                      filename = true,
                      types = {types_sql})
        ORDER BY patient_id, ICULOS
    """)

    n_rows = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    n_patients = con.execute("SELECT COUNT(DISTINCT patient_id) FROM observations").fetchone()[0]
    print(f"Loaded {n_rows:,} rows for {n_patients:,} patients into {DB_PATH}")
    con.close()


if __name__ == "__main__":
    main()